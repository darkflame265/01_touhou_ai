# env/obs_builder.py
import time
import cv2
import numpy as np

from env.reimu_detector import ReimuDetector


class ObsBuilder:
    """
    ✅ 2-Stream 8채널 관측 (Global 4ch + Local 4ch)
      - Global:
        g0: playfield_gray (0..1)
        g1: absdiff(playfield_gray, prev_playfield_gray) (0..1)
        g2: bullet_mask_global (0..1)
        g3: risk_global (0..1)

      - Local:
        l0: crop_gray (0..1) + meta pixels(x,y,conf)
        l1: absdiff(crop_gray, prev_crop_gray) (0..1)
        l2: bullet_mask_local (0..1)
        l3: risk_centered_local (0..1)

    (OPT)
      - heavy(HSV/DT)는 heavy_every 프레임마다만 갱신, 나머지는 캐시 재사용
      - Global heavy는 global_proc_size에서만 계산 후 obs_out_size로 업샘플
      - Local heavy는 obs_out_size에서만 계산(=crop_size에서 DT 금지)
      - det도 det_every 프레임마다만 호출 + 불안정(저conf/lost)일 때만 강제 det
      - debug 창은 show_obs_debug=False면 완전 미사용
      - 프로파일은 obs_prof_enable=True면 obs_prof_every마다 출력(출력 후 구간 리셋)
    """

    # -------------------------
    # init
    # -------------------------
    def __init__(self, screen, debug_viz=None, obs_out_size=128, crop_size=256, use_fallback_full_preprocess=True):
        self.screen = screen
        self.debug = debug_viz

        self.obs_out_size = int(obs_out_size)
        self.crop_size = int(crop_size)
        self.use_fallback_full_preprocess = bool(use_fallback_full_preprocess)

        img0 = self.screen.capture()
        self.H, self.W = img0.shape[:2]

        # playfield width 캐시
        self._playfield_ratio = float(getattr(self.screen, "PLAYFIELD_RIGHT_RATIO", 0.70))
        self._playfield_w = max(1, min(self.W, int(self.W * self._playfield_ratio)))

        # -------------------------
        # Detector
        # -------------------------
        self.det = ReimuDetector(
            screen=self.screen,
            weight_path="weights/reimu_heatmap_best.pt",
            beta=12.0,
            prior_strength=1.0,
            ema_alpha=0.85,
            device=None,
            track_prior_strength=2.0,
            track_prior_sigma=0.08,
            lock_conf_thr=0.015,
            max_jump_norm=0.22,
            jump_allow_conf_gain=1.8,
            lost_patience=8,
            use_fp16=True,
            track_prior_every=2,
            print_prof=True,
            prof_every=200,
        )

        # -------------------------
        # State
        # -------------------------
        self.player_center = (self.W // 2, int(self.H * 0.78))
        self.last_xy_norm = (0.5, 0.78)
        self.last_conf = 0.0

        # gating
        self.conf_update_thr = 0.02
        self.max_jump_norm_obs = 0.18
        self.jump_allow_conf_gain_obs = 2.0
        self.lost_patience_obs = 10
        self._lost_obs = 0

        self.edge_margin_norm = 0.035
        self.edge_jump_conf_gain = 3.0
        self.edge_jump_min_conf = 0.90

        # local crop smoothing
        self.center_max_speed_px = 10.0
        self.center_micro_ema_alpha = 0.0
        self._crop_center_f = None

        # meta pixels
        self.meta_patch = 4

        # prev frames
        self._prev_crop_gray_u8 = None
        self._prev_play_gray_u8 = None

        # -------------------------
        # Bullet / Risk params
        # -------------------------
        self.enable_bullet_channels = True
        self.bullet_hsv_s_min = 40
        self.bullet_hsv_v_min = 140
        self.bullet_hsv_v_max = 255
        self.bullet_close_morph = 0

        self.risk_tau_px_local = 8.0
        self.risk_tau_px_global = 10.0
        self.center_sigma_px = float(self.crop_size) * 0.35
        self.risk_clip_max = 1.0

        # risk hold
        self.risk_decay = 0.70
        self._risk_hold_crop_01 = None

        # crop clamp + valid mask
        self.clamp_crop_inside = True
        self.enable_valid_mask = True
        self.valid_mask_soft_blur = 7

        # -------------------------
        # OPT knobs
        # -------------------------
        self.heavy_every = 2
        self.det_every = 4
        self.global_proc_size = (96, 72)  # (W,H)
        self._step_i = 0

        # caches (all obs_out_size float32 0..1)
        self._g_bullet_cache_01 = None
        self._g_risk_cache_01 = None
        self._l_bullet_cache_01 = None
        self._l_risk_cache_01 = None

        # detector cache
        self._last_det = None  # (x_n, y_n, conf, logits)

        # debug windows
        self.show_obs_debug = False
        self.win_local_risk = "OBS_LOCAL_RISK"
        self.win_global_risk = "OBS_GLOBAL_RISK"
        self._obs_win_inited = False
        self._obs_win_pos_local = (1600, 60)
        self._obs_win_pos_global = (1600, 720)
        self._obs_win_size = (520, 520)

        # profiling
        self.obs_prof_enable = True
        self.obs_prof_every = 200
        self._prof_cnt = 0
        self._prof_sum = None
        self._prof_reset()

    # -------------------------
    # lifecycle
    # -------------------------
    def reset(self):
        if hasattr(self.det, "reset"):
            self.det.reset()

        self.player_center = (self.W // 2, int(self.H * 0.78))
        self.last_xy_norm = (0.5, 0.78)
        self.last_conf = 0.0
        self._lost_obs = 0

        self._prev_crop_gray_u8 = None
        self._prev_play_gray_u8 = None
        self._crop_center_f = None
        self._risk_hold_crop_01 = None

        self._step_i = 0
        self._g_bullet_cache_01 = None
        self._g_risk_cache_01 = None
        self._l_bullet_cache_01 = None
        self._l_risk_cache_01 = None

        self._last_det = None

        self._prof_cnt = 0
        self._prof_reset()

    def on_player_death(self):
        try:
            if hasattr(self.det, "on_player_death"):
                self.det.on_player_death()
        except Exception:
            pass

    # -------------------------
    # small helpers
    # -------------------------
    def _prof_reset(self):
        self._prof_sum = {
            "total": 0.0,
            "det": 0.0,
            "g_graydiff": 0.0,
            "g_heavy": 0.0,
            "g_cache": 0.0,
            "l_crop": 0.0,
            "l_graydiff": 0.0,
            "l_heavy": 0.0,
            "l_cache": 0.0,
            "meta_stack": 0.0,
            "debug": 0.0,
        }

    def _tick(self, key: str, t0: float):
        if self.obs_prof_enable:
            self._prof_sum[key] += (time.perf_counter() - t0)

    def _maybe_print_prof(self, do_heavy: bool):
        if not self.obs_prof_enable:
            return
        self._prof_cnt += 1
        if (self._prof_cnt % int(self.obs_prof_every)) != 0:
            return

        def ms(k):  # average over last window
            return (self._prof_sum[k] / float(self.obs_prof_every)) * 1000.0

        print(
            "[OBS_PROF] avg_ms/call | "
            f"total={ms('total'):.2f} det={ms('det'):.2f} "
            f"g(gray+diff)={ms('g_graydiff'):.2f} g(heavy)={ms('g_heavy'):.2f} g(cache)={ms('g_cache'):.2f} | "
            f"l(crop)={ms('l_crop'):.2f} l(gray+diff)={ms('l_graydiff'):.2f} l(heavy)={ms('l_heavy'):.2f} l(cache)={ms('l_cache'):.2f} | "
            f"meta+stack={ms('meta_stack'):.2f} dbg={ms('debug'):.2f} "
            f"(heavy_every={self.heavy_every}, do_heavy={do_heavy})"
        )
        self._prof_reset()

    def _ensure_obs_window(self):
        if self._obs_win_inited:
            return
        try:
            cv2.namedWindow(self.win_local_risk, cv2.WINDOW_NORMAL)
            cv2.moveWindow(self.win_local_risk, int(self._obs_win_pos_local[0]), int(self._obs_win_pos_local[1]))
            cv2.resizeWindow(self.win_local_risk, int(self._obs_win_size[0]), int(self._obs_win_size[1]))

            cv2.namedWindow(self.win_global_risk, cv2.WINDOW_NORMAL)
            cv2.moveWindow(self.win_global_risk, int(self._obs_win_pos_global[0]), int(self._obs_win_pos_global[1]))
            cv2.resizeWindow(self.win_global_risk, int(self._obs_win_size[0]), int(self._obs_win_size[1]))
        except Exception:
            pass
        self._obs_win_inited = True

    @staticmethod
    def _is_near_edge(x_n: float, y_n: float, m: float) -> bool:
        return (x_n <= m) or (x_n >= 1.0 - m) or (y_n <= m) or (y_n >= 1.0 - m)

    def _playfield_norm_to_full_xy(self, x_n: float, y_n: float) -> tuple[int, int]:
        cx = int(np.clip(x_n * self._playfield_w, 0, self._playfield_w - 1))
        cy = int(np.clip(y_n * self.H, 0, self.H - 1))
        return cx, cy

    @staticmethod
    def _dist_norm(a_xy, b_xy) -> float:
        dx = float(a_xy[0] - b_xy[0])
        dy = float(a_xy[1] - b_xy[1])
        return float((dx * dx + dy * dy) ** 0.5)

    def _gate_xy_update(self, x_n, y_n, conf):
        prev_xy = self.last_xy_norm
        prev_c = float(self.last_conf)

        x_n = float(np.clip(x_n, 0.0, 1.0))
        y_n = float(np.clip(y_n, 0.0, 1.0))
        conf = float(conf)

        if conf < float(self.conf_update_thr):
            self._lost_obs += 1
            if self._lost_obs >= int(self.lost_patience_obs):
                self._lost_obs = 0
                return (x_n, y_n, conf, True, "FORCE_LOWCONF")
            return (float(prev_xy[0]), float(prev_xy[1]), float(prev_c), False, "LOWCONF_HOLD")

        d = self._dist_norm((x_n, y_n), prev_xy)
        if d > float(self.max_jump_norm_obs):
            need = max(1e-6, prev_c) * float(self.jump_allow_conf_gain_obs)

            m = float(self.edge_margin_norm)
            prev_edge = self._is_near_edge(float(prev_xy[0]), float(prev_xy[1]), m)
            now_edge = self._is_near_edge(x_n, y_n, m)

            if now_edge and (not prev_edge):
                need *= float(self.edge_jump_conf_gain)
                if conf < float(self.edge_jump_min_conf):
                    self._lost_obs += 1
                    if self._lost_obs >= int(self.lost_patience_obs):
                        self._lost_obs = 0
                        return (float(prev_xy[0]), float(prev_xy[1]), float(prev_c), True, "FORCE_EDGE_HOLD")
                    return (float(prev_xy[0]), float(prev_xy[1]), float(prev_c), False, "EDGE_JUMP_REJECT")

            if conf >= need:
                self._lost_obs = 0
                return (x_n, y_n, conf, True, "JUMP_ACCEPT")

            self._lost_obs += 1
            if self._lost_obs >= int(self.lost_patience_obs):
                self._lost_obs = 0
                return (x_n, y_n, conf, True, "FORCE_JUMP")
            return (float(prev_xy[0]), float(prev_xy[1]), float(prev_c), False, "JUMP_REJECT")

        self._lost_obs = 0
        return (x_n, y_n, conf, True, "OK")

    def _inject_meta_pixels_local_ch0(self, local_ch0_01: np.ndarray) -> np.ndarray:
        try:
            x_n, y_n = self.last_xy_norm
            c = float(self.last_conf)

            x_n = float(np.clip(x_n, 0.0, 1.0))
            y_n = float(np.clip(y_n, 0.0, 1.0))
            c = float(np.clip(c, 0.0, 1.0))

            p = int(self.meta_patch)
            if local_ch0_01.shape[0] >= p and local_ch0_01.shape[1] >= p * 3:
                local_ch0_01[0:p, 0:p] = x_n
                local_ch0_01[0:p, p:2 * p] = y_n
                local_ch0_01[0:p, 2 * p:3 * p] = c
        except Exception:
            pass
        return local_ch0_01

    def _clamp_crop_center(self, cx: int, cy: int, size: int) -> tuple[int, int]:
        half = int(size) // 2
        if half <= 0:
            return int(cx), int(cy)
        cx_min, cx_max = half, max(half, self.W - half - 1)
        cy_min, cy_max = half, max(half, self.H - half - 1)
        return int(np.clip(cx, cx_min, cx_max)), int(np.clip(cy, cy_min, cy_max))

    def _crop_square_bgr_with_mask(self, img_bgr, cx, cy, size):
        h, w = img_bgr.shape[:2]
        size = int(size)
        half = size // 2

        x1, y1 = int(cx - half), int(cy - half)
        x2, y2 = x1 + size, y1 + size

        if (0 <= x1) and (0 <= y1) and (x2 <= w) and (y2 <= h):
            crop = img_bgr[y1:y2, x1:x2]
            valid = np.full((size, size), 255, dtype=np.uint8)
            return crop, valid

        pad_l, pad_t = max(0, -x1), max(0, -y1)
        pad_r, pad_b = max(0, x2 - w), max(0, y2 - h)

        img_pad = cv2.copyMakeBorder(img_bgr, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_REFLECT_101)
        base_valid = np.full((h, w), 255, dtype=np.uint8)
        valid_pad = cv2.copyMakeBorder(base_valid, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_CONSTANT, value=0)

        x1p, y1p = x1 + pad_l, y1 + pad_t
        x2p, y2p = x2 + pad_l, y2 + pad_t

        crop = img_pad[y1p:y2p, x1p:x2p]
        valid = valid_pad[y1p:y2p, x1p:x2p]

        if crop.shape[:2] != (size, size):
            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
        if valid.shape[:2] != (size, size):
            valid = cv2.resize(valid, (size, size), interpolation=cv2.INTER_NEAREST)
        return crop, valid

    def _center_speed_limit(self, prev_xy_f, target_xy_i):
        px, py = float(prev_xy_f[0]), float(prev_xy_f[1])
        tx, ty = float(target_xy_i[0]), float(target_xy_i[1])
        dx, dy = tx - px, ty - py
        dist = float((dx * dx + dy * dy) ** 0.5)

        vmax = max(1e-6, float(self.center_max_speed_px))
        if dist <= vmax:
            return (tx, ty)
        s = vmax / dist
        return (px + dx * s, py + dy * s)

    # -------------------------
    # bullet / risk
    # -------------------------
    def _compute_bullet_mask_u8(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        s, v = hsv[:, :, 1], hsv[:, :, 2]

        mask = (s >= int(self.bullet_hsv_s_min)) & (v >= int(self.bullet_hsv_v_min)) & (v <= int(self.bullet_hsv_v_max))
        mask_u8 = (mask.astype(np.uint8) * 255)

        k = int(self.bullet_close_morph)
        if k > 0:
            ksz = 2 * k + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask_u8

    def _compute_risk_heat(self, bullet_mask_u8: np.ndarray, tau_px: float) -> np.ndarray:
        inv = cv2.bitwise_not(bullet_mask_u8)  # 탄=0, 배경=255
        dist = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)

        tau = max(1e-6, float(tau_px))
        risk = np.exp(-dist / tau).astype(np.float32)

        m = float(risk.max())
        if m > 1e-6:
            risk = risk / m
        if self.risk_clip_max is not None:
            risk = np.clip(risk, 0.0, float(self.risk_clip_max))
        return risk.astype(np.float32, copy=False)

    def _center_weight_local(self, risk01: np.ndarray, center_xy=None) -> np.ndarray:
        h, w = risk01.shape[:2]
        if center_xy is None:
            cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
        else:
            cx, cy = float(center_xy[0]), float(center_xy[1])

        cx = float(np.clip(cx, 0.0, w - 1.0))
        cy = float(np.clip(cy, 0.0, h - 1.0))

        yy, xx = np.indices((h, w), dtype=np.float32)
        rr2 = (xx - cx) ** 2 + (yy - cy) ** 2

        # 256 기준 sigma를 128 좌표계로 스케일
        sigma = max(1e-6, float(self.center_sigma_px) * (float(w) / float(self.crop_size)))
        w_center = np.exp(-rr2 / (2.0 * sigma * sigma)).astype(np.float32)

        out = risk01 * w_center
        m = float(out.max())
        if m > 1e-6:
            out = out / m
        if self.risk_clip_max is not None:
            out = np.clip(out, 0.0, float(self.risk_clip_max))
        return out.astype(np.float32, copy=False)

    # -------------------------
    # main
    # -------------------------
    def make_state(self, img_bgr):
        t_all = time.perf_counter()
        self._step_i += 1

        do_heavy = (
            self.enable_bullet_channels
            and (self.heavy_every <= 1 or (self._step_i % int(self.heavy_every) == 0))
        )

        # -------------------------
        # Global gray+diff -> g0,g1
        # -------------------------
        tg = time.perf_counter()
        play_gray = self.screen.get_playfield_gray(img_bgr)

        if self._prev_play_gray_u8 is None or self._prev_play_gray_u8.shape != play_gray.shape:
            play_diff = np.zeros_like(play_gray)
        else:
            play_diff = cv2.absdiff(play_gray, self._prev_play_gray_u8)
        self._prev_play_gray_u8 = play_gray

        interp_g = cv2.INTER_AREA if max(play_gray.shape[:2]) >= self.obs_out_size else cv2.INTER_LINEAR
        g0 = cv2.resize(play_gray, (self.obs_out_size, self.obs_out_size), interpolation=interp_g).astype(np.float32) / 255.0
        g1 = cv2.resize(play_diff, (self.obs_out_size, self.obs_out_size), interpolation=interp_g).astype(np.float32) / 255.0
        self._tick("g_graydiff", tg)

        # -------------------------
        # Global heavy -> g2,g3 (cached)
        # -------------------------
        if do_heavy:
            th = time.perf_counter()
            try:
                play_bgr = self.screen.get_playfield_bgr(img_bgr)
            except Exception:
                pw = int(self._playfield_w)
                play_bgr = img_bgr[:, :pw]

            gpw, gph = int(self.global_proc_size[0]), int(self.global_proc_size[1])
            play_small = cv2.resize(play_bgr, (gpw, gph), interpolation=cv2.INTER_AREA)

            g_bullet_u8 = self._compute_bullet_mask_u8(play_small)
            g_risk = self._compute_risk_heat(g_bullet_u8, tau_px=float(self.risk_tau_px_global))

            g2 = cv2.resize(g_bullet_u8, (self.obs_out_size, self.obs_out_size), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
            g3 = cv2.resize(g_risk, (self.obs_out_size, self.obs_out_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)

            self._g_bullet_cache_01, self._g_risk_cache_01 = g2, g3
            self._tick("g_heavy", th)
        else:
            tc = time.perf_counter()
            if self._g_bullet_cache_01 is None:
                self._g_bullet_cache_01 = np.zeros((self.obs_out_size, self.obs_out_size), dtype=np.float32)
            if self._g_risk_cache_01 is None:
                self._g_risk_cache_01 = np.zeros((self.obs_out_size, self.obs_out_size), dtype=np.float32)
            g2, g3 = self._g_bullet_cache_01, self._g_risk_cache_01
            self._tick("g_cache", tc)

        # -------------------------
        # Detector (conditional) -> player_center
        # -------------------------
        td = time.perf_counter()
        force_det = (float(self.last_conf) < 0.02) or (int(self._lost_obs) >= 2)
        do_det = force_det or (self.det_every <= 1) or (self._step_i % int(self.det_every) == 0)

        if do_det:
            self._last_det = self.det.step(img_bgr)
        det = self._last_det
        self._tick("det", td)

        if det is None:
            cx, cy = self.player_center
        else:
            if not do_det:
                cx, cy = self.player_center
            else:
                x_n, y_n, conf, _ = det
                x_use, y_use, c_use, used, _ = self._gate_xy_update(x_n, y_n, conf)
                self.last_xy_norm = (float(x_use), float(y_use))
                self.last_conf = float(c_use)

                cx_new, cy_new = self._playfield_norm_to_full_xy(x_use, y_use)
                if used:
                    cx, cy = cx_new, cy_new
                    self.player_center = (cx, cy)
                else:
                    cx, cy = self.player_center

        # -------------------------
        # Crop + valid mask (prep to 128)
        # -------------------------
        tcrop = time.perf_counter()
        cx_i, cy_i = int(cx), int(cy)

        if self._crop_center_f is None:
            self._crop_center_f = (float(cx_i), float(cy_i))
        else:
            nx, ny = self._center_speed_limit(self._crop_center_f, (cx_i, cy_i))
            a = float(self.center_micro_ema_alpha)
            if a > 0.0:
                px, py = self._crop_center_f
                nx = a * px + (1.0 - a) * nx
                ny = a * py + (1.0 - a) * ny
            self._crop_center_f = (nx, ny)

        cx_s = int(round(self._crop_center_f[0]))
        cy_s = int(round(self._crop_center_f[1]))
        if self.clamp_crop_inside:
            cx_s, cy_s = self._clamp_crop_center(cx_s, cy_s, self.crop_size)
            self._crop_center_f = (float(cx_s), float(cy_s))

        crop_bgr, valid_u8 = self._crop_square_bgr_with_mask(img_bgr, cx_s, cy_s, self.crop_size)

        valid01 = None
        if self.enable_valid_mask:
            v = (valid_u8.astype(np.float32) / 255.0)
            k = int(self.valid_mask_soft_blur)
            if k and k >= 3 and (k % 2 == 1):
                v = cv2.GaussianBlur(v, (k, k), 0)
            v = np.clip(v, 0.0, 1.0).astype(np.float32, copy=False)
            valid01 = cv2.resize(v, (self.obs_out_size, self.obs_out_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        self._tick("l_crop", tcrop)

        # -------------------------
        # Local gray+diff -> l0,l1
        # -------------------------
        tl = time.perf_counter()
        crop_gray_u8 = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if self._prev_crop_gray_u8 is None or self._prev_crop_gray_u8.shape != crop_gray_u8.shape:
            diff_u8 = np.zeros_like(crop_gray_u8)
        else:
            diff_u8 = cv2.absdiff(crop_gray_u8, self._prev_crop_gray_u8)
        self._prev_crop_gray_u8 = crop_gray_u8

        interp_l = cv2.INTER_AREA if self.crop_size >= self.obs_out_size else cv2.INTER_LINEAR
        l0 = cv2.resize(crop_gray_u8, (self.obs_out_size, self.obs_out_size), interpolation=interp_l).astype(np.float32) / 255.0
        l1 = cv2.resize(diff_u8, (self.obs_out_size, self.obs_out_size), interpolation=interp_l).astype(np.float32) / 255.0
        self._tick("l_graydiff", tl)

        # -------------------------
        # Local heavy -> l2,l3 (cached)
        # -------------------------
        if do_heavy:
            tlh = time.perf_counter()
            l_bgr_small = cv2.resize(crop_bgr, (self.obs_out_size, self.obs_out_size), interpolation=cv2.INTER_AREA)

            l_bullet_u8 = self._compute_bullet_mask_u8(l_bgr_small)
            l_risk_base = self._compute_risk_heat(l_bullet_u8, tau_px=float(self.risk_tau_px_local))

            # center_xy (obs 좌표계)
            try:
                half = 0.5 * float(self.obs_out_size)
                if det is None:
                    center_xy = (half, half)
                else:
                    cx_target, cy_target = self._playfield_norm_to_full_xy(self.last_xy_norm[0], self.last_xy_norm[1])
                    sx = float(self.obs_out_size) / float(self.crop_size)
                    dx = float(cx_target - cx_s) * sx
                    dy = float(cy_target - cy_s) * sx
                    center_xy = (half + dx, half + dy)
            except Exception:
                center_xy = None

            l_risk_now = self._center_weight_local(l_risk_base, center_xy=center_xy)

            if self._risk_hold_crop_01 is None or self._risk_hold_crop_01.shape != l_risk_now.shape:
                self._risk_hold_crop_01 = l_risk_now.astype(np.float32, copy=True)
            else:
                decay = float(np.clip(self.risk_decay, 0.0, 1.0))
                self._risk_hold_crop_01 = np.maximum(self._risk_hold_crop_01 * decay, l_risk_now).astype(np.float32, copy=False)

            l2 = (l_bullet_u8.astype(np.float32) / 255.0)
            l3 = self._risk_hold_crop_01

            self._l_bullet_cache_01, self._l_risk_cache_01 = l2, l3
            self._tick("l_heavy", tlh)
        else:
            tlc = time.perf_counter()
            if self._l_bullet_cache_01 is None:
                self._l_bullet_cache_01 = np.zeros((self.obs_out_size, self.obs_out_size), dtype=np.float32)
            if self._l_risk_cache_01 is None:
                self._l_risk_cache_01 = np.zeros((self.obs_out_size, self.obs_out_size), dtype=np.float32)
            l2, l3 = self._l_bullet_cache_01, self._l_risk_cache_01
            self._tick("l_cache", tlc)

        # apply valid mask (obs 기준)
        if valid01 is not None:
            l0 = (l0 * valid01).astype(np.float32)
            l1 = (l1 * valid01).astype(np.float32)
            l2 = (l2 * valid01).astype(np.float32)
            l3 = (l3 * valid01).astype(np.float32)

        # meta + stack
        tms = time.perf_counter()
        l0 = self._inject_meta_pixels_local_ch0(l0)
        obs8 = np.stack([g0, g1, g2, g3, l0, l1, l2, l3], axis=0).astype(np.float32, copy=False)
        self._tick("meta_stack", tms)

        # debug (optional)
        tdbg = time.perf_counter()
        if self.show_obs_debug:
            try:
                self._ensure_obs_window()

                visL = (np.clip(l3, 0.0, 1.0) * 255.0).astype(np.uint8)
                visL = cv2.resize(visL, self._obs_win_size, interpolation=cv2.INTER_NEAREST)
                cv2.imshow(self.win_local_risk, cv2.cvtColor(visL, cv2.COLOR_GRAY2BGR))

                visG = (np.clip(g3, 0.0, 1.0) * 255.0).astype(np.uint8)
                visG = cv2.resize(visG, self._obs_win_size, interpolation=cv2.INTER_NEAREST)
                cv2.imshow(self.win_global_risk, cv2.cvtColor(visG, cv2.COLOR_GRAY2BGR))

                cv2.waitKey(1)
            except Exception:
                pass
        self._tick("debug", tdbg)

        # total + print
        self._tick("total", t_all)
        self._maybe_print_prof(do_heavy=do_heavy)

        return obs8
