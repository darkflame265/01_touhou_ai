# env/obs_builder.py
import cv2
import numpy as np
import time

from env.reimu_detector import ReimuDetector


class ObsBuilder:
    """
    ✅ 2-Stream 8채널 관측 (Global 4ch + Local 4ch)
      - Global(전체 플레이필드, center-free)
        g0: playfield_gray (0..1)
        g1: absdiff(playfield_gray, prev_playfield_gray) (0..1)
        g2: bullet_candidate_mask_global (0..1)
        g3: risk_heatmap_global (0..1)

      - Local(레이무 주변 crop)
        l0: crop_gray (0..1) + meta pixels(x,y,conf)
        l1: absdiff(crop_gray, prev_crop_gray) (0..1)
        l2: bullet_candidate_mask_local (0..1)
        l3: risk_heatmap_centered_local (0..1)

    ✅ 리턴 shape: (8, obs_out_size, obs_out_size) float32

    =========================
    (OPT) 즉시 체감 최적화
      - heavy(HSV/DT)는 heavy_every 프레임마다만 갱신, 나머지는 캐시 재사용
      - Global bullet/risk: 저해상도(global_proc_size)에서만 계산
      - Local bullet/risk : crop_size(256)에서 하지 않고 obs_out_size(128)에서만 계산
      - 디버그는 업샘플해서 보기 좋게만 표시(학습 입력에는 영향 없음)
    =========================
    """

    def __init__(self, screen, debug_viz=None, obs_out_size=128, crop_size=256, use_fallback_full_preprocess=True):
        self.screen = screen
        self.debug = debug_viz

        self.obs_out_size = int(obs_out_size)
        self.crop_size = int(crop_size)
        self.use_fallback_full_preprocess = bool(use_fallback_full_preprocess)

        self.global_channels = 4
        self.local_channels = 4
        self.obs_channels = self.global_channels + self.local_channels  # 8

        img0 = self.screen.capture()
        h0, w0 = img0.shape[:2]
        self.H, self.W = h0, w0

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

        self.player_center = (w0 // 2, int(h0 * 0.78))
        self.conf_update_thr = 0.02

        # 점프 억제 게이트(Obs측)
        self.max_jump_norm_obs = 0.18
        self.jump_allow_conf_gain_obs = 2.0
        self.lost_patience_obs = 10
        self._lost_obs = 0

        # 코너/가장자리 갑툭튀 점프 억제
        self.edge_margin_norm = 0.035
        self.edge_jump_conf_gain = 3.0
        self.edge_jump_min_conf = 0.90

        self.last_xy_norm = (0.5, 0.78)
        self.last_conf = 0.0

        # meta pixels(로컬 ch0에만)
        self.meta_patch = 4

        # -------------------------
        # Debug window
        # -------------------------
        self.show_obs_debug = True
        self.win_local_risk = "OBS_LOCAL_RISK"
        self.win_global_risk = "OBS_GLOBAL_RISK"
        self._obs_win_inited = False
        self._obs_win_pos_local = (1600, 60)
        self._obs_win_pos_global = (1600, 720)
        self._obs_win_size = (520, 520)

        # -------------------------
        # prev frames
        # -------------------------
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

        # 화면 안정화(로컬 crop 중심)
        self.center_max_speed_px = 10.0
        self.center_micro_ema_alpha = 0.0
        self._crop_center_f = None

        # local risk 안정화 (max-hold + decay)
        self.risk_decay = 0.70
        self._risk_hold_crop_01 = None

        # crop 바깥 clamp(거울 패딩 원천 차단)
        self.clamp_crop_inside = True

        # padding 마스크(혹시 발생하면)
        self.enable_valid_mask = True
        self.valid_mask_soft_blur = 7

        # =========================
        # ✅ (NEW) 최적화 스위치
        # =========================
        self.heavy_every = 2               # HSV/DT 같은 무거운 채널 갱신 주기
        self._step_i = 0

        # 전역 bullet/risk 계산 해상도 (작을수록 빠름)
        self.global_proc_size = (96, 72)   # (W,H) 추천: (96,72) / (128,96)

        # 캐시(heavy_every 사이 프레임에서 재사용) — 모두 "obs_out_size" 기준 float32(0..1)
        self._g_bullet_cache_01 = None
        self._g_risk_cache_01 = None
        self._l_bullet_cache_01 = None
        self._l_risk_cache_01 = None

        self.obs_prof_enable = True
        self.obs_prof_every = 200  # 몇 프레임마다 출력할지

        self._prof_cnt = 0
        self._prof_sum = {
            "det": 0.0,            # detector step
            "g_graydiff": 0.0,      # global gray + diff + resize(g0,g1)
            "g_heavy": 0.0,         # global heavy (resize+HSV+DT+upsample)
            "g_cache": 0.0,         # global cache reuse
            "l_crop": 0.0,          # crop + valid mask prep
            "l_graydiff": 0.0,      # local gray + diff + resize(l0,l1)
            "l_heavy": 0.0,         # local heavy (resize+HSV+DT+center-weight)
            "l_cache": 0.0,         # local cache reuse
            "meta_stack": 0.0,      # meta inject + stack
            "debug": 0.0,           # debug imshow
            "total": 0.0,           # total make_state
        }

        # =========================
        # ✅ (NEW) detector도 매 프레임 안 돌리기
        # =========================
        self.det_every = 4         # 2면 2프레임마다 1번만 det.step()
        self._last_det = None      # (x_n, y_n, conf, logits) 캐시
        self._last_det_reason = "INIT"


    # -------------------------
    # lifecycle
    # -------------------------
    def reset(self):
        if hasattr(self.det, "reset"):
            self.det.reset()

        self.player_center = (self.W // 2, int(self.H * 0.78))
        self._lost_obs = 0

        self.last_xy_norm = (0.5, 0.78)
        self.last_conf = 0.0

        self._prev_crop_gray_u8 = None
        self._prev_play_gray_u8 = None

        self._crop_center_f = None
        self._risk_hold_crop_01 = None

        self._step_i = 0
        self._g_bullet_cache_01 = None
        self._g_risk_cache_01 = None
        self._l_bullet_cache_01 = None
        self._l_risk_cache_01 = None

        # OBS PROF
        self._prof_cnt = 0
        for k in self._prof_sum:
            self._prof_sum[k] = 0.0

        self._last_det = None
        self._last_det_reason = "RESET"



    def on_player_death(self):
        try:
            if hasattr(self.det, "on_player_death"):
                self.det.on_player_death()
        except Exception:
            pass

    # -------------------------
    # debug windows
    # -------------------------
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

    # -------------------------
    # utils
    # -------------------------
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

            if (now_edge and (not prev_edge)):
                need = need * float(self.edge_jump_conf_gain)
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
        h, w = self.H, self.W
        half = int(size) // 2
        if half <= 0:
            return int(cx), int(cy)
        cx_min = half
        cx_max = max(half, w - half - 1)
        cy_min = half
        cy_max = max(half, h - half - 1)
        return int(np.clip(cx, cx_min, cx_max)), int(np.clip(cy, cy_min, cy_max))

    def _crop_square_bgr_with_mask(self, img_bgr, cx, cy, size):
        h, w = img_bgr.shape[:2]
        size = int(size)
        half = size // 2

        x1 = int(cx - half)
        y1 = int(cy - half)
        x2 = x1 + size
        y2 = y1 + size

        if (0 <= x1) and (0 <= y1) and (x2 <= w) and (y2 <= h):
            crop = img_bgr[y1:y2, x1:x2]
            valid = np.full((size, size), 255, dtype=np.uint8)
            return crop, valid

        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - w)
        pad_b = max(0, y2 - h)

        img_pad = cv2.copyMakeBorder(img_bgr, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_REFLECT_101)

        base_valid = np.full((h, w), 255, dtype=np.uint8)
        valid_pad = cv2.copyMakeBorder(base_valid, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_CONSTANT, value=0)

        x1p = x1 + pad_l
        y1p = y1 + pad_t
        x2p = x2 + pad_l
        y2p = y2 + pad_t

        crop = img_pad[y1p:y2p, x1p:x2p]
        valid = valid_pad[y1p:y2p, x1p:x2p]

        if crop.shape[0] != size or crop.shape[1] != size:
            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
        if valid.shape[0] != size or valid.shape[1] != size:
            valid = cv2.resize(valid, (size, size), interpolation=cv2.INTER_NEAREST)

        return crop, valid

    def _center_speed_limit(self, prev_xy_f, target_xy_i):
        px, py = float(prev_xy_f[0]), float(prev_xy_f[1])
        tx, ty = float(target_xy_i[0]), float(target_xy_i[1])
        dx = tx - px
        dy = ty - py
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
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
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

    def _compute_risk_heat_centered_local(self, risk01: np.ndarray, center_xy=None) -> np.ndarray:
        h, w = risk01.shape[:2]
        if center_xy is None:
            cx = (w - 1) * 0.5
            cy = (h - 1) * 0.5
        else:
            cx = float(center_xy[0])
            cy = float(center_xy[1])

        cx = float(np.clip(cx, 0.0, w - 1.0))
        cy = float(np.clip(cy, 0.0, h - 1.0))

        yy, xx = np.indices((h, w), dtype=np.float32)
        rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
        # 256기준 sigma를 128좌표계로 스케일
        sigma = max(1e-6, float(self.center_sigma_px) * (float(w) / float(self.crop_size)))
        center_w = np.exp(-rr2 / (2.0 * sigma * sigma)).astype(np.float32)

        risk_centered = risk01 * center_w
        m = float(risk_centered.max())
        if m > 1e-6:
            risk_centered = risk_centered / m
        if self.risk_clip_max is not None:
            risk_centered = np.clip(risk_centered, 0.0, float(self.risk_clip_max))
        return risk_centered.astype(np.float32, copy=False)

    # -------------------------
    # main
    # -------------------------
    def make_state(self, img_bgr):
        t_all0 = time.perf_counter()

        self._step_i += 1
        do_heavy = (self.enable_bullet_channels and (self.heavy_every <= 1 or (self._step_i % int(self.heavy_every) == 0)))

        # =========================================================
        # A) Global stream
        # =========================================================
        t0 = time.perf_counter()

        play_gray = self.screen.get_playfield_gray(img_bgr)  # uint8

        if self._prev_play_gray_u8 is None or self._prev_play_gray_u8.shape != play_gray.shape:
            play_diff = np.zeros_like(play_gray)
        else:
            play_diff = cv2.absdiff(play_gray, self._prev_play_gray_u8)
        self._prev_play_gray_u8 = play_gray

        interp_g = cv2.INTER_AREA if max(play_gray.shape[:2]) >= self.obs_out_size else cv2.INTER_LINEAR
        g0 = cv2.resize(play_gray, (self.obs_out_size, self.obs_out_size), interpolation=interp_g).astype(np.float32) / 255.0
        g1 = cv2.resize(play_diff, (self.obs_out_size, self.obs_out_size), interpolation=interp_g).astype(np.float32) / 255.0

        t1 = time.perf_counter()
        if self.obs_prof_enable:
            self._prof_sum["g_graydiff"] += (t1 - t0)

        if do_heavy:
            th0 = time.perf_counter()

            try:
                play_bgr = self.screen.get_playfield_bgr(img_bgr)
            except Exception:
                pw = int(self._playfield_w)
                play_bgr = img_bgr[:, :pw].copy()

            gpw, gph = int(self.global_proc_size[0]), int(self.global_proc_size[1])
            play_bgr_small = cv2.resize(play_bgr, (gpw, gph), interpolation=cv2.INTER_AREA)

            g_bullet_u8_small = self._compute_bullet_mask_u8(play_bgr_small)
            g_risk_01_small = self._compute_risk_heat(g_bullet_u8_small, tau_px=float(self.risk_tau_px_global))

            g2 = cv2.resize(g_bullet_u8_small, (self.obs_out_size, self.obs_out_size), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
            g3 = cv2.resize(g_risk_01_small, (self.obs_out_size, self.obs_out_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)

            self._g_bullet_cache_01 = g2
            self._g_risk_cache_01 = g3

            th1 = time.perf_counter()
            if self.obs_prof_enable:
                self._prof_sum["g_heavy"] += (th1 - th0)
        else:
            tc0 = time.perf_counter()

            if self._g_bullet_cache_01 is None:
                self._g_bullet_cache_01 = np.zeros((self.obs_out_size, self.obs_out_size), dtype=np.float32)
            if self._g_risk_cache_01 is None:
                self._g_risk_cache_01 = np.zeros((self.obs_out_size, self.obs_out_size), dtype=np.float32)
            g2 = self._g_bullet_cache_01
            g3 = self._g_risk_cache_01

            tc1 = time.perf_counter()
            if self.obs_prof_enable:
                self._prof_sum["g_cache"] += (tc1 - tc0)

        # =========================================================
        # B) Local stream
        # =========================================================
        td0 = time.perf_counter()


        # B) Local stream - DET 캐시 (조건부 호출)
        # =========================
        # det_every를 4 정도로 올리고, 불안정할 때만 강제로 det
        base_every = int(self.det_every)  # 예: 4
        force = False

        # 1) conf가 낮으면 강제 det
        if float(self.last_conf) < (float(self.conf_update_thr) + 0.01):
            force = True

        # 2) 최근에 lost가 쌓였으면 강제 det
        if int(self._lost_obs) >= 2:
            force = True

        # 3) 너무 오래 det 안 돌렸으면 강제 det
        do_det = force or (base_every <= 1) or (self._step_i % base_every == 0)

        if do_det:
            det = self.det.step(img_bgr)
            self._last_det = det
            self._last_det_reason = "DET_STEP"
        else:
            det = self._last_det
            self._last_det_reason = "DET_CACHE"



        td1 = time.perf_counter()
        if self.obs_prof_enable:
            self._prof_sum["det"] += (td1 - td0)

        if det is None:
            cx, cy = self.player_center
        else:
            if not do_det:
                # 캐시 프레임이면 좌표 업데이트 스킵(센터 유지)
                cx, cy = self.player_center
            else:
                x_n, y_n, conf, _logits = det
                x_use, y_use, c_use, used, _reason = self._gate_xy_update(x_n, y_n, conf)

                self.last_xy_norm = (float(x_use), float(y_use))
                self.last_conf = float(c_use)

                cx_new, cy_new = self._playfield_norm_to_full_xy(x_use, y_use)
                if used:
                    cx, cy = cx_new, cy_new
                    self.player_center = (cx, cy)
                else:
                    cx, cy = self.player_center


        tcrop0 = time.perf_counter()

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

        # valid mask를 128로 줄여서 준비
        if self.enable_valid_mask:
            valid01_full = (valid_u8.astype(np.float32) / 255.0)
            k = int(self.valid_mask_soft_blur)
            if k and k >= 3 and (k % 2 == 1):
                valid01_full = cv2.GaussianBlur(valid01_full, (k, k), 0)
            valid01_full = np.clip(valid01_full, 0.0, 1.0).astype(np.float32, copy=False)
            valid01 = cv2.resize(valid01_full, (self.obs_out_size, self.obs_out_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        else:
            valid01 = None

        tcrop1 = time.perf_counter()
        if self.obs_prof_enable:
            self._prof_sum["l_crop"] += (tcrop1 - tcrop0)

        tld0 = time.perf_counter()

        crop_gray_u8 = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if self._prev_crop_gray_u8 is None or self._prev_crop_gray_u8.shape != crop_gray_u8.shape:
            diff_u8_full = np.zeros_like(crop_gray_u8)
        else:
            diff_u8_full = cv2.absdiff(crop_gray_u8, self._prev_crop_gray_u8)
        self._prev_crop_gray_u8 = crop_gray_u8

        interp_l = cv2.INTER_AREA if self.crop_size >= self.obs_out_size else cv2.INTER_LINEAR
        l0 = cv2.resize(crop_gray_u8, (self.obs_out_size, self.obs_out_size), interpolation=interp_l).astype(np.float32) / 255.0
        l1 = cv2.resize(diff_u8_full, (self.obs_out_size, self.obs_out_size), interpolation=interp_l).astype(np.float32) / 255.0

        tld1 = time.perf_counter()
        if self.obs_prof_enable:
            self._prof_sum["l_graydiff"] += (tld1 - tld0)

        if do_heavy:
            tlh0 = time.perf_counter()

            # 128 기준 bgr 만들기(HSV/DT 여기서만)
            l_bgr_small = cv2.resize(crop_bgr, (self.obs_out_size, self.obs_out_size), interpolation=cv2.INTER_AREA)
            l_bullet_u8_small = self._compute_bullet_mask_u8(l_bgr_small)
            l_risk_base = self._compute_risk_heat(l_bullet_u8_small, tau_px=float(self.risk_tau_px_local))

            # 중심가중치(128 좌표계)
            try:
                half = 0.5 * float(self.obs_out_size)
                if det is None:
                    center_xy = (half, half)
                else:
                    cx_target, cy_target = self._playfield_norm_to_full_xy(self.last_xy_norm[0], self.last_xy_norm[1])
                    dx = float(cx_target - cx_s) * (float(self.obs_out_size) / float(self.crop_size))
                    dy = float(cy_target - cy_s) * (float(self.obs_out_size) / float(self.crop_size))
                    center_xy = (half + dx, half + dy)
            except Exception:
                center_xy = None

            l_risk_now = self._compute_risk_heat_centered_local(l_risk_base, center_xy=center_xy)

            # max-hold + decay
            if self._risk_hold_crop_01 is None or self._risk_hold_crop_01.shape != l_risk_now.shape:
                self._risk_hold_crop_01 = l_risk_now.astype(np.float32, copy=True)
            else:
                decay = float(np.clip(self.risk_decay, 0.0, 1.0))
                self._risk_hold_crop_01 = np.maximum(self._risk_hold_crop_01 * decay, l_risk_now).astype(np.float32, copy=False)

            l2 = (l_bullet_u8_small.astype(np.float32) / 255.0)
            l3 = self._risk_hold_crop_01.astype(np.float32, copy=False)

            self._l_bullet_cache_01 = l2
            self._l_risk_cache_01 = l3

            tlh1 = time.perf_counter()
            if self.obs_prof_enable:
                self._prof_sum["l_heavy"] += (tlh1 - tlh0)
        else:
            tlc0 = time.perf_counter()

            if self._l_bullet_cache_01 is None:
                self._l_bullet_cache_01 = np.zeros((self.obs_out_size, self.obs_out_size), dtype=np.float32)
            if self._l_risk_cache_01 is None:
                self._l_risk_cache_01 = np.zeros((self.obs_out_size, self.obs_out_size), dtype=np.float32)
            l2 = self._l_bullet_cache_01
            l3 = self._l_risk_cache_01

            tlc1 = time.perf_counter()
            if self.obs_prof_enable:
                self._prof_sum["l_cache"] += (tlc1 - tlc0)

        # valid mask 적용(128 기준)
        if valid01 is not None:
            l0 = (l0 * valid01).astype(np.float32)
            l1 = (l1 * valid01).astype(np.float32)
            l2 = (l2 * valid01).astype(np.float32)
            l3 = (l3 * valid01).astype(np.float32)

        tms0 = time.perf_counter()
        l0 = self._inject_meta_pixels_local_ch0(l0)
        obs8 = np.stack([g0, g1, g2, g3, l0, l1, l2, l3], axis=0).astype(np.float32, copy=False)
        tms1 = time.perf_counter()
        if self.obs_prof_enable:
            self._prof_sum["meta_stack"] += (tms1 - tms0)

        # debug show (업샘플)
        tdbg0 = time.perf_counter()
        if self.show_obs_debug:
            try:
                self._ensure_obs_window()

                visL = (np.clip(l3, 0.0, 1.0) * 255.0).astype(np.uint8)
                visL = cv2.resize(visL, self._obs_win_size, interpolation=cv2.INTER_NEAREST)
                visL = cv2.cvtColor(visL, cv2.COLOR_GRAY2BGR)
                cv2.imshow(self.win_local_risk, visL)

                visG = (np.clip(g3, 0.0, 1.0) * 255.0).astype(np.uint8)
                visG = cv2.resize(visG, self._obs_win_size, interpolation=cv2.INTER_NEAREST)
                visG = cv2.cvtColor(visG, cv2.COLOR_GRAY2BGR)
                cv2.imshow(self.win_global_risk, visG)

                cv2.waitKey(1)
            except Exception:
                pass
        tdbg1 = time.perf_counter()
        if self.obs_prof_enable:
            self._prof_sum["debug"] += (tdbg1 - tdbg0)

        # =========================
        # PROF PRINT
        # =========================
        t_all1 = time.perf_counter()
        if self.obs_prof_enable:
            self._prof_sum["total"] += (t_all1 - t_all0)
            self._prof_cnt += 1
            if (self._prof_cnt % int(self.obs_prof_every)) == 0:
                n = max(1, int(self._prof_cnt))
                # 최근 구간 평균만 보고 싶으면, 출력 후 sum을 0으로 리셋하는 방식으로 바꿔도 됨.
                def ms(key): 
                    return (self._prof_sum[key] / float(self.obs_prof_every)) * 1000.0

                print(
                    "[OBS_PROF] avg_ms/call | "
                    f"total={ms('total'):.2f} "
                    f"det={ms('det'):.2f} "
                    f"g(gray+diff)={ms('g_graydiff'):.2f} g(heavy)={ms('g_heavy'):.2f} g(cache)={ms('g_cache'):.2f} | "
                    f"l(crop)={ms('l_crop'):.2f} l(gray+diff)={ms('l_graydiff'):.2f} l(heavy)={ms('l_heavy'):.2f} l(cache)={ms('l_cache'):.2f} | "
                    f"meta+stack={ms('meta_stack'):.2f} dbg={ms('debug'):.2f} "
                    f"(heavy_every={self.heavy_every}, do_heavy={do_heavy})"
                )

                # ✅ “최근 obs_prof_every 구간”만 보려고 평균 안정화시키고 싶으면 아래 리셋 추천
                for k in self._prof_sum:
                    self._prof_sum[k] = 0.0

        return obs8
