# env/obs_builder.py
import cv2
import numpy as np

from env.reimu_detector import ReimuDetector


class ObsBuilder:
    """
    ✅ 2-Stream 8채널 관측 (Global 4ch + Local 4ch)
      - Global(전체 플레이필드, center-free)
        g0: playfield_gray (0..1)
        g1: absdiff(playfield_gray, prev_playfield_gray) (0..1)
        g2: bullet_candidate_mask_global (0..1)
        g3: risk_heatmap_global (0..1)  [distanceTransform 기반]

      - Local(레이무 주변 crop)
        l0: crop_gray (0..1) + meta pixels(x,y,conf)
        l1: absdiff(crop_gray, prev_crop_gray) (0..1)
        l2: bullet_candidate_mask_local (0..1)
        l3: risk_heatmap_centered_local (0..1) [distanceTransform + 중심가중치]

    ✅ 리턴 shape: (8, obs_out_size, obs_out_size) float32
    """

    def __init__(self, screen, debug_viz=None, obs_out_size=128, crop_size=256, use_fallback_full_preprocess=True):
        self.screen = screen
        self.debug = debug_viz

        self.obs_out_size = int(obs_out_size)
        self.crop_size = int(crop_size)
        self.use_fallback_full_preprocess = bool(use_fallback_full_preprocess)

        # ✅ 프레임당 채널: Global 4 + Local 4 = 8
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
        # Detector (기존 유지)
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

        # 기본 초기 위치/신뢰도
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

        # 정책/리워드용 좌표/신뢰도
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
        # Bullet / Risk params (공용)
        # -------------------------
        self.enable_bullet_channels = True

        self.bullet_hsv_s_min = 40
        self.bullet_hsv_v_min = 140
        self.bullet_hsv_v_max = 255
        self.bullet_close_morph = 0

        # risk (local/global 각각 사용)
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

        # post blur
        self.post_resize_blur_ksize = 0

        # crop 바깥 clamp(거울 패딩 원천 차단)
        self.clamp_crop_inside = True

        # padding 마스크(혹시 발생하면)
        self.enable_valid_mask = True
        self.valid_mask_soft_blur = 7

        # global에도 “패딩” 같은 건 없지만, playfield 절단/리사이즈에서 과신호 줄이려면 쓸 수 있음(기본 off)
        self.global_apply_soft_mask = False

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

        # 1) 너무 낮은 conf는 홀드
        if conf < float(self.conf_update_thr):
            self._lost_obs += 1
            if self._lost_obs >= int(self.lost_patience_obs):
                self._lost_obs = 0
                return (x_n, y_n, conf, True, "FORCE_LOWCONF")
            return (float(prev_xy[0]), float(prev_xy[1]), float(prev_c), False, "LOWCONF_HOLD")

        d = self._dist_norm((x_n, y_n), prev_xy)

        # 2) 점프면 accept/reject
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
        """
        local ch0에만 meta 삽입(기존 전략 유지)
        """
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
        """
        crop + valid_mask_u8(원본=255, padding=0)
        """
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

    def _compute_risk_heat_centered_local(self, bullet_mask_u8: np.ndarray, center_xy=None) -> np.ndarray:
        # base risk
        risk = self._compute_risk_heat(bullet_mask_u8, tau_px=float(self.risk_tau_px_local))

        h, w = risk.shape[:2]
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
        sigma = max(1e-6, float(self.center_sigma_px))
        center_w = np.exp(-rr2 / (2.0 * sigma * sigma)).astype(np.float32)

        risk_centered = risk * center_w
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
        # =========================================================
        # A) Global stream (전체 플레이필드)
        # =========================================================
        # playfield는 screen.get_playfield_gray/img에서 쓰던 그 영역이 가장 안정적
        play_gray = self.screen.get_playfield_gray(img_bgr)  # (Hpf, Wpf) uint8

        # global diff
        if self._prev_play_gray_u8 is None or self._prev_play_gray_u8.shape != play_gray.shape:
            play_diff = np.zeros_like(play_gray)
        else:
            play_diff = cv2.absdiff(play_gray, self._prev_play_gray_u8)
        self._prev_play_gray_u8 = play_gray

        # global bullet/risk
        if self.enable_bullet_channels:
            # playfield bgr이 없다면, 전체 img에서 playfield 영역만 잘라서 사용
            # Screen에 get_playfield_bgr가 없을 수도 있어서 안전하게 slice를 구성
            try:
                play_bgr = self.screen.get_playfield_bgr(img_bgr)  # 있으면 최고
            except Exception:
                # fallback: 화면 좌측 playfield 정도만 추정 (ratio 기반)
                pw = int(self._playfield_w)
                play_bgr = img_bgr[:, :pw].copy()

            g_bullet_u8 = self._compute_bullet_mask_u8(play_bgr)
            g_risk_01 = self._compute_risk_heat(g_bullet_u8, tau_px=float(self.risk_tau_px_global))
        else:
            g_bullet_u8 = np.zeros_like(play_gray, dtype=np.uint8)
            g_risk_01 = np.zeros_like(play_gray, dtype=np.float32)

        # resize to obs_out_size
        interp_g = cv2.INTER_AREA if max(play_gray.shape[:2]) >= self.obs_out_size else cv2.INTER_LINEAR
        g0 = cv2.resize(play_gray, (self.obs_out_size, self.obs_out_size), interpolation=interp_g).astype(np.float32) / 255.0
        g1 = cv2.resize(play_diff, (self.obs_out_size, self.obs_out_size), interpolation=interp_g).astype(np.float32) / 255.0
        g2 = cv2.resize(g_bullet_u8, (self.obs_out_size, self.obs_out_size), interpolation=interp_g).astype(np.float32) / 255.0
        g3 = cv2.resize(g_risk_01, (self.obs_out_size, self.obs_out_size), interpolation=interp_g).astype(np.float32)

        # =========================================================
        # B) Local stream (레이무 주변)
        # =========================================================
        det = self.det.step(img_bgr)

        if det is None:
            cx, cy = self.player_center
        else:
            x_n, y_n, conf, logits = det
            x_use, y_use, c_use, used, reason = self._gate_xy_update(x_n, y_n, conf)

            self.last_xy_norm = (float(x_use), float(y_use))
            self.last_conf = float(c_use)

            cx_new, cy_new = self._playfield_norm_to_full_xy(x_use, y_use)
            if used:
                cx, cy = cx_new, cy_new
                self.player_center = (cx, cy)
            else:
                cx, cy = self.player_center

        # crop center smoothing
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

        # valid mask
        if self.enable_valid_mask:
            valid01 = (valid_u8.astype(np.float32) / 255.0)
            k = int(self.valid_mask_soft_blur)
            if k and k >= 3 and (k % 2 == 1):
                valid01 = cv2.GaussianBlur(valid01, (k, k), 0)
            valid01 = np.clip(valid01, 0.0, 1.0).astype(np.float32, copy=False)
        else:
            valid01 = None

        crop_gray_u8 = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if self._prev_crop_gray_u8 is None or self._prev_crop_gray_u8.shape != crop_gray_u8.shape:
            diff_u8 = np.zeros_like(crop_gray_u8)
        else:
            diff_u8 = cv2.absdiff(crop_gray_u8, self._prev_crop_gray_u8)
        self._prev_crop_gray_u8 = crop_gray_u8

        if self.enable_bullet_channels:
            l_bullet_u8 = self._compute_bullet_mask_u8(crop_bgr)

            try:
                half = 0.5 * float(self.crop_size)
                if det is None:
                    center_xy = (half, half)
                else:
                    cx_target, cy_target = self._playfield_norm_to_full_xy(self.last_xy_norm[0], self.last_xy_norm[1])
                    dx = float(cx_target - cx_s)
                    dy = float(cy_target - cy_s)
                    center_xy = (half + dx, half + dy)
            except Exception:
                center_xy = None

            l_risk_now = self._compute_risk_heat_centered_local(l_bullet_u8, center_xy=center_xy)

            if self._risk_hold_crop_01 is None or self._risk_hold_crop_01.shape != l_risk_now.shape:
                self._risk_hold_crop_01 = l_risk_now.astype(np.float32, copy=True)
            else:
                decay = float(np.clip(self.risk_decay, 0.0, 1.0))
                self._risk_hold_crop_01 = np.maximum(self._risk_hold_crop_01 * decay, l_risk_now).astype(np.float32, copy=False)

            l_risk_01 = self._risk_hold_crop_01
        else:
            l_bullet_u8 = np.zeros((self.crop_size, self.crop_size), dtype=np.uint8)
            l_risk_01 = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)

        # apply valid mask
        if valid01 is not None:
            crop_gray_u8 = np.clip(crop_gray_u8.astype(np.float32) * valid01, 0, 255).astype(np.uint8)
            diff_u8 = np.clip(diff_u8.astype(np.float32) * valid01, 0, 255).astype(np.uint8)
            l_bullet_u8 = np.clip(l_bullet_u8.astype(np.float32) * valid01, 0, 255).astype(np.uint8)
            l_risk_01 = (l_risk_01.astype(np.float32) * valid01).astype(np.float32)

        # resize local to obs_out_size
        interp_l = cv2.INTER_AREA if self.crop_size >= self.obs_out_size else cv2.INTER_LINEAR
        l0 = cv2.resize(crop_gray_u8, (self.obs_out_size, self.obs_out_size), interpolation=interp_l).astype(np.float32) / 255.0
        l1 = cv2.resize(diff_u8, (self.obs_out_size, self.obs_out_size), interpolation=interp_l).astype(np.float32) / 255.0
        l2 = cv2.resize(l_bullet_u8, (self.obs_out_size, self.obs_out_size), interpolation=interp_l).astype(np.float32) / 255.0
        l3 = cv2.resize(l_risk_01, (self.obs_out_size, self.obs_out_size), interpolation=interp_l).astype(np.float32)

        # post blur (optional)
        k = int(self.post_resize_blur_ksize)
        if k and k >= 3 and (k % 2 == 1):
            g0 = cv2.GaussianBlur(g0, (k, k), 0)
            g1 = cv2.GaussianBlur(g1, (k, k), 0)
            g2 = cv2.GaussianBlur(g2, (k, k), 0)
            g3 = cv2.GaussianBlur(g3, (k, k), 0)
            l0 = cv2.GaussianBlur(l0, (k, k), 0)
            l1 = cv2.GaussianBlur(l1, (k, k), 0)
            l2 = cv2.GaussianBlur(l2, (k, k), 0)
            l3 = cv2.GaussianBlur(l3, (k, k), 0)

        # meta -> local ch0 only
        l0 = self._inject_meta_pixels_local_ch0(l0)

        obs8 = np.stack([g0, g1, g2, g3, l0, l1, l2, l3], axis=0).astype(np.float32, copy=False)

        # debug show: risk maps
        if self.show_obs_debug:
            try:
                self._ensure_obs_window()
                visL = (np.clip(l3, 0.0, 1.0) * 255.0).astype(np.uint8)
                visL = cv2.cvtColor(visL, cv2.COLOR_GRAY2BGR)
                cv2.imshow(self.win_local_risk, visL)

                visG = (np.clip(g3, 0.0, 1.0) * 255.0).astype(np.uint8)
                visG = cv2.cvtColor(visG, cv2.COLOR_GRAY2BGR)
                cv2.imshow(self.win_global_risk, visG)

                cv2.waitKey(1)
            except Exception:
                pass

        return obs8
