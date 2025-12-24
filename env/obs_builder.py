# env/obs_builder.py
import cv2
import numpy as np

from env.reimu_detector import ReimuDetector


class ObsBuilder:
    """
    ✅ 4채널 관측
      - ch0: 현재 crop_gray (0..1) + meta pixels(레이무 xy/conf)
      - ch1: absdiff(current_crop_gray, prev_crop_gray) (0..1)
      - ch2: bullet_candidate_mask (0..1)  [HSV/밝기/채도 기반]
      - ch3: risk_heatmap_centered (0..1)  [distanceTransform + (레이무기반) 중심 가중치]

    ✅ 리턴 shape: (4, obs_out_size, obs_out_size) float32
    """

    def __init__(self, screen, debug_viz=None, obs_out_size=84, crop_size=160, use_fallback_full_preprocess=True):
        self.screen = screen
        self.debug = debug_viz

        self.obs_out_size = int(obs_out_size)
        self.crop_size = int(crop_size)
        self.use_fallback_full_preprocess = bool(use_fallback_full_preprocess)

        # ✅ 외부에서 채널 수를 알 수 있게
        self.obs_channels = 4

        img0 = self.screen.capture()
        h0, w0 = img0.shape[:2]
        self.H, self.W = h0, w0

        # ✅ 레이무 검출기(히트맵)
        self.det = ReimuDetector(
            screen=self.screen,
            weight_path="weights/reimu_heatmap_best.pt",
            beta=12.0,
            prior_strength=1.0,
            ema_alpha=0.85,
            device=None,

            # tracking 옵션
            track_prior_strength=2.0,
            track_prior_sigma=0.08,
            lock_conf_thr=0.015,
            max_jump_norm=0.22,
            jump_allow_conf_gain=1.8,
            lost_patience=8,

            # 성능 옵션
            use_fp16=True,
            track_prior_every=2,
            print_prof=True,
            prof_every=200,
        )

        # 기본 초기 위치/신뢰도
        self.player_center = (w0 // 2, int(h0 * 0.78))
        self.conf_update_thr = 0.02

        # 점프 억제 게이트
        self.max_jump_norm_obs = 0.18
        self.jump_allow_conf_gain_obs = 2.0
        self.lost_patience_obs = 10
        self._lost_obs = 0

        # 디버그용(원하면 확장)
        self._dbg_last = None

        # 정책/리워드용 좌표/신뢰도
        self.last_xy_norm = (0.5, 0.78)
        self.last_conf = 0.0

        # ✅ 메타 픽셀 설정 (ch0에만)
        self.meta_patch = 4

        # playfield width 캐시
        self._playfield_ratio = float(getattr(self.screen, "PLAYFIELD_RIGHT_RATIO", 0.70))
        self._playfield_w = max(1, min(self.W, int(self.W * self._playfield_ratio)))

        # -------------------------
        # ✅ OBS 디버그 창(가운데 화면만 크게)
        # -------------------------
        self.show_obs_debug = True
        self.win_crop = "OBS_CROP"
        self._obs_win_inited = False
        self._obs_win_pos = (1600, 60)
        self._obs_win_size = (600, 600)

        self._prev_crop_gray_u8 = None

        # -------------------------
        # ✅ 탄 후보 마스크 / 위험도 히트맵
        # -------------------------
        self.enable_bullet_channels = True

        # HSV 기반 탄 후보
        self.bullet_hsv_s_min = 40     # 채도 하한
        self.bullet_hsv_v_min = 140    # 밝기 하한
        self.bullet_hsv_v_max = 255
        self.bullet_close_morph = 0    # 0이면 off, 1~2 추천

        # 위험도 변환 파라미터
        self.risk_tau_px = 8.0

        # 중심 가중치 sigma
        self.center_sigma_px = float(self.crop_size) * 0.35

        # 안전장치 clip
        self.risk_clip_max = 1.0

    def reset(self):
        if hasattr(self.det, "reset"):
            self.det.reset()

        self.player_center = (self.W // 2, int(self.H * 0.78))
        self._lost_obs = 0

        self.last_xy_norm = (0.5, 0.78)
        self.last_conf = 0.0
        self._dbg_last = None

        self._prev_crop_gray_u8 = None

    def _ensure_obs_window(self):
        if self._obs_win_inited:
            return
        try:
            cv2.namedWindow(self.win_crop, cv2.WINDOW_NORMAL)
            cv2.moveWindow(self.win_crop, int(self._obs_win_pos[0]), int(self._obs_win_pos[1]))
            cv2.resizeWindow(self.win_crop, int(self._obs_win_size[0]), int(self._obs_win_size[1]))
        except Exception:
            pass
        self._obs_win_inited = True

    def _crop_square_bgr(self, img_bgr, cx, cy, size):
        h, w = img_bgr.shape[:2]
        size = int(size)
        half = size // 2

        x1 = int(cx - half)
        y1 = int(cy - half)
        x2 = x1 + size
        y2 = y1 + size

        if (0 <= x1) and (0 <= y1) and (x2 <= w) and (y2 <= h):
            return img_bgr[y1:y2, x1:x2]

        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - w)
        pad_b = max(0, y2 - h)

        img_pad = cv2.copyMakeBorder(
            img_bgr,
            pad_t, pad_b, pad_l, pad_r,
            borderType=cv2.BORDER_REFLECT_101
        )

        x1 += pad_l
        y1 += pad_t
        x2 += pad_l
        y2 += pad_t

        crop = img_pad[y1:y2, x1:x2]
        if crop.shape[0] != size or crop.shape[1] != size:
            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
        return crop

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
            if conf >= need:
                self._lost_obs = 0
                return (x_n, y_n, conf, True, "JUMP_ACCEPT")
            else:
                self._lost_obs += 1
                if self._lost_obs >= int(self.lost_patience_obs):
                    self._lost_obs = 0
                    return (x_n, y_n, conf, True, "FORCE_JUMP")
                return (float(prev_xy[0]), float(prev_xy[1]), float(prev_c), False, "JUMP_REJECT")

        self._lost_obs = 0
        return (x_n, y_n, conf, True, "OK")

    def _inject_meta_pixels_ch0_only(self, obs_ch0_01: np.ndarray) -> np.ndarray:
        try:
            x_n, y_n = self.last_xy_norm
            c = float(self.last_conf)

            x_n = float(np.clip(x_n, 0.0, 1.0))
            y_n = float(np.clip(y_n, 0.0, 1.0))
            c = float(np.clip(c, 0.0, 1.0))

            p = int(self.meta_patch)
            if obs_ch0_01.shape[0] >= p and obs_ch0_01.shape[1] >= p * 3:
                obs_ch0_01[0:p, 0:p] = x_n
                obs_ch0_01[0:p, p:p * 2] = y_n
                obs_ch0_01[0:p, p * 2:p * 3] = c
        except Exception:
            pass
        return obs_ch0_01

    def _compute_bullet_mask_u8(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        crop_bgr: (crop_size, crop_size, 3) BGR
        return: bullet_mask_u8 (0 or 255), shape (crop_size, crop_size)
        """
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
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

    def _compute_risk_heat_centered(self, bullet_mask_u8: np.ndarray, center_xy=None) -> np.ndarray:
        """
        bullet_mask_u8: 0/255, 탄 후보 픽셀=255
        center_xy: (cx, cy) in crop pixel coords. None이면 crop center 사용
        return: risk01 float32 (crop_size, crop_size), 0..1
        """
        inv = cv2.bitwise_not(bullet_mask_u8)  # 탄=0, 배경=255
        dist = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)

        tau = max(1e-6, float(self.risk_tau_px))
        risk = np.exp(-dist / tau).astype(np.float32)

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

    def make_state(self, img_bgr):
        # 1) detector
        det = self.det.step(img_bgr)

        if det is None:
            cx, cy = self.player_center
            self._dbg_last = None
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

            self._dbg_last = (float(x_use), float(y_use), float(c_use), logits, str(reason))

        # 2) crop
        crop_bgr = self._crop_square_bgr(img_bgr, cx, cy, self.crop_size)

        # 3) gray + diff
        crop_gray_u8 = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if self._prev_crop_gray_u8 is None or self._prev_crop_gray_u8.shape != crop_gray_u8.shape:
            diff_u8 = np.zeros_like(crop_gray_u8)
        else:
            diff_u8 = cv2.absdiff(crop_gray_u8, self._prev_crop_gray_u8)
        self._prev_crop_gray_u8 = crop_gray_u8

        # 3.5) bullet mask + risk (레이무 위치 기반 중심)
        if self.enable_bullet_channels:
            bullet_mask_u8 = self._compute_bullet_mask_u8(crop_bgr)

            # 레이무 추정 좌표를 crop 내부 픽셀로 변환해 center로 사용
            try:
                half = 0.5 * float(self.crop_size)
                if det is None:
                    center_xy = (half, half)
                else:
                    # last_xy_norm은 "게이트 적용된 추정치"
                    cx_target, cy_target = self._playfield_norm_to_full_xy(self.last_xy_norm[0], self.last_xy_norm[1])
                    dx = float(cx_target - cx)
                    dy = float(cy_target - cy)
                    center_xy = (half + dx, half + dy)
            except Exception:
                center_xy = None

            risk_crop_01 = self._compute_risk_heat_centered(bullet_mask_u8, center_xy=center_xy)
        else:
            bullet_mask_u8 = np.zeros_like(crop_gray_u8, dtype=np.uint8)
            risk_crop_01 = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)

        # 4) resize
        interp = cv2.INTER_AREA if self.crop_size >= self.obs_out_size else cv2.INTER_LINEAR

        ch0 = cv2.resize(crop_gray_u8, (self.obs_out_size, self.obs_out_size), interpolation=interp).astype(np.float32) / 255.0
        ch1 = cv2.resize(diff_u8, (self.obs_out_size, self.obs_out_size), interpolation=interp).astype(np.float32) / 255.0
        ch2 = cv2.resize(bullet_mask_u8, (self.obs_out_size, self.obs_out_size), interpolation=interp).astype(np.float32) / 255.0
        ch3 = cv2.resize(risk_crop_01, (self.obs_out_size, self.obs_out_size), interpolation=interp).astype(np.float32)

        # 5) meta는 ch0에만
        ch0 = self._inject_meta_pixels_ch0_only(ch0)

        obs4 = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32, copy=False)

        # 6) 디버그: bullet_mask만 크게 표시
        if self.show_obs_debug:
            try:
                self._ensure_obs_window()
                vis = cv2.cvtColor(bullet_mask_u8, cv2.COLOR_GRAY2BGR)
                cv2.imshow(self.win_crop, vis)
                cv2.waitKey(1)
            except Exception:
                pass

        return obs4
