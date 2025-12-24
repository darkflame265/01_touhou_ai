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

        self.obs_channels = 4

        img0 = self.screen.capture()
        h0, w0 = img0.shape[:2]
        self.H, self.W = h0, w0

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

        # ✅ 코너/가장자리 "갑툭튀 점프" 추가 억제
        self.edge_margin_norm = 0.035          # (0~1) 가장자리 마진
        self.edge_jump_conf_gain = 3.0         # 가장자리로 점프할 때 필요한 conf 배수(기존 need에 곱)
        self.edge_jump_min_conf = 0.90         # 가장자리 점프는 최소 이 conf 이상만 고려

        self._dbg_last = None

        # 정책/리워드용 좌표/신뢰도
        self.last_xy_norm = (0.5, 0.78)
        self.last_conf = 0.0

        # meta pixels
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

        self.bullet_hsv_s_min = 40
        self.bullet_hsv_v_min = 140
        self.bullet_hsv_v_max = 255
        self.bullet_close_morph = 0

        self.risk_tau_px = 8.0
        self.center_sigma_px = float(self.crop_size) * 0.35
        self.risk_clip_max = 1.0

        # -------------------------
        # ✅ 화면 안정화(딜레이 최소화 버전)
        # -------------------------
        # (A) crop 중심: "속도 제한" (프레임당 최대 이동 픽셀)
        self.center_max_speed_px = 10.0

        # 선택 옵션: 미세 EMA (0이면 OFF). speed-limit 후에 아주 약하게만 적용 가능.
        self.center_micro_ema_alpha = 0.0

        self._crop_center_f = None  # (cx_f, cy_f) float

        # (B) risk 안정화: "max-hold + decay"
        self.risk_decay = 0.70
        self._risk_hold_crop_01 = None  # (crop_size, crop_size) float32

        # resize 후 약한 블러로 aliasing 완화 (0이면 off, 3 또는 5 추천)
        self.post_resize_blur_ksize = 0

        # ✅ (핵심) crop이 화면 밖으로 나가며 "거울 코너" 패턴 생기는 걸 원천 차단
        #  - True면 cx_s/cy_s를 항상 화면 안쪽으로 clamp → padding 자체가 거의 발생 안 함
        self.clamp_crop_inside = True

        # ✅ 혹시라도 padding이 생기면, 그 영역(거울/패딩)이 입력에서 과신호가 되지 않게 마스크 적용
        self.enable_valid_mask = True
        self.valid_mask_soft_blur = 7  # 0이면 off, 홀수 권장(5~11)

    def reset(self):
        if hasattr(self.det, "reset"):
            self.det.reset()

        self.player_center = (self.W // 2, int(self.H * 0.78))
        self._lost_obs = 0

        self.last_xy_norm = (0.5, 0.78)
        self.last_conf = 0.0
        self._dbg_last = None

        self._prev_crop_gray_u8 = None
        self._crop_center_f = None
        self._risk_hold_crop_01 = None

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

    @staticmethod
    def _is_near_edge(x_n: float, y_n: float, m: float) -> bool:
        return (x_n <= m) or (x_n >= 1.0 - m) or (y_n <= m) or (y_n >= 1.0 - m)

    def _clamp_crop_center(self, cx: int, cy: int, size: int) -> tuple[int, int]:
        """crop가 화면 밖으로 나가 padding/거울이 생기지 않도록 중심을 clamp"""
        h, w = self.H, self.W
        half = int(size) // 2

        # size가 화면보다 큰 경우 방어
        if half <= 0:
            return int(cx), int(cy)

        # 중심이 가질 수 있는 범위: [half, w-half-1], [half, h-half-1]
        cx_min = half
        cx_max = max(half, w - half - 1)
        cy_min = half
        cy_max = max(half, h - half - 1)

        cx_c = int(np.clip(cx, cx_min, cx_max))
        cy_c = int(np.clip(cy, cy_min, cy_max))
        return cx_c, cy_c

    def _crop_square_bgr_with_mask(self, img_bgr, cx, cy, size):
        """
        crop과 함께:
          - valid_mask_u8: 실제 화면에서 온 픽셀=255, padding=0 (size x size)
        를 리턴.
        """
        h, w = img_bgr.shape[:2]
        size = int(size)
        half = size // 2

        x1 = int(cx - half)
        y1 = int(cy - half)
        x2 = x1 + size
        y2 = y1 + size

        # 완전히 화면 안
        if (0 <= x1) and (0 <= y1) and (x2 <= w) and (y2 <= h):
            crop = img_bgr[y1:y2, x1:x2]
            valid = np.full((size, size), 255, dtype=np.uint8)
            return crop, valid

        # padding 발생 (clamp를 켜면 이 케이스가 거의 사라짐)
        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - w)
        pad_b = max(0, y2 - h)

        img_pad = cv2.copyMakeBorder(
            img_bgr, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_REFLECT_101
        )

        base_valid = np.full((h, w), 255, dtype=np.uint8)
        valid_pad = cv2.copyMakeBorder(
            base_valid, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_CONSTANT, value=0
        )

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

        # 1) 너무 낮은 conf는 기본 홀드
        if conf < float(self.conf_update_thr):
            self._lost_obs += 1
            if self._lost_obs >= int(self.lost_patience_obs):
                self._lost_obs = 0
                return (x_n, y_n, conf, True, "FORCE_LOWCONF")
            return (float(prev_xy[0]), float(prev_xy[1]), float(prev_c), False, "LOWCONF_HOLD")

        d = self._dist_norm((x_n, y_n), prev_xy)

        # 2) 점프면: 기본 need 기준으로 accept/reject
        if d > float(self.max_jump_norm_obs):
            need = max(1e-6, prev_c) * float(self.jump_allow_conf_gain_obs)

            # ✅ "가장자리/코너로 갑자기 점프"는 더 빡세게
            m = float(self.edge_margin_norm)
            prev_edge = self._is_near_edge(float(prev_xy[0]), float(prev_xy[1]), m)
            now_edge = self._is_near_edge(x_n, y_n, m)

            if (now_edge and (not prev_edge)):
                need = need * float(self.edge_jump_conf_gain)
                if conf < float(self.edge_jump_min_conf):
                    self._lost_obs += 1
                    if self._lost_obs >= int(self.lost_patience_obs):
                        # ✅ 여기서 "가장자리로 FORCE"를 해버리면,
                        #   코너 가짜패턴(거울)과 결합해서 lock이 코너로 빨려들기 쉬움.
                        #   그래서 FORCE_EDGE_JUMP 대신 FORCE_JUMP(일반)로만 처리.
                        self._lost_obs = 0
                        return (float(prev_xy[0]), float(prev_xy[1]), float(prev_c), True, "FORCE_EDGE_HOLD")
                    return (float(prev_xy[0]), float(prev_xy[1]), float(prev_c), False, "EDGE_JUMP_REJECT")

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
                obs_ch0_01[0:p, p:2 * p] = y_n
                obs_ch0_01[0:p, 2 * p:3 * p] = c
        except Exception:
            pass
        return obs_ch0_01

    def _compute_bullet_mask_u8(self, crop_bgr: np.ndarray) -> np.ndarray:
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

    def _center_speed_limit(self, prev_xy_f, target_xy_i):
        """프레임당 이동량을 max_speed로 제한 (딜레이 최소화 스무딩)"""
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

        # 2) crop 중심 안정화: speed-limit + (옵션) micro EMA
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

        # ✅ (핵심) crop이 화면 밖으로 나가서 거울/코너 패턴 생기지 않도록 clamp
        if self.clamp_crop_inside:
            cx_s, cy_s = self._clamp_crop_center(cx_s, cy_s, self.crop_size)
            # clamp 후 center_f도 같이 맞춰줘서 "다음 프레임에 다시 밖으로 밀리는" 현상 방지
            self._crop_center_f = (float(cx_s), float(cy_s))

        crop_bgr, valid_u8 = self._crop_square_bgr_with_mask(img_bgr, cx_s, cy_s, self.crop_size)

        # ✅ valid mask (0..1), soft blur 옵션
        if self.enable_valid_mask:
            valid01 = (valid_u8.astype(np.float32) / 255.0)
            k = int(self.valid_mask_soft_blur)
            if k and k >= 3 and (k % 2 == 1):
                valid01 = cv2.GaussianBlur(valid01, (k, k), 0)
            valid01 = np.clip(valid01, 0.0, 1.0).astype(np.float32, copy=False)
        else:
            valid01 = None

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

            risk_now = self._compute_risk_heat_centered(bullet_mask_u8, center_xy=center_xy)

            # ✅ max-hold + decay (증가 즉시, 감소 완만)
            if self._risk_hold_crop_01 is None or self._risk_hold_crop_01.shape != risk_now.shape:
                self._risk_hold_crop_01 = risk_now.astype(np.float32, copy=True)
            else:
                decay = float(np.clip(self.risk_decay, 0.0, 1.0))
                self._risk_hold_crop_01 = np.maximum(self._risk_hold_crop_01 * decay, risk_now).astype(np.float32, copy=False)

            risk_crop_01 = self._risk_hold_crop_01
        else:
            bullet_mask_u8 = np.zeros((self.crop_size, self.crop_size), dtype=np.uint8)
            risk_crop_01 = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)

        # ✅ padding(거울) 영역이 입력에서 과신호가 되지 않도록 마스크 적용
        if valid01 is not None:
            # uint8에 바로 곱하면 경계가 너무 날카로울 수 있어서 float로 처리 후 복원
            crop_gray_u8 = np.clip(crop_gray_u8.astype(np.float32) * valid01, 0, 255).astype(np.uint8)
            diff_u8 = np.clip(diff_u8.astype(np.float32) * valid01, 0, 255).astype(np.uint8)
            bullet_mask_u8 = np.clip(bullet_mask_u8.astype(np.float32) * valid01, 0, 255).astype(np.uint8)
            risk_crop_01 = (risk_crop_01.astype(np.float32) * valid01).astype(np.float32)

        # 4) resize
        interp = cv2.INTER_AREA if self.crop_size >= self.obs_out_size else cv2.INTER_LINEAR

        ch0 = cv2.resize(crop_gray_u8, (self.obs_out_size, self.obs_out_size), interpolation=interp).astype(np.float32) / 255.0
        ch1 = cv2.resize(diff_u8, (self.obs_out_size, self.obs_out_size), interpolation=interp).astype(np.float32) / 255.0
        ch2 = cv2.resize(bullet_mask_u8, (self.obs_out_size, self.obs_out_size), interpolation=interp).astype(np.float32) / 255.0
        ch3 = cv2.resize(risk_crop_01, (self.obs_out_size, self.obs_out_size), interpolation=interp).astype(np.float32)

        # (옵션) post blur로 깜빡임/alias 완화
        k = int(self.post_resize_blur_ksize)
        if k and k >= 3 and (k % 2 == 1):
            ch0 = cv2.GaussianBlur(ch0, (k, k), 0)
            ch1 = cv2.GaussianBlur(ch1, (k, k), 0)
            ch2 = cv2.GaussianBlur(ch2, (k, k), 0)
            ch3 = cv2.GaussianBlur(ch3, (k, k), 0)

        # 5) meta는 ch0에만
        ch0 = self._inject_meta_pixels_ch0_only(ch0)

        obs4 = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32, copy=False)

        # 6) 디버그: ch3(최종 입력과 동일) 표시
        if self.show_obs_debug:
            try:
                self._ensure_obs_window()
                dbg = (np.clip(ch3, 0.0, 1.0) * 255.0).astype(np.uint8)
                vis = cv2.cvtColor(dbg, cv2.COLOR_GRAY2BGR)
                cv2.imshow(self.win_crop, vis)
                cv2.waitKey(1)
            except Exception:
                pass

        return obs4
