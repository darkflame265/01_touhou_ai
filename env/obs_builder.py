# env/obs_builder.py
import cv2
import numpy as np

from env.reimu_detector import ReimuDetector


class ObsBuilder:
    def __init__(self, screen, debug_viz=None, obs_out_size=84, crop_size=160, use_fallback_full_preprocess=True):
        self.screen = screen
        self.debug = debug_viz

        self.obs_out_size = obs_out_size
        self.crop_size = crop_size
        self.use_fallback_full_preprocess = use_fallback_full_preprocess

        img0 = self.screen.capture()
        h0, w0 = img0.shape[:2]
        self.H, self.W = h0, w0

        # === UI 마스킹 파라미터(기존 유지) ===
        self.ui_cut_ratio = 0.66
        self.ui_cut_bottom_ratio = 1.00

        # ✅ 레이무 검출기(히트맵)
        self.det = ReimuDetector(
            screen=self.screen,
            weight_path="weights/reimu_heatmap_best.pt",
            beta=12.0,
            prior_strength=1.0,
            ema_alpha=0.75,
        )

        # 마지막으로 믿을만한 플레이어 중심(풀프레임 좌표)
        self.player_center = (w0 // 2, int(h0 * 0.78))
        self._last_conf = 0.0

        # conf가 너무 낮으면 위치 업데이트 안 함
        self.conf_update_thr = 0.02

        # 디버그용 (GameEnv에서 사용)
        self._dbg_last = None

        # ✅ 정책 입력(관측 이미지)에 박아 넣을 좌표/신뢰도 캐시
        # det이 None이거나 conf가 낮아도 "마지막 값"을 유지해서 안정적으로 제공
        self.last_xy_norm = (0.5, 0.78)  # playfield norm 0..1
        self.last_conf = 0.0

        # ✅ 메타 픽셀 설정 (CNN이 잘 읽게 4x4로 큼직하게)
        self.meta_patch = 4
        # 배치: x / y / conf 를 좌상단에 나란히
        # (0:4, 0:4)=x, (0:4, 4:8)=y, (0:4, 8:12)=conf

    def reset(self):
        if hasattr(self.det, "reset"):
            self.det.reset()
        self.player_center = (self.W // 2, int(self.H * 0.78))
        self._last_conf = 0.0

        # 좌표도 기본값으로 리셋
        self.last_xy_norm = (0.5, 0.78)
        self.last_conf = 0.0
        self._dbg_last = None

    def _crop_square_bgr(self, img_bgr, cx, cy, size):
        h, w = img_bgr.shape[:2]
        size = int(size)
        half = size // 2

        x1 = int(round(cx - half))
        y1 = int(round(cy - half))
        x2 = x1 + size
        y2 = y1 + size

        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - w)
        pad_b = max(0, y2 - h)

        if pad_l or pad_t or pad_r or pad_b:
            img_bgr = cv2.copyMakeBorder(
                img_bgr,
                pad_t, pad_b, pad_l, pad_r,
                borderType=cv2.BORDER_REFLECT_101
            )
            x1 += pad_l
            y1 += pad_t
            x2 += pad_l
            y2 += pad_t

        crop = img_bgr[y1:y2, x1:x2]
        if crop.shape[0] != size or crop.shape[1] != size:
            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
        return crop

    def _mask_out_ui_region(self, img_bgr: np.ndarray) -> np.ndarray:
        out = img_bgr.copy()
        x_ui = int(self.W * self.ui_cut_ratio)
        out[:, x_ui:, :] = 0
        if self.ui_cut_bottom_ratio < 1.0:
            y0 = int(self.H * self.ui_cut_bottom_ratio)
            out[y0:, x_ui:, :] = 0
        return out

    def _playfield_norm_to_full_xy(self, x_n: float, y_n: float) -> tuple[int, int]:
        """
        detector는 screen.get_playfield_gray() 기준 0..1 좌표를 내놓는다.
        screen.get_playfield_gray()는 기본적으로:
          - full gray에서 좌측 playfield만 자름 (w * PLAYFIELD_RIGHT_RATIO)
          - y는 전체 높이 그대로
        """
        playfield_w = int(self.W * getattr(self.screen, "PLAYFIELD_RIGHT_RATIO", 0.70))
        playfield_w = max(1, min(self.W, playfield_w))
        cx = int(np.clip(x_n * playfield_w, 0, playfield_w - 1))
        cy = int(np.clip(y_n * self.H, 0, self.H - 1))
        return cx, cy

    def on_player_death(self):
        if hasattr(self.det, "on_player_death"):
            self.det.on_player_death()

    def _inject_meta_pixels(self, obs01: np.ndarray) -> np.ndarray:
        """
        obs01: float32 0..1 (H,W)
        좌상단에 x,y,conf를 큰 패치로 박아 넣는다.
        """
        try:
            x_n, y_n = self.last_xy_norm
            c = float(self.last_conf)

            x_n = float(np.clip(x_n, 0.0, 1.0))
            y_n = float(np.clip(y_n, 0.0, 1.0))
            c = float(np.clip(c, 0.0, 1.0))

            p = int(self.meta_patch)
            # 크기 안전장치
            if obs01.shape[0] < p or obs01.shape[1] < p * 3:
                return obs01

            obs01[0:p, 0:p] = x_n
            obs01[0:p, p:p * 2] = y_n
            obs01[0:p, p * 2:p * 3] = c
        except Exception:
            pass
        return obs01

    def make_state(self, img_bgr):
        # (선택) 디버그용 UI 마스킹된 화면 (현재는 사용 안 하지만 유지)
        _ = self._mask_out_ui_region(img_bgr)

        det = self.det.step(img_bgr)

        if det is None:
            cx, cy = self.player_center
            conf = 0.0
            self._dbg_last = None
            # last_xy_norm/last_conf는 "이전 값" 유지 (정책 입력 안정화)
        else:
            x_n, y_n, conf, logits = det
            x_n = float(np.clip(x_n, 0.0, 1.0))
            y_n = float(np.clip(y_n, 0.0, 1.0))
            conf = float(conf)

            # ✅ 정책 입력용 좌표는 항상 업데이트(신뢰도와 함께)
            # conf가 낮아도 값은 넣되, conf도 같이 넣어서 "불확실함"을 모델이 알게 함
            self.last_xy_norm = (x_n, y_n)
            self.last_conf = conf

            cx_new, cy_new = self._playfield_norm_to_full_xy(x_n, y_n)

            # conf가 충분할 때만 크롭 중심 업데이트
            if conf >= self.conf_update_thr:
                cx, cy = cx_new, cy_new
                self.player_center = (cx, cy)
                self._last_conf = conf
            else:
                cx, cy = self.player_center

            # ✅ raw 좌표(표시용)도 같이 캐시
            x_raw, y_raw = x_n, y_n
            try:
                if hasattr(self.det, "last_raw_xy") and (self.det.last_raw_xy is not None):
                    x_raw, y_raw = self.det.last_raw_xy
            except Exception:
                pass

            # 🔥 디버그용 캐시: (lock_xy, conf, logits, raw_xy)
            self._dbg_last = (x_n, y_n, conf, logits, float(x_raw), float(y_raw))

        # ===== (옵션) fallback: 완전 못 찾는 상황이면 전체 preprocess를 관측으로 =====
        if self.use_fallback_full_preprocess:
            if (det is None) or (float(conf) <= 1e-6):
                full = self.screen.preprocess(img_bgr)

                if full.shape != (self.obs_out_size, self.obs_out_size):
                    full = cv2.resize(
                        full, (self.obs_out_size, self.obs_out_size),
                        interpolation=cv2.INTER_AREA
                    )

                if full.dtype != np.float32:
                    full = full.astype(np.float32)

                # ✅ fallback에서도 메타 픽셀 주입
                full = self._inject_meta_pixels(full)
                return full

        # ===== 플레이어 중심 crop 관측 =====
        crop_bgr = self._crop_square_bgr(img_bgr, cx, cy, self.crop_size)

        # ✅ (선택) UI 노이즈 줄이고 싶으면 여기서 UI영역 마스킹 후 crop하는 방식도 가능하지만,
        # 지금은 입력을 바꾸지 않기 위해 유지.

        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        obs = cv2.resize(crop_gray, (self.obs_out_size, self.obs_out_size), interpolation=cv2.INTER_LINEAR)
        obs = obs.astype(np.float32) / 255.0

        # ✅ 메타 픽셀 주입 (정책이 x,y,conf를 직접 읽게 됨)
        obs = self._inject_meta_pixels(obs)
        return obs
