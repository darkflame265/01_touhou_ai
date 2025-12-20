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
            prior_strength=1.0,  # 너가 넣은 아래쪽 선호 프라이어 유지
            ema_alpha=0.75,
        )

        # 마지막으로 믿을만한 플레이어 중심(풀프레임 좌표)
        self.player_center = (w0 // 2, int(h0 * 0.78))
        self._last_conf = 0.0

        # conf가 너무 낮으면 위치 업데이트 안 함
        self.conf_update_thr = 0.02  # soft-argmax peak prob는 작을 수 있음. 너무 빡세게 잡지 말기.

        self._dbg_last = None

    def reset(self):
        if hasattr(self.det, "reset"):
            self.det.reset()
        self.player_center = (self.W // 2, int(self.H * 0.78))
        self._last_conf = 0.0

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
        따라서 full 좌표 변환은:
          full_x = x_n * playfield_width
          full_y = y_n * full_height
        """
        playfield_w = int(self.W * getattr(self.screen, "PLAYFIELD_RIGHT_RATIO", 0.70))
        playfield_w = max(1, min(self.W, playfield_w))
        cx = int(np.clip(x_n * playfield_w, 0, playfield_w - 1))
        cy = int(np.clip(y_n * self.H, 0, self.H - 1))
        return cx, cy
        
    def on_player_death(self):
        if hasattr(self.det, "on_player_death"):
            self.det.on_player_death()


    def make_state(self, img_bgr):
        # (선택) 디버그용 UI 마스킹된 화면
        img_for_dbg = self._mask_out_ui_region(img_bgr)

        det = self.det.step(img_bgr)

        if det is None:
            cx, cy = self.player_center
            conf = 0.0
            self._dbg_last = None
        else:
            x_n, y_n, conf, logits = det
            cx_new, cy_new = self._playfield_norm_to_full_xy(x_n, y_n)

            if conf >= self.conf_update_thr:
                cx, cy = cx_new, cy_new
                self.player_center = (cx, cy)
                self._last_conf = conf
            else:
                cx, cy = self.player_center

            # 🔥 디버그용 캐시
            self._dbg_last = (x_n, y_n, conf, logits)


        # ===== (옵션) fallback: 완전 못 찾는 상황이면 전체 preprocess를 관측으로 =====
        if self.use_fallback_full_preprocess:
            if (det is None) or (float(conf) <= 1e-6):
                full = self.screen.preprocess(img_bgr)  # mode="low"면 기본 84x84

                # ✅ 어떤 크기로 오든 obs_out_size로 강제 통일 (중요!)
                if full.shape != (self.obs_out_size, self.obs_out_size):
                    full = cv2.resize(
                        full, (self.obs_out_size, self.obs_out_size),
                        interpolation=cv2.INTER_AREA
                    )

                # preprocess는 이미 float32 0..1 이므로 그대로 반환
                if full.dtype != np.float32:
                    full = full.astype(np.float32)
                return full


        # ===== 플레이어 중심 crop 관측 =====
        crop_bgr = self._crop_square_bgr(img_bgr, cx, cy, self.crop_size)
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        obs = cv2.resize(crop_gray, (self.obs_out_size, self.obs_out_size), interpolation=cv2.INTER_LINEAR)
        obs = obs.astype(np.float32) / 255.0

        # (참고) 기존 debug.show_tracker는 tracker 객체가 필요해서 여기선 호출 안 함.
        # 원하면 DebugViz에 "show_reimu(cx,cy,conf)" 같은 함수를 따로 만들어 연결하면 됨.

        return obs
