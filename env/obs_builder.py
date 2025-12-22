# env/obs_builder.py
import time
import cv2
import numpy as np

from env.reimu_detector import ReimuDetector


class ObsBuilder:
    def __init__(self, screen, debug_viz=None, obs_out_size=84, crop_size=160, use_fallback_full_preprocess=True):
        self.screen = screen
        self.debug = debug_viz

        self.obs_out_size = int(obs_out_size)
        self.crop_size = int(crop_size)
        self.use_fallback_full_preprocess = bool(use_fallback_full_preprocess)

        img0 = self.screen.capture()
        h0, w0 = img0.shape[:2]
        self.H, self.W = h0, w0

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
        self.last_xy_norm = (0.5, 0.78)  # playfield norm 0..1
        self.last_conf = 0.0

        # ✅ 메타 픽셀 설정 (CNN이 잘 읽게 4x4)
        self.meta_patch = 4

        # -------------------------
        # PROFILING (ObsBuilder 내부)
        # -------------------------
        self.prof_enabled = True
        self.prof_every = 200

        self._prof_i = 0
        self._t_det = 0.0
        self._t_crop = 0.0
        self._t_gray_resize = 0.0
        self._t_meta = 0.0
        self._t_fallback = 0.0

        # playfield width 캐시 (ratio가 고정이면 매번 계산할 필요 없음)
        self._playfield_ratio = float(getattr(self.screen, "PLAYFIELD_RIGHT_RATIO", 0.70))
        self._playfield_w = max(1, min(self.W, int(self.W * self._playfield_ratio)))

    def reset(self):
        if hasattr(self.det, "reset"):
            self.det.reset()
        self.player_center = (self.W // 2, int(self.H * 0.78))
        self._last_conf = 0.0
        self.last_xy_norm = (0.5, 0.78)
        self.last_conf = 0.0
        self._dbg_last = None

        self._prof_i = 0
        self._t_det = 0.0
        self._t_crop = 0.0
        self._t_gray_resize = 0.0
        self._t_meta = 0.0
        self._t_fallback = 0.0

    def _crop_square_bgr(self, img_bgr, cx, cy, size):
        """
        ✅ fast-path:
        - 크롭 영역이 프레임 내부에 완전히 들어오면 -> 슬라이스만(복사 없음)
        - 프레임 밖으로 나가면 -> 그 때만 copyMakeBorder
        """
        h, w = img_bgr.shape[:2]
        size = int(size)
        half = size // 2

        x1 = int(cx - half)
        y1 = int(cy - half)
        x2 = x1 + size
        y2 = y1 + size

        # ✅ padding 필요 없는 일반 케이스 (대부분 여기로 들어오길 기대)
        if (0 <= x1) and (0 <= y1) and (x2 <= w) and (y2 <= h):
            crop = img_bgr[y1:y2, x1:x2]
            # crop은 보통 contiguous가 아닐 수 있지만, cv2.cvtColor가 잘 처리함
            return crop

        # padding 필요한 케이스만 border
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
        # 안전장치
        if crop.shape[0] != size or crop.shape[1] != size:
            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
        return crop

    def _playfield_norm_to_full_xy(self, x_n: float, y_n: float) -> tuple[int, int]:
        # ✅ playfield_w는 캐시 사용
        cx = int(np.clip(x_n * self._playfield_w, 0, self._playfield_w - 1))
        cy = int(np.clip(y_n * self.H, 0, self.H - 1))
        return cx, cy

    def on_player_death(self):
        if hasattr(self.det, "on_player_death"):
            self.det.on_player_death()

    def _inject_meta_pixels(self, obs01: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()

        try:
            x_n, y_n = self.last_xy_norm
            c = float(self.last_conf)

            x_n = float(np.clip(x_n, 0.0, 1.0))
            y_n = float(np.clip(y_n, 0.0, 1.0))
            c = float(np.clip(c, 0.0, 1.0))

            p = int(self.meta_patch)
            if obs01.shape[0] >= p and obs01.shape[1] >= p * 3:
                obs01[0:p, 0:p] = x_n
                obs01[0:p, p:p * 2] = y_n
                obs01[0:p, p * 2:p * 3] = c
        except Exception:
            pass

        if self.prof_enabled:
            self._t_meta += (time.perf_counter() - t0)
        return obs01

    def _prof_print_if_needed(self):
        if not self.prof_enabled:
            return
        self._prof_i += 1
        if (self._prof_i % self.prof_every) != 0:
            return

        n = float(self._prof_i)
        # ms/step
        det_ms = (self._t_det / n) * 1000.0
        crop_ms = (self._t_crop / n) * 1000.0
        gray_ms = (self._t_gray_resize / n) * 1000.0
        meta_ms = (self._t_meta / n) * 1000.0
        fb_ms = (self._t_fallback / n) * 1000.0

        print(
            f"[OBS_PROF] avg_ms/step | det={det_ms:.2f} crop={crop_ms:.2f} "
            f"gray+resize={gray_ms:.2f} meta={meta_ms:.2f} fallback={fb_ms:.2f}"
        )

    def make_state(self, img_bgr):
        # -------------------------
        # 1) detector
        # -------------------------
        t0 = time.perf_counter()
        det = self.det.step(img_bgr)
        if self.prof_enabled:
            self._t_det += (time.perf_counter() - t0)

        if det is None:
            cx, cy = self.player_center
            conf = 0.0
            self._dbg_last = None
        else:
            x_n, y_n, conf, logits = det
            x_n = float(np.clip(x_n, 0.0, 1.0))
            y_n = float(np.clip(y_n, 0.0, 1.0))
            conf = float(conf)

            # 정책 입력 안정화
            self.last_xy_norm = (x_n, y_n)
            self.last_conf = conf

            cx_new, cy_new = self._playfield_norm_to_full_xy(x_n, y_n)

            if conf >= self.conf_update_thr:
                cx, cy = cx_new, cy_new
                self.player_center = (cx, cy)
                self._last_conf = conf
            else:
                cx, cy = self.player_center

            x_raw, y_raw = x_n, y_n
            try:
                if hasattr(self.det, "last_raw_xy") and (self.det.last_raw_xy is not None):
                    x_raw, y_raw = self.det.last_raw_xy
            except Exception:
                pass

            self._dbg_last = (x_n, y_n, conf, logits, float(x_raw), float(y_raw))

        # -------------------------
        # 2) fallback
        # -------------------------
        if self.use_fallback_full_preprocess and ((det is None) or (float(conf) <= 1e-6)):
            t0 = time.perf_counter()

            full = self.screen.preprocess(img_bgr)  # float32 0..1, 보통 84x84
            if full.shape != (self.obs_out_size, self.obs_out_size):
                full = cv2.resize(full, (self.obs_out_size, self.obs_out_size), interpolation=cv2.INTER_AREA)
            if full.dtype != np.float32:
                full = full.astype(np.float32)

            full = self._inject_meta_pixels(full)

            if self.prof_enabled:
                self._t_fallback += (time.perf_counter() - t0)
                self._prof_print_if_needed()
            return full

        # -------------------------
        # 3) crop
        # -------------------------
        t0 = time.perf_counter()
        crop_bgr = self._crop_square_bgr(img_bgr, cx, cy, self.crop_size)
        if self.prof_enabled:
            self._t_crop += (time.perf_counter() - t0)

        # -------------------------
        # 4) gray + resize
        # -------------------------
        t0 = time.perf_counter()
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        interp = cv2.INTER_AREA if self.crop_size >= self.obs_out_size else cv2.INTER_LINEAR
        obs = cv2.resize(crop_gray, (self.obs_out_size, self.obs_out_size), interpolation=interp)
        obs = obs.astype(np.float32) / 255.0
        if self.prof_enabled:
            self._t_gray_resize += (time.perf_counter() - t0)

        # 5) meta
        obs = self._inject_meta_pixels(obs)

        if self.prof_enabled:
            self._prof_print_if_needed()
        return obs
