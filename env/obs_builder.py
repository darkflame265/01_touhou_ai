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

        # ✅ 레이무 검출기
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

        # player center (fallback 기준)
        self.player_center = (w0 // 2, int(h0 * 0.78))
        self._last_conf = 0.0
        self.conf_update_thr = 0.02

        # 게이트
        self.max_jump_norm_obs = 0.18
        self.jump_allow_conf_gain_obs = 2.0
        self.lost_patience_obs = 10
        self._lost_obs = 0

        # 디버그용
        self._dbg_last = None
        self.last_xy_norm = (0.5, 0.78)
        self.last_conf = 0.0
        self.meta_patch = 4

        # -------------------------
        # ✅ (CHANGED) crop 디버그 창: 1개만
        # -------------------------
        self.show_obs_debug = False
        self.win_crop = "OBS_CROP"

        # 창 위치(더 오른쪽으로)
        # - 모니터 해상도에 따라 더 키워도 됨
        self.win_x = 1650
        self.win_y = 60

        # 보기 좋은 표시 크기(윈도우 크기)
        # - 실제 데이터는 crop_size지만, 화면에 크게 보여주기 위해 resizeWindow만 키움
        self.win_w = 520
        self.win_h = 520

        self._crop_win_inited = False

        # 디버그 캐시
        self.last_crop_gray_u8 = None   # crop_size x crop_size (uint8)
        self.last_obs_u8 = None         # obs_out_size x obs_out_size (uint8)  # (창은 안 띄우지만 저장은 가능)

        # PROF
        self.prof_enabled = True
        self.prof_every = 200
        self._prof_i = 0
        self._t_det = 0.0
        self._t_crop = 0.0
        self._t_gray_resize = 0.0
        self._t_meta = 0.0
        self._t_fallback = 0.0

        # playfield width 캐시
        self._playfield_ratio = float(getattr(self.screen, "PLAYFIELD_RIGHT_RATIO", 0.70))
        self._playfield_w = max(1, min(self.W, int(self.W * self._playfield_ratio)))

    def reset(self):
        if hasattr(self.det, "reset"):
            self.det.reset()

        self.player_center = (self.W // 2, int(self.H * 0.78))
        self._last_conf = 0.0
        self._lost_obs = 0

        self.last_xy_norm = (0.5, 0.78)
        self.last_conf = 0.0
        self._dbg_last = None

        self.last_crop_gray_u8 = None
        self.last_obs_u8 = None

        self._prof_i = 0
        self._t_det = 0.0
        self._t_crop = 0.0
        self._t_gray_resize = 0.0
        self._t_meta = 0.0
        self._t_fallback = 0.0

    # -------------------------
    # (NEW) crop 디버그창 1개만 init
    # -------------------------
    def _ensure_crop_window(self):
        if self._crop_win_inited:
            return
        try:
            cv2.namedWindow(self.win_crop, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.win_crop, int(self.win_w), int(self.win_h))
            cv2.moveWindow(self.win_crop, int(self.win_x), int(self.win_y))
        except Exception:
            pass
        self._crop_win_inited = True

    def _show_crop_debug(self, crop_gray_u8: np.ndarray):
        if not self.show_obs_debug:
            return
        try:
            self._ensure_crop_window()

            # crop_gray_u8는 crop_size 크기(예: 256x256) 그대로.
            # 창 크기만 키워서 보기 편하게.
            cv2.imshow(self.win_crop, crop_gray_u8)
            cv2.waitKey(1)
        except Exception:
            pass

    # -------------------------
    # crop util
    # -------------------------
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

    def on_player_death(self):
        if hasattr(self.det, "on_player_death"):
            self.det.on_player_death()
        self._lost_obs = 0

    def _inject_meta_pixels(self, obs01: np.ndarray) -> np.ndarray:
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
        return obs01

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

        # 1) low conf -> hold
        if conf < float(self.conf_update_thr):
            self._lost_obs += 1
            if self._lost_obs >= int(self.lost_patience_obs):
                self._lost_obs = 0
                return (x_n, y_n, conf, True, "FORCE_LOWCONF")
            return (float(prev_xy[0]), float(prev_xy[1]), float(prev_c), False, "LOWCONF_HOLD")

        # 2) jump gate
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

        # 3) ok
        self._lost_obs = 0
        return (x_n, y_n, conf, True, "OK")

    def make_state(self, img_bgr):
        # detector
        det = self.det.step(img_bgr)

        if det is None:
            cx, cy = self.player_center
            conf = 0.0
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
                self._last_conf = float(c_use)
            else:
                cx, cy = self.player_center

            x_raw, y_raw = float(x_n), float(y_n)
            try:
                if hasattr(self.det, "last_raw_xy") and (self.det.last_raw_xy is not None):
                    x_raw, y_raw = self.det.last_raw_xy
            except Exception:
                pass

            self._dbg_last = (float(x_use), float(y_use), float(c_use), logits, float(x_raw), float(y_raw), str(reason))

        # ✅ fallback: full preprocess
        if self.use_fallback_full_preprocess and ((det is None) or (float(conf) <= 1e-6)):
            full = self.screen.preprocess(img_bgr)  # float32 0..1
            if full.shape != (self.obs_out_size, self.obs_out_size):
                full = cv2.resize(full, (self.obs_out_size, self.obs_out_size), interpolation=cv2.INTER_AREA)
            if full.dtype != np.float32:
                full = full.astype(np.float32)

            full = self._inject_meta_pixels(full)

            # (선택) fallback 상황에서도 "뭔가 보이게" 하고 싶으면,
            # 84/128 관측을 크게 띄워서 보면 됨.
            # 지금 요청은 "crop 하나만"이지만, fallback 때 crop이 없으니
            # 대신 full을 보여주도록 유지(원치 않으면 아래 4줄 주석 처리).
            if self.show_obs_debug:
                u8 = (np.clip(full, 0, 1) * 255).astype(np.uint8)
                self.last_obs_u8 = u8
                self._show_crop_debug(u8)

            return full

        # crop
        crop_bgr = self._crop_square_bgr(img_bgr, cx, cy, self.crop_size)
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        # ✅ crop 창 1개만 표시
        self.last_crop_gray_u8 = crop_gray
        self._show_crop_debug(crop_gray)

        # gray + resize -> obs
        interp = cv2.INTER_AREA if self.crop_size >= self.obs_out_size else cv2.INTER_LINEAR
        obs = cv2.resize(crop_gray, (self.obs_out_size, self.obs_out_size), interpolation=interp)
        obs = obs.astype(np.float32) / 255.0

        # meta
        obs = self._inject_meta_pixels(obs)

        # obs 캐시는 유지(창은 안 띄움)
        try:
            self.last_obs_u8 = (np.clip(obs, 0, 1) * 255).astype(np.uint8)
        except Exception:
            pass

        return obs
