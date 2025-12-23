# env/obs_builder.py
import time
import cv2
import numpy as np

from env.reimu_detector import ReimuDetector


class ObsBuilder:
    def __init__(
        self,
        screen,
        debug_viz=None,
        obs_out_size=84,
        crop_size=160,
        use_fallback_full_preprocess=True,
        # 디버그(탄막 강조) 옵션
        debug_bullets=True,
    ):
        self.screen = screen
        self.debug = debug_viz

        self.obs_out_size = int(obs_out_size)
        self.crop_size = int(crop_size)
        self.use_fallback_full_preprocess = bool(use_fallback_full_preprocess)
        self.debug_bullets = bool(debug_bullets)

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

        # 마지막으로 믿을만한 플레이어 중심(풀프레임 좌표)
        self.player_center = (w0 // 2, int(h0 * 0.78))
        self._last_conf = 0.0

        # conf가 너무 낮으면 위치 업데이트 안 함
        self.conf_update_thr = 0.02

        # -------------------------
        # ✅ ObsBuilder 추가 점프 억제 게이트
        # -------------------------
        self.max_jump_norm_obs = 0.18
        self.jump_allow_conf_gain_obs = 2.0
        self.lost_patience_obs = 10
        self._lost_obs = 0

        # 디버그용(외부에서 읽을 수 있게)
        self._dbg_last = None

        # 정책 입력(관측 이미지)에 박아 넣을 좌표/신뢰도 캐시
        self.last_xy_norm = (0.5, 0.78)  # playfield norm 0..1
        self.last_conf = 0.0

        # 메타 픽셀 설정 (CNN이 잘 읽게 4x4)
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
        self._t_dbg = 0.0

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

        self._prof_i = 0
        self._t_det = 0.0
        self._t_crop = 0.0
        self._t_gray_resize = 0.0
        self._t_meta = 0.0
        self._t_fallback = 0.0
        self._t_dbg = 0.0

    def on_player_death(self):
        if hasattr(self.det, "on_player_death"):
            self.det.on_player_death()
        self._lost_obs = 0

    # -------------------------
    # utils
    # -------------------------
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
        det_ms = (self._t_det / n) * 1000.0
        crop_ms = (self._t_crop / n) * 1000.0
        gray_ms = (self._t_gray_resize / n) * 1000.0
        meta_ms = (self._t_meta / n) * 1000.0
        fb_ms = (self._t_fallback / n) * 1000.0
        dbg_ms = (self._t_dbg / n) * 1000.0

        # print(
        #     f"[OBS_PROF] avg_ms/step | det={det_ms:.2f} crop={crop_ms:.2f} "
        #     f"gray+resize={gray_ms:.2f} meta={meta_ms:.2f} fallback={fb_ms:.2f} dbg={dbg_ms:.2f}"
        # )

    def _gate_xy_update(self, x_n, y_n, conf):
        """
        ✅ detector가 주는 (x,y)가 순간적으로 튀는 문제를 ObsBuilder에서 한 번 더 방지.
        - conf가 낮으면: 이전 last_xy_norm 유지
        - conf가 충분해도 "점프"면: conf가 이전 대비 충분히 좋아질 때만 점프 허용
        - 낮은 conf/거절이 너무 오래 지속되면: 강제로 새 값 허용
        """
        prev_xy = self.last_xy_norm
        prev_c = float(self.last_conf)

        x_n = float(np.clip(x_n, 0.0, 1.0))
        y_n = float(np.clip(y_n, 0.0, 1.0))
        conf = float(conf)

        # 1) conf 낮으면 업데이트 금지
        if conf < float(self.conf_update_thr):
            self._lost_obs += 1
            if self._lost_obs >= int(self.lost_patience_obs):
                self._lost_obs = 0
                return (x_n, y_n, conf, True, "FORCE_LOWCONF")
            return (float(prev_xy[0]), float(prev_xy[1]), float(prev_c), False, "LOWCONF_HOLD")

        # 2) 점프 검사
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

        # 3) 정상 업데이트
        self._lost_obs = 0
        return (x_n, y_n, conf, True, "OK")

    # -------------------------
    # ✅ Debug: 탄막 강조 맵 만들기 (학습 입력은 건드리지 않음)
    # -------------------------
    def _make_bullet_emphasis(self, crop_gray_01: np.ndarray) -> np.ndarray:
        """
        crop_gray_01: float32 0..1 (obs_out_size x obs_out_size)
        반환: float32 0..1 (탄막/작은 밝은 점 위주로 강조)
        """
        g8 = np.clip(crop_gray_01 * 255.0, 0, 255).astype(np.uint8)

        # 작은 밝은 점(탄막)을 강조하기 위한 blackhat
        # 커널은 너무 크면 글로우/배경까지 먹으니 5~9 사이 추천
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bh = cv2.morphologyEx(g8, cv2.MORPH_BLACKHAT, k)

        # 약간의 대비 확장
        bh = cv2.GaussianBlur(bh, (3, 3), 0)

        # 정규화 -> 0..1
        bh_f = bh.astype(np.float32)
        mx = float(bh_f.max()) if bh_f.size else 0.0
        if mx > 1e-6:
            bh_f = bh_f / mx
        else:
            bh_f[:] = 0.0
        return bh_f

    def _maybe_debug_show(self, crop_bgr, obs01, bullet01=None):
        """
        debug_viz가 특정 메서드를 갖고 있으면 호출하는 방식(프로젝트 호환성 유지).
        - DebugViz / ReimuDebugViz 어떤 걸 쓰든 에러 없이 넘어가게 try/hasattr 처리.
        """
        if self.debug is None:
            return
        t0 = time.perf_counter()
        try:
            # 1) raw crop 보여주기(가능하면)
            if hasattr(self.debug, "show_obs_crop"):
                # show_obs_crop(crop_bgr, obs01, bullet01)
                try:
                    self.debug.show_obs_crop(crop_bgr, obs01, bullet01)
                except TypeError:
                    # 시그니처가 다르면 최소한 crop만
                    self.debug.show_obs_crop(crop_bgr)
            elif hasattr(self.debug, "show_crop"):
                self.debug.show_crop(crop_bgr)

            # 2) obs / bullet 맵 따로
            if hasattr(self.debug, "show_obs"):
                self.debug.show_obs(obs01)
            if bullet01 is not None and hasattr(self.debug, "show_bullets"):
                self.debug.show_bullets(bullet01)
        except Exception:
            pass
        finally:
            if self.prof_enabled:
                self._t_dbg += (time.perf_counter() - t0)

    # -------------------------
    # main
    # -------------------------
    def make_state(self, img_bgr):
        # -------------------------
        # 1) detector
        # -------------------------
        t0 = time.perf_counter()
        det = self.det.step(img_bgr)
        if self.prof_enabled:
            self._t_det += (time.perf_counter() - t0)

        used_conf = 0.0  # ✅ 게이트 통과 conf(=정책/판단에 쓸 값)

        if det is None:
            cx, cy = self.player_center
            self._dbg_last = None
            # last_xy_norm/last_conf는 유지 (이전 값 기반으로 계속 crop)
            used_conf = float(self.last_conf)
        else:
            x_n, y_n, conf_raw, logits = det

            # ✅ ObsBuilder 게이트(점프 억제/저신뢰 홀드)
            x_use, y_use, c_use, used, reason = self._gate_xy_update(x_n, y_n, conf_raw)
            used_conf = float(c_use)

            # 정책 입력/리워드 shaping 안정화를 위해 "게이트 통과값"을 저장
            self.last_xy_norm = (float(x_use), float(y_use))
            self.last_conf = float(c_use)

            cx_new, cy_new = self._playfield_norm_to_full_xy(x_use, y_use)

            if used:
                cx, cy = cx_new, cy_new
                self.player_center = (cx, cy)
                self._last_conf = float(c_use)
            else:
                cx, cy = self.player_center

            # raw 좌표는 디버그용
            x_raw, y_raw = float(x_n), float(y_n)
            try:
                if hasattr(self.det, "last_raw_xy") and (self.det.last_raw_xy is not None):
                    x_raw, y_raw = self.det.last_raw_xy
            except Exception:
                pass

            # dbg: (used_x, used_y, used_conf, logits, raw_x, raw_y, gate_reason)
            self._dbg_last = (float(x_use), float(y_use), float(c_use), logits, float(x_raw), float(y_raw), str(reason))

        # -------------------------
        # 2) fallback (✅ raw conf가 아니라 used_conf 기준!)
        # -------------------------
        if self.use_fallback_full_preprocess and (det is None or used_conf <= 1e-6):
            t0 = time.perf_counter()

            full = self.screen.preprocess(img_bgr)  # float32 0..1 (보통 84x84)
            if full.shape != (self.obs_out_size, self.obs_out_size):
                full = cv2.resize(full, (self.obs_out_size, self.obs_out_size), interpolation=cv2.INTER_AREA)
            if full.dtype != np.float32:
                full = full.astype(np.float32)

            full = self._inject_meta_pixels(full)

            if self.prof_enabled:
                self._t_fallback += (time.perf_counter() - t0)
                self._prof_print_if_needed()

            # 디버그 표시(가능하면)
            # fallback은 crop_bgr가 없으니 obs만
            self._maybe_debug_show(crop_bgr=None, obs01=full, bullet01=None)
            return full

        # -------------------------
        # 3) crop
        # -------------------------
        t0 = time.perf_counter()
        crop_bgr = self._crop_square_bgr(img_bgr, cx, cy, self.crop_size)
        if self.prof_enabled:
            self._t_crop += (time.perf_counter() - t0)

        # -------------------------
        # 4) gray + resize -> obs01
        # -------------------------
        t0 = time.perf_counter()
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        interp = cv2.INTER_AREA if self.crop_size >= self.obs_out_size else cv2.INTER_LINEAR
        obs = cv2.resize(crop_gray, (self.obs_out_size, self.obs_out_size), interpolation=interp)
        obs01 = obs.astype(np.float32) / 255.0
        if self.prof_enabled:
            self._t_gray_resize += (time.perf_counter() - t0)

        # 5) meta
        obs01 = self._inject_meta_pixels(obs01)

        # -------------------------
        # 6) debug: 탄막 강조 맵(선택)
        # -------------------------
        bullet01 = None
        if self.debug_bullets:
            try:
                bullet01 = self._make_bullet_emphasis(obs01)
            except Exception:
                bullet01 = None

        self._maybe_debug_show(crop_bgr=crop_bgr, obs01=obs01, bullet01=bullet01)

        if self.prof_enabled:
            self._prof_print_if_needed()
        return obs01
