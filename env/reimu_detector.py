# env/reimu_detector.py
import os
import time
from collections import deque

import cv2
import numpy as np
import torch

from vision.models.heatmap_net import HeatmapNet, soft_argmax_2d


class ReimuDetector:
    """
    Heatmap detector that returns:
      (x_norm, y_norm, conf, logits)

    ✅ 성능 최적화 포인트:
    - deque -> 고정 numpy 버퍼로 교체 (np.stack 제거)
    - torch 텐서 생성 최소화 (CPU면 from_numpy view, CUDA면 prealloc + copy_)
    - torch.inference_mode() 사용
    - (옵션) track_prior를 매 프레임이 아니라 N프레임마다 적용 가능
    - (옵션) CUDA FP16
    """

    def __init__(
            self,
            screen,
            weight_path="weights/reimu_heatmap_best.pt",
            beta=12.0,
            prior_strength=1.0,

            # ✅ 변경(튜닝): 더 강하게 스무딩해서 점프 억제
            ema_alpha=0.90,

            device=None,

            # ===== tracking 옵션 =====

            # ✅ 변경(튜닝): 락 주변으로 더 끌어당김(점프 억제 도움)
            track_prior_strength=2.4,

            # ✅ 변경(튜닝): prior를 더 “좁게” (락 근처 선호)
            track_prior_sigma=0.06,

            # ✅ 변경(튜닝): 애매한 conf로 락 잡지 않게
            lock_conf_thr=0.030,

            # ✅ 변경(튜닝): 한번 락 잡으면 멀리 점프 더 어렵게
            max_jump_norm=0.14,

            # ✅ 변경(튜닝): 락을 깨고 점프하려면 훨씬 높은 conf 필요
            jump_allow_conf_gain=2.6,

            # ✅ 변경(튜닝): 잠깐 흔들려도 락을 더 오래 유지
            lost_patience=16,

            # ===== 성능 옵션 =====
            use_fp16=True,            # CUDA에서만 의미 있음
            track_prior_every=1,       # 1=매프레임, 2=2프레임마다, 3=3프레임마다...
            print_prof=True,          # det 내부 프로파일 출력
            prof_every=200,            # 몇 step마다 출력할지
        ):
        self.screen = screen
        self.weight_path = weight_path
        self.beta = float(beta)
        self.prior_strength = float(prior_strength)
        self.ema_alpha = float(ema_alpha)

        self.track_prior_strength = float(track_prior_strength)
        self.track_prior_sigma = float(track_prior_sigma)
        self.lock_conf_thr = float(lock_conf_thr)
        self.max_jump_norm = float(max_jump_norm)
        self.jump_allow_conf_gain = float(jump_allow_conf_gain)
        self.lost_patience = int(lost_patience)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_fp16 = bool(use_fp16 and (self.device.type == "cuda"))
        self.track_prior_every = max(1, int(track_prior_every))

        self.print_prof = bool(print_prof)
        self.prof_every = max(10, int(prof_every))
        self._prof_step = 0

        # CUDA 성능 힌트
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        ckpt = torch.load(self.weight_path, map_location=self.device)
        cfg = ckpt.get("cfg", {})

        self.out_w = int(cfg.get("w", 160))
        self.out_h = int(cfg.get("h", 120))
        self.stack = int(cfg.get("stack", 4))

        self.model = HeatmapNet(in_ch=self.stack, base=32).to(self.device)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.eval()

        if self.use_fp16:
            self.model.half()

        # ===== 고정 버퍼 (stack,H,W) =====
        self._buf_np = np.zeros((self.stack, self.out_h, self.out_w), dtype=np.float32)
        self._buf_len = 0  # warmup 카운트

        # CUDA일 때는 입력 텐서를 미리 만들어두고 copy_만 함
        self._x_cuda = None
        if self.device.type == "cuda":
            dtype = torch.float16 if self.use_fp16 else torch.float32
            self._x_cuda = torch.empty((1, self.stack, self.out_h, self.out_w), device=self.device, dtype=dtype)

        # EMA(부드러운 위치)
        self._ema_xy = None  # np([x,y]) in [0,1]

        # Tracking lock 상태
        self._lock_xy = None
        self._lock_conf = 0.0
        self._lost_count = 0

        # tracking prior용 mesh 캐시
        self._yy = None
        self._xx = None

        # 디버그 표시용
        self.last_raw_xy = None
        self.last_lock_xy = None

        # step 카운터
        self._step_i = 0
        print(f"[DET] device={self.device} fp16={self.use_fp16} track_prior_every={self.track_prior_every} stack={self.stack} out={self.out_w}x{self.out_h}")


    def reset(self):
        self._buf_np.fill(0.0)
        self._buf_len = 0
        self._ema_xy = None
        self._lock_xy = None
        self._lock_conf = 0.0
        self._lost_count = 0
        self.last_raw_xy = None
        self.last_lock_xy = None
        self._step_i = 0

    def on_player_death(self):
        self._lock_xy = None
        self._lock_conf = 0.0
        self._lost_count = 0

    def _apply_bottom_prior(self, logits: torch.Tensor) -> torch.Tensor:
        if self.prior_strength <= 0:
            return logits
        H = logits.shape[-2]
        yy = torch.linspace(0.0, 1.0, H, device=logits.device, dtype=logits.dtype).view(1, 1, H, 1)
        penalty = (1.0 - yy)  # top=1, bottom=0
        return logits - self.prior_strength * penalty

    def _ensure_mesh(self, device, dtype):
        if (self._yy is None) or (self._xx is None) or (self._yy.device != device) or (self._yy.dtype != dtype):
            yy = torch.linspace(0.0, 1.0, self.out_h, device=device, dtype=dtype).view(1, 1, self.out_h, 1)
            xx = torch.linspace(0.0, 1.0, self.out_w, device=device, dtype=dtype).view(1, 1, 1, self.out_w)
            self._yy = yy
            self._xx = xx

    def _apply_track_prior(self, logits: torch.Tensor) -> torch.Tensor:
        if self.track_prior_strength <= 0 or self._lock_xy is None:
            return logits

        self._ensure_mesh(logits.device, logits.dtype)
        x0 = float(self._lock_xy[0])
        y0 = float(self._lock_xy[1])

        dx = (self._xx - x0)
        dy = (self._yy - y0)
        d2 = dx * dx + dy * dy
        sigma2 = max(1e-6, self.track_prior_sigma * self.track_prior_sigma)
        bonus = torch.exp(-0.5 * d2 / sigma2)  # (1,1,H,W)

        return logits + (self.track_prior_strength * bonus)

    def _ema(self, x, y):
        v = np.array([x, y], dtype=np.float32)
        if self._ema_xy is None:
            self._ema_xy = v
        else:
            a = self.ema_alpha
            self._ema_xy = a * v + (1.0 - a) * self._ema_xy
        return float(self._ema_xy[0]), float(self._ema_xy[1])

    @staticmethod
    def _dist(a_xy, b_xy) -> float:
        dx = float(a_xy[0] - b_xy[0])
        dy = float(a_xy[1] - b_xy[1])
        return float((dx * dx + dy * dy) ** 0.5)

    def _push_frame(self, small01: np.ndarray):
        """
        small01: (H,W) float32 0..1
        stack 버퍼에 넣기: [1:] <- [:-1], [0] <- new
        """
        if self.stack > 1:
            self._buf_np[1:] = self._buf_np[:-1]
        self._buf_np[0] = small01
        self._buf_len = min(self.stack, self._buf_len + 1)

    def _make_input_tensor(self):
        """
        (1, C, H, W) 텐서를 만든다.
        - CPU: torch.from_numpy로 view (복사 최소)
        - CUDA: prealloc 텐서에 copy_
        """
        if self.device.type == "cpu":
            x = torch.from_numpy(self._buf_np).unsqueeze(0)  # float32
            return x
        else:
            assert self._x_cuda is not None
            # numpy -> torch(cpu) -> cuda copy (비용 있지만, 매번 새 텐서 생성은 피함)
            x_cpu = torch.from_numpy(self._buf_np).unsqueeze(0)  # float32 CPU
            if self.use_fp16:
                x_cpu = x_cpu.half()
            self._x_cuda.copy_(x_cpu, non_blocking=False)
            return self._x_cuda

    def step(self, img_bgr):
        """
        Returns:
        None (during warmup)
        or (x_norm, y_norm, conf, logits)

        ✅ 디버그 추가(원인 추적용):
        - lock 획득/유지/해제
        - 점프(d > max_jump_norm) 허용/거부
        - low conf로 lock 유지/해제
        - raw/ema/lock 좌표 및 conf, d, lost_count 출력

        사용법(원하면 __init__에 넣거나 런타임에 setattr로 켜도 됨):
        self.track_debug = True
        self.track_debug_every = 60   # N스텝마다 정기 출력
        """
        self._step_i += 1
        t0 = time.perf_counter()

        # -------------------------
        # (DEBUG 옵션)
        # -------------------------
        dbg_on = bool(getattr(self, "track_debug", False))
        dbg_every = int(getattr(self, "track_debug_every", 60))
        if not hasattr(self, "_dbg_i"):
            self._dbg_i = 0
        self._dbg_i += 1

        def _dbg(msg: str):
            if not dbg_on:
                return
            # 너무 스팸이면 every로 제한
            if (dbg_every > 1) and ((self._dbg_i % dbg_every) != 0):
                return
            print(msg)

        # -------------------------
        # 1) playfield gray + resize + normalize
        # -------------------------
        play = self.screen.get_playfield_gray(img_bgr)  # gray playfield
        small = cv2.resize(play, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
        small = small.astype(np.float32)
        small *= (1.0 / 255.0)
        t1 = time.perf_counter()

        # -------------------------
        # 2) stack buffer push
        # -------------------------
        self._push_frame(small)
        if self._buf_len < self.stack:
            return None
        t2 = time.perf_counter()

        # -------------------------
        # 3) input tensor 준비
        # -------------------------
        x = self._make_input_tensor()
        if self.device.type == "cuda" and not self.use_fp16:
            x = x.float()
        t3 = time.perf_counter()

        # -------------------------
        # 4) forward
        # -------------------------
        with torch.inference_mode():
            logits = self.model(x)                      # (1,1,H,W)
            logits = self._apply_bottom_prior(logits)

            # track prior는 매 프레임이 아니라 N프레임마다만 (옵션)
            if (self.track_prior_every == 1) or ((self._step_i % self.track_prior_every) == 0):
                logits = self._apply_track_prior(logits)

            xy, conf = soft_argmax_2d(logits, beta=self.beta)
        t4 = time.perf_counter()

        # -------------------------
        # 5) to python float
        # -------------------------
        x_raw = float(xy[0, 0].item())
        y_raw = float(xy[0, 1].item())
        c = float(conf[0, 0].item())

        x_raw = float(np.clip(x_raw, 0.0, 1.0))
        y_raw = float(np.clip(y_raw, 0.0, 1.0))
        self.last_raw_xy = (x_raw, y_raw)

        # EMA
        x_n, y_n = self._ema(x_raw, y_raw)
        x_n = float(np.clip(x_n, 0.0, 1.0))
        y_n = float(np.clip(y_n, 0.0, 1.0))

        cur = np.array([x_n, y_n], dtype=np.float32)

        # =========================================================
        # lock / gating
        # =========================================================
        # (A) 아직 lock이 없으면: conf가 충분할 때만 lock 잡기
        if self._lock_xy is None:
            if c >= self.lock_conf_thr:
                self._lock_xy = cur.copy()
                self._lock_conf = c
                self._lost_count = 0

                if dbg_on:
                    print(
                        f"[DET][LOCK_ACQUIRE] step={self._step_i} "
                        f"raw=({x_raw:.3f},{y_raw:.3f}) ema=({x_n:.3f},{y_n:.3f}) "
                        f"conf={c:.4f} thr={self.lock_conf_thr:.4f}"
                    )
            else:
                _dbg(
                    f"[DET][NO_LOCK] step={self._step_i} "
                    f"raw=({x_raw:.3f},{y_raw:.3f}) ema=({x_n:.3f},{y_n:.3f}) "
                    f"conf={c:.4f} < thr={self.lock_conf_thr:.4f}"
                )
            return x_n, y_n, c, logits

        # (B) lock이 있는 상태
        d = self._dist(cur, self._lock_xy)

        # (B1) conf가 낮으면: lost_count 증가, patience 전까지 lock 좌표로 고정
        if c < self.lock_conf_thr:
            self._lost_count += 1

            if dbg_on:
                print(
                    f"[DET][LOW_CONF] step={self._step_i} conf={c:.4f} < thr={self.lock_conf_thr:.4f} "
                    f"lost={self._lost_count}/{self.lost_patience} "
                    f"d={d:.4f} lock=({float(self._lock_xy[0]):.3f},{float(self._lock_xy[1]):.3f}) "
                    f"ema=({x_n:.3f},{y_n:.3f})"
                )

            if self._lost_count >= self.lost_patience:
                if dbg_on:
                    print(
                        f"[DET][LOCK_DROP_BY_LOST] step={self._step_i} "
                        f"lost={self._lost_count}/{self.lost_patience} conf={c:.4f}"
                    )
                self._lock_xy = None
                self._lock_conf = 0.0
                self._lost_count = 0
                return x_n, y_n, c, logits
            else:
                # lock 유지 (반환 좌표는 lock)
                x_n, y_n = float(self._lock_xy[0]), float(self._lock_xy[1])
                return x_n, y_n, c, logits

        # (B2) 점프 거리 초과면: 매우 높은 conf일 때만 점프 허용
        if d > self.max_jump_norm:
            allow_thr = float(self._lock_conf * self.jump_allow_conf_gain)

            if c >= allow_thr:
                if dbg_on:
                    print(
                        f"[DET][JUMP_ACCEPT] step={self._step_i} d={d:.4f} > max_jump={self.max_jump_norm:.4f} "
                        f"conf={c:.4f} >= allow_thr={allow_thr:.4f} "
                        f"lock_conf={self._lock_conf:.4f} gain={self.jump_allow_conf_gain:.2f} "
                        f"new_lock=({x_n:.3f},{y_n:.3f}) old_lock=({float(self._lock_xy[0]):.3f},{float(self._lock_xy[1]):.3f})"
                    )
                self._lock_xy = cur.copy()
                self._lock_conf = c
                self._lost_count = 0
                return x_n, y_n, c, logits
            else:
                self._lost_count += 1

                if dbg_on:
                    print(
                        f"[DET][JUMP_REJECT] step={self._step_i} d={d:.4f} > max_jump={self.max_jump_norm:.4f} "
                        f"conf={c:.4f} < allow_thr={allow_thr:.4f} "
                        f"lost={self._lost_count}/{self.lost_patience} "
                        f"keep_lock=({float(self._lock_xy[0]):.3f},{float(self._lock_xy[1]):.3f}) "
                        f"ema=({x_n:.3f},{y_n:.3f}) raw=({x_raw:.3f},{y_raw:.3f})"
                    )

                if self._lost_count >= self.lost_patience:
                    if dbg_on:
                        print(
                            f"[DET][LOCK_DROP_BY_JUMP] step={self._step_i} "
                            f"lost={self._lost_count}/{self.lost_patience} conf={c:.4f}"
                        )
                    self._lock_xy = None
                    self._lock_conf = 0.0
                    self._lost_count = 0
                    return x_n, y_n, c, logits

                # lock 유지 (반환 좌표는 lock)
                x_n, y_n = float(self._lock_xy[0]), float(self._lock_xy[1])
                return x_n, y_n, c, logits

        # (B3) 정상 범위면: lock을 부드럽게 업데이트
        a = 0.35
        self._lock_xy = (a * cur + (1.0 - a) * self._lock_xy).astype(np.float32)
        self._lock_conf = max(self._lock_conf * 0.90, c)
        self._lost_count = 0

        _dbg(
            f"[DET][LOCK_UPDATE] step={self._step_i} d={d:.4f} "
            f"lock=({float(self._lock_xy[0]):.3f},{float(self._lock_xy[1]):.3f}) "
            f"conf={c:.4f} lock_conf={self._lock_conf:.4f} "
            f"ema=({x_n:.3f},{y_n:.3f}) raw=({x_raw:.3f},{y_raw:.3f})"
        )

        # ===== 내부 프로파일 출력 =====
        if self.print_prof:
            self._prof_step += 1
            if (self._prof_step % self.prof_every) == 0:
                ms_pre = (t1 - t0) * 1000.0
                ms_buf = (t2 - t1) * 1000.0
                ms_in = (t3 - t2) * 1000.0
                ms_fw = (t4 - t3) * 1000.0
                ms_all = (t4 - t0) * 1000.0
                print(f"[DET_PROF] pre={ms_pre:.2f} buf={ms_buf:.2f} in={ms_in:.2f} fw={ms_fw:.2f} total={ms_all:.2f}")

        return float(self._lock_xy[0]), float(self._lock_xy[1]), c, logits
