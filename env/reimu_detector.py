# env/reimu_detector.py
import time

import cv2
import numpy as np
import torch

from vision.models.heatmap_net import HeatmapNet, soft_argmax_2d


class ReimuDetector:
    """
    Heatmap detector that returns:
      (x_norm, y_norm, conf, logits)

    ✅ 목표(이번 완성본):
    - 보스 등장/이펙트 변화 때 "잠깐" 강한 peak가 생겨도 lock이 즉시 갈아타지 않게
      => "Switch-stability gate" (N프레임 연속일 때만 lock 전환)

    ✅ 유지:
    - numpy 고정 버퍼 stack
    - torch.inference_mode()
    - (옵션) CUDA FP16
    - (옵션) track_prior_every
    """

    def __init__(
        self,
        screen,
        weight_path="weights/reimu_heatmap_best.pt",
        beta=12.0,
        prior_strength=1.0,
        ema_alpha=0.90,
        device=None,

        # ===== tracking 옵션 =====
        track_prior_strength=2.4,
        track_prior_sigma=0.06,
        lock_conf_thr=0.030,
        max_jump_norm=0.14,
        jump_allow_conf_gain=2.6,
        lost_patience=16,

        # ===== 보스 튐 방지(기존) =====
        y_gate_min=0.40,
        y_gate_softness=0.06,

        topk=8,
        cand_use_softmax=True,
        cand_prior_w=2.2,
        cand_bottom_w=0.8,
        cand_jump_w=1.6,
        cand_prior_sigma=0.10,

        upjump_extra_gain=1.7,
        upjump_margin=0.04,

        # ===== ✅ NEW: lock "전환" 안정화 게이트 =====
        # 멀리 점프해서 lock을 바꾸려면, 같은 방향의 후보가 N프레임 연속으로
        # 충분히 강한 근거(conf/score)를 보여야만 전환한다.
        switch_patience=6,              # 4~10 추천. (보스 등장 순간 튐이면 6~8이 안정적)
        switch_min_conf=0.55,           # switch 후보의 최소 conf (너무 낮으면 카운트 안 쌓음)
        switch_score_margin=0.10,       # 현재 lock 근처 best_score 대비 얼마나 좋아야 카운트 인정할지
        switch_min_dist=0.22,           # lock과 이 정도 이상 멀면 "switch 후보"로 간주(너무 가까우면 그냥 update)
        switch_decay=0.70,              # switch 후보가 끊기면 count를 얼마나 유지할지(0~1, 낮을수록 빨리 리셋)

        # ===== 성능 옵션 =====
        use_fp16=True,
        track_prior_every=1,
        print_prof=True,
        prof_every=200,
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

        # ---- boss 튐 방지 파라미터 ----
        self.y_gate_min = float(y_gate_min)
        self.y_gate_softness = float(y_gate_softness)
        self.topk = int(max(1, topk))
        self.cand_use_softmax = bool(cand_use_softmax)
        self.cand_prior_w = float(cand_prior_w)
        self.cand_bottom_w = float(cand_bottom_w)
        self.cand_jump_w = float(cand_jump_w)
        self.cand_prior_sigma = float(cand_prior_sigma)
        self.upjump_extra_gain = float(upjump_extra_gain)
        self.upjump_margin = float(upjump_margin)

        # ---- NEW: switch gate ----
        self.switch_patience = int(max(1, switch_patience))
        self.switch_min_conf = float(switch_min_conf)
        self.switch_score_margin = float(switch_score_margin)
        self.switch_min_dist = float(switch_min_dist)
        self.switch_decay = float(np.clip(switch_decay, 0.0, 1.0))

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_fp16 = bool(use_fp16 and (self.device.type == "cuda"))
        self.track_prior_every = max(1, int(track_prior_every))

        self.print_prof = bool(print_prof)
        self.prof_every = max(10, int(prof_every))
        self._prof_step = 0

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
        self._buf_len = 0

        # CUDA prealloc
        self._x_cuda = None
        if self.device.type == "cuda":
            dtype = torch.float16 if self.use_fp16 else torch.float32
            self._x_cuda = torch.empty((1, self.stack, self.out_h, self.out_w), device=self.device, dtype=dtype)

        # EMA
        self._ema_xy = None  # np([x,y])

        # Tracking lock
        self._lock_xy = None
        self._lock_conf = 0.0
        self._lost_count = 0

        # NEW: switch 후보 누적 상태
        self._sw_xy = None
        self._sw_count = 0.0
        self._sw_best_score = None

        # mesh cache
        self._yy = None
        self._xx = None

        # debug
        self.last_raw_xy = None
        self.last_lock_xy = None
        self.last_cands = None

        self._step_i = 0
        print(
            f"[DET] device={self.device} fp16={self.use_fp16} "
            f"track_prior_every={self.track_prior_every} stack={self.stack} out={self.out_w}x{self.out_h} "
            f"y_gate_min={self.y_gate_min:.2f} topk={self.topk} switch_patience={self.switch_patience}"
        )

    def reset(self):
        self._buf_np.fill(0.0)
        self._buf_len = 0
        self._ema_xy = None
        self._lock_xy = None
        self._lock_conf = 0.0
        self._lost_count = 0

        self._sw_xy = None
        self._sw_count = 0.0
        self._sw_best_score = None

        self.last_raw_xy = None
        self.last_lock_xy = None
        self.last_cands = None
        self._step_i = 0

    def on_player_death(self):
        self._lock_xy = None
        self._lock_conf = 0.0
        self._lost_count = 0

        self._sw_xy = None
        self._sw_count = 0.0
        self._sw_best_score = None

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
        if self.stack > 1:
            self._buf_np[1:] = self._buf_np[:-1]
        self._buf_np[0] = small01
        self._buf_len = min(self.stack, self._buf_len + 1)

    def _make_input_tensor(self):
        if self.device.type == "cpu":
            return torch.from_numpy(self._buf_np).unsqueeze(0)  # float32
        assert self._x_cuda is not None
        x_cpu = torch.from_numpy(self._buf_np).unsqueeze(0)  # float32 CPU
        if self.use_fp16:
            x_cpu = x_cpu.half()
        self._x_cuda.copy_(x_cpu, non_blocking=False)
        return self._x_cuda

    def _y_gate_penalty(self, y_norm: float) -> float:
        yg = self.y_gate_min
        s = max(1e-6, self.y_gate_softness)
        if y_norm >= yg:
            return 0.0
        t = (yg - y_norm) / s
        return float(np.clip(t, 0.0, 1.0))

    def _pick_from_topk(self, logits: torch.Tensor):
        """
        return: (x_norm, y_norm, cand_conf, score, (ix,iy))
        """
        hm = logits[0, 0]  # (H,W)
        H, W = hm.shape[-2], hm.shape[-1]

        if self.cand_use_softmax:
            flat = (hm * float(self.beta)).reshape(-1)
            prob = torch.softmax(flat, dim=0).reshape(H, W)
            score_base = prob
        else:
            score_base = hm

        flat2 = score_base.reshape(-1)
        k = min(self.topk, flat2.numel())
        vals, idxs = torch.topk(flat2, k=k, largest=True, sorted=True)

        lock_xy = self._lock_xy
        lock_x = float(lock_xy[0]) if lock_xy is not None else None
        lock_y = float(lock_xy[1]) if lock_xy is not None else None

        sigma2 = max(1e-6, self.cand_prior_sigma * self.cand_prior_sigma)

        best = None
        cand_list = []

        for j in range(k):
            v = float(vals[j].item())
            idx = int(idxs[j].item())
            iy = idx // W
            ix = idx - iy * W

            x = (ix + 0.5) / float(W)
            y = (iy + 0.5) / float(H)

            gate = self._y_gate_penalty(y)
            gate_mul = 1.0 - gate

            if lock_xy is None:
                prior_bonus = 0.0
                jump_cost = 0.0
            else:
                dx = x - lock_x
                dy = y - lock_y
                d2 = dx * dx + dy * dy
                prior_bonus = float(np.exp(-0.5 * d2 / sigma2))
                jump_cost = float(np.sqrt(d2))

            bottom_bonus = y

            score = (v * gate_mul) + (self.cand_prior_w * prior_bonus) + (self.cand_bottom_w * bottom_bonus) - (
                self.cand_jump_w * jump_cost
            )

            cand_list.append((x, y, v, score))
            if (best is None) or (score > best[3]):
                best = (x, y, v, score, (ix, iy))

        self.last_cands = cand_list
        return best

    def _switch_gate_update(self, x_new: float, y_new: float, conf: float, best_score: float) -> bool:
        """
        멀리 점프하여 lock을 갈아타려는 경우,
        - switch 후보를 누적하고
        - N프레임 연속으로 조건을 만족하면 True(전환 허용)
        """
        if self._lock_xy is None:
            self._sw_xy = None
            self._sw_count = 0.0
            self._sw_best_score = None
            return True

        lockx, locky = float(self._lock_xy[0]), float(self._lock_xy[1])
        d = self._dist((x_new, y_new), (lockx, locky))

        # 가까우면 switch가 아니라 그냥 update 흐름
        if d < self.switch_min_dist:
            self._sw_xy = None
            self._sw_count = 0.0
            self._sw_best_score = None
            return True

        # conf/score 조건이 약하면 카운트 쌓지 말고 감쇠
        if conf < self.switch_min_conf:
            self._sw_count *= self.switch_decay
            if self._sw_count < 0.5:
                self._sw_xy = None
                self._sw_best_score = None
            return False

        # "현재 lock 근처" 후보 대비 score가 충분히 좋아야 카운트 인정
        # (best_score는 pick 후보 score. lock 근처로 고정되어 있을 땐 이 점수가 튈 때가 있어 margin으로 필터)
        if self._sw_best_score is None:
            self._sw_best_score = float(best_score)

        # 후보가 자주 바뀌면(서로 다른 방향) 카운트가 의미 없으니, 위치가 크게 달라지면 리셋
        if self._sw_xy is None:
            self._sw_xy = (float(x_new), float(y_new))
            self._sw_count = 1.0
            self._sw_best_score = float(best_score)
        else:
            sd = self._dist((x_new, y_new), self._sw_xy)
            if sd > 0.10:
                # switch 후보 방향이 바뀜 => 다시 쌓기
                self._sw_xy = (float(x_new), float(y_new))
                self._sw_count = 1.0
                self._sw_best_score = float(best_score)
            else:
                # 같은 방향 => 카운트 증가 (score가 더 좋아지면 갱신)
                if (best_score + self.switch_score_margin) >= self._sw_best_score:
                    self._sw_count += 1.0
                    self._sw_best_score = max(self._sw_best_score, float(best_score))
                else:
                    # 살짝 약해지면 감쇠
                    self._sw_count *= self.switch_decay

        return self._sw_count >= float(self.switch_patience)

    def step(self, img_bgr):
        """
        Returns:
          None (warmup)
          or (x_norm, y_norm, conf, logits)
        """
        self._step_i += 1
        t0 = time.perf_counter()

        dbg_on = bool(getattr(self, "track_debug", False))
        dbg_every = int(getattr(self, "track_debug_every", 200))
        if not hasattr(self, "_dbg_i"):
            self._dbg_i = 0
        self._dbg_i += 1

        def _dbg(msg: str):
            if not dbg_on:
                return
            if (dbg_every > 1) and ((self._dbg_i % dbg_every) != 0):
                return
            print(msg)

        # 1) playfield gray + resize + normalize
        play = self.screen.get_playfield_gray(img_bgr)
        small = cv2.resize(play, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
        small = small.astype(np.float32) * (1.0 / 255.0)
        t1 = time.perf_counter()

        # 2) stack buffer push
        self._push_frame(small)
        if self._buf_len < self.stack:
            return None
        t2 = time.perf_counter()

        # 3) input tensor
        x = self._make_input_tensor()
        if self.device.type == "cuda" and not self.use_fp16:
            x = x.float()
        t3 = time.perf_counter()

        # 4) forward
        with torch.inference_mode():
            logits = self.model(x)  # (1,1,H,W)
            logits = self._apply_bottom_prior(logits)

            if (self.track_prior_every == 1) or ((self._step_i % self.track_prior_every) == 0):
                logits = self._apply_track_prior(logits)

            xy_sa, conf_sa = soft_argmax_2d(logits, beta=self.beta)
            best = self._pick_from_topk(logits)
        t4 = time.perf_counter()

        # 5) to python float
        x_raw = float(np.clip(float(xy_sa[0, 0].item()), 0.0, 1.0))
        y_raw = float(np.clip(float(xy_sa[0, 1].item()), 0.0, 1.0))
        c = float(conf_sa[0, 0].item())
        self.last_raw_xy = (x_raw, y_raw)

        if best is None:
            x_pick, y_pick = x_raw, y_raw
            cand_conf, cand_score = c, 0.0
        else:
            x_pick, y_pick, cand_conf, cand_score, _ = best

        # EMA는 pick 좌표에 적용
        x_n, y_n = self._ema(float(x_pick), float(y_pick))
        x_n = float(np.clip(x_n, 0.0, 1.0))
        y_n = float(np.clip(y_n, 0.0, 1.0))
        cur = np.array([x_n, y_n], dtype=np.float32)

        # =========================================================
        # lock / gating
        # =========================================================
        if self._lock_xy is None:
            self._sw_xy = None
            self._sw_count = 0.0
            self._sw_best_score = None

            if c >= self.lock_conf_thr:
                self._lock_xy = cur.copy()
                self._lock_conf = c
                self._lost_count = 0
                if dbg_on:
                    print(
                        f"[DET][LOCK_ACQUIRE] step={self._step_i} "
                        f"pick=({x_pick:.3f},{y_pick:.3f}) ema=({x_n:.3f},{y_n:.3f}) "
                        f"rawSA=({x_raw:.3f},{y_raw:.3f}) confSA={c:.4f} thr={self.lock_conf_thr:.4f} "
                        f"cand_conf={float(cand_conf):.6f} cand_score={float(cand_score):.4f}"
                    )
            else:
                _dbg(
                    f"[DET][NO_LOCK] step={self._step_i} pick=({x_pick:.3f},{y_pick:.3f}) "
                    f"ema=({x_n:.3f},{y_n:.3f}) confSA={c:.4f} < thr={self.lock_conf_thr:.4f}"
                )
            return x_n, y_n, c, logits

        # lock 있음
        d = self._dist(cur, self._lock_xy)

        # conf 낮으면 유지
        if c < self.lock_conf_thr:
            self._lost_count += 1
            if dbg_on:
                print(
                    f"[DET][LOW_CONF] step={self._step_i} confSA={c:.4f} < thr={self.lock_conf_thr:.4f} "
                    f"lost={self._lost_count}/{self.lost_patience} d={d:.4f} "
                    f"lock=({float(self._lock_xy[0]):.3f},{float(self._lock_xy[1]):.3f}) "
                    f"ema=({x_n:.3f},{y_n:.3f})"
                )

            if self._lost_count >= self.lost_patience:
                if dbg_on:
                    print(f"[DET][LOCK_DROP_BY_LOST] step={self._step_i} lost={self._lost_count}/{self.lost_patience} confSA={c:.4f}")
                self._lock_xy = None
                self._lock_conf = 0.0
                self._lost_count = 0
                return x_n, y_n, c, logits

            return float(self._lock_xy[0]), float(self._lock_xy[1]), c, logits

        # 점프 거리 초과
        if d > self.max_jump_norm:
            allow_thr = float(self._lock_conf * self.jump_allow_conf_gain)

            prev_y = float(self._lock_xy[1])
            if (prev_y - y_n) > self.upjump_margin:
                allow_thr *= self.upjump_extra_gain

            # ✅ NEW: switch-stability gate
            # "즉시 갈아타기" 대신, N프레임 연속으로 확신일 때만 갈아탄다.
            if c >= allow_thr:
                ok_switch = self._switch_gate_update(x_n, y_n, c, float(cand_score))
                if ok_switch:
                    if dbg_on:
                        print(
                            f"[DET][SWITCH_ACCEPT] step={self._step_i} d={d:.4f} confSA={c:.4f} >= allow_thr={allow_thr:.4f} "
                            f"sw_count={self._sw_count:.1f}/{self.switch_patience} new_lock=({x_n:.3f},{y_n:.3f}) "
                            f"old_lock=({float(self._lock_xy[0]):.3f},{float(self._lock_xy[1]):.3f})"
                        )
                    self._lock_xy = cur.copy()
                    self._lock_conf = c
                    self._lost_count = 0
                    self._sw_xy = None
                    self._sw_count = 0.0
                    self._sw_best_score = None
                    return x_n, y_n, c, logits
                else:
                    self._lost_count += 1
                    if dbg_on:
                        print(
                            f"[DET][SWITCH_HOLD] step={self._step_i} d={d:.4f} confSA={c:.4f} >= allow_thr={allow_thr:.4f} "
                            f"sw_count={self._sw_count:.1f}/{self.switch_patience} keep_lock=({float(self._lock_xy[0]):.3f},{float(self._lock_xy[1]):.3f}) "
                            f"cand_score={float(cand_score):.3f}"
                        )

                    if self._lost_count >= self.lost_patience:
                        if dbg_on:
                            print(f"[DET][LOCK_DROP_BY_SWITCH] step={self._step_i} lost={self._lost_count}/{self.lost_patience} confSA={c:.4f}")
                        self._lock_xy = None
                        self._lock_conf = 0.0
                        self._lost_count = 0
                        self._sw_xy = None
                        self._sw_count = 0.0
                        self._sw_best_score = None
                        return x_n, y_n, c, logits

                    return float(self._lock_xy[0]), float(self._lock_xy[1]), c, logits

            # 기존 JUMP_REJECT (allow_thr 미만)
            self._lost_count += 1
            if dbg_on:
                print(
                    f"[DET][JUMP_REJECT] step={self._step_i} d={d:.4f} > max_jump={self.max_jump_norm:.4f} "
                    f"confSA={c:.4f} < allow_thr={allow_thr:.4f} "
                    f"lost={self._lost_count}/{self.lost_patience} "
                    f"keep_lock=({float(self._lock_xy[0]):.3f},{float(self._lock_xy[1]):.3f}) "
                    f"ema=({x_n:.3f},{y_n:.3f}) pick=({x_pick:.3f},{y_pick:.3f}) rawSA=({x_raw:.3f},{y_raw:.3f})"
                )

            if self._lost_count >= self.lost_patience:
                if dbg_on:
                    print(f"[DET][LOCK_DROP_BY_JUMP] step={self._step_i} lost={self._lost_count}/{self.lost_patience} confSA={c:.4f}")
                self._lock_xy = None
                self._lock_conf = 0.0
                self._lost_count = 0
                self._sw_xy = None
                self._sw_count = 0.0
                self._sw_best_score = None
                return x_n, y_n, c, logits

            return float(self._lock_xy[0]), float(self._lock_xy[1]), c, logits

        # 정상 범위: lock 부드럽게 업데이트
        a = 0.35
        self._lock_xy = (a * cur + (1.0 - a) * self._lock_xy).astype(np.float32)
        self._lock_conf = max(self._lock_conf * 0.90, c)
        self._lost_count = 0

        # switch 후보는 서서히 잊기
        self._sw_count *= self.switch_decay
        if self._sw_count < 0.5:
            self._sw_xy = None
            self._sw_best_score = None

        self.last_lock_xy = (float(self._lock_xy[0]), float(self._lock_xy[1]))

        _dbg(
            f"[DET][LOCK_UPDATE] step={self._step_i} d={d:.4f} "
            f"lock=({float(self._lock_xy[0]):.3f},{float(self._lock_xy[1]):.3f}) "
            f"confSA={c:.4f} lock_conf={self._lock_conf:.4f} "
            f"ema=({x_n:.3f},{y_n:.3f}) pick=({x_pick:.3f},{y_pick:.3f}) rawSA=({x_raw:.3f},{y_raw:.3f})"
        )

        if self.print_prof:
            self._prof_step += 1
            if (self._prof_step % self.prof_every) == 0:
                ms_all = (t4 - t0) * 1000.0
                # 필요하면 프린트 풀기
                # print(f"[DET_PROF] total={ms_all:.2f}ms")
                _ = ms_all

        return float(self._lock_xy[0]), float(self._lock_xy[1]), c, logits
