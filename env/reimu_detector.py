# env/reimu_detector.py
import os
from collections import deque

import cv2
import numpy as np
import torch

from vision.models.heatmap_net import HeatmapNet, soft_argmax_2d


class ReimuDetector:
    """
    Heatmap detector that returns:
      (x_norm, y_norm, conf, logits)

    개선:
      - tracking prior(가우시안 앵커)로 "잡은 레이무" 유지
      - 점프 억제(gating)로 보스/아이템에 튀는 현상 감소
      - lost 누적 시 재탐색(reacquire)
      - on_player_death()로 죽으면 lock 해제
    """

    def __init__(
        self,
        screen,
        weight_path="weights/reimu_heatmap_best.pt",
        beta=12.0,
        prior_strength=1.0,
        ema_alpha=0.75,
        device=None,
        # ===== tracking 옵션 =====
        track_prior_strength=2.0,     # 1.0~4.0 추천 (클수록 현재 트랙 고집)
        track_prior_sigma=0.08,       # 0.06~0.12 추천 (작을수록 더 빡세게 고정)
        lock_conf_thr=0.015,          # 이 이상이면 "유효 추적"로 간주
        max_jump_norm=0.22,           # 한 프레임에 허용할 점프(정규화) (0.18~0.30)
        jump_allow_conf_gain=1.8,     # 멀리 튈 때 conf가 (기존 대비) 이만큼 좋아야 갈아탐
        lost_patience=8,              # 튀거나 conf 낮음이 몇 프레임 누적되면 lock 약화/해제
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

        ckpt = torch.load(self.weight_path, map_location=self.device)
        cfg = ckpt.get("cfg", {})

        self.out_w = int(cfg.get("w", 160))
        self.out_h = int(cfg.get("h", 120))
        self.stack = int(cfg.get("stack", 4))

        self.model = HeatmapNet(in_ch=self.stack, base=32).to(self.device)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.eval()

        self.buf = deque(maxlen=self.stack)

        # EMA(부드러운 위치)
        self._ema_xy = None  # np([x,y]) in [0,1]

        # Tracking lock 상태
        self._lock_xy = None        # np([x,y]) : "현재 믿는 레이무"
        self._lock_conf = 0.0
        self._lost_count = 0

        # tracking prior용 mesh 캐시
        self._yy = None
        self._xx = None

        #디버그 표시용.좌표.
        self.last_raw_xy = None  # (x,y) in [0,1]
        self.last_lock_xy = None


    def reset(self):
        self.buf.clear()
        self._ema_xy = None
        self._lock_xy = None
        self._lock_conf = 0.0
        self._lost_count = 0

    def on_player_death(self):
        """GameEnv에서 호출해주면: 죽었을 때만 추적 lock을 푼다."""
        self._lock_xy = None
        self._lock_conf = 0.0
        self._lost_count = 0
        # buf는 유지해도 되지만, 깔끔하게 초기화하고 싶으면 아래 활성화
        # self.buf.clear()
        # self._ema_xy = None

    def _apply_bottom_prior(self, logits: torch.Tensor) -> torch.Tensor:
        if self.prior_strength <= 0:
            return logits
        H = logits.shape[-2]
        yy = torch.linspace(0.0, 1.0, H, device=logits.device, dtype=logits.dtype).view(1, 1, H, 1)
        penalty = (1.0 - yy)  # top=1, bottom=0
        return logits - self.prior_strength * penalty

    def _ensure_mesh(self, device, dtype):
        # (1,1,H,W)로 브로드캐스트 가능한 xx/yy
        if (self._yy is None) or (self._xx is None) or (self._yy.device != device) or (self._yy.dtype != dtype):
            yy = torch.linspace(0.0, 1.0, self.out_h, device=device, dtype=dtype).view(1, 1, self.out_h, 1)
            xx = torch.linspace(0.0, 1.0, self.out_w, device=device, dtype=dtype).view(1, 1, 1, self.out_w)
            self._yy = yy
            self._xx = xx

    def _apply_track_prior(self, logits: torch.Tensor) -> torch.Tensor:
        """
        현재 lock_xy 주변에 가우시안 보너스를 더해 logits를 '고정'한다.
        """
        if self.track_prior_strength <= 0:
            return logits
        if self._lock_xy is None:
            return logits

        self._ensure_mesh(logits.device, logits.dtype)
        x0 = float(self._lock_xy[0])
        y0 = float(self._lock_xy[1])

        # 가우시안: exp(-d^2 / (2*sigma^2))
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

    def step(self, img_bgr):
        """
        Returns:
          None (during warmup)
          or (x_norm, y_norm, conf, logits)
        """
        play = self.screen.get_playfield_gray(img_bgr)  # gray playfield
        small = cv2.resize(play, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

        self.buf.appendleft(small)
        if len(self.buf) < self.stack:
            return None

        x_np = np.stack(list(self.buf), axis=0)  # (C,H,W)
        x = torch.from_numpy(x_np).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.model(x)                      # (1,1,H,W)
            logits = self._apply_bottom_prior(logits)   # 기존 prior
            logits = self._apply_track_prior(logits)    # ✅ 추적 prior (핵심)
            xy, conf = soft_argmax_2d(logits, beta=self.beta)

        x_raw = float(xy[0, 0].detach().cpu())
        y_raw = float(xy[0, 1].detach().cpu())
        c = float(conf[0, 0].detach().cpu())

        # raw 저장 (표시용)
        x_raw = float(np.clip(x_raw, 0.0, 1.0))
        y_raw = float(np.clip(y_raw, 0.0, 1.0))
        self.last_raw_xy = (x_raw, y_raw)


        x_n = float(xy[0, 0].detach().cpu())
        y_n = float(xy[0, 1].detach().cpu())
        c = float(conf[0, 0].detach().cpu())

        # EMA로 일단 부드럽게
        x_n, y_n = self._ema(x_n, y_n)
        x_n = float(np.clip(x_n, 0.0, 1.0))
        y_n = float(np.clip(y_n, 0.0, 1.0))

        # ====== lock / gating 로직 ======
        cur = np.array([x_n, y_n], dtype=np.float32)

        if self._lock_xy is None:
            # 처음 lock: conf가 조금이라도 있으면 시작
            if c >= self.lock_conf_thr:
                self._lock_xy = cur.copy()
                self._lock_conf = c
                self._lost_count = 0
            return x_n, y_n, c, logits

        # 이미 lock 중이면 "점프" 검사
        d = self._dist(cur, self._lock_xy)

        if c < self.lock_conf_thr:
            # conf 낮음 -> lock 유지 + lost 누적
            self._lost_count += 1
            if self._lost_count >= self.lost_patience:
                # 너무 오래 못 믿으면 lock 약화/해제 (재탐색)
                self._lock_xy = None
                self._lock_conf = 0.0
                self._lost_count = 0
            else:
                # lock 위치로 되돌려 반환(튐 방지)
                x_n, y_n = float(self._lock_xy[0]), float(self._lock_xy[1])
            return x_n, y_n, c, logits

        # conf 충분한데 멀리 튄 경우 -> 갈아타기 조건을 더 까다롭게
        if d > self.max_jump_norm:
            # "정말로 레이무가 순간이동했거나" / "기존 lock이 틀렸거나"만 허용
            # conf가 이전보다 충분히 좋아야 lock을 바꿈
            if c >= (self._lock_conf * self.jump_allow_conf_gain):
                # 갈아타기 허용
                self._lock_xy = cur.copy()
                self._lock_conf = c
                self._lost_count = 0
                return x_n, y_n, c, logits
            else:
                # 갈아타기 거부 -> lock 유지, lost 누적
                self._lost_count += 1
                if self._lost_count >= self.lost_patience:
                    self._lock_xy = None
                    self._lock_conf = 0.0
                    self._lost_count = 0
                    return x_n, y_n, c, logits  # 다음 프레임부터 재탐색
                # 반환은 lock로 고정
                x_n, y_n = float(self._lock_xy[0]), float(self._lock_xy[1])
                # conf는 네트워크 conf를 그대로 주되, obs_builder가 update_thr로 막아주게 둘 수도 있고
                return x_n, y_n, c, logits

        # 정상 범위 이동: lock을 천천히 업데이트
        # (너무 빠르게 따라가면 튐에 취약해져서 a를 낮게)
        a = 0.35
        self._lock_xy = (a * cur + (1.0 - a) * self._lock_xy).astype(np.float32)
        self._lock_conf = max(self._lock_conf * 0.90, c)  # 점진 갱신
        self._lost_count = 0

        # 반환은 lock 기반으로 (일관성)
        return float(self._lock_xy[0]), float(self._lock_xy[1]), c, logits
