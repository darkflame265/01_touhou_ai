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
      (x_norm, y_norm, peak_prob)
    where x_norm,y_norm are normalized on PLAYFIELD image space (0..1).
    """
    def __init__(
        self,
        screen,
        weight_path="weights/reimu_heatmap_best.pt",
        beta=12.0,
        prior_strength=1.0,
        ema_alpha=0.75,
        device=None,
    ):
        self.screen = screen
        self.weight_path = weight_path
        self.beta = float(beta)
        self.prior_strength = float(prior_strength)
        self.ema_alpha = float(ema_alpha)

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
        self._ema_xy = None  # np([x,y]) in [0,1]

    def reset(self):
        self.buf.clear()
        self._ema_xy = None

    def _apply_bottom_prior(self, logits: torch.Tensor) -> torch.Tensor:
        if self.prior_strength <= 0:
            return logits
        H = logits.shape[-2]
        yy = torch.linspace(0.0, 1.0, H, device=logits.device, dtype=logits.dtype).view(1, 1, H, 1)
        penalty = (1.0 - yy)  # top=1, bottom=0
        return logits - self.prior_strength * penalty

    def _ema(self, x, y):
        v = np.array([x, y], dtype=np.float32)
        if self._ema_xy is None:
            self._ema_xy = v
        else:
            a = self.ema_alpha
            self._ema_xy = a * v + (1.0 - a) * self._ema_xy
        return float(self._ema_xy[0]), float(self._ema_xy[1])

    def step(self, img_bgr):
        """
        Returns:
          None (during warmup)
          or (x_norm, y_norm, peak_prob)
        """
        play = self.screen.get_playfield_gray(img_bgr)  # gray playfield
        small = cv2.resize(play, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

        self.buf.appendleft(small)
        if len(self.buf) < self.stack:
            return None

        x_np = np.stack(list(self.buf), axis=0)  # (C,H,W)
        x = torch.from_numpy(x_np).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.model(x)                     # (1,1,H,W)
            logits = self._apply_bottom_prior(logits)  # optional prior
            xy, conf = soft_argmax_2d(logits, beta=self.beta)

        x_n = float(xy[0, 0].detach().cpu())
        y_n = float(xy[0, 1].detach().cpu())
        c = float(conf[0, 0].detach().cpu())

        x_n, y_n = self._ema(x_n, y_n)
        x_n = float(np.clip(x_n, 0.0, 1.0))
        y_n = float(np.clip(y_n, 0.0, 1.0))
        return x_n, y_n, c, logits
