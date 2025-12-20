import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticCNN(nn.Module):
    """
    입력: x shape = (B, C, H, W)  (C는 frame_stack 채널)

    ObsBuilder가 좌상단 메타 픽셀로 (x_norm, y_norm, conf)를 박아 넣으면,
    이 네트워크가 그 값을 읽어서 CNN feature에 concat한다.

    메타 픽셀 레이아웃(ObsBuilder 기준):
      patch = 4 일 때
      - (0:4, 0:4)   = x_norm
      - (0:4, 4:8)   = y_norm
      - (0:4, 8:12)  = conf
    """

    def __init__(self, input_channels, num_actions, meta_patch: int = 4):
        super().__init__()
        self.meta_patch = int(meta_patch)
        self.meta_dim = 3  # x, y, conf

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc_img = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )

        # 메타 (x,y,conf) -> 작은 임베딩
        self.fc_meta = nn.Sequential(
            nn.Linear(self.meta_dim, 32),
            nn.ReLU(),
        )

        feat_dim = 512 + 32

        self.policy_head = nn.Linear(feat_dim, num_actions)
        self.value_head = nn.Linear(feat_dim, 1)

    @staticmethod
    def _normalize_input(x: torch.Tensor) -> torch.Tensor:
        # obs가 보통 float32 0..1 이지만, 혹시 0..255로 들어오면 자동으로 보정
        if x.dtype.is_floating_point:
            if x.max().item() > 1.5:
                return x / 255.0
            return x
        return x.float() / 255.0

    def _extract_meta(self, x01: torch.Tensor) -> torch.Tensor:
        """
        x01: (B,C,H,W) float 0..1
        가장 최근 프레임(채널 -1)의 메타 픽셀을 평균으로 읽어 (B,3)로 만든다.
        """
        B, C, H, W = x01.shape
        p = self.meta_patch
        need_w = p * 3

        if (H < p) or (W < need_w):
            return torch.zeros((B, self.meta_dim), device=x01.device, dtype=x01.dtype)

        last = x01[:, -1]  # (B,H,W)

        x_val = last[:, 0:p, 0:p].mean(dim=(1, 2))
        y_val = last[:, 0:p, p:2 * p].mean(dim=(1, 2))
        c_val = last[:, 0:p, 2 * p:3 * p].mean(dim=(1, 2))

        meta = torch.stack([x_val, y_val, c_val], dim=1)  # (B,3)
        meta = torch.nan_to_num(meta, nan=0.0, posinf=1.0, neginf=0.0)
        meta = torch.clamp(meta, 0.0, 1.0)
        return meta

    def forward(self, x):
        x01 = self._normalize_input(x)

        meta = self._extract_meta(x01)     # (B,3)
        meta_feat = self.fc_meta(meta)     # (B,32)

        z = self.conv(x01)
        z = self.pool(z)
        z = z.view(z.size(0), -1)
        img_feat = self.fc_img(z)          # (B,512)

        feat = torch.cat([img_feat, meta_feat], dim=1)

        logits = self.policy_head(feat)
        value = self.value_head(feat)
        return logits, value
