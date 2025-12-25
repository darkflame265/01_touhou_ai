# models/shared/two_stream_actor_critic.py
import torch
import torch.nn as nn


class TwoStreamActorCriticCNN(nn.Module):
    """
    입력: x shape = (B, C_total, H, W)
    - 프레임 스택 concat 구조 유지
    - per_frame = global_channels + local_channels (=8)
    - global: frame마다 [0:4]
    - local : frame마다 [4:8]

    meta(x,y,conf)는 "마지막 프레임의 local ch0" 좌상단 패치에 있음.
    """

    def __init__(
        self,
        input_channels: int,
        num_actions: int,
        obs_channels_per_frame: int = 8,
        global_channels: int = 4,
        local_channels: int = 4,
        meta_patch: int = 4,
        meta_local_channel_offset: int = 0,  # local 내부에서 meta가 들어있는 채널(보통 0=local ch0)
    ):
        super().__init__()

        self.input_channels = int(input_channels)
        self.num_actions = int(num_actions)

        self.obs_channels_per_frame = int(obs_channels_per_frame)
        self.global_channels = int(global_channels)
        self.local_channels = int(local_channels)

        assert self.global_channels + self.local_channels == self.obs_channels_per_frame, \
            "global_channels + local_channels must equal obs_channels_per_frame"

        self.meta_patch = int(meta_patch)
        self.meta_dim = 3
        self.meta_local_channel_offset = int(meta_local_channel_offset)

        # ---------
        # Global CNN branch
        # ---------
        self.conv_g = nn.Sequential(
            nn.Conv2d(self._global_total_channels_guess(), 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.pool_g = nn.AdaptiveAvgPool2d((7, 7))
        self.fc_g = nn.Sequential(
            nn.Linear(64 * 7 * 7, 384),
            nn.ReLU(),
        )

        # ---------
        # Local CNN branch
        # ---------
        self.conv_l = nn.Sequential(
            nn.Conv2d(self._local_total_channels_guess(), 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.pool_l = nn.AdaptiveAvgPool2d((7, 7))
        self.fc_l = nn.Sequential(
            nn.Linear(64 * 7 * 7, 384),
            nn.ReLU(),
        )

        # meta
        self.fc_meta = nn.Sequential(
            nn.Linear(self.meta_dim, 32),
            nn.ReLU(),
        )

        feat_dim = 384 + 384 + 32
        self.policy_head = nn.Linear(feat_dim, self.num_actions)
        self.value_head = nn.Linear(feat_dim, 1)

    # ---- helpers for init ----
    def _global_total_channels_guess(self) -> int:
        # input_channels는 C_total인데, 여기서는 "어떤 값이 와도" forward에서 맞춰 split함.
        # Conv2d in_channels는 고정이어야 하므로, 여기서는 placeholder가 필요.
        # 해결: forward에서 첫 호출 때 재구성하는 방식이 번거로우니,
        #       설계상 input_channels는 항상 per_frame*T 형태이므로,
        #       여기서는 "input_channels 중 global portion"을 그대로 받기 위해
        #       conv를 LazyConv2d로 쓰는 방법도 있음.
        # 하지만 너 프로젝트는 torch 2.x고, 안정성 위해 LazyConv2d 사용하자.
        return 1  # dummy (LazyConv2d로 실제는 무시)

    def _local_total_channels_guess(self) -> int:
        return 1  # dummy (LazyConv2d로 실제는 무시)

    @staticmethod
    def _normalize_input(x: torch.Tensor) -> torch.Tensor:
        if x.dtype.is_floating_point:
            try:
                if x.max().item() > 1.5:
                    return x / 255.0
            except Exception:
                pass
            return x
        return x.float() / 255.0

    def _split_global_local(self, x01: torch.Tensor):
        """
        x01: (B, C_total, H, W)
        return:
          xg: (B, T*global_channels, H, W)
          xl: (B, T*local_channels,  H, W)
        """
        B, C, H, W = x01.shape
        per = max(1, int(self.obs_channels_per_frame))
        T = max(1, C // per)

        # 이상치 방어: C가 per로 안나눠떨어지면 뒤쪽은 버림
        C_use = T * per
        x01 = x01[:, :C_use]

        # (B, T, per, H, W)
        xt = x01.view(B, T, per, H, W)

        xg = xt[:, :, 0:self.global_channels].contiguous().view(B, T * self.global_channels, H, W)
        xl = xt[:, :, self.global_channels:self.global_channels + self.local_channels].contiguous().view(B, T * self.local_channels, H, W)
        return xg, xl, T

    def _extract_meta_from_local_last(self, x01: torch.Tensor) -> torch.Tensor:
        """
        meta는 마지막 프레임의 local ch0 패치에 존재.
        x01: (B, C_total, H, W) float 0..1
        return: (B,3)
        """
        B, C, H, W = x01.shape
        per = max(1, int(self.obs_channels_per_frame))
        T = max(1, C // per)

        C_use = T * per
        x01 = x01[:, :C_use]

        last_base = (T - 1) * per
        # local ch0 index = last_base + global_channels + meta_local_channel_offset
        ch_idx = last_base + self.global_channels + self.meta_local_channel_offset
        ch_idx = max(0, min(int(ch_idx), int(C_use - 1)))

        p = int(self.meta_patch)
        need_w = p * 3
        if (H < p) or (W < need_w):
            return torch.zeros((B, self.meta_dim), device=x01.device, dtype=x01.dtype)

        m = x01[:, ch_idx]  # (B,H,W)
        x_val = m[:, 0:p, 0:p].mean(dim=(1, 2))
        y_val = m[:, 0:p, p:2 * p].mean(dim=(1, 2))
        c_val = m[:, 0:p, 2 * p:3 * p].mean(dim=(1, 2))

        meta = torch.stack([x_val, y_val, c_val], dim=1)
        meta = torch.nan_to_num(meta, nan=0.0, posinf=1.0, neginf=0.0)
        meta = torch.clamp(meta, 0.0, 1.0)
        return meta

    def forward(self, x: torch.Tensor):
        x01 = self._normalize_input(x)

        # split
        xg, xl, _ = self._split_global_local(x01)

        # LazyConv2d로 바꾸기: in_channels를 런타임에 맞춤
        # (한 번만 교체)
        if not isinstance(self.conv_g[0], nn.LazyConv2d):
            pass

        # conv 첫 레이어를 Lazy로 교체(안정적으로 1회)
        if isinstance(self.conv_g[0], nn.Conv2d) and self.conv_g[0].in_channels == 1:
            self.conv_g[0] = nn.LazyConv2d(32, kernel_size=8, stride=4).to(x01.device)
        if isinstance(self.conv_l[0], nn.Conv2d) and self.conv_l[0].in_channels == 1:
            self.conv_l[0] = nn.LazyConv2d(32, kernel_size=8, stride=4).to(x01.device)

        # meta
        meta = self._extract_meta_from_local_last(x01)
        meta_feat = self.fc_meta(meta)

        # global branch
        zg = self.conv_g(xg)
        zg = self.pool_g(zg)
        zg = zg.view(zg.size(0), -1)
        fg = self.fc_g(zg)

        # local branch
        zl = self.conv_l(xl)
        zl = self.pool_l(zl)
        zl = zl.view(zl.size(0), -1)
        fl = self.fc_l(zl)

        feat = torch.cat([fg, fl, meta_feat], dim=1)
        logits = self.policy_head(feat)
        value = self.value_head(feat)
        return logits, value
