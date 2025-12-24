# models/shared/cnn_actor_critic.py
import torch
import torch.nn as nn


class ActorCriticCNN(nn.Module):
    """
    입력: x shape = (B, C_total, H, W)

    이 프로젝트는 "프레임 스택 concat" 구조:
      - ObsBuilder가 프레임당 obs_channels=4 를 만들고
      - GameEnv가 frame_stack_size=4 를 채널축 concat => C_total = 4*4 = 16

    ObsBuilder는 메타(x_norm,y_norm,conf)를 "프레임의 ch0" 좌상단 패치에만 박음.
    따라서 "가장 최근 프레임의 ch0"에서 meta를 읽는다.

    meta_channel_offset:
      - 프레임 내부에서 meta가 박힌 채널의 오프셋
      - 현재 ObsBuilder는 ch0에 박으니 기본값 0
    """

    def __init__(
        self,
        input_channels: int,
        num_actions: int,
        obs_channels_per_frame: int = 4,
        meta_patch: int = 4,
        meta_channel_offset: int = 0,
    ):
        super().__init__()

        self.input_channels = int(input_channels)
        self.num_actions = int(num_actions)

        self.obs_channels_per_frame = int(obs_channels_per_frame)
        self.meta_patch = int(meta_patch)
        self.meta_channel_offset = int(meta_channel_offset)

        self.meta_dim = 3  # x, y, conf

        # CNN trunk
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # 입력 크기 변화에도 안전하게 고정 feature 크기 만들기
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc_img = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )

        # meta (x,y,conf) -> 임베딩
        self.fc_meta = nn.Sequential(
            nn.Linear(self.meta_dim, 32),
            nn.ReLU(),
        )

        feat_dim = 512 + 32
        self.policy_head = nn.Linear(feat_dim, self.num_actions)
        self.value_head = nn.Linear(feat_dim, 1)

    @staticmethod
    def _normalize_input(x: torch.Tensor) -> torch.Tensor:
        """
        obs가 float32 0..1 이 기본이지만,
        혹시 0..255로 들어오면 자동 보정.
        """
        if x.dtype.is_floating_point:
            # max()가 큰 경우만 255 스케일로 간주
            try:
                if x.max().item() > 1.5:
                    return x / 255.0
            except Exception:
                pass
            return x
        return x.float() / 255.0

    def _meta_channel_index(self, c_total: int) -> int:
        """
        가장 최근 프레임의 meta 채널 인덱스를 계산.
        - T = C_total // obs_channels_per_frame
        - last_frame_base = (T-1)*obs_channels_per_frame
        - idx = last_frame_base + meta_channel_offset
        """
        per = max(1, int(self.obs_channels_per_frame))

        if c_total <= 0:
            return 0

        if c_total < per:
            # 이상 케이스: 그냥 마지막 채널에서 읽기
            return max(0, c_total - 1)

        T = max(1, c_total // per)
        last_base = (T - 1) * per
        idx = last_base + int(self.meta_channel_offset)

        # 안전 클램프
        idx = max(0, min(int(idx), int(c_total - 1)))
        return idx

    def _extract_meta(self, x01: torch.Tensor) -> torch.Tensor:
        """
        x01: (B, C_total, H, W) float 0..1
        지정 채널(최근 프레임의 ch0 등)에서 meta patch를 평균으로 읽어 (B,3)로 반환
        """
        B, C_total, H, W = x01.shape
        p = int(self.meta_patch)
        need_w = p * 3

        if (H < p) or (W < need_w):
            return torch.zeros((B, self.meta_dim), device=x01.device, dtype=x01.dtype)

        ch_idx = self._meta_channel_index(int(C_total))
        m = x01[:, ch_idx]  # (B,H,W)

        x_val = m[:, 0:p, 0:p].mean(dim=(1, 2))
        y_val = m[:, 0:p, p:2 * p].mean(dim=(1, 2))
        c_val = m[:, 0:p, 2 * p:3 * p].mean(dim=(1, 2))

        meta = torch.stack([x_val, y_val, c_val], dim=1)  # (B,3)
        meta = torch.nan_to_num(meta, nan=0.0, posinf=1.0, neginf=0.0)
        meta = torch.clamp(meta, 0.0, 1.0)
        return meta

    def forward(self, x: torch.Tensor):
        x01 = self._normalize_input(x)

        # meta
        meta = self._extract_meta(x01)     # (B,3)
        meta_feat = self.fc_meta(meta)     # (B,32)

        # img
        z = self.conv(x01)
        z = self.pool(z)
        z = z.view(z.size(0), -1)
        img_feat = self.fc_img(z)          # (B,512)

        feat = torch.cat([img_feat, meta_feat], dim=1)
        logits = self.policy_head(feat)
        value = self.value_head(feat)
        return logits, value
