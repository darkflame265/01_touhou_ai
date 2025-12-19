# env/position_shaping.py
from __future__ import annotations
from dataclasses import dataclass

from env.playfield_utils import get_target_point, get_playfield_rect_safe


@dataclass
class ShapingConfig:
    target_y_ratio: float = 0.78
    shaping_k: float = 0.35
    shaping_clip: float = 0.25

    stuck_dist_px: int = 2
    stuck_need: int = 10
    stuck_pen: float = 0.20

    edge_guard_px: int = 24
    edge_guard_pen: float = 0.08

    # ✅ 절대 y거리 페널티
    abs_y_penalty_k: float = 0.06

    # ✅ 위쪽 성향 억제(soft band)
    # None이면 꺼짐
    top_limit_px: int | None = None
    top_soft_band_px: int = 160
    top_soft_pen: float = 0.08


class PositionShaper:
    """
    EnvState(self.s)에 저장된 prev_dist_norm, prev_pc, stuck_run 등을 사용/갱신한다.
    통계 edge60_cnt, top270_cnt도 여기서 카운트한다.
    """
    def __init__(self, screen, env_state, cfg: ShapingConfig):
        self.screen = screen
        self.s = env_state
        self.cfg = cfg

    def reset(self):
        self.s.prev_dist_norm = None
        self.s.prev_pc = None
        self.s.stuck_run = 0
        self.s.edge60_cnt = 0
        self.s.top270_cnt = 0

    def step_reward(self, img_bgr, player_center):
        """
        player_center가 None이면 0.0 반환.
        """
        if player_center is None:
            return 0.0

        px, py = int(player_center[0]), int(player_center[1])

        tx, ty, (l, t, r, b) = get_target_point(self.screen, img_bgr, self.cfg.target_y_ratio)
        w = max(1.0, float(r - l))
        h = max(1.0, float(b - t))

        dx = (px - tx) / w
        dy = (py - ty) / h
        dist_norm = float((dx * dx + dy * dy) ** 0.5)

        reward = 0.0

        # (A) delta shaping
        if self.s.prev_dist_norm is not None:
            delta = float(self.s.prev_dist_norm - dist_norm)
            shape = float(self.cfg.shaping_k * delta)
            if shape > self.cfg.shaping_clip:
                shape = self.cfg.shaping_clip
            elif shape < -self.cfg.shaping_clip:
                shape = -self.cfg.shaping_clip
            reward += shape
        self.s.prev_dist_norm = dist_norm

        # (B) ✅ 절대 y거리 패널티(매 프레임)
        abs_y_norm = float(abs(py - ty) / max(1.0, h))
        reward -= float(self.cfg.abs_y_penalty_k * abs_y_norm)

        # (C) stuck 패널티
        if self.s.prev_pc is not None:
            dxp = px - int(self.s.prev_pc[0])
            dyp = py - int(self.s.prev_pc[1])
            d2 = dxp * dxp + dyp * dyp
            if d2 <= (self.cfg.stuck_dist_px * self.cfg.stuck_dist_px):
                self.s.stuck_run += 1
            else:
                self.s.stuck_run = 0

            if self.s.stuck_run >= self.cfg.stuck_need:
                reward -= float(self.cfg.stuck_pen)
                self.s.stuck_run = int(self.cfg.stuck_need * 0.6)

        self.s.prev_pc = (px, py)

        # (D) edge guard 패널티
        edge_px = min(px - l, r - px, py - t, b - py)
        if edge_px <= self.cfg.edge_guard_px:
            x = float((self.cfg.edge_guard_px - edge_px) / max(1, self.cfg.edge_guard_px))
            reward -= float(self.cfg.edge_guard_pen * (x * x))

        # 통계
        if edge_px <= 60:
            self.s.edge60_cnt += 1
        if py < (t + 270):
            self.s.top270_cnt += 1

        # (E) 위쪽 성향 억제(soft band) - 옵션
        if self.cfg.top_limit_px is not None:
            top_line = t + int(self.cfg.top_limit_px)
            band = max(1, int(self.cfg.top_soft_band_px))
            soft_start = top_line + band  # 이보다 아래는 패널티 0

            if py < soft_start:
                ratio = float((soft_start - py) / band)
                if ratio > 1.0:
                    ratio = 1.0
                reward -= float(self.cfg.top_soft_pen * (ratio * ratio))

        return float(reward)
