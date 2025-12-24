# env/action_masking.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from env.actions import ACTIONS
from env.playfield_utils import get_playfield_rect_safe


@dataclass
class MaskingConfig:
    margin_px: int = 200
    use_flip: bool = True

    # 상단 제한(원하면 켜기)
    top_limit_px: Optional[int] = None
    top_limit_fudge_px: int = 10


class ActionMasker:
    def __init__(self, screen, obs, cfg: MaskingConfig):
        self.screen = screen
        self.obs = obs
        self.cfg = cfg

    def _action_dir(self, action_enum) -> Tuple[int, int]:
        """
        value는 ["LEFT"], ["UP","RIGHT"] 같이 방향키만 들어있다고 가정.
        """
        keys = set(action_enum.value)
        dx = (-1 if "LEFT" in keys else (1 if "RIGHT" in keys else 0))
        dy = (-1 if "UP" in keys else (1 if "DOWN" in keys else 0))
        return dx, dy

    def _flip_action(self, action_enum, flip_x: bool = False, flip_y: bool = False):
        keys = list(action_enum.value)

        def repl(k: str) -> str:
            if flip_x:
                if k == "LEFT": return "RIGHT"
                if k == "RIGHT": return "LEFT"
            if flip_y:
                if k == "UP": return "DOWN"
                if k == "DOWN": return "UP"
            return k

        new_keys = [repl(k) for k in keys]

        for a in ACTIONS:
            if list(a.value) == new_keys:
                return a
        return action_enum

    def get_action_mask(self, img_bgr) -> np.ndarray:
        mask = np.ones((len(ACTIONS),), dtype=np.bool_)

        pc = getattr(self.obs, "player_center", None)
        if pc is None:
            return mask

        px, py = int(pc[0]), int(pc[1])
        l, t, r, b = get_playfield_rect_safe(self.screen, img_bgr)

        margin_px = int(self.cfg.margin_px)

        left_d = px - l
        right_d = r - px
        top_d = py - t
        bot_d = b - py

        near_left = (left_d <= margin_px)
        near_right = (right_d <= margin_px)
        near_top = (top_d <= margin_px)
        near_bot = (bot_d <= margin_px)

        # top-limit: py가 너무 위면 UP 방향 금지
        top_forbid = False
        if self.cfg.top_limit_px is not None:
            top_line = t + int(self.cfg.top_limit_px) - int(self.cfg.top_limit_fudge_px)
            top_forbid = (py <= top_line)

        for i, a in enumerate(ACTIONS):
            if a.name == "NONE":
                continue

            dx, dy = self._action_dir(a)

            if near_left and dx < 0:
                mask[i] = False
            if near_right and dx > 0:
                mask[i] = False
            if near_top and dy < 0:
                mask[i] = False
            if near_bot and dy > 0:
                mask[i] = False

            if top_forbid and dy < 0:
                mask[i] = False

        return mask

    def apply_action_mask(self, action_idx: int, img_bgr):
        # ✅ 안전장치: idx 범위 밖이면 NONE으로
        try:
            ai = int(action_idx)
        except Exception:
            ai = 0
        if not (0 <= ai < len(ACTIONS)):
            ai = 0

        mask = self.get_action_mask(img_bgr)

        # 이미 유효하면 그대로
        if bool(mask[ai]):
            return ai, False, mask

        orig = ACTIONS[ai]
        pc = getattr(self.obs, "player_center", None)

        flip_x = False
        flip_y = False

        if pc is not None:
            px, py = int(pc[0]), int(pc[1])
            l, t, r, b = get_playfield_rect_safe(self.screen, img_bgr)

            margin_px = int(self.cfg.margin_px)

            left_d = px - l
            right_d = r - px
            top_d = py - t
            bot_d = b - py

            dx, dy = self._action_dir(orig)

            if (left_d <= margin_px and dx < 0) or (right_d <= margin_px and dx > 0):
                flip_x = True
            if (top_d <= margin_px and dy < 0) or (bot_d <= margin_px and dy > 0):
                flip_y = True

            # top-limit 걸리면 UP은 DOWN으로 치환 시도
            if self.cfg.top_limit_px is not None:
                top_forbid = (py <= (t + int(self.cfg.top_limit_px)))
                if top_forbid and dy < 0:
                    flip_y = True

        alt = orig
        if self.cfg.use_flip:
            alt = self._flip_action(orig, flip_x=flip_x, flip_y=flip_y)

        try:
            alt_idx = ACTIONS.index(alt)
        except ValueError:
            alt_idx = 0  # NONE

        if 0 <= alt_idx < len(ACTIONS) and bool(mask[alt_idx]):
            return int(alt_idx), True, mask

        # 마지막 보정: UP 계열이면 DOWN 계열 강제 시도
        if pc is not None:
            dx, dy = self._action_dir(orig)
            if dy < 0:
                forced = self._flip_action(orig, flip_y=True)
                try:
                    forced_idx = ACTIONS.index(forced)
                    if bool(mask[forced_idx]):
                        return int(forced_idx), True, mask
                except ValueError:
                    pass

        return 0, True, mask
