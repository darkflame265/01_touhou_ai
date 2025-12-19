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
    # None이면 기능 꺼짐
    top_limit_px: Optional[int] = None
    top_limit_fudge_px: int = 10  # 너 코드에 있던 -10 여유


class ActionMasker:
    def __init__(self, screen, obs, cfg: MaskingConfig):
        self.screen = screen
        self.obs = obs
        self.cfg = cfg

    def _action_dir(self, action_enum) -> Tuple[int, int, bool]:
        keys = set(action_enum.value)
        dx = (-1 if "LEFT" in keys else (1 if "RIGHT" in keys else 0))
        dy = (-1 if "UP" in keys else (1 if "DOWN" in keys else 0))
        is_slow = ("SLOW" in keys)
        return dx, dy, is_slow

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
            dx, dy, _ = self._action_dir(a)
            if a.name == "NONE":
                continue

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
        mask = self.get_action_mask(img_bgr)

        if 0 <= int(action_idx) < len(ACTIONS) and bool(mask[int(action_idx)]):
            return int(action_idx), False, mask

        orig = ACTIONS[int(action_idx)]
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

            dx, dy, _ = self._action_dir(orig)

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
            dx, dy, _ = self._action_dir(orig)
            if dy < 0:
                forced = self._flip_action(orig, flip_y=True)
                try:
                    forced_idx = ACTIONS.index(forced)
                    if bool(mask[forced_idx]):
                        return int(forced_idx), True, mask
                except ValueError:
                    pass

        return 0, True, mask
