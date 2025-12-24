import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field


@dataclass
class EnvState:
    lives: int = 3
    prev_ui_lives: int | None = None
    last_hit_time: float = 0.0
    hit_cooldown: float = 0.6

    prev_state: np.ndarray | None = None

    frame_stack_size: int = 4
    frame_stack: deque = field(default_factory=deque)

    slow_streak: int = 0
    slow_streak_max: int = 10

    debug_every: int = 20
    step_i: int = 0

    action_repeat: int = 2
    frame_sleep: float = 0.03

    ui_absent_count: int = 0
    ui_absent_needed: int = 2

    episode_terminated: bool = False
    terminate_until: float = 0.0
    terminate_cooldown_sec: float = 0.8

    prev_action_idx: int | None = None
    same_action_count: int = 0

    def __post_init__(self):
        # ✅ frame_stack_size와 deque(maxlen)를 항상 일치시킨다
        try:
            n = int(self.frame_stack_size)
            if n <= 0:
                n = 1
        except Exception:
            n = 4
            self.frame_stack_size = 4

        # 이미 deque면 maxlen만 맞춰서 재생성
        if isinstance(self.frame_stack, deque):
            if self.frame_stack.maxlen != n:
                old = list(self.frame_stack)
                self.frame_stack = deque(old[-n:], maxlen=n)
        else:
            self.frame_stack = deque(maxlen=n)
