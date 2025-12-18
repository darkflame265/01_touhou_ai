# env/env_state.py
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
    frame_stack: deque = field(default_factory=lambda: deque(maxlen=4))

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
