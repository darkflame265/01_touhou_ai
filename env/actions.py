# env/actions.py
from enum import Enum

class Action(Enum):
    NONE = []

    # SLOW 8방향만 유지 (Shift는 set_always_slow(True)로 항상 눌림)
    SLOW_LEFT = ["LEFT"]
    SLOW_RIGHT = ["RIGHT"]
    SLOW_UP = ["UP"]
    SLOW_DOWN = ["DOWN"]

    SLOW_UP_LEFT = ["UP", "LEFT"]
    SLOW_UP_RIGHT = ["UP", "RIGHT"]
    SLOW_DOWN_LEFT = ["DOWN", "LEFT"]
    SLOW_DOWN_RIGHT = ["DOWN", "RIGHT"]

ACTIONS = list(Action)
