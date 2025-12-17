# env/actions.py
from enum import Enum


class Action(Enum):
    NONE = []

    # FAST 8방향
    LEFT = ["LEFT"]
    RIGHT = ["RIGHT"]
    UP = ["UP"]
    DOWN = ["DOWN"]

    UP_LEFT = ["UP", "LEFT"]
    UP_RIGHT = ["UP", "RIGHT"]
    DOWN_LEFT = ["DOWN", "LEFT"]
    DOWN_RIGHT = ["DOWN", "RIGHT"]

    # SLOW 8방향 (Shift + 방향)
    SLOW_LEFT = ["SLOW", "LEFT"]
    SLOW_RIGHT = ["SLOW", "RIGHT"]
    SLOW_UP = ["SLOW", "UP"]
    SLOW_DOWN = ["SLOW", "DOWN"]

    SLOW_UP_LEFT = ["SLOW", "UP", "LEFT"]
    SLOW_UP_RIGHT = ["SLOW", "UP", "RIGHT"]
    SLOW_DOWN_LEFT = ["SLOW", "DOWN", "LEFT"]
    SLOW_DOWN_RIGHT = ["SLOW", "DOWN", "RIGHT"]


ACTIONS = list(Action)
