# env/actions.py
from enum import Enum


class Action(Enum):
    """
    ✅ 상시 SLOW 전제:
    - Shift는 set_always_slow(True) 같은 방식으로 환경에서 항상 눌려있다고 가정
    - 여기 액션은 '방향'만 담당 (8방향)
    """

    SLOW_LEFT = ["LEFT"]
    SLOW_RIGHT = ["RIGHT"]
    SLOW_UP = ["UP"]
    SLOW_DOWN = ["DOWN"]

    SLOW_UP_LEFT = ["UP", "LEFT"]
    SLOW_UP_RIGHT = ["UP", "RIGHT"]
    SLOW_DOWN_LEFT = ["DOWN", "LEFT"]
    SLOW_DOWN_RIGHT = ["DOWN", "RIGHT"]


ACTIONS = list(Action)
