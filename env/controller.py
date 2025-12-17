# env/controller.py
import pydirectinput

# 공격 키 (항상 누르고 있을 키)
ATTACK_KEY = "z"

# 이동/모디파이어 키 목록
MOVE_KEYS = {
    "LEFT": "left",
    "RIGHT": "right",
    "UP": "up",
    "DOWN": "down",
    "SLOW": "shift",
}

# ---- internal state ----
_HELD = set()            # 실제로 keyDown 했다고 "우리가" 믿는 키들
_ATTACK_HOLD = True      # 공격키 자동 유지 on/off


def set_attack_hold(enabled: bool):
    """
    공격키(ATTACK_KEY)를 자동으로 누른 상태로 유지할지.
    - 종료/로비 전환 시 False로 두고 싶으면 사용.
    """
    global _ATTACK_HOLD
    _ATTACK_HOLD = bool(enabled)
    if not _ATTACK_HOLD:
        # 즉시 공격키를 떼고 상태에서도 제거
        _key_up(ATTACK_KEY)


def _key_down(key: str):
    if key not in _HELD:
        pydirectinput.keyDown(key)
        _HELD.add(key)


def _key_up(key: str):
    if key in _HELD:
        pydirectinput.keyUp(key)
        _HELD.discard(key)
    else:
        # 혹시 held 추적이 꼬였더라도 keyUp은 한 번 보내는 게 안전할 때가 많음
        pydirectinput.keyUp(key)


def press_keys(action_keys):
    """
    action_keys: ["LEFT", "UP"] 같은 리스트
    """
    # 공격 키는 (옵션에 따라) 누른 상태 유지
    if _ATTACK_HOLD:
        _key_down(ATTACK_KEY)
    else:
        _key_up(ATTACK_KEY)

    # 이동/모디파이어 키 처리
    for name, key in MOVE_KEYS.items():
        if name in action_keys:
            _key_down(key)
        else:
            _key_up(key)


def release_all():
    # 이동 키 전부 떼기 (SLOW 포함)
    for key in MOVE_KEYS.values():
        _key_up(key)

    # 공격 키도 떼기
    _key_up(ATTACK_KEY)
