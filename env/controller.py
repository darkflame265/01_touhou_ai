# env/controller.py
import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)

# SendInput signature (menu.py와 동일)
user32.SendInput.argtypes = (wintypes.UINT, ctypes.c_void_p, ctypes.c_int)
user32.SendInput.restype = wintypes.UINT

INPUT_KEYBOARD = 1

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

# ULONG_PTR 호환 (menu.py와 동일)
if ctypes.sizeof(ctypes.c_void_p) == 8:
    ULONG_PTR = ctypes.c_uint64
else:
    ULONG_PTR = ctypes.c_uint32


# --- Windows INPUT full layout (menu.py와 동일) ---
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", INPUT_UNION),
    ]


def _send_input(inp: INPUT) -> bool:
    arr = (INPUT * 1)(inp)
    p = ctypes.byref(arr[0])
    sent = user32.SendInput(1, p, ctypes.sizeof(INPUT))
    return sent == 1


def _send_scancode(scan: int, is_down: bool, extended: bool = False) -> bool:
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_EXTENDEDKEY if extended else 0)
    if not is_down:
        flags |= KEYEVENTF_KEYUP

    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki = KEYBDINPUT(wVk=0, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=0)
    return _send_input(inp)


# =========================
# Key mapping (Scan codes)
# =========================
# Set 1 scancodes + extended flags (menu.py와 동일한 개념)
SCAN = {
    "z": (0x2C, False),
    "shift": (0x2A, False),
    "left": (0x4B, True),
    "right": (0x4D, True),
    "up": (0x48, True),
    "down": (0x50, True),
}

ATTACK_KEY = "z"

MOVE_KEYS = {
    "LEFT": "left",
    "RIGHT": "right",
    "UP": "up",
    "DOWN": "down",
    "SLOW": "shift",
}

# ---- internal state ----
_HELD = set()
_ATTACK_HOLD = True

# ✅ 항상 SLOW(Shift) 유지 모드
_ALWAYS_SLOW = False


def set_attack_hold(enabled: bool):
    global _ATTACK_HOLD
    _ATTACK_HOLD = bool(enabled)
    if not _ATTACK_HOLD:
        _key_up(ATTACK_KEY)


def set_always_slow(enabled: bool):
    """
    ✅ enabled=True면 이동 시 Shift(SLOW)를 항상 누른 상태로 유지.
    - 액션 공간은 그대로 두고, 실제 입력만 전부 SLOW로 바꿈.
    - reset()에서 호출해 두면 키 꼬임에도 강함.
    """
    global _ALWAYS_SLOW
    _ALWAYS_SLOW = bool(enabled)

    # 즉시 반영: 켜면 눌러두고, 끄면 해제
    slow_key = MOVE_KEYS["SLOW"]
    if _ALWAYS_SLOW:
        _key_down(slow_key)
    else:
        _key_up(slow_key)


def _key_down(key: str):
    if key in _HELD:
        return
    sc, ext = SCAN[key]
    _send_scancode(sc, is_down=True, extended=ext)
    _HELD.add(key)


def _key_up(key: str):
    sc, ext = SCAN[key]
    _send_scancode(sc, is_down=False, extended=ext)
    _HELD.discard(key)


def press_keys(action_keys):
    """
    action_keys: ["LEFT"], ["SLOW","UP"], ["UP","RIGHT"] 같은 형태
    ✅ _ALWAYS_SLOW가 True면 action_keys에 SLOW가 없어도 Shift를 항상 누름.
    """
    # 공격키 유지
    if _ATTACK_HOLD:
        _key_down(ATTACK_KEY)
    else:
        _key_up(ATTACK_KEY)

    # --- SLOW(shift)는 특별 처리 ---
    slow_name = "SLOW"
    slow_key = MOVE_KEYS[slow_name]
    if _ALWAYS_SLOW:
        _key_down(slow_key)
    else:
        if slow_name in action_keys:
            _key_down(slow_key)
        else:
            _key_up(slow_key)

    # --- 방향키 처리 ---
    # SLOW는 이미 처리했으니 제외하고 나머지만
    for name, key in MOVE_KEYS.items():
        if name == slow_name:
            continue
        if name in action_keys:
            _key_down(key)
        else:
            _key_up(key)


def release_all():
    """
    모든 키 해제.
    - 안전하게 항상 다 풀되, _ALWAYS_SLOW가 켜져 있으면 shift는 유지.
    """
    for name, key in MOVE_KEYS.items():
        if name == "SLOW" and _ALWAYS_SLOW:
            continue
        _key_up(key)
    _key_up(ATTACK_KEY)
