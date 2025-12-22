# env/controller.py
import ctypes
from ctypes import wintypes
import time

user32 = ctypes.WinDLL("user32", use_last_error=True)

# SendInput signature
user32.SendInput.argtypes = (wintypes.UINT, ctypes.c_void_p, ctypes.c_int)
user32.SendInput.restype = wintypes.UINT

INPUT_KEYBOARD = 1

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

# ULONG_PTR 호환
if ctypes.sizeof(ctypes.c_void_p) == 8:
    ULONG_PTR = ctypes.c_uint64
else:
    ULONG_PTR = ctypes.c_uint32


# --- Windows INPUT full layout ---
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
    inp.union.ki = KEYBDINPUT(
        wVk=0,
        wScan=scan,
        dwFlags=flags,
        time=0,
        dwExtraInfo=0,
    )
    return _send_input(inp)


# =========================
# Key mapping (Scan codes)
# =========================
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


# =========================
# Public controls
# =========================
def set_attack_hold(enabled: bool):
    global _ATTACK_HOLD
    _ATTACK_HOLD = bool(enabled)
    if not _ATTACK_HOLD:
        _key_up(ATTACK_KEY)


def set_always_slow(enabled: bool):
    """
    enabled=True면 Shift를 항상 누른 상태로 유지.
    """
    global _ALWAYS_SLOW
    _ALWAYS_SLOW = bool(enabled)

    slow_key = MOVE_KEYS["SLOW"]
    if _ALWAYS_SLOW:
        _key_down(slow_key)
    else:
        _key_up(slow_key)


# =========================
# Low-level key ops
# =========================
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


# =========================
# Main input API
# =========================
def press_keys(action_keys):
    """
    action_keys: ["LEFT"], ["SLOW","UP"], ["UP","RIGHT"] 등
    """
    # 공격키 유지
    if _ATTACK_HOLD:
        _key_down(ATTACK_KEY)
    else:
        _key_up(ATTACK_KEY)

    # --- SLOW 처리 ---
    slow_name = "SLOW"
    slow_key = MOVE_KEYS[slow_name]

    if _ALWAYS_SLOW:
        _key_down(slow_key)
    else:
        if slow_name in action_keys:
            _key_down(slow_key)
        else:
            _key_up(slow_key)

    # --- 방향키 ---
    for name, key in MOVE_KEYS.items():
        if name == slow_name:
            continue
        if name in action_keys:
            _key_down(key)
        else:
            _key_up(key)


def release_all(force: bool = False):
    """
    모든 키 해제.

    force=True:
      - ALWAYS_SLOW 무시
      - Shift 포함 전부 해제
    """
    global _ALWAYS_SLOW

    if force:
        _ALWAYS_SLOW = False

    for name, key in MOVE_KEYS.items():
        _key_up(key)
    _key_up(ATTACK_KEY)


# =========================
# ✅ 종료 안전장치 (핵심)
# =========================
def cleanup_inputs_on_exit():
    """
    🔥 학습 종료 시 반드시 호출해야 하는 함수 🔥

    - ALWAYS_SLOW 강제 해제
    - 모든 키 KeyUp
    - Shift 토글 탭(Down → Up)으로
      Windows 키 상태 꼬임을 물리적으로 복구
    """
    global _ALWAYS_SLOW
    _ALWAYS_SLOW = False

    # 1) 논리적 해제
    release_all(force=True)
    time.sleep(0.05)

    # 2) 물리적 안전장치: Shift 토글 탭
    sc, ext = SCAN["shift"]
    _send_scancode(sc, is_down=True, extended=ext)
    time.sleep(0.03)
    _send_scancode(sc, is_down=False, extended=ext)

    print("[CTRL] cleanup_inputs_on_exit: Shift safely released")
