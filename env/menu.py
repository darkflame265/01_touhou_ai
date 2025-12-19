# env/menu.py
import time
import ctypes
from ctypes import wintypes
import win32gui

from env.screen import find_touhou_window

# =========================
# WinAPI SendInput (DEBUG + FULL STRUCT)
# =========================

user32 = ctypes.WinDLL("user32", use_last_error=True)

# SendInput signature (중요)
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

# --- Windows INPUT full layout (매우 중요) ---
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

LPINPUT = ctypes.POINTER(INPUT)


# DirectInput scancodes (키보드 기준)
SC_Z = 0x2C
SC_X = 0x2D
SC_UP = 0x48  # extended

# Virtual-Key codes
VK_Z = 0x5A
VK_X = 0x58
VK_UP = 0x26


def focus_touhou_window():
    hwnd = find_touhou_window()
    if not hwnd:
        print("[MENU][ERR] Touhou window not found")
        return False
    try:
        win32gui.ShowWindow(hwnd, 9)  # SW_RESTORE
        win32gui.SetForegroundWindow(hwnd)
        return True
    except Exception as e:
        print("[MENU][ERR] focus failed:", repr(e))
        return False


# =========================
# Debug helpers
# =========================

def debug_sendinput_layout():
    print("========== [MENU][DEBUG] ctypes layout ==========")
    print("Pointer size:", ctypes.sizeof(ctypes.c_void_p))
    print("sizeof(ULONG_PTR):", ctypes.sizeof(ULONG_PTR))
    print("sizeof(MOUSEINPUT):", ctypes.sizeof(MOUSEINPUT))
    print("sizeof(KEYBDINPUT):", ctypes.sizeof(KEYBDINPUT))
    print("sizeof(HARDWAREINPUT):", ctypes.sizeof(HARDWAREINPUT))
    print("sizeof(INPUT_UNION):", ctypes.sizeof(INPUT_UNION))
    print("sizeof(INPUT):", ctypes.sizeof(INPUT))
    print("align(INPUT):", ctypes.alignment(INPUT))
    print("offset INPUT.type:", INPUT.type.offset)
    print("offset INPUT.union:", INPUT.union.offset)
    print("=================================================")


def _send_input(inp: INPUT) -> bool:
    arr = (INPUT * 1)(inp)
    # 포인터는 첫 원소를 넘기는 게 가장 안전
    p = ctypes.byref(arr[0])
    sent = user32.SendInput(1, p, ctypes.sizeof(INPUT))
    if sent != 1:
        err = ctypes.get_last_error()
        print(f"[MENU][ERR] SendInput failed err={err}")
        return False
    return True


def tap_vk(vk: int, press=0.02, gap=0.03, label=""):
    if label:
        print(f"[MENU] tap_vk {label} vk=0x{vk:02X}")

    # key down (VK)
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki = KEYBDINPUT(wVk=vk, wScan=0, dwFlags=0, time=0, dwExtraInfo=0)
    _send_input(inp)

    time.sleep(press)

    # key up (VK)
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki = KEYBDINPUT(wVk=vk, wScan=0, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=0)
    _send_input(inp)

    time.sleep(gap)


def tap_scancode(scan: int, extended=False, press=0.02, gap=0.03, label=""):
    if label:
        print(f"[MENU] tap_sc {label} sc=0x{scan:02X} ext={extended}")

    flags_down = KEYEVENTF_SCANCODE | (KEYEVENTF_EXTENDEDKEY if extended else 0)
    flags_up = flags_down | KEYEVENTF_KEYUP

    # down
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki = KEYBDINPUT(wVk=0, wScan=scan, dwFlags=flags_down, time=0, dwExtraInfo=0)
    _send_input(inp)

    time.sleep(press)

    # up
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki = KEYBDINPUT(wVk=0, wScan=scan, dwFlags=flags_up, time=0, dwExtraInfo=0)
    _send_input(inp)

    time.sleep(gap)


def self_test_sendinput():
    """
    1) 구조체 크기 출력
    2) VK 방식으로 Z 입력 테스트
    3) Scancode 방식으로 Z 입력 테스트
    """
    debug_sendinput_layout()

    print("[MENU][DEBUG] Focus window and test keys...")
    focus_touhou_window()
    time.sleep(0.1)

    print("[MENU][DEBUG] VK test: Z")
    tap_vk(VK_Z, label="VK_Z")

    print("[MENU][DEBUG] SC test: Z")
    tap_scancode(SC_Z, label="SC_Z")

    print("[MENU][DEBUG] done")


# =========================
# Menu routines
# =========================

def enter_practice_from_cursor():
    print("[MENU] FAST practice entry (SendInput FULL)")
    focus_touhou_window()
    time.sleep(0.05)

    # 우선 안정성을 위해 scancode로 시도
    t0 = time.time()
    for i in range(6):
        tap_scancode(SC_Z, label=f"Z{i+1}/6", press=0.02, gap=0.02)
        time.sleep(0.7)
    dt = time.time() - t0
    print(f"[MENU] Z x6 done in {dt:.3f}s")


def recover_to_practice_from_lobby():
    print("[MENU][RECOVER] start (SendInput FULL)")
    focus_touhou_window()
    time.sleep(0.05)

    for i in range(10):
        tap_scancode(SC_X, label=f"X{i+1}/12", press=0.02, gap=0.02)
        time.sleep(0.2)

    for i in range(5):
        tap_scancode(SC_UP, extended=True, label=f"UP{i+1}/5", press=0.02, gap=0.02)
        time.sleep(0.2)

    for i in range(6):
        tap_scancode(SC_Z, label=f"Z{i+1}/6", press=0.02, gap=0.02)
        time.sleep(0.7)

    print("[MENU][RECOVER] done")

def recover_from_score_to_lobby(screen, max_sec=3.0):
    """
    Score 화면이면 X/Z로 빠져나와서(Score가 아닐 때까지) 로비로 복귀 시도.
    성공 기준: score_screen 감지가 False로 바뀜.
    """
    print("[MENU][RECOVER_SCORE] start")
    focus_touhou_window()
    time.sleep(0.05)

    t0 = time.time()
    tries = 0

    while (time.time() - t0) < max_sec:
        img = screen.capture()
        if not screen.is_score_screen(img):
            print(f"[MENU][RECOVER_SCORE] done (tries={tries})")
            return True

        # Score 화면에서 흔히 통하는 탈출 시퀀스
        tap_scancode(SC_X, label=f"X{tries+1}", press=0.02, gap=0.02)
        time.sleep(0.08)

        # 가끔 확인/exit가 Z인 경우가 있어서 섞어줌
        if (tries % 3) == 2:
            tap_scancode(SC_Z, label=f"Z{tries+1}", press=0.02, gap=0.02)
            time.sleep(0.12)

        tries += 1

    print(f"[MENU][RECOVER_SCORE] timeout (tries={tries})")
    return False

