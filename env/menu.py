# env/menu.py
import time
import os
import ctypes
from ctypes import wintypes

import win32gui
import cv2
import numpy as np

from env.screen import find_touhou_window

# =========================
# WinAPI SendInput (FULL STRUCT)
# =========================
user32 = ctypes.WinDLL("user32", use_last_error=True)

user32.SendInput.argtypes = (wintypes.UINT, ctypes.c_void_p, ctypes.c_int)
user32.SendInput.restype = wintypes.UINT

INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

if ctypes.sizeof(ctypes.c_void_p) == 8:
    ULONG_PTR = ctypes.c_uint64
else:
    ULONG_PTR = ctypes.c_uint32


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


# =========================
# DirectInput scancodes
# =========================
SC_Z = 0x2C
SC_X = 0x2D
SC_UP = 0x48  # extended=True로 보내야 함


# =========================
# Templates
# =========================
TEMPLATES: dict[str, np.ndarray] = {}


def _load_gray(path: str) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None or g.size == 0:
        raise RuntimeError(f"template load failed: {path}")
    return g


def load_lobby_templates():
    """assets/lobby_practice.png, assets/lobby_quit.png 를 로드"""
    base = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "assets"))
    TEMPLATES["practice"] = _load_gray(os.path.join(base, "lobby_practice.png"))
    TEMPLATES["quit"] = _load_gray(os.path.join(base, "lobby_quit.png"))
    print("[MENU] lobby templates loaded:", list(TEMPLATES.keys()))


# 모듈 import 시 자동 로드 시도(없으면 경고만)
try:
    load_lobby_templates()
except Exception as e:
    print("[MENU][WARN] lobby templates not loaded:", repr(e))


def _match_template(gray: np.ndarray, tmpl: np.ndarray) -> float:
    th, tw = tmpl.shape[:2]
    if gray.shape[0] < th or gray.shape[1] < tw:
        return 0.0
    res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
    return float(res.max())


# =========================
# Focus helper (강제 포커스)
# =========================
def focus_touhou_window(max_try: int = 5, sleep: float = 0.05) -> bool:
    """
    SetForegroundWindow가 정책상 실패하는 경우가 많아서
    - topmost 토글
    - AttachThreadInput 트릭
    을 섞어서 포커스를 최대한 잡는다.

    NOTE:
      GetCurrentThreadId 는 user32가 아니라 kernel32!
    """
    hwnd = find_touhou_window()
    if not hwnd:
        print("[MENU][ERR] Touhou window not found")
        return False

    u32 = ctypes.WinDLL("user32", use_last_error=True)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    GetForegroundWindow = u32.GetForegroundWindow
    GetForegroundWindow.restype = wintypes.HWND

    GetWindowThreadProcessId = u32.GetWindowThreadProcessId
    GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    GetWindowThreadProcessId.restype = wintypes.DWORD

    AttachThreadInput = u32.AttachThreadInput
    AttachThreadInput.argtypes = (wintypes.DWORD, wintypes.DWORD, wintypes.BOOL)
    AttachThreadInput.restype = wintypes.BOOL

    SetForegroundWindow = u32.SetForegroundWindow
    SetForegroundWindow.argtypes = (wintypes.HWND,)
    SetForegroundWindow.restype = wintypes.BOOL

    SetActiveWindow = u32.SetActiveWindow
    SetActiveWindow.argtypes = (wintypes.HWND,)
    SetActiveWindow.restype = wintypes.HWND

    SetFocus = u32.SetFocus
    SetFocus.argtypes = (wintypes.HWND,)
    SetFocus.restype = wintypes.HWND

    BringWindowToTop = u32.BringWindowToTop
    BringWindowToTop.argtypes = (wintypes.HWND,)
    BringWindowToTop.restype = wintypes.BOOL

    SetWindowPos = u32.SetWindowPos
    SetWindowPos.argtypes = (
        wintypes.HWND, wintypes.HWND,
        wintypes.INT, wintypes.INT, wintypes.INT, wintypes.INT,
        wintypes.UINT
    )
    SetWindowPos.restype = wintypes.BOOL

    GetCurrentThreadId = k32.GetCurrentThreadId
    GetCurrentThreadId.restype = wintypes.DWORD

    SW_RESTORE = 9
    HWND_TOPMOST = wintypes.HWND(-1)
    HWND_NOTOPMOST = wintypes.HWND(-2)
    SWP_NOMOVE = 0x0002
    SWP_NOSIZE = 0x0001
    SWP_SHOWWINDOW = 0x0040

    def _is_fg():
        return GetForegroundWindow() == hwnd

    for _ in range(max_try):
        try:
            win32gui.ShowWindow(hwnd, SW_RESTORE)
        except Exception:
            pass

        # 1) 기본 시도
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass
        if _is_fg():
            return True

        # 2) topmost 토글
        try:
            SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
            SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
        except Exception:
            pass

        try:
            BringWindowToTop(hwnd)
        except Exception:
            pass

        # 3) AttachThreadInput
        fg = GetForegroundWindow()
        fg_pid = wintypes.DWORD(0)
        target_pid = wintypes.DWORD(0)

        fg_tid = GetWindowThreadProcessId(fg, ctypes.byref(fg_pid)) if fg else 0
        target_tid = GetWindowThreadProcessId(hwnd, ctypes.byref(target_pid))
        cur_tid = GetCurrentThreadId()

        try:
            if fg_tid and fg_tid != cur_tid:
                AttachThreadInput(fg_tid, cur_tid, True)
            if target_tid and target_tid != cur_tid:
                AttachThreadInput(target_tid, cur_tid, True)

            SetActiveWindow(hwnd)
            SetFocus(hwnd)
            SetForegroundWindow(hwnd)

        finally:
            try:
                if fg_tid and fg_tid != cur_tid:
                    AttachThreadInput(fg_tid, cur_tid, False)
            except Exception:
                pass
            try:
                if target_tid and target_tid != cur_tid:
                    AttachThreadInput(target_tid, cur_tid, False)
            except Exception:
                pass

        if _is_fg():
            return True

        time.sleep(sleep)

    print("[MENU][ERR] focus failed: cannot bring Touhou window to foreground")
    return False


# =========================
# SendInput helpers
# =========================
def _send_input(inp: INPUT) -> bool:
    arr = (INPUT * 1)(inp)
    p = ctypes.byref(arr[0])
    sent = user32.SendInput(1, p, ctypes.sizeof(INPUT))
    return sent == 1


def tap_scancode(scan: int, extended=False, press=0.02, gap=0.03, label=""):
    # if label:
    #     print(f"[MENU] tap_sc {label} sc=0x{scan:02X} ext={extended}")

    flags_down = KEYEVENTF_SCANCODE | (KEYEVENTF_EXTENDEDKEY if extended else 0)
    flags_up = flags_down | KEYEVENTF_KEYUP

    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki = KEYBDINPUT(wVk=0, wScan=scan, dwFlags=flags_down, time=0, dwExtraInfo=0)
    _send_input(inp)

    time.sleep(press)

    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki = KEYBDINPUT(wVk=0, wScan=scan, dwFlags=flags_up, time=0, dwExtraInfo=0)
    _send_input(inp)

    time.sleep(gap)


# =========================
# Menu routines
# =========================
def enter_practice_from_cursor():
    print("[MENU] FAST practice entry")
    if not focus_touhou_window():
        print("[MENU][ERR] focus failed -> skip key inputs")
        return
    time.sleep(0.05)

    t0 = time.time()
    for i in range(6):
        tap_scancode(SC_Z, label=f"Z{i+1}/6", press=0.02, gap=0.02)
        time.sleep(0.7)
    dt = time.time() - t0
    print(f"[MENU] Z x6 done in {dt:.3f}s")


def recover_to_practice_from_lobby():
    print("[MENU][RECOVER] start")
    if not focus_touhou_window():
        print("[MENU][ERR] focus failed -> skip key inputs")
        return
    time.sleep(0.05)

    for i in range(3):
        tap_scancode(SC_X, label=f"X{i+1}/3", press=0.02, gap=0.02)
        time.sleep(0.2)

    for i in range(5):
        tap_scancode(SC_UP, extended=True, label=f"UP{i+1}/5", press=0.02, gap=0.02)
        time.sleep(0.2)

    for i in range(6):
        tap_scancode(SC_Z, label=f"Z{i+1}/6", press=0.02, gap=0.02)
        time.sleep(0.7)

    print("[MENU][RECOVER] done")


def recover_from_score_to_lobby(screen, max_sec=3.0):
    print("[MENU][RECOVER_SCORE] start")
    if not focus_touhou_window():
        print("[MENU][ERR] focus failed -> skip key inputs")
        return False
    time.sleep(0.05)

    t0 = time.time()
    tries = 0

    while (time.time() - t0) < max_sec:
        img = screen.capture()
        if not screen.is_score_screen(img):
            print(f"[MENU][RECOVER_SCORE] done (tries={tries})")
            return True

        tap_scancode(SC_X, label=f"X{tries+1}", press=0.02, gap=0.02)
        time.sleep(0.08)

        if (tries % 3) == 2:
            tap_scancode(SC_Z, label=f"Z{tries+1}", press=0.02, gap=0.02)
            time.sleep(0.12)

        tries += 1

    print(f"[MENU][RECOVER_SCORE] timeout (tries={tries})")
    return False


# =========================
# Location detection (템플릿은 "로비/일러스트 구분"에만)
# =========================
def _roi(img_bgr, x1r, y1r, x2r, y2r):
    h, w = img_bgr.shape[:2]
    x1 = int(np.clip(w * x1r, 0, w - 1))
    x2 = int(np.clip(w * x2r, 0, w))
    y1 = int(np.clip(h * y1r, 0, h - 1))
    y2 = int(np.clip(h * y2r, 0, h))
    if x2 <= x1 or y2 <= y1:
        return img_bgr[0:1, 0:1]
    return img_bgr[y1:y2, x1:x2]


def _menu_highlight_score(img_bgr_roi):
    """
    선택된 메뉴는 흰색 글로우가 강함:
      - mean (밝기)
      - white_ratio (아주 밝은 픽셀 비율)
      - std (글로우/윤곽으로 표준편차도 상승)
    """
    if img_bgr_roi.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    g = cv2.cvtColor(img_bgr_roi, cv2.COLOR_BGR2GRAY)
    mean = float(g.mean())
    white_ratio = float((g >= 210).mean())   # 200~225 사이에서 튜닝 가능
    std = float(g.std())

    # 조합 점수 (경험적으로 안정)
    score = (mean * 0.55) + (white_ratio * 420.0) + (std * 1.1)
    return score, mean, white_ratio, std


def detect_location(screen):
    """
    return dict:
      state: 'SCORE' | 'IN_GAME' | 'LOBBY' | 'ILLUST' | 'UNKNOWN'
      selected_name: 'PRACTICE' | 'QUIT' | None
      scores: debug dict or None
    """
    img = screen.capture()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) SCORE
    try:
        if screen.is_score_screen(img):
            return {"state": "SCORE", "selected_name": None, "scores": None}
    except Exception:
        pass

    # 2) IN_GAME 힌트 (UI 패널)
    in_game_hint = False
    try:
        in_game_hint = bool(screen.ui_panel_present(img))
    except Exception:
        in_game_hint = False

    # 3) 로비 메뉴 "존재" 판정: 오른쪽 메뉴 영역에서 템플릿 존재 확인
    #    (중요: 전체 화면이 아니라 메뉴 ROI로 제한!)
    menu_roi = _roi(img, 0.55, 0.28, 0.98, 0.92)
    menu_gray = cv2.cvtColor(menu_roi, cv2.COLOR_BGR2GRAY)

    practice_t = TEMPLATES.get("practice", None)
    quit_t = TEMPLATES.get("quit", None)

    practice_tm = _match_template(menu_gray, practice_t) if practice_t is not None else 0.0
    quit_tm = _match_template(menu_gray, quit_t) if quit_t is not None else 0.0

    menu_present = (max(practice_tm, quit_tm) >= 0.70)  # 존재 판정은 0.65~0.75 사이 튜닝

    if not menu_present:
        # 메뉴가 없으면 보통 일러스트 화면. 다만 게임 중이면 IN_GAME 우선.
        if in_game_hint:
            return {"state": "IN_GAME", "selected_name": None, "scores": {"practice_tm": practice_tm, "quit_tm": quit_tm}}
        return {"state": "ILLUST", "selected_name": None, "scores": {"practice_tm": practice_tm, "quit_tm": quit_tm}}

    # 4) 선택(커서) 판정: 템플릿이 아니라 "강조 점수"로 결정
    #    (메뉴 존재가 확정된 상태에서만!)
    practice_roi = _roi(img, 0.67, 0.40, 0.97, 0.58)
    quit_roi = _roi(img, 0.67, 0.76, 0.97, 0.92)

    pr_score, pr_mean, pr_wr, pr_std = _menu_highlight_score(practice_roi)
    qt_score, qt_mean, qt_wr, qt_std = _menu_highlight_score(quit_roi)

    selected = None
    # 점수 차이가 충분히 나야 선택 판정 (흔들림 방지)
    # margin을 너무 크게 잡으면 None이 많아짐. 8~20 사이에서 조정.
    margin = 12.0
    if pr_score >= qt_score + margin:
        selected = "PRACTICE"
    elif qt_score >= pr_score + margin:
        selected = "QUIT"

    return {
        "state": "LOBBY",
        "selected_name": selected,
        "scores": {
            "practice_tm": practice_tm,
            "quit_tm": quit_tm,
            "pr_score": pr_score,
            "qt_score": qt_score,
            "pr_mean": pr_mean,
            "qt_mean": qt_mean,
            "pr_white_ratio": pr_wr,
            "qt_white_ratio": qt_wr,
            "pr_std": pr_std,
            "qt_std": qt_std,
        }
    }


# =========================
# Lobby -> Practice align
# =========================
def ensure_practice_cursor_from_lobby(screen, verify=True, max_try=3):
    """
    - ILLUST면 Z 1회 눌러 LOBBY 진입
    - LOBBY면:
        X 1회 (Quit 기준점)
        UP 5회 (Practice로)
      이후 verify면 selected=PRACTICE 확인
    """
    if not focus_touhou_window():
        print("[MENU][BOOT] focus failed -> cannot send keys")
        return False

    for attempt in range(max_try):
        st = detect_location(screen)
        print(f"[MENU][BOOT] try {attempt+1}/{max_try} state={st.get('state')} selected={st.get('selected_name')}")

        if st.get("state") == "ILLUST":
            print("[MENU][BOOT] ILLUST detected -> tap Z")
            tap_scancode(SC_Z, label="Z(enter lobby)", press=0.02, gap=0.02)
            time.sleep(0.45)
            continue

        if st.get("state") != "LOBBY":
            print(f"[MENU][BOOT] not in LOBBY (state={st.get('state')}) -> cannot align")
            return False

        # 기준점: Quit로
        tap_scancode(SC_X, label="X(to Quit)", press=0.02, gap=0.02)
        time.sleep(0.25)

        # Practice로 UP 5회
        for i in range(5):
            tap_scancode(SC_UP, extended=True, label=f"UP{i+1}/5", press=0.02, gap=0.02)
            time.sleep(0.12)

        # 커서 이동/글로우 반영 대기
        time.sleep(0.30)

        if not verify:
            return True

        st2 = detect_location(screen)
        print(f"[MENU][BOOT] verify result: state={st2.get('state')} selected={st2.get('selected_name')}")
        if st2.get("state") == "LOBBY" and st2.get("selected_name") == "PRACTICE":
            return True

        print("[MENU][BOOT] verify failed -> retry")
        time.sleep(0.25)

    print("[MENU][BOOT] failed to align PRACTICE after retries")
    return False
