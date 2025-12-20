# env/menu.py
import time
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


# DirectInput scancodes
SC_Z = 0x2C
SC_X = 0x2D
SC_UP = 0x48  # extended

# Virtual-Key codes (debug용)
VK_Z = 0x5A
VK_X = 0x58
VK_UP = 0x26


def focus_touhou_window(max_try: int = 5, sleep: float = 0.05) -> bool:
    """
    ✅ Windows 포커스 정책 때문에 SetForegroundWindow가 실패하는 경우가 많아서
    AttachThreadInput 트릭 + topmost 토글까지 써서 최대한 강제로 포커스를 잡는다.

    핵심:
      - user32: GetForegroundWindow, GetWindowThreadProcessId, AttachThreadInput, SetForegroundWindow...
      - kernel32: GetCurrentThreadId  (※ user32에 없음!)
    """
    hwnd = find_touhou_window()
    if not hwnd:
        print("[MENU][ERR] Touhou window not found")
        return False

    u32 = ctypes.WinDLL("user32", use_last_error=True)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    # ---------- user32 ----------
    GetForegroundWindow = u32.GetForegroundWindow
    GetForegroundWindow.restype = wintypes.HWND

    GetWindowThreadProcessId = u32.GetWindowThreadProcessId
    GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    GetWindowThreadProcessId.restype = wintypes.DWORD

    AttachThreadInput = u32.AttachThreadInput
    AttachThreadInput.argtypes = (wintypes.DWORD, wintypes.DWORD, wintypes.BOOL)
    AttachThreadInput.restype = wintypes.BOOL

    BringWindowToTop = u32.BringWindowToTop
    BringWindowToTop.argtypes = (wintypes.HWND,)
    BringWindowToTop.restype = wintypes.BOOL

    SetForegroundWindow = u32.SetForegroundWindow
    SetForegroundWindow.argtypes = (wintypes.HWND,)
    SetForegroundWindow.restype = wintypes.BOOL

    SetFocus = u32.SetFocus
    SetFocus.argtypes = (wintypes.HWND,)
    SetFocus.restype = wintypes.HWND

    SetActiveWindow = u32.SetActiveWindow
    SetActiveWindow.argtypes = (wintypes.HWND,)
    SetActiveWindow.restype = wintypes.HWND

    SetWindowPos = u32.SetWindowPos
    SetWindowPos.argtypes = (
        wintypes.HWND, wintypes.HWND,
        wintypes.INT, wintypes.INT, wintypes.INT, wintypes.INT,
        wintypes.UINT
    )
    SetWindowPos.restype = wintypes.BOOL

    # ---------- kernel32 ----------
    GetCurrentThreadId = k32.GetCurrentThreadId
    GetCurrentThreadId.restype = wintypes.DWORD

    SW_RESTORE = 9
    HWND_TOPMOST = wintypes.HWND(-1)
    HWND_NOTOPMOST = wintypes.HWND(-2)
    SWP_NOMOVE = 0x0002
    SWP_NOSIZE = 0x0001
    SWP_SHOWWINDOW = 0x0040

    def _is_foreground():
        return GetForegroundWindow() == hwnd

    for _ in range(max_try):
        # 최소화면 복구
        try:
            win32gui.ShowWindow(hwnd, SW_RESTORE)
        except Exception:
            pass

        # 1) 기본 시도
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass

        if _is_foreground():
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

        # 3) AttachThreadInput 트릭
        fg = GetForegroundWindow()
        fg_pid = wintypes.DWORD(0)
        target_pid = wintypes.DWORD(0)

        fg_tid = GetWindowThreadProcessId(fg, ctypes.byref(fg_pid)) if fg else 0
        target_tid = GetWindowThreadProcessId(hwnd, ctypes.byref(target_pid))
        cur_tid = GetCurrentThreadId()

        try:
            # fg <-> current
            if fg_tid and fg_tid != cur_tid:
                AttachThreadInput(fg_tid, cur_tid, True)

            # target <-> current
            if target_tid and target_tid != cur_tid:
                AttachThreadInput(target_tid, cur_tid, True)

            SetActiveWindow(hwnd)
            SetFocus(hwnd)
            SetForegroundWindow(hwnd)

        finally:
            # detach (실패해도 무시)
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

        if _is_foreground():
            return True

        time.sleep(sleep)

    print("[MENU][ERR] focus failed: cannot bring Touhou window to foreground")
    return False



def _send_input(inp: INPUT) -> bool:
    arr = (INPUT * 1)(inp)
    p = ctypes.byref(arr[0])
    sent = user32.SendInput(1, p, ctypes.sizeof(INPUT))
    return sent == 1


def tap_scancode(scan: int, extended=False, press=0.02, gap=0.03, label=""):
    if label:
        print(f"[MENU] tap_sc {label} sc=0x{scan:02X} ext={extended}")

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
# Menu routines (기존 유지)
# =========================
def enter_practice_from_cursor():
    print("[MENU] FAST practice entry")
    if not focus_touhou_window():
        print("[MENU][ERR] focus failed -> skip key inputs")
        return
    focus_touhou_window()
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
    focus_touhou_window()
    time.sleep(0.05)

    for i in range(3):
        tap_scancode(SC_X, label=f"X{i+1}/10", press=0.02, gap=0.02)
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
    focus_touhou_window()
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
# ✅ NEW: Location detection (improved)
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


def _edge_ratio(gray_roi):
    if gray_roi.size == 0:
        return 0.0
    edges = cv2.Canny(gray_roi, 50, 140)
    return float((edges > 0).mean())


def _menu_highlight_score(img_bgr_roi):
    """
    메뉴에서 '선택된 항목'은 흰색으로 강하게 강조됨.
    -> 평균 밝기(mean)만 보면 흔들릴 수 있어서:
       - mean
       - white_ratio(밝은 픽셀 비율)
       - contrast(roi 평균 - 주변 평균)
    을 같이 쓴다.
    """
    if img_bgr_roi.size == 0:
        return 0.0, 0.0, 0.0

    g = cv2.cvtColor(img_bgr_roi, cv2.COLOR_BGR2GRAY)

    mean = float(g.mean())
    # 밝은 픽셀 비율 (강조 텍스트는 여기서 차이가 크게 남)
    white_ratio = float((g >= 210).mean())  # 200~225 사이에서 튜닝 가능

    # 대비(contrast): roi 내부가 주변보다 얼마나 밝은지
    # roi 전체 평균만 쓰면 배경 밝기에 영향을 받음 -> 대비로 보정
    # 간단히 roi의 상/하 패딩을 포함한 약간 큰 영역의 평균을 빼는 방식 대신,
    # 여기서는 g 자체 std를 보조로 사용
    contrast = float(g.std())  # 선택된 글자는 윤곽+글로우로 std가 올라감

    # 점수 조합 (가중치는 경험적으로 안정적인 값)
    score = (mean * 0.6) + (white_ratio * 300.0) + (contrast * 1.2)
    return score, mean, white_ratio


def _red_dom_score(img_bgr_roi):
    """Quit 빨강 강조 같은 케이스가 있을 때 보조로 쓰는 값."""
    if img_bgr_roi.size == 0:
        return 0.0
    b, g, r = cv2.split(img_bgr_roi)
    r = r.astype(np.float32)
    b = b.astype(np.float32)
    g = g.astype(np.float32)
    red_dom = float(np.mean(np.maximum(0.0, r - np.maximum(b, g))))
    return red_dom


def detect_location(screen):
    """
    return dict:
      state: 'SCORE' | 'IN_GAME' | 'LOBBY' | 'ILLUST' | 'UNKNOWN'
      selected_name: 'PRACTICE' | 'QUIT' | None
      scores: debugging numbers
    """
    img = screen.capture()

    # 1) SCORE (가장 명확)
    try:
        if screen.is_score_screen(img):
            return {"state": "SCORE", "selected_name": None, "scores": {}}
    except Exception:
        pass

    # 2) IN_GAME 힌트(우측 UI 패널)
    try:
        in_game_hint = bool(screen.ui_panel_present(img))
    except Exception:
        in_game_hint = False

    # 3) 로비 메뉴 유무: 오른쪽 메뉴 글자 윤곽(edge) 밀도
    menu_roi = _roi(img, 0.56, 0.30, 0.96, 0.86)
    menu_gray = cv2.cvtColor(menu_roi, cv2.COLOR_BGR2GRAY)
    er = _edge_ratio(menu_gray)

    # 4) PRACTICE / QUIT 선택 판정 (정확도 강화)
    #    - practice: 'Practice Start' 줄 영역
    practice_roi = _roi(img, 0.70, 0.43, 0.95, 0.56)
    practice_score, practice_mean, practice_white = _menu_highlight_score(practice_roi)

    #    - quit: 'Quit' 줄 영역
    quit_roi = _roi(img, 0.72, 0.78, 0.93, 0.90)
    quit_score, quit_mean, quit_white = _menu_highlight_score(quit_roi)

    # 빨강 우세(보조)
    red_dom = _red_dom_score(quit_roi)

    # 5) 상태 결정
    # 로비 메뉴는 글자 윤곽이 많아서 edge ratio가 올라감
    if er >= 0.028:
        state = "LOBBY"
    elif er < 0.020:
        state = "ILLUST"
    else:
        state = "IN_GAME" if in_game_hint else "UNKNOWN"

    selected = None
    if state == "LOBBY":
        # ✅ 핵심: practice_score vs quit_score 비교로 선택 판단
        # 점수차가 충분하면 확정.
        # (이렇게 하면 환경/밝기 변화에 훨씬 강해짐)
        if practice_score > quit_score + 6.0:
            selected = "PRACTICE"
        elif quit_score > practice_score + 6.0:
            selected = "QUIT"
        else:
            # 애매하면 보조 신호로 판단(기준 완화)
            # 기존보다 훨씬 완화된 기준
            if red_dom > 10.0:
                selected = "QUIT"
            elif practice_mean > 75.0:
                selected = "PRACTICE"
            else:
                selected = None

    scores = {
        "menu_edge_ratio": er,
        "practice_score": practice_score,
        "practice_mean": practice_mean,
        "practice_white": practice_white,
        "quit_score": quit_score,
        "quit_mean": quit_mean,
        "quit_white": quit_white,
        "red_dom": red_dom,
    }
    return {"state": state, "selected_name": selected, "scores": scores}


# =========================
# ✅ NEW: Lobby → Practice 커서 정렬
# =========================
def ensure_practice_cursor_from_lobby(screen, verify=True, max_try=3):
    """
    목표: 로비 메뉴 상태에서
      1) X 1번 (Quit로 이동)
      2) UP 5번 (Practice Start로 이동)
    - verify=True면 마지막에 detect_location으로 PRACTICE 추정 확인
    - 일러스트 화면(메뉴 없음)이면 Z 1번 눌러 로비 진입 시도
    """
     # ✅ 반드시 포커스 확보, 실패시 즉시 중단.
    if not focus_touhou_window():
        print("[MENU][BOOT] focus failed -> cannot send keys")
        return False

    for attempt in range(max_try):
        st = detect_location(screen)
        state = st.get("state")

        if state == "ILLUST":
            print("[MENU][BOOT] ILLUST detected -> tap Z to enter lobby...")
            tap_scancode(SC_Z, label="Z(enter lobby)", press=0.02, gap=0.02)
            time.sleep(0.35)
            continue

        if state != "LOBBY":
            print(f"[MENU][BOOT] not in LOBBY (state={state}) -> cannot align cursor")
            return False

        print("[MENU][BOOT] LOBBY detected -> aligning cursor to PRACTICE...")
        # 1) Quit로 이동
        tap_scancode(SC_X, label="X(to Quit)", press=0.02, gap=0.02)
        time.sleep(0.20)

        # 2) UP 5번
        for i in range(5):
            tap_scancode(SC_UP, extended=True, label=f"UP{i+1}/5", press=0.02, gap=0.02)
            time.sleep(0.12)

        time.sleep(0.20)

        if not verify:
            return True

        st2 = detect_location(screen)
        sel = st2.get("selected_name")
        sc = st2.get("scores", {}) or {}
        ps = float(sc.get("practice_score", 0.0))
        qs = float(sc.get("quit_score", 0.0))

        print(f"[MENU][BOOT] verify: state={st2.get('state')} selected={sel} scores={sc}")

        # ✅ 성공 조건(강화):
        # 1) selected가 PRACTICE면 즉시 성공
        # 2) selected가 None이어도 practice_score가 quit_score보다 충분히 크면 성공
        if st2.get("state") == "LOBBY":
            if sel == "PRACTICE":
                return True
            if ps > qs + 6.0:
                print("[MENU][BOOT] verify: PRACTICE inferred by score (no explicit selected)")
                return True


        # verify 실패해도, 로비라는 것만 확실하면 “실행은 가능”하게 두되 재시도 1~2번
        print("[MENU][BOOT] verify failed -> retry alignment")
        time.sleep(0.25)

    return False
