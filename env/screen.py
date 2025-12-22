# env/screen.py
import win32gui
import mss
import cv2
import numpy as np
import os
import json


def find_touhou_window():
    target_hwnd = None

    def enum_handler(hwnd, _):
        nonlocal target_hwnd
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if "동방홍마향" in title:
                target_hwnd = hwnd

    win32gui.EnumWindows(enum_handler, None)
    return target_hwnd


class Screen:
    PLAYFIELD_RIGHT_RATIO = 0.70

    PLAYFIELD_TOP_CROP = 0.00
    PLAYFIELD_BOTTOM_CROP = 1.00
    PLAYFIELD_LEFT_CROP = 0.00
    PLAYFIELD_RIGHT_CROP = 1.00

    UI_EDGE_RATIO_THR = 0.040
    UI_STD_MIN = 15.0
    UI_STD_MAX = 80.0
    UI_MEAN_MIN = 20.0
    UI_MEAN_MAX = 200.0

    def __init__(self, mode="low"):
        self.mode = mode
        self.sct = mss.mss()
        self.hwnd = find_touhou_window()

        if not self.hwnd:
            raise Exception("동방홍마향 창을 찾을 수 없음")

        title = win32gui.GetWindowText(self.hwnd)
        rect = win32gui.GetWindowRect(self.hwnd)

        print("[DEBUG] 잡은 창 제목:", title)
        print("[DEBUG] 창 좌표 (left, top, right, bottom):", rect)
        self.win_rect = rect

        # ✅ capture 성능 최적화: rect 캐시 + 주기적으로만 갱신
        self._cap_rect = rect
        self._cap_i = 0
        self._cap_refresh_every = 30  # 30프레임마다 1번만 GetWindowRect

        # ===== Score screen template (optional) =====
        self.score_tmpl = None
        self.score_roi = None  # (x,y,w,h)

        print("[SCREEN] optimized capture active")

        try:
            cfg_path = os.path.join(os.path.dirname(__file__), "ui_config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if "score_roi" in cfg:
                    r = cfg["score_roi"]
                    self.score_roi = (int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
        except Exception as e:
            print("[DEBUG] ui_config score_roi load failed:", repr(e))

        try:
            tmpl_path = os.path.join(os.path.dirname(__file__), "..", "assets", "score_template.png")
            tmpl_path = os.path.normpath(tmpl_path)
            if os.path.exists(tmpl_path):
                g = cv2.imread(tmpl_path, cv2.IMREAD_GRAYSCALE)
                if g is not None and g.size > 0:
                    self.score_tmpl = g
                    print("[DEBUG] score template loaded:", tmpl_path, "shape=", g.shape)
                else:
                    print("[DEBUG] score template exists but failed to read:", tmpl_path)
            else:
                print("[DEBUG] score template not found (optional):", tmpl_path)
        except Exception as e:
            print("[DEBUG] score template load failed:", repr(e))

    def _get_capture_rect(self):
        # 주기적으로만 갱신
        self._cap_i += 1
        if (self._cap_i % self._cap_refresh_every) == 0:
            try:
                self._cap_rect = win32gui.GetWindowRect(self.hwnd)
            except Exception:
                pass
        return self._cap_rect

    def capture(self):
        """
        - mss.grab() -> BGRA (uint8)
        - BGRA[:,:,:3] 는 이미 BGR 이므로 cvtColor(BGRA2BGR) 불필요
        - 다만 슬라이싱 결과가 non-contiguous일 수 있으니,
          ✅ '필요할 때만' contiguous로 복사
        """
        left, top, right, bottom = self._get_capture_rect()
        monitor = {
            "left": left,
            "top": top,
            "width": right - left,
            "height": bottom - top
        }

        img = np.asarray(self.sct.grab(monitor))  # BGRA, dtype=uint8
        bgr = img[..., :3]  # view (대개 non-contiguous 가능)

        # ✅ 조건부 contiguous: 꼭 필요할 때만 복사
        # OpenCV에 넘기거나 numpy 연산에서 예상치 못한 느림/에러 방지
        if not bgr.flags["C_CONTIGUOUS"]:
            bgr = np.ascontiguousarray(bgr)

        return bgr

    def get_playfield_gray(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        x2 = int(w * self.PLAYFIELD_RIGHT_RATIO)
        play = gray[:, :x2]

        ph, pw = play.shape
        x1 = int(pw * self.PLAYFIELD_LEFT_CROP)
        x2 = int(pw * self.PLAYFIELD_RIGHT_CROP)
        y1 = int(ph * self.PLAYFIELD_TOP_CROP)
        y2 = int(ph * self.PLAYFIELD_BOTTOM_CROP)

        play = play[y1:y2, x1:x2]
        return play

    def preprocess(self, img_bgr):
        play = self.get_playfield_gray(img_bgr)

        if self.mode == "low":
            resized = cv2.resize(play, (84, 84), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(play, (160, 120), interpolation=cv2.INTER_AREA)

        return (resized.astype(np.float32) / 255.0)

    def ui_panel_present(self, img_bgr) -> bool:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        x1 = int(w * self.PLAYFIELD_RIGHT_RATIO)
        panel = gray[:, x1:]

        if panel.size == 0:
            return False

        mean = float(panel.mean())
        std = float(panel.std())
        edges = cv2.Canny(panel, 80, 160)
        edge_ratio = float((edges > 0).mean())

        ok = (
            (edge_ratio >= self.UI_EDGE_RATIO_THR) and
            (self.UI_STD_MIN <= std <= self.UI_STD_MAX) and
            (self.UI_MEAN_MIN <= mean <= self.UI_MEAN_MAX)
        )
        return bool(ok)

    def detect_death(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        full_brightness = gray.mean() / 255.0
        gameover = full_brightness > 0.82

        x1 = int(w * 0.35)
        x2 = int(w * 0.65)
        y1 = int(h * 0.60)
        y2 = int(h * 0.95)

        roi = gray[y1:y2, x1:x2]
        bright_ratio = float((roi > 225).mean())
        hit = bright_ratio > 0.020

        return hit, gameover

    def playfield_motion_score(self, prev_play_gray, curr_play_gray):
        diff = cv2.absdiff(prev_play_gray, curr_play_gray)
        return float(diff.mean()) / 255.0

    def danger_from_playfield(self, play_gray, return_parts=False):
        h, w = play_gray.shape

        y1 = int(h * 0.60)
        y2 = int(h * 0.98)
        x1 = int(w * 0.20)
        x2 = int(w * 0.80)

        roi = play_gray[y1:y2, x1:x2]
        if roi.size == 0:
            if return_parts:
                return 0.0, 0.0, 0.0, 0.0
            return 0.0

        edges = cv2.Canny(roi, 50, 120)
        edge_ratio = float((edges > 0).mean())
        bright_ratio = float((roi > 160).mean())
        std_norm = float(roi.std()) / 255.0

        danger = 0.0
        danger += 4.0 * edge_ratio
        danger += 2.0 * bright_ratio
        danger += 1.2 * std_norm
        danger = max(0.0, min(1.0, danger))

        if return_parts:
            return float(danger), float(edge_ratio), float(bright_ratio), float(std_norm)
        return float(danger)

    def _crop_roi(self, gray: np.ndarray, roi):
        if roi is None:
            return gray, 0, 0
        x, y, w, h = roi
        H, W = gray.shape
        x0 = max(0, min(W, x))
        y0 = max(0, min(H, y))
        x1 = max(0, min(W, x0 + w))
        y1 = max(0, min(H, y0 + h))
        return gray[y0:y1, x0:x1], x0, y0

    def is_score_screen(self, img_bgr, thr: float = 0.75) -> bool:
        if self.score_tmpl is None:
            return False

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        src, _, _ = self._crop_roi(gray, self.score_roi)

        th, tw = self.score_tmpl.shape[:2]
        if src.shape[0] < th or src.shape[1] < tw:
            return False

        res = cv2.matchTemplate(src, self.score_tmpl, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, _ = cv2.minMaxLoc(res)
        return bool(maxv >= float(thr))
