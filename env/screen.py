# env/screen.py
import os
import json
import time
import win32gui
import win32con
import mss
import cv2
import numpy as np


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


def _safe_mkdir(p: str):
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass


class Screen:
    # ===== playfield split =====
    PLAYFIELD_RIGHT_RATIO = 0.70

    PLAYFIELD_TOP_CROP = 0.00
    PLAYFIELD_BOTTOM_CROP = 1.00
    PLAYFIELD_LEFT_CROP = 0.00
    PLAYFIELD_RIGHT_CROP = 1.00

    # ===== UI panel heuristics =====
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
        win_rect = win32gui.GetWindowRect(self.hwnd)
        print("[DEBUG] 잡은 창 제목:", title)
        print("[DEBUG] 창 좌표 (left, top, right, bottom):", win_rect)
        self.win_rect = win_rect

        # ===== capture 성능 최적화 =====
        # 1) "윈도우 전체" 대신 "클라이언트 영역"을 캡쳐 (타이틀바/테두리 제외 → 픽셀↓ → capture ms↓)
        # 2) rect를 캐시하고 주기적으로만 갱신
        self._cap_rect = self._get_client_rect_screen()  # (l,t,r,b)
        self._cap_i = 0
        self._cap_refresh_every = 30  # 30프레임마다 갱신

        print("[SCREEN] optimized capture active (client-rect capture + rect cache)")

        # ===== Score screen template (optional) =====
        self.score_tmpl = None
        self.score_roi = None  # (x,y,w,h) in "captured image coords"

        # ===== capture debug dump options =====
        self.debug_dump_on_start = False
        self.debug_dump_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "debug_caps"))
        self.debug_dump_annotated = True

        # ui_config.json 로드
        self._load_ui_config()

        # 템플릿 로드
        self._load_score_template()

        # 시작 시 캡쳐영역 덤프 (옵션)
        if self.debug_dump_on_start:
            self.dump_capture_debug(tag="start")

    # ----------------------------
    # Config / Template
    # ----------------------------
    def _load_ui_config(self):
        try:
            cfg_path = os.path.join(os.path.dirname(__file__), "ui_config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)

                # score_roi: 캡쳐 이미지 기준 좌표
                if "score_roi" in cfg:
                    r = cfg["score_roi"]
                    self.score_roi = (int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))

                # 캡쳐 디버그 덤프 옵션
                # {
                #   "capture_debug_dump_on_start": true,
                #   "capture_debug_dump_dir": "C:/01_touhou_ai/debug_caps",
                #   "capture_debug_dump_annotated": true
                # }
                if "capture_debug_dump_on_start" in cfg:
                    self.debug_dump_on_start = bool(cfg["capture_debug_dump_on_start"])
                if "capture_debug_dump_dir" in cfg and isinstance(cfg["capture_debug_dump_dir"], str):
                    self.debug_dump_dir = os.path.normpath(cfg["capture_debug_dump_dir"])
                if "capture_debug_dump_annotated" in cfg:
                    self.debug_dump_annotated = bool(cfg["capture_debug_dump_annotated"])

        except Exception as e:
            print("[DEBUG] ui_config load failed:", repr(e))

    def _load_score_template(self):
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

    # ----------------------------
    # Rect helpers
    # ----------------------------
    def _get_client_rect_screen(self):
        """
        윈도우 "클라이언트 영역"을 스크린 좌표로 변환해 반환.
        - GetWindowRect 보다 작아서 캡쳐 픽셀 수 감소 → capture ms 감소에 도움
        """
        try:
            # 클라이언트 좌표 (0,0)-(w,h)
            l, t, r, b = win32gui.GetClientRect(self.hwnd)
            # 스크린 좌표로 변환
            (sx0, sy0) = win32gui.ClientToScreen(self.hwnd, (l, t))
            (sx1, sy1) = win32gui.ClientToScreen(self.hwnd, (r, b))
            # 안전장치: 최소 크기 체크
            if sx1 > sx0 and sy1 > sy0:
                return (sx0, sy0, sx1, sy1)
        except Exception:
            pass

        # 실패 시: window rect로 폴백
        try:
            return win32gui.GetWindowRect(self.hwnd)
        except Exception:
            return (0, 0, 640, 480)

    def _get_capture_rect(self):
        # 주기적으로만 갱신
        self._cap_i += 1
        if (self._cap_i % self._cap_refresh_every) == 0:
            try:
                self._cap_rect = self._get_client_rect_screen()
            except Exception:
                pass
        return self._cap_rect

    # ----------------------------
    # Public: capture + debug dump
    # ----------------------------
    def capture(self):
        """
        - mss.grab() -> BGRA (uint8)
        - BGRA[:,:,:3] 는 이미 BGR 이므로 cvtColor(BGRA2BGR) 불필요
        - 다만 슬라이싱 결과가 non-contiguous일 수 있으니,
          ✅ OpenCV 연산에 넘길 가능성이 있으면 contiguous로 복사
        """
        left, top, right, bottom = self._get_capture_rect()
        w = max(1, int(right - left))
        h = max(1, int(bottom - top))

        monitor = {"left": int(left), "top": int(top), "width": w, "height": h}

        img = np.asarray(self.sct.grab(monitor))  # BGRA
        bgr = img[..., :3]  # view

        if not bgr.flags["C_CONTIGUOUS"]:
            bgr = np.ascontiguousarray(bgr)

        return bgr

    def dump_capture_debug(self, tag: str = "manual"):
        """
        현재 캡쳐 영역이 어디인지 확인용:
        - debug_dump_dir/capture_debug_raw_{tag}.png
        - debug_dump_dir/capture_debug_annotated_{tag}.png (playfield/ui/score_roi 표시)
        """
        try:
            _safe_mkdir(self.debug_dump_dir)

            # 최신 rect로 갱신 후 캡쳐
            self._cap_rect = self._get_client_rect_screen()
            img = self.capture()

            ts = time.strftime("%Y%m%d_%H%M%S")
            raw_path = os.path.join(self.debug_dump_dir, f"capture_debug_raw_{tag}_{ts}.png")
            cv2.imwrite(raw_path, img)

            if self.debug_dump_annotated:
                ann = img.copy()
                h, w = ann.shape[:2]

                # playfield 분리선 표시
                x_pf = int(w * self.PLAYFIELD_RIGHT_RATIO)
                cv2.line(ann, (x_pf, 0), (x_pf, h - 1), (0, 255, 255), 2)

                # playfield crop 영역 표시 (현재는 full height 기준)
                # get_playfield_gray에서 실제로 쓰는 crop을 BGR 좌표로 역산해서 표시
                # (주의: get_playfield_gray는 playfield에서 또 crop 적용)
                pw = x_pf
                ph = h
                x1 = int(pw * self.PLAYFIELD_LEFT_CROP)
                x2 = int(pw * self.PLAYFIELD_RIGHT_CROP)
                y1 = int(ph * self.PLAYFIELD_TOP_CROP)
                y2 = int(ph * self.PLAYFIELD_BOTTOM_CROP)
                cv2.rectangle(ann, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)

                # score_roi 표시 (설정되어 있으면)
                if self.score_roi is not None:
                    sx, sy, sw, sh = self.score_roi
                    cv2.rectangle(ann, (sx, sy), (sx + sw - 1, sy + sh - 1), (255, 0, 0), 2)

                # 캡쳐 rect 정보 텍스트
                l, t, r, b = self._cap_rect
                txt = f"CAP_RECT client(screen) L{l} T{t} R{r} B{b} | size={w}x{h}"
                cv2.putText(ann, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                ann_path = os.path.join(self.debug_dump_dir, f"capture_debug_annotated_{tag}_{ts}.png")
                cv2.imwrite(ann_path, ann)

            print(f"[SCREEN][DUMP] saved: {raw_path}")
            if self.debug_dump_annotated:
                print(f"[SCREEN][DUMP] saved: {ann_path}")

        except Exception as e:
            print("[SCREEN][DUMP] failed:", repr(e))

    # ----------------------------
    # Existing APIs (kept compatible)
    # ----------------------------
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
