# env/screen.py
import os
import json
import time
import win32gui
import cv2
import numpy as np

# dxcam은 선택적으로 import (없으면 mss fallback)
try:
    import dxcam
    _HAS_DXCAM = True
except Exception:
    _HAS_DXCAM = False

try:
    import mss
    _HAS_MSS = True
except Exception:
    _HAS_MSS = False


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
    PLAYFIELD_RIGHT_RATIO = 0.67

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
        self.hwnd = find_touhou_window()

        if not self.hwnd:
            raise Exception("동방홍마향 창을 찾을 수 없음")

        title = win32gui.GetWindowText(self.hwnd)
        win_rect = win32gui.GetWindowRect(self.hwnd)
        print("[DEBUG] 잡은 창 제목:", title)
        print("[DEBUG] 창 좌표 (left, top, right, bottom):", win_rect)
        self.win_rect = win_rect

        # ===== capture rect cache (client rect in screen coords) =====
        self._cap_rect = self._get_client_rect_screen()  # (l,t,r,b)
        self._cap_i = 0
        self._cap_refresh_every = 30

        # ===== gray cache (per-frame) =====
        self._gray_cache_src_id = None
        self._gray_cache = None

        # ===== Score screen template (optional) =====
        self.score_tmpl = None
        self.score_roi = None  # (x,y,w,h) in "captured image coords"

        # ===== capture debug dump options =====
        self.debug_dump_on_start = False
        self.debug_dump_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "debug_caps"))
        self.debug_dump_annotated = True

        # ui_config / template
        self._load_ui_config()
        self._load_score_template()

        # ===== Capture backend =====
        self._use_dxcam = False
        self._dx = None
        self._dx_region = None
        self._dx_target_fps = 60  # 필요하면 60~240 사이로 조절

        self._mss = None

        self._init_capture_backend()

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

                if "score_roi" in cfg:
                    r = cfg["score_roi"]
                    self.score_roi = (int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))

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
        """
        try:
            l, t, r, b = win32gui.GetClientRect(self.hwnd)
            (sx0, sy0) = win32gui.ClientToScreen(self.hwnd, (l, t))
            (sx1, sy1) = win32gui.ClientToScreen(self.hwnd, (r, b))
            if sx1 > sx0 and sy1 > sy0:
                return (int(sx0), int(sy0), int(sx1), int(sy1))
        except Exception:
            pass

        try:
            L, T, R, B = win32gui.GetWindowRect(self.hwnd)
            return (int(L), int(T), int(R), int(B))
        except Exception:
            return (0, 0, 640, 480)

    def _get_capture_rect(self):
        self._cap_i += 1
        if (self._cap_i % self._cap_refresh_every) == 0:
            try:
                self._cap_rect = self._get_client_rect_screen()
            except Exception:
                pass
        return self._cap_rect

    # ----------------------------
    # Capture backend init / region sync
    # ----------------------------
    def _init_capture_backend(self):
        # dxcam 우선
        if _HAS_DXCAM:
            try:
                # output_color="BGR"로 바로 받으면 변환 비용 최소
                self._dx = dxcam.create(output_color="BGR")
                if self._dx is not None:
                    self._use_dxcam = True
                    self._dx_region = self._cap_rect  # (l,t,r,b)
                    # start() + get_latest_frame() 방식
                    self._dx.start(region=self._dx_region, target_fps=self._dx_target_fps)
                    print("[SCREEN] capture backend = dxcam (BGR) | region=", self._dx_region)
                    return
            except Exception as e:
                print("[SCREEN] dxcam init failed -> fallback:", repr(e))
                self._use_dxcam = False
                self._dx = None

        # fallback: mss
        if _HAS_MSS:
            try:
                self._mss = mss.mss()
                print("[SCREEN] capture backend = mss (fallback)")
                return
            except Exception as e:
                print("[SCREEN] mss init failed:", repr(e))

        raise Exception("capture backend init failed: dxcam/mss 모두 사용 불가")

    def _ensure_dxcam_region(self, rect):
        """
        client rect가 변하면 dxcam region을 재시작해서 맞춰준다.
        (창 이동/크기변경 대응)
        """
        if not self._use_dxcam or self._dx is None:
            return
        rect = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
        if self._dx_region == rect:
            return
        try:
            self._dx.stop()
        except Exception:
            pass
        try:
            self._dx_region = rect
            self._dx.start(region=self._dx_region, target_fps=self._dx_target_fps)
            # print("[SCREEN] dxcam region updated:", self._dx_region)
        except Exception as e:
            print("[SCREEN] dxcam region update failed:", repr(e))

    # ----------------------------
    # Per-frame gray cache
    # ----------------------------
    def gray(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        같은 img_bgr 객체에 대해 cvtColor를 1번만 수행하도록 캐시.
        """
        if img_bgr is None:
            return None
        src_id = id(img_bgr)
        if self._gray_cache_src_id == src_id and self._gray_cache is not None:
            return self._gray_cache
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        self._gray_cache_src_id = src_id
        self._gray_cache = g
        return g

    # ----------------------------
    # Public: capture + debug dump
    # ----------------------------
    def capture(self):
        """
        return: BGR uint8 (H,W,3)
        """
        left, top, right, bottom = self._get_capture_rect()
        w = max(1, int(right - left))
        h = max(1, int(bottom - top))

        # 프레임 바뀌면 gray 캐시도 초기화
        self._gray_cache_src_id = None
        self._gray_cache = None

        if self._use_dxcam and self._dx is not None:
            # 창 이동/리사이즈 대응
            self._ensure_dxcam_region((left, top, right, bottom))

            frame = None
            try:
                frame = self._dx.get_latest_frame()
            except Exception:
                frame = None

            # get_latest_frame()이 None이면, 아주 짧게 1회 재시도
            if frame is None:
                time.sleep(0.001)
                try:
                    frame = self._dx.get_latest_frame()
                except Exception:
                    frame = None

            if frame is None:
                # 안전 폴백: 빈 프레임
                return np.zeros((h, w, 3), dtype=np.uint8)

            # dxcam output_color="BGR"이면 이미 BGR
            bgr = frame
            if bgr.ndim != 3 or bgr.shape[2] != 3:
                # 혹시 모르는 포맷 방어
                bgr = np.asarray(bgr, dtype=np.uint8)
                if bgr.ndim == 2:
                    bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
                elif bgr.ndim == 3 and bgr.shape[2] == 4:
                    bgr = bgr[:, :, :3]

            if not bgr.flags["C_CONTIGUOUS"]:
                bgr = np.ascontiguousarray(bgr)
            return bgr

        # ---- fallback: mss ----
        if self._mss is None:
            # mss가 init 실패했는데 여기까지 오면 예외 케이스
            return np.zeros((h, w, 3), dtype=np.uint8)

        monitor = {"left": int(left), "top": int(top), "width": int(w), "height": int(h)}
        img = np.asarray(self._mss.grab(monitor))  # BGRA
        bgr = img[..., :3]
        if not bgr.flags["C_CONTIGUOUS"]:
            bgr = np.ascontiguousarray(bgr)
        return bgr

    def dump_capture_debug(self, tag: str = "manual"):
        """
        현재 캡쳐 영역이 어디인지 확인용 덤프.
        - raw / annotated 저장
        """
        try:
            _safe_mkdir(self.debug_dump_dir)

            self._cap_rect = self._get_client_rect_screen()
            img = self.capture()

            ts = time.strftime("%Y%m%d_%H%M%S")
            raw_path = os.path.join(self.debug_dump_dir, f"capture_debug_raw_{tag}_{ts}.png")
            cv2.imwrite(raw_path, img)

            ann_path = None
            if self.debug_dump_annotated:
                ann = img.copy()
                H, W = ann.shape[:2]

                x_pf = int(W * self.PLAYFIELD_RIGHT_RATIO)
                cv2.line(ann, (x_pf, 0), (x_pf, H - 1), (0, 255, 255), 2)

                pw = x_pf
                ph = H
                x1 = int(pw * self.PLAYFIELD_LEFT_CROP)
                x2 = int(pw * self.PLAYFIELD_RIGHT_CROP)
                y1 = int(ph * self.PLAYFIELD_TOP_CROP)
                y2 = int(ph * self.PLAYFIELD_BOTTOM_CROP)
                cv2.rectangle(ann, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)

                if self.score_roi is not None:
                    sx, sy, sw, sh = self.score_roi
                    cv2.rectangle(ann, (sx, sy), (sx + sw - 1, sy + sh - 1), (255, 0, 0), 2)

                l, t, r, b = self._cap_rect
                txt = f"CAP_RECT client(screen) L{l} T{t} R{r} B{b} | size={W}x{H}"
                cv2.putText(ann, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                ann_path = os.path.join(self.debug_dump_dir, f"capture_debug_annotated_{tag}_{ts}.png")
                cv2.imwrite(ann_path, ann)

            print(f"[SCREEN][DUMP] saved: {raw_path}")
            if ann_path is not None:
                print(f"[SCREEN][DUMP] saved: {ann_path}")

        except Exception as e:
            print("[SCREEN][DUMP] failed:", repr(e))

    # ----------------------------
    # Existing APIs (gray 재사용 지원)
    # ----------------------------
    def get_playfield_gray(self, img_bgr, gray=None):
        if gray is None:
            gray = self.gray(img_bgr)
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

    def preprocess(self, img_bgr, gray=None):
        play = self.get_playfield_gray(img_bgr, gray=gray)

        if self.mode == "low":
            resized = cv2.resize(play, (84, 84), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(play, (160, 120), interpolation=cv2.INTER_AREA)

        return (resized.astype(np.float32) / 255.0)

    def ui_panel_present(self, img_bgr, gray=None) -> bool:
        if gray is None:
            gray = self.gray(img_bgr)
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

    def detect_death(self, img_bgr, gray=None):
        if gray is None:
            gray = self.gray(img_bgr)
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

    def is_score_screen(self, img_bgr, thr: float = 0.75, gray=None) -> bool:
        if self.score_tmpl is None:
            return False

        if gray is None:
            gray = self.gray(img_bgr)

        src, _, _ = self._crop_roi(gray, self.score_roi)

        th, tw = self.score_tmpl.shape[:2]
        if src.shape[0] < th or src.shape[1] < tw:
            return False

        res = cv2.matchTemplate(src, self.score_tmpl, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, _ = cv2.minMaxLoc(res)
        return bool(maxv >= float(thr))
