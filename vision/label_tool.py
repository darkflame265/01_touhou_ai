# vision/label_tool.py
import os
import json
import time
import shutil
import cv2
import numpy as np

RAW_DIR = os.path.join("vision", "datasets", "raw_playfield")
LABEL_DIR = os.path.join("vision", "datasets", "labels")
LABEL_PATH = os.path.join(LABEL_DIR, "labels_playfield.json")

WIN_NAME = "label_tool"


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _try_load_json(path: str):
    """깨진/빈 json이면 None 반환."""
    if not os.path.exists(path):
        return None
    try:
        if os.path.getsize(path) == 0:
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_labels(path: str) -> dict:
    """
    1) labels_playfield.json 정상 로드
    2) 깨졌으면 .tmp / .bak에서 복구 시도
    3) 그래도 안되면 {}로 시작하되, 깨진 파일은 .corrupt로 백업
    """
    data = _try_load_json(path)
    if isinstance(data, dict):
        return data

    # tmp 후보들(지난 저장 도중 남았을 수 있음)
    candidates = [
        path + ".tmp",
        path + ".tmp1",
        path + ".tmp2",
        path + ".bak",
    ]
    for c in candidates:
        data = _try_load_json(c)
        if isinstance(data, dict):
            print(f"[label] recovered labels from: {c}")
            return data

    # 여기까지 왔으면 로드 불가 → 기존 파일 백업하고 새로 시작
    if os.path.exists(path):
        ts = time.strftime("%Y%m%d_%H%M%S")
        corrupt = path + f".corrupt_{ts}"
        try:
            shutil.copy2(path, corrupt)
            print(f"[label] labels file was invalid; backed up to: {corrupt}")
        except Exception as e:
            print("[label] failed to backup corrupt labels:", repr(e))

    print("[label] failed to load labels (empty/corrupt). starting fresh {}")
    return {}


def safe_save_json(path: str, data: dict, retries: int = 20, delay: float = 0.05):
    """
    Windows에서 os.replace가 파일 잠금(WinError 5)으로 실패할 수 있어서:
    - tmp에 먼저 쓰고 flush+fsync
    - 기존 파일은 .bak로 백업(가능하면)
    - os.replace를 여러 번 재시도
    - 그래도 실패하면 마지막 수단으로 직접 write (덮어쓰기) 시도
    """
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    bak = path + ".bak"

    # 1) tmp에 안전하게 쓰기
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    # 2) 기존 파일 백업(선택)
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            shutil.copy2(path, bak)
    except Exception:
        pass

    # 3) 원자적 교체 재시도
    last_err = None
    for _ in range(retries):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError as e:
            last_err = e
            time.sleep(delay)
        except Exception as e:
            last_err = e
            time.sleep(delay)

    # 4) 최후 수단: path를 직접 열어 덮어쓰기(잠금이 풀렸으면 성공)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        # tmp 정리
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return True
    except Exception as e:
        print("[label] FATAL: failed to save labels:", repr(last_err), " / ", repr(e))
        return False


def list_frames(folder: str):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    files.sort()
    return files


class LabelTool:
    def __init__(self, show_only_unlabeled=True):
        ensure_dir(LABEL_DIR)

        if not os.path.isdir(RAW_DIR):
            raise FileNotFoundError(f"raw folder not found: {RAW_DIR}")

        all_frames = list_frames(RAW_DIR)
        if not all_frames:
            raise RuntimeError(f"No images found in: {RAW_DIR}")

        self.labels = load_labels(LABEL_PATH)

        if show_only_unlabeled:
            self.frames = [f for f in all_frames if f not in self.labels]
        else:
            self.frames = all_frames

        if not self.frames:
            raise RuntimeError("No unlabeled frames left 🎉")

        self.i = 0
        self.curr_bgr = None
        self.curr_size = None  # (H,W)
        self._needs_reload = True

        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN_NAME, self.on_mouse)

        print(f"[label] total frames: {len(all_frames)}")
        print(f"[label] unlabeled frames: {len(self.frames)}")
        print(f"[label] labels loaded: {len(self.labels)}")

    def _read_frame_bgr(self, fname: str):
        path = os.path.join(RAW_DIR, fname)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None or img.size == 0:
            raise RuntimeError(f"Failed to read image: {path}")

        if len(img.shape) == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            H, W = img.shape[:2]
        else:
            if img.shape[2] == 4:
                bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                bgr = img
            H, W = bgr.shape[:2]
        return bgr, (H, W)

    def load_current(self):
        fname = self.frames[self.i]
        bgr, (H, W) = self._read_frame_bgr(fname)
        self.curr_bgr = bgr
        self.curr_size = (H, W)
        self._needs_reload = False

    def current_fname(self):
        return self.frames[self.i]

    def set_label(self, x_norm: float, y_norm: float):
        fname = self.current_fname()
        self.labels[fname] = {
            "x": float(np.clip(x_norm, 0.0, 1.0)),
            "y": float(np.clip(y_norm, 0.0, 1.0)),
            "conf": 1,
        }

    def delete_label(self):
        fname = self.current_fname()
        if fname in self.labels:
            del self.labels[fname]

    def next_frame(self):
        if self.i < len(self.frames) - 1:
            self.i += 1
            self._needs_reload = True
        else:
            print("[label] all unlabeled frames processed 🎉")
            self.i = len(self.frames) - 1

    def on_mouse(self, event, x, y, flags, param):
        if self.curr_size is None:
            return

        H, W = self.curr_size
        x_norm = x / float(W)
        y_norm = y / float(H)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.set_label(x_norm, y_norm)
            ok = safe_save_json(LABEL_PATH, self.labels)
            if not ok:
                print("[label] save failed (file lock). try again or close editors/antivirus scan.")
            self.next_frame()

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.delete_label()
            safe_save_json(LABEL_PATH, self.labels)
            self._needs_reload = True

    def draw_overlay(self, disp):
        fname = self.current_fname()
        total = len(self.frames)
        msg = f"[{self.i+1}/{total}] {fname} | LClick=label+next | RClick=delete | q=quit"
        cv2.putText(disp, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(disp, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 1, cv2.LINE_AA)

    def run(self):
        print("[label] mode: show ONLY unlabeled frames")
        while True:
            if self._needs_reload or self.curr_bgr is None:
                self.load_current()

            disp = self.curr_bgr.copy()
            self.draw_overlay(disp)
            cv2.imshow(WIN_NAME, disp)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break

        safe_save_json(LABEL_PATH, self.labels)
        print(f"[label] auto-saved on exit ({len(self.labels)} labels)")
        cv2.destroyAllWindows()


def main():
    tool = LabelTool(show_only_unlabeled=True)
    tool.run()


if __name__ == "__main__":
    main()
