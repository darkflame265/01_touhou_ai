# vision/infer_live.py
import os
import time
from collections import deque

import cv2
import numpy as np
import torch

from env.screen import Screen
from vision.models.center_regressor import CenterRegressor


WEIGHT_PATH = os.path.join("weights", "reimu_center_stack_best.pt")
WIN = "reimu_infer_stack"


class EMA:
    def __init__(self, alpha=0.65):
        self.alpha = alpha
        self.v = None

    def update(self, x):
        x = np.array(x, dtype=np.float32)
        if self.v is None:
            self.v = x
        else:
            self.v = self.alpha * x + (1 - self.alpha) * self.v
        return self.v


def load_model(weight_path: str, device):
    ckpt = torch.load(weight_path, map_location=device)
    cfg = ckpt.get("cfg", {"in_ch": 4, "w": 160, "h": 120, "stack": 4})
    in_ch = int(cfg.get("in_ch", cfg.get("stack", 4)))
    model = CenterRegressor(in_ch=in_ch, base=32).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, cfg


def preprocess_playfield_stack(img_bgr, screen: Screen, out_w: int, out_h: int):
    play = screen.get_playfield_gray(img_bgr)  # (H,W)
    ph, pw = play.shape
    small = cv2.resize(play, (out_w, out_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return play, (ph, pw), small  # small: (out_h,out_w)


def main():
    if not os.path.exists(WEIGHT_PATH):
        raise FileNotFoundError(f"weight not found: {WEIGHT_PATH} (먼저 python -m vision.train 실행)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(WEIGHT_PATH, device)
    out_w = int(cfg.get("w", 160))
    out_h = int(cfg.get("h", 120))
    stack = int(cfg.get("stack", cfg.get("in_ch", 4)))

    screen = Screen(mode="high")

    ema = EMA(alpha=0.65)
    conf_thr = 0.35

    buf = deque(maxlen=stack)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    print("[infer] q=quit")
    while True:
        img_bgr = screen.capture()
        play_gray, (ph, pw), small = preprocess_playfield_stack(img_bgr, screen, out_w=out_w, out_h=out_h)

        buf.appendleft(small)  # 최신이 0번: [t, t-1, t-2, t-3]

        if len(buf) < stack:
            disp = cv2.cvtColor(play_gray, cv2.COLOR_GRAY2BGR)
            cv2.putText(disp, f"warming up stack... ({len(buf)}/{stack})",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(disp, f"warming up stack... ({len(buf)}/{stack})",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,0,0), 1, cv2.LINE_AA)
            cv2.imshow(WIN, disp)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            continue

        x_np = np.stack(list(buf), axis=0)  # (C,H,W)
        x = torch.from_numpy(x_np).unsqueeze(0).float().to(device)  # (1,C,H,W)

        with torch.no_grad():
            pred = model(x)[0].detach().cpu().numpy()

        x_n, y_n, conf = float(pred[0]), float(pred[1]), float(pred[2])
        px = int(x_n * pw)
        py = int(y_n * ph)

        sm = ema.update([px, py])
        spx, spy = int(sm[0]), int(sm[1])

        disp = cv2.cvtColor(play_gray, cv2.COLOR_GRAY2BGR)
        txt = f"x={x_n:.3f} y={y_n:.3f} conf={conf:.3f} stack={stack}"
        cv2.putText(disp, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(disp, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)

        if conf >= conf_thr:
            cv2.drawMarker(disp, (spx, spy), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
            cv2.circle(disp, (spx, spy), 10, (0, 255, 0), 2)
        else:
            cv2.putText(disp, "LOW CONF (not updating)", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(disp, "LOW CONF (not updating)", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)

        cv2.imshow(WIN, disp)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break

        time.sleep(0.001)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
