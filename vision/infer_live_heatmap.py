# vision/infer_live_heatmap.py
import os
import time
from collections import deque

import cv2
import numpy as np
import torch

from env.screen import Screen
from vision.models.heatmap_net import HeatmapNet, soft_argmax_2d


WEIGHT_PATH = os.path.join("weights", "reimu_heatmap_best.pt")
WIN = "reimu_infer_heatmap"

# ✅ 아래쪽 선호 프라이어 강도 (0.3~2.0에서 튜닝)
PRIOR_STRENGTH = 1.0

# ✅ soft-argmax 뾰족함(8~20 사이 튜닝)
SOFTARG_BETA = 12.0


class EMA:
    def __init__(self, alpha=0.75):
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
    cfg = ckpt.get("cfg", {"w": 160, "h": 120, "stack": 4, "sigma": 2.0})
    model = HeatmapNet(in_ch=int(cfg.get("stack", 4)), base=32).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, cfg


def preprocess(img_bgr, screen: Screen, out_w: int, out_h: int):
    play = screen.get_playfield_gray(img_bgr)
    ph, pw = play.shape
    small = cv2.resize(play, (out_w, out_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return play, (ph, pw), small


def apply_bottom_prior(logits: torch.Tensor, strength: float) -> torch.Tensor:
    """
    logits: (B,1,H,W)
    strength: penalty scale for upper region.
      - y=0 (top): subtract ~strength
      - y=1 (bottom): subtract ~0
    """
    if strength <= 0:
        return logits
    H, W = logits.shape[-2], logits.shape[-1]
    yy = torch.linspace(0.0, 1.0, H, device=logits.device, dtype=logits.dtype).view(1, 1, H, 1)
    penalty = (1.0 - yy)  # top=1, bottom=0
    return logits - strength * penalty


def main():
    if not os.path.exists(WEIGHT_PATH):
        raise FileNotFoundError(f"weight not found: {WEIGHT_PATH} (먼저 python -m vision.train_heatmap 실행)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(WEIGHT_PATH, device)
    out_w = int(cfg.get("w", 160))
    out_h = int(cfg.get("h", 120))
    stack = int(cfg.get("stack", 4))

    screen = Screen(mode="high")
    buf = deque(maxlen=stack)
    ema = EMA(alpha=0.75)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    print("[infer_heatmap] q=quit")
    print(f"[infer_heatmap] using weights: {os.path.abspath(WEIGHT_PATH)}")
    print(f"[infer_heatmap] PRIOR_STRENGTH={PRIOR_STRENGTH}  SOFTARG_BETA={SOFTARG_BETA}  stack={stack}")

    while True:
        img_bgr = screen.capture()
        play_gray, (ph, pw), small = preprocess(img_bgr, screen, out_w, out_h)
        buf.appendleft(small)

        disp = cv2.cvtColor(play_gray, cv2.COLOR_GRAY2BGR)

        if len(buf) < stack:
            cv2.putText(disp, f"warming up stack... ({len(buf)}/{stack})",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(WIN, disp)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            continue

        x_np = np.stack(list(buf), axis=0)  # (C,H,W)
        x = torch.from_numpy(x_np).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits = model(x)  # (1,1,H,W) logits
            # ✅ 아래쪽 선호 프라이어 적용
            logits = apply_bottom_prior(logits, PRIOR_STRENGTH)

            xy, conf = soft_argmax_2d(logits, beta=SOFTARG_BETA)
            x_n = float(xy[0, 0].cpu())
            y_n = float(xy[0, 1].cpu())
            c = float(conf[0, 0].cpu())

        px = int(x_n * pw)
        py = int(y_n * ph)

        sm = ema.update([px, py])
        spx, spy = int(sm[0]), int(sm[1])

        txt = f"x={x_n:.3f} y={y_n:.3f} conf={c:.4f} stack={stack} prior={PRIOR_STRENGTH}"
        cv2.putText(disp, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.drawMarker(disp, (spx, spy), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
        cv2.circle(disp, (spx, spy), 10, (0, 255, 0), 2)

        # heatmap preview (sigmoid after prior)
        hm = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        hm_u8 = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
        hm_show = cv2.resize(hm_u8, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("heatmap_160x120", hm_show)

        cv2.imshow(WIN, disp)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break

        time.sleep(0.001)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
