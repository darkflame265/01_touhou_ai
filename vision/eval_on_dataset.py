import os, json, random
import cv2
import numpy as np
import torch
from collections import deque

from vision.models.center_regressor import CenterRegressor

RAW_DIR = os.path.join("vision", "datasets", "raw_playfield")
LABEL_PATH = os.path.join("vision", "datasets", "labels", "labels_playfield.json")
WEIGHT_PATH = os.path.join("weights", "reimu_center_stack_best.pt")

OUT_W, OUT_H = 160, 120
STACK = 4

def read_small(fname):
    p = os.path.join(RAW_DIR, fname)
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None, None
    H, W = g.shape[:2]
    small = cv2.resize(g, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return g, small

def main(n=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(WEIGHT_PATH, map_location=device)
    cfg = ckpt.get("cfg", {})
    stack = int(cfg.get("stack", cfg.get("in_ch", STACK)))

    model = CenterRegressor(in_ch=stack, base=32).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    files = sorted([f for f in os.listdir(RAW_DIR) if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".webp"))])
    idx_map = {f:i for i,f in enumerate(files)}

    keys = [k for k in labels.keys() if k in idx_map and idx_map[k] - (stack-1) >= 0]
    random.shuffle(keys)
    keys = keys[:n]

    cv2.namedWindow("eval", cv2.WINDOW_NORMAL)

    for k in keys:
        i = idx_map[k]

        # stack 만들기: [t, t-1, ...]
        st = []
        base_gray = None
        H = W = None
        for d in range(stack):
            g, small = read_small(files[i - d])
            if g is None:
                st = None
                break
            if d == 0:
                base_gray = g
                H, W = g.shape[:2]
            st.append(small)
        if st is None:
            continue

        x = np.stack(st, axis=0)  # (C,H,W) in small space
        x_t = torch.from_numpy(x).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred = model(x_t)[0].detach().cpu().numpy()
        x_n, y_n, conf = float(pred[0]), float(pred[1]), float(pred[2])

        # pred 픽셀(원본 플레이필드 기준)
        px = int(x_n * W)
        py = int(y_n * H)

        # GT 픽셀
        gt = labels[k]
        gx = int(float(gt["x"]) * W)
        gy = int(float(gt["y"]) * H)

        disp = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(disp, (gx, gy), (255, 0, 0), cv2.MARKER_CROSS, 22, 2)   # GT: 파랑
        cv2.drawMarker(disp, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)   # Pred: 초록

        txt = f"{k}  pred=({x_n:.3f},{y_n:.3f}) conf={conf:.3f}  gt=({gt['x']:.3f},{gt['y']:.3f})"
        cv2.putText(disp, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(disp, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

        cv2.imshow("eval", disp)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
