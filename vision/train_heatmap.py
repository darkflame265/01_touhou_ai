# vision/train_heatmap.py
import os
import json
import random
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from vision.models.heatmap_net import HeatmapNet, bce_dice_loss


RAW_DIR = os.path.join("vision", "datasets", "raw_playfield")
LABEL_PATH = os.path.join("vision", "datasets", "labels", "labels_playfield.json")
OUT_DIR = "weights"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_load_labels(path: str) -> dict:
    for p in [path, path + ".tmp", path + ".bak"]:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
    raise RuntimeError(f"labels json is empty/corrupt: {path}")


def list_frames(folder: str):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    files.sort()
    return files


def read_gray(path: str):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None or g.size == 0:
        raise RuntimeError(f"Failed to read: {path}")
    return g


def make_gaussian_heatmap(h: int, w: int, x_px: float, y_px: float, sigma: float):
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-((xx - x_px) ** 2 + (yy - y_px) ** 2) / (2 * sigma * sigma))
    g = g.astype(np.float32)
    g /= (g.max() + 1e-8)
    return g


class HeatmapStackDataset(Dataset):
    """
    Input: 4-frame stack (C,H,W) float in [0,1]
    Target: 1 heatmap (1,H,W) gaussian centered at label (pixel space of resized input)
    """
    def __init__(self, raw_dir, labels_path, out_w=160, out_h=120, stack=4, sigma=2.0, augment=True):
        self.raw_dir = raw_dir
        self.out_w = out_w
        self.out_h = out_h
        self.stack = stack
        self.sigma = sigma
        self.augment = augment

        self.files = list_frames(raw_dir)
        idx_map = {f:i for i,f in enumerate(self.files)}
        labels = safe_load_labels(labels_path)

        items = []
        for fname, v in labels.items():
            if fname not in idx_map:
                continue
            i = idx_map[fname]
            if i - (stack - 1) < 0:
                continue
            x = float(v["x"])
            y = float(v["y"])
            conf = float(v.get("conf", 1))
            if conf < 0.5:
                continue
            items.append((i, x, y))

        if not items:
            raise RuntimeError("No usable labeled items.")

        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        i, x_n, y_n = self.items[idx]

        # stack 만들기
        frames = []
        for k in range(self.stack):
            f = self.files[i - k]
            g = read_gray(os.path.join(self.raw_dir, f))
            g = cv2.resize(g, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            frames.append(g)
        x_np = np.stack(frames, axis=0)  # (C,H,W)

        # 라벨을 resized pixel로 변환
        x_px = x_n * (self.out_w - 1)
        y_px = y_n * (self.out_h - 1)

        # --- augment ---
        if self.augment:
            # 밝기/대비(채널 공통)
            a = 1.0 + np.random.uniform(-0.10, 0.10)
            b = np.random.uniform(-0.08, 0.08)
            x_np = np.clip(x_np * a + b, 0.0, 1.0)

            # ✅ translation + 라벨 동반 이동 (heatmap에서 특히 효과 큼)
            max_dx = int(self.out_w * 0.10)
            max_dy = int(self.out_h * 0.10)
            dx = np.random.randint(-max_dx, max_dx + 1)
            dy = np.random.randint(-max_dy, max_dy + 1)
            if dx != 0 or dy != 0:
                M = np.array([[1, 0, dx],
                              [0, 1, dy]], dtype=np.float32)
                warped = []
                for c in range(x_np.shape[0]):
                    w = cv2.warpAffine(
                        x_np[c], M, (self.out_w, self.out_h),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0.0
                    )
                    warped.append(w)
                x_np = np.stack(warped, axis=0)
                x_px = np.clip(x_px + dx, 0, self.out_w - 1)
                y_px = np.clip(y_px + dy, 0, self.out_h - 1)

        # heatmap 타겟 생성
        hm = make_gaussian_heatmap(self.out_h, self.out_w, x_px, y_px, sigma=self.sigma)  # (H,W)
        hm = hm[None, ...]  # (1,H,W)

        x_t = torch.from_numpy(x_np).float()
        y_t = torch.from_numpy(hm).float()
        return x_t, y_t


def split_indices(n, val_ratio, seed):
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    nv = max(1, int(n * val_ratio))
    return idxs[nv:], idxs[:nv]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=160)
    ap.add_argument("--h", type=int, default=120)
    ap.add_argument("--stack", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[train_heatmap] device:", device)

    ds_full = HeatmapStackDataset(RAW_DIR, LABEL_PATH, out_w=args.w, out_h=args.h,
                                  stack=args.stack, sigma=args.sigma, augment=True)
    tr_idx, va_idx = split_indices(len(ds_full), args.val_ratio, args.seed)

    ds_full_noaug = HeatmapStackDataset(RAW_DIR, LABEL_PATH, out_w=args.w, out_h=args.h,
                                        stack=args.stack, sigma=args.sigma, augment=False)

    ds_tr = torch.utils.data.Subset(ds_full, tr_idx)
    ds_va = torch.utils.data.Subset(ds_full_noaug, va_idx)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0)

    model = HeatmapNet(in_ch=args.stack, base=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = 1e9
    best_path = os.path.join(OUT_DIR, "reimu_heatmap_best.pt")
    last_path = os.path.join(OUT_DIR, "reimu_heatmap_last.pt")

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = tr_bce = tr_dice = 0.0
        ntr = 0

        for x, y in dl_tr:
            x = x.to(device)  # (B,C,H,W)
            y = y.to(device)  # (B,1,H,W)

            logits = model(x)
            loss, l_bce, l_dice = bce_dice_loss(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = x.size(0)
            tr_loss += float(loss.detach().cpu()) * bs
            tr_bce += float(l_bce.cpu()) * bs
            tr_dice += float(l_dice.cpu()) * bs
            ntr += bs

        tr_loss /= max(1, ntr)
        tr_bce  /= max(1, ntr)
        tr_dice /= max(1, ntr)

        model.eval()
        va_loss = va_bce = va_dice = 0.0
        nva = 0
        with torch.no_grad():
            for x, y in dl_va:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss, l_bce, l_dice = bce_dice_loss(logits, y)

                bs = x.size(0)
                va_loss += float(loss.cpu()) * bs
                va_bce  += float(l_bce.cpu()) * bs
                va_dice += float(l_dice.cpu()) * bs
                nva += bs

        va_loss /= max(1, nva)
        va_bce  /= max(1, nva)
        va_dice /= max(1, nva)

        print(f"[ep {ep:03d}] tr: loss={tr_loss:.4f} (bce={tr_bce:.4f}, dice={tr_dice:.4f}) | "
              f"va: loss={va_loss:.4f} (bce={va_bce:.4f}, dice={va_dice:.4f})")

        torch.save({"model": model.state_dict(),
                    "cfg": {"w": args.w, "h": args.h, "stack": args.stack, "sigma": args.sigma}}, last_path)

        if va_loss < best:
            best = va_loss
            torch.save({"model": model.state_dict(),
                        "cfg": {"w": args.w, "h": args.h, "stack": args.stack, "sigma": args.sigma}}, best_path)
            print(f"  -> best saved: {best_path} (val={best:.4f})")

    print("[train_heatmap] done.")
    print(" best:", best_path)
    print(" last:", last_path)


if __name__ == "__main__":
    main()
