# vision/train.py
import os
import json
import random
import argparse
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from vision.models.center_regressor import CenterRegressor, loss_center_regression


# ✅ 플레이필드 전용
RAW_DIR = os.path.join("vision", "datasets", "raw_playfield")
LABEL_PATH = os.path.join("vision", "datasets", "labels", "labels_playfield.json")
OUT_DIR = "weights"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _safe_load_labels(labels_path: str) -> dict:
    for p in [labels_path, labels_path + ".tmp", labels_path + ".bak"]:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if p != labels_path:
                        print(f"[train] recovered labels from: {p}")
                    return data
            except Exception:
                pass
    raise RuntimeError(
        f"[train] labels json is empty/corrupt.\n"
        f"  - main: {labels_path}\n"
        f"  - also checked: .tmp / .bak\n"
        f"해결: label_tool로 다시 저장 후 재시도."
    )


def read_gray_resized(path: str, out_w: int, out_h: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        raise RuntimeError(f"Failed to read: {path}")
    gray = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return (gray.astype(np.float32) / 255.0)


def aug_gray(gray: np.ndarray) -> np.ndarray:
    # 아주 약하게만 (좌표학습이라 과한 증강 금지)
    a = 1.0 + np.random.uniform(-0.10, 0.10)  # contrast
    b = np.random.uniform(-0.08, 0.08)        # brightness
    x = np.clip(gray * a + b, 0.0, 1.0)
    return x


class ReimuCenterStackDataset(Dataset):
    """
    4-frame grayscale stack: [t, t-1, t-2, t-3] as channels.
    라벨은 t 프레임 기준으로 사용.
    """
    def __init__(self, raw_dir: str, labels_path: str, out_w: int, out_h: int, stack: int = 4, augment: bool = True):
        labels = _safe_load_labels(labels_path)

        # 파일 목록 정렬해서 index 기반 접근 가능하게
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        all_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(exts)]
        all_files.sort()
        if not all_files:
            raise RuntimeError(f"No images found in: {raw_dir}")

        # 파일명 -> 인덱스
        idx_map = {f: i for i, f in enumerate(all_files)}

        items = []
        for fname, v in labels.items():
            if fname not in idx_map:
                continue
            i = idx_map[fname]
            # stack에 필요한 과거 프레임이 있어야 함
            if i - (stack - 1) < 0:
                continue
            x = float(v["x"])
            y = float(v["y"])
            conf = float(v.get("conf", 1))
            items.append((i, fname, x, y, conf))

        if not items:
            raise RuntimeError("No usable labeled items for stacking. (초반 프레임 라벨은 stack 때문에 제외될 수 있음)")

        self.raw_dir = raw_dir
        self.files = all_files
        self.items = items
        self.out_w = out_w
        self.out_h = out_h
        self.stack = stack
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        i, fname, x, y, conf = self.items[idx]

        # t, t-1, t-2, t-3
        frames = []
        for k in range(self.stack):
            fk = self.files[i - k]
            path = os.path.join(self.raw_dir, fk)
            g = read_gray_resized(path, self.out_w, self.out_h)
            frames.append(g)

        # (stack,H,W)
        x_np = np.stack(frames, axis=0)

        if self.augment:
            # (1) 밝기/대비: 채널 공통 적용
            a = 1.0 + np.random.uniform(-0.10, 0.10)
            b = np.random.uniform(-0.08, 0.08)
            x_np = np.clip(x_np * a + b, 0.0, 1.0)

            # (2) ✅ 랜덤 평행이동(라벨도 같이 이동)  <<< 이게 핵심
            max_dx = int(self.out_w * 0.08)  # 폭의 8%
            max_dy = int(self.out_h * 0.08)  # 높이의 8%
            dx = np.random.randint(-max_dx, max_dx + 1)
            dy = np.random.randint(-max_dy, max_dy + 1)

            if dx != 0 or dy != 0:
                M = np.array([[1, 0, dx],
                              [0, 1, dy]], dtype=np.float32)
                # 각 채널에 동일 변환 적용
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

                # 라벨도 같이 이동(정규화)
                x = np.clip(x + dx / float(self.out_w), 0.0, 1.0)
                y = np.clip(y + dy / float(self.out_h), 0.0, 1.0)
                target_xy = torch.tensor([x, y], dtype=torch.float32)


        x_t = torch.from_numpy(x_np).float()              # (C,H,W)
        target_xy = torch.tensor([x, y], dtype=torch.float32)
        target_conf = torch.tensor([conf], dtype=torch.float32)
        return x_t, target_xy, target_conf

    def make_weights_by_grid(self, gx=10, gy=10):
        # 그리드 빈도 역수로 샘플 가중치
        counts = {}
        bins = []
        for (_, _, x, y, conf) in self.items:
            cx = int(np.clip(x * gx, 0, gx - 1))
            cy = int(np.clip(y * gy, 0, gy - 1))
            b = (cx, cy)
            bins.append(b)
            counts[b] = counts.get(b, 0) + 1

        w = [1.0 / float(counts[b]) for b in bins]
        return torch.tensor(w, dtype=torch.float32)


@dataclass
class TrainConfig:
    out_w: int = 160
    out_h: int = 120
    stack: int = 4
    batch: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_ratio: float = 0.15
    seed: int = 42
    lambda_conf: float = 1.0
    num_workers: int = 0
    balanced_sampling: bool = True


def split_indices(n: int, val_ratio: float, seed: int):
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    nv = max(1, int(n * val_ratio))
    val = idxs[:nv]
    tr = idxs[nv:]
    return tr, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=160)
    ap.add_argument("--h", type=int, default=120)
    ap.add_argument("--stack", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda_conf", type=float, default=1.0)
    ap.add_argument("--no_balance", action="store_true", help="disable balanced sampling")
    args = ap.parse_args()

    cfg = TrainConfig(
        out_w=args.w, out_h=args.h, stack=args.stack,
        batch=args.batch, epochs=args.epochs, lr=args.lr,
        val_ratio=args.val_ratio, seed=args.seed, lambda_conf=args.lambda_conf,
        balanced_sampling=(not args.no_balance),
    )

    set_seed(cfg.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[train] device:", device)

    ds_full = ReimuCenterStackDataset(RAW_DIR, LABEL_PATH, cfg.out_w, cfg.out_h, stack=cfg.stack, augment=True)
    tr_idx, va_idx = split_indices(len(ds_full), cfg.val_ratio, cfg.seed)

    # val은 augment=False
    ds_full_noaug = ReimuCenterStackDataset(RAW_DIR, LABEL_PATH, cfg.out_w, cfg.out_h, stack=cfg.stack, augment=False)

    ds_tr = torch.utils.data.Subset(ds_full, tr_idx)
    ds_va = torch.utils.data.Subset(ds_full_noaug, va_idx)

    if cfg.balanced_sampling:
        weights = ds_full.make_weights_by_grid(10, 10)
        w_tr = weights[tr_idx]
        sampler = WeightedRandomSampler(weights=w_tr, num_samples=len(w_tr), replacement=True)
        dl_tr = DataLoader(ds_tr, batch_size=cfg.batch, sampler=sampler, num_workers=cfg.num_workers, drop_last=True)
        print("[train] balanced sampling: ON")
    else:
        dl_tr = DataLoader(ds_tr, batch_size=cfg.batch, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
        print("[train] balanced sampling: OFF")

    dl_va = DataLoader(ds_va, batch_size=cfg.batch, shuffle=False, num_workers=cfg.num_workers)

    model = CenterRegressor(in_ch=cfg.stack, base=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = 1e9
    best_path = os.path.join(OUT_DIR, "reimu_center_stack_best.pt")
    last_path = os.path.join(OUT_DIR, "reimu_center_stack_last.pt")

    for ep in range(1, cfg.epochs + 1):
        model.train()
        tr_loss = tr_coord = tr_conf = 0.0
        ntr = 0

        for x, t_xy, t_c in dl_tr:
            x = x.to(device)          # (B,C,H,W)
            t_xy = t_xy.to(device)    # (B,2)
            t_c = t_c.to(device)      # (B,1)

            pred = model(x)
            loss, l_coord, l_conf = loss_center_regression(pred, t_xy, t_c, lambda_conf=cfg.lambda_conf)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = x.size(0)
            tr_loss += float(loss.detach().cpu()) * bs
            tr_coord += float(l_coord.cpu()) * bs
            tr_conf += float(l_conf.cpu()) * bs
            ntr += bs

        tr_loss /= max(1, ntr)
        tr_coord /= max(1, ntr)
        tr_conf /= max(1, ntr)

        model.eval()
        va_loss = va_coord = va_conf = 0.0
        nva = 0
        with torch.no_grad():
            for x, t_xy, t_c in dl_va:
                x = x.to(device)
                t_xy = t_xy.to(device)
                t_c = t_c.to(device)

                pred = model(x)
                loss, l_coord, l_conf = loss_center_regression(pred, t_xy, t_c, lambda_conf=cfg.lambda_conf)

                bs = x.size(0)
                va_loss += float(loss.cpu()) * bs
                va_coord += float(l_coord.cpu()) * bs
                va_conf += float(l_conf.cpu()) * bs
                nva += bs

        va_loss /= max(1, nva)
        va_coord /= max(1, nva)
        va_conf /= max(1, nva)

        print(f"[ep {ep:03d}] tr: loss={tr_loss:.4f} (coord={tr_coord:.4f}, conf={tr_conf:.4f}) | "
              f"va: loss={va_loss:.4f} (coord={va_coord:.4f}, conf={va_conf:.4f})")

        torch.save({
            "model": model.state_dict(),
            "cfg": {"w": cfg.out_w, "h": cfg.out_h, "in_ch": cfg.stack, "stack": cfg.stack},
        }, last_path)

        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "model": model.state_dict(),
                "cfg": {"w": cfg.out_w, "h": cfg.out_h, "in_ch": cfg.stack, "stack": cfg.stack},
            }, best_path)
            print(f"  -> best saved: {best_path} (val={best_val:.4f})")

    print("[train] done.")
    print(" best:", best_path)
    print(" last:", last_path)


if __name__ == "__main__":
    main()
