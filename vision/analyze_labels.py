import os, json, math
import numpy as np

LABEL_PATH = os.path.join("vision", "datasets", "labels", "labels_playfield.json")

def main():
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    xs, ys = [], []
    for k, v in labels.items():
        if float(v.get("conf", 1)) < 0.5:
            continue
        xs.append(float(v["x"]))
        ys.append(float(v["y"]))

    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    mx, my = float(xs.mean()), float(ys.mean())
    sx, sy = float(xs.std()), float(ys.std())

    # “항상 평균 찍기” 베이스라인의 평균 오차(정규화 거리)
    d = np.sqrt((xs - mx) ** 2 + (ys - my) ** 2)
    baseline_rmse = float(np.sqrt(np.mean(d**2)))
    baseline_mae = float(np.mean(d))

    # 10x10 그리드 점유율로 다양성 대충 보기
    gx = np.clip((xs * 10).astype(int), 0, 9)
    gy = np.clip((ys * 10).astype(int), 0, 9)
    occ = np.zeros((10, 10), dtype=np.int32)
    for a, b in zip(gx, gy):
        occ[b, a] += 1

    top_cells = sorted([(int(occ[r, c]), r, c) for r in range(10) for c in range(10)], reverse=True)[:5]

    print(f"[labels] N={len(xs)}")
    print(f"[labels] mean(x,y)=({mx:.3f}, {my:.3f})  std(x,y)=({sx:.3f}, {sy:.3f})")
    print(f"[baseline] always-mean  RMSE={baseline_rmse:.4f}  MAE={baseline_mae:.4f}")
    print("[grid] top occupied cells (count, row(y), col(x)):", top_cells)

    # 경험적으로: std가 너무 작으면(예: sx<0.08, sy<0.06) 평균수렴 위험 매우 큼
    if sx < 0.08 and sy < 0.06:
        print("[warn] position diversity looks LOW -> mean-collapse likely")
    else:
        print("[ok] position diversity seems non-trivial (still can be improved)")

if __name__ == "__main__":
    main()
