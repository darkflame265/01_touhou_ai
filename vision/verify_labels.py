import os, json, random, cv2

RAW_DIR = os.path.join("vision", "datasets", "raw_playfield")
LABEL_PATH = os.path.join("vision", "datasets", "labels", "labels_playfield.json")

def main(n=30):
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    keys = list(labels.keys())
    random.shuffle(keys)
    keys = keys[:n]

    cv2.namedWindow("verify", cv2.WINDOW_NORMAL)

    for k in keys:
        p = os.path.join(RAW_DIR, k)
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if len(img.shape) == 2:
            disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            H, W = img.shape[:2]
        else:
            disp = img.copy()
            H, W = disp.shape[:2]

        x = int(labels[k]["x"] * W)
        y = int(labels[k]["y"] * H)

        cv2.drawMarker(disp, (x,y), (0,255,0), cv2.MARKER_CROSS, 22, 2)
        cv2.putText(disp, k, (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("verify", disp)

        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
