# vision/collect_frames.py
import os
import time
import cv2
import numpy as np

from env.screen import Screen


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def next_index(folder: str, prefix="frame_", ext=".png") -> int:
    mx = 0
    for name in os.listdir(folder):
        if not (name.startswith(prefix) and name.endswith(ext)):
            continue
        mid = name[len(prefix):-len(ext)]
        if mid.isdigit():
            mx = max(mx, int(mid))
    return mx + 1


def main():
    # ✅ 플레이필드 전용 폴더(새로)
    out_dir = os.path.join("vision", "datasets", "raw_playfield")
    ensure_dir(out_dir)

    screen = Screen(mode="high")

    target_fps = 15      # 10~20 추천
    every_n = 1          # 1이면 매 프레임 저장

    idx = next_index(out_dir)
    i = 0
    last = time.time()

    print("[collect] PLAYFIELD-ONLY mode")
    print("[collect] keys: q=quit, space=pause/resume")
    paused = False

    cv2.namedWindow("collect_preview", cv2.WINDOW_NORMAL)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" "):
            paused = not paused
            print("[collect] paused =", paused)

        if paused:
            time.sleep(0.05)
            continue

        img_bgr = screen.capture()

        # ✅ 플레이필드만 추출 (gray)
        play_gray = screen.get_playfield_gray(img_bgr)  # (H,W) gray

        # 미리보기
        cv2.imshow("collect_preview", play_gray)

        i += 1
        if i % every_n == 0:
            name = f"frame_{idx:06d}.png"
            path = os.path.join(out_dir, name)
            ok = cv2.imwrite(path, play_gray)
            if ok:
                idx += 1
                if idx % 200 == 0:
                    print(f"[collect] saved: {idx-1} frames")
            else:
                print("[collect] imwrite failed:", path)

        # FPS 맞추기
        dt = time.time() - last
        wait = (1.0 / target_fps) - dt
        if wait > 0:
            time.sleep(wait)
        last = time.time()

    cv2.destroyAllWindows()
    print("[collect] done.")


if __name__ == "__main__":
    main()
