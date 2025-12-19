# env/playfield_utils.py
from __future__ import annotations
from typing import Tuple


def get_playfield_rect_safe(screen, img_bgr) -> Tuple[int, int, int, int]:
    """
    플레이필드 경계(= UI 제외) 좌표를 "항상" 얻는다.
    좌표계는 캡처 이미지 좌표계 (0..W, 0..H)
    """
    H, W = img_bgr.shape[:2]

    r = int(W * float(screen.PLAYFIELD_RIGHT_RATIO))
    r = max(1, min(W, r))

    l = 0
    t = 0
    b = H

    # (옵션) Screen에 PLAYFIELD_*_CROP이 있으면 반영
    try:
        l = int(W * float(screen.PLAYFIELD_LEFT_CROP))
        t = int(H * float(screen.PLAYFIELD_TOP_CROP))
        r = int(r * float(screen.PLAYFIELD_RIGHT_CROP))
        b = int(H * float(screen.PLAYFIELD_BOTTOM_CROP))
    except Exception:
        pass

    # 안전 클램프
    l = max(0, min(W - 1, l))
    t = max(0, min(H - 1, t))
    r = max(l + 1, min(W, r))
    b = max(t + 1, min(H, b))

    return (l, t, r, b)


def get_target_point(screen, img_bgr, target_y_ratio: float):
    """
    타겟 포인트(tx,ty) + 플레이필드 rect 반환
    """
    l, t, r, b = get_playfield_rect_safe(screen, img_bgr)
    w = max(1, int(r - l))
    h = max(1, int(b - t))
    tx = int(l + w * 0.5)
    ty = int(t + h * float(target_y_ratio))
    return tx, ty, (l, t, r, b)
