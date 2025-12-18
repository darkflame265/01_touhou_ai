# env/player_tracker.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class TrackResult:
    x: int
    y: int
    conf: float
    found: bool

class PlayerTracker:
    """
    색 기반 후보 -> 컨투어 -> last_pos 기반 스코어링 -> EMA 스무딩
    """
    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        init_xy: tuple[int, int] | None = None,
        ema_alpha: float = 0.35,
        base_search_radius: int = 200,
    ):
        self.w = frame_w
        self.h = frame_h
        self.ema_alpha = ema_alpha
        self.base_r = base_search_radius

        if init_xy is None:
            init_xy = (frame_w // 2, int(frame_h * 0.75))

        self.last_x, self.last_y = init_xy
        self.smooth_x, self.smooth_y = float(self.last_x), float(self.last_y)
        self.miss_count = 0

    def _roi_from_last(self, frame: np.ndarray) -> tuple[np.ndarray, int, int]:
        # miss가 늘수록 탐색 반경 확대
        r = int(self.base_r * (1.0 + min(self.miss_count, 10) * 0.18))
        x0 = max(0, self.last_x - r)
        y0 = max(0, self.last_y - r)
        x1 = min(self.w, self.last_x + r)
        y1 = min(self.h, self.last_y + r)
        return frame[y0:y1, x0:x1], x0, y0

    def _player_mask(self, bgr: np.ndarray) -> np.ndarray:
        """
        여기서 핵심은 '게임마다 플레이어 색이 거의 고정'이라는 점을 이용하는 것.
        기본값은 "밝은(흰색/연한색) + 채도 낮음" 위주.
        필요하면 아래 threshold를 너 게임(동방) 스프라이트에 맞게 조정하면 됨.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # (1) 밝고 채도 낮은 영역: 흰색/연한색 스프라이트 계열
        lower1 = np.array([0, 0, 190], dtype=np.uint8)
        upper1 = np.array([180, 80, 255], dtype=np.uint8)
        m1 = cv2.inRange(hsv, lower1, upper1)

        # (2) 푸른기/청록기 플레이어라면 도움이 되는 보조 마스크(필요 없으면 지워도 됨)
        lower2 = np.array([80, 40, 120], dtype=np.uint8)
        upper2 = np.array([120, 255, 255], dtype=np.uint8)
        m2 = cv2.inRange(hsv, lower2, upper2)

        mask = cv2.bitwise_or(m1, m2)

        # 노이즈 제거(탄막 점/아이템 반짝임 등 제거)
        mask = cv2.medianBlur(mask, 5)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
        return mask

    def _score_candidate(self, cx: int, cy: int, area: float, bbox_wh: tuple[int, int]) -> float:
        # last_pos 가까울수록 +, 너무 큰/작은 덩어리면 -
        dx = cx - self.last_x
        dy = cy - self.last_y
        dist = (dx*dx + dy*dy) ** 0.5

        w, h = bbox_wh
        aspect = w / max(1, h)

        # 경험적으로 플레이어는 "아주 작은 점"도 아니고 "큰 폭발 이펙트"도 아님
        # (너 캡쳐 해상도에 맞춰 area 범위만 좀 튜닝하면 정확도가 크게 튐)
        area_ok = 1.0
        if area < 15:
            area_ok = 0.2
        elif area > 1200:
            area_ok = 0.1

        # 세로로 긴/가로로 넓은 이상한 덩어리도 감점
        aspect_ok = 1.0
        if aspect < 0.35 or aspect > 2.8:
            aspect_ok = 0.5

        # 거리가 멀면 감점 (miss_count 높을 때는 반경 커져도, 그래도 가까운 걸 선호)
        dist_score = np.exp(-dist / (120.0 + 30.0 * min(self.miss_count, 10)))

        return float(dist_score * area_ok * aspect_ok)

    def update(self, frame_bgr: np.ndarray) -> TrackResult:
        # 1) ROI만 잘라서 (연산량↓, 오탐↓)
        roi, ox, oy = self._roi_from_last(frame_bgr)

        # 2) 마스크 생성
        mask = self._player_mask(roi)

        # 3) 후보 컨투어
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = 0.0

        for c in cnts:
            area = cv2.contourArea(c)
            if area <= 5:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx = ox + x + w // 2
            cy = oy + y + h // 2

            score = self._score_candidate(cx, cy, area, (w, h))
            if score > best_score:
                best_score = score
                best = (cx, cy)

        # 4) 결과 확정 + EMA 스무딩
        if best is not None and best_score >= 0.12:
            self.miss_count = 0
            self.last_x, self.last_y = int(best[0]), int(best[1])

            a = self.ema_alpha
            self.smooth_x = (1 - a) * self.smooth_x + a * self.last_x
            self.smooth_y = (1 - a) * self.smooth_y + a * self.last_y

            return TrackResult(int(self.smooth_x), int(self.smooth_y), float(best_score), True)

        # 못 찾으면 miss 증가, 위치는 유지
        self.miss_count += 1
        return TrackResult(int(self.smooth_x), int(self.smooth_y), float(best_score), False)
