# env/det_track_tracker.py
from __future__ import annotations
from dataclasses import dataclass
import time
import cv2
import numpy as np

from env.template_tracker import MultiTemplateTracker


@dataclass
class TrackResult:
    x: int
    y: int
    conf: float
    found: bool
    method: str  # "track" or "detect" or "hold"


def _create_cv_tracker(prefer: str = "CSRT"):
    name = prefer.upper()
    legacy = getattr(cv2, "legacy", None)

    def _try(fn):
        try:
            return fn()
        except Exception:
            return None

    if name == "CSRT":
        if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
            t = _try(legacy.TrackerCSRT_create)
            if t is not None:
                return t
        if hasattr(cv2, "TrackerCSRT_create"):
            t = _try(cv2.TrackerCSRT_create)
            if t is not None:
                return t

    if name == "MOSSE":
        if legacy is not None and hasattr(legacy, "TrackerMOSSE_create"):
            t = _try(legacy.TrackerMOSSE_create)
            if t is not None:
                return t
        if hasattr(cv2, "TrackerMOSSE_create"):
            t = _try(cv2.TrackerMOSSE_create)
            if t is not None:
                return t

    if legacy is not None and hasattr(legacy, "TrackerKCF_create"):
        t = _try(legacy.TrackerKCF_create)
        if t is not None:
            return t
    if hasattr(cv2, "TrackerKCF_create"):
        t = _try(cv2.TrackerKCF_create)
        if t is not None:
            return t

    raise RuntimeError("No supported OpenCV tracker found (CSRT/MOSSE/KCF).")


class DetTrackTracker:
    """
    Detector(템플릿) + Tracker(OpenCV) 하이브리드

    추가(중요):
    - on_player_death() 이후 death_cooldown_sec 동안은 "hold" 상태로 추적 정지
    - 쿨다운이 끝나면 N프레임 동안 detector를 강제(갈아타기 방지 + 재획득 안정화)
    """

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        template_paths: list[str],
        init_xy: tuple[int, int] | None = None,

        tracker_prefer: str = "CSRT",
        init_box: int = 56,
        track_fail_to_detect: int = 2,
        redetect_every: int = 15,

        lost_fail_to_respawn: int = 8,
        respawn_xy: tuple[int, int] | None = None,

        # ===== NEW: death 직후 "추적 정지" + "재탐지 강화" =====
        death_cooldown_sec: float = 1.2,        # 1.0~2.0 권장
        post_death_force_detect_frames: int = 10,  # 쿨다운 직후 N프레임은 detect 강제
        post_death_boost_miss: int = 10,        # detector ROI 반경 키우기 위한 miss_count 부스팅(간접)

        **detector_kwargs,
    ):
        self.w = frame_w
        self.h = frame_h

        self.init_box = int(init_box)
        self.track_fail_to_detect = int(track_fail_to_detect)
        self.redetect_every = int(redetect_every)
        self.lost_fail_to_respawn = int(lost_fail_to_respawn)

        # NEW
        self.death_cooldown_sec = float(death_cooldown_sec)
        self.post_death_force_detect_frames = int(post_death_force_detect_frames)
        self.post_death_boost_miss = int(post_death_boost_miss)

        self._hold_until = 0.0
        self._post_death_frames_left = 0

        self.detector = MultiTemplateTracker(
            frame_w=frame_w,
            frame_h=frame_h,
            template_paths=template_paths,
            init_xy=init_xy,
            **detector_kwargs,
        )

        if init_xy is None:
            init_xy = (frame_w // 2, int(frame_h * 0.78))
        self.last_x, self.last_y = int(init_xy[0]), int(init_xy[1])

        if respawn_xy is None:
            respawn_xy = (frame_w // 2, int(frame_h * 0.78))
        self.respawn_xy = (int(respawn_xy[0]), int(respawn_xy[1]))

        self._tracker_prefer = str(tracker_prefer)
        self._cv_tracker = None
        self._track_ok = False
        self._track_fail_streak = 0
        self._detect_fail_streak = 0
        self._frame_i = 0

        # Debug passthrough
        self.miss_count = 0
        self.last_match_box = None

        self.dbg_best = None
        self.dbg_second = None
        self.dbg_margin = None
        self.dbg_red = None
        self.dbg_white = None
        self.dbg_reject = None
        self.dbg_candidate_center = None
        self.dbg_confirm = None
        self.dbg_votes = None
        self.dbg_similarity = None
        self.dbg_similarity_pct = None
        self.dbg_ignore_hit = None

    # -----------------------------

    def _clamp_box(self, x, y, w, h):
        x = int(max(0, min(self.w - 1, x)))
        y = int(max(0, min(self.h - 1, y)))
        w = int(max(8, min(self.w - x, w)))
        h = int(max(8, min(self.h - y, h)))
        return x, y, w, h

    def _box_from_center(self, cx: int, cy: int, size: int):
        half = size // 2
        x = cx - half
        y = cy - half
        return self._clamp_box(x, y, size, size)

    def _init_tracker(self, frame_bgr: np.ndarray, cx: int, cy: int):
        self._cv_tracker = _create_cv_tracker(self._tracker_prefer)
        x, y, w, h = self._box_from_center(cx, cy, self.init_box)
        ok = bool(self._cv_tracker.init(frame_bgr, (x, y, w, h)))
        self._track_ok = ok
        self._track_fail_streak = 0
        if ok:
            self.last_match_box = (x, y, w, h)

    def _update_tracker(self, frame_bgr: np.ndarray):
        if self._cv_tracker is None:
            return False, None
        ok, box = self._cv_tracker.update(frame_bgr)
        if not ok:
            return False, None
        x, y, w, h = box
        x, y, w, h = self._clamp_box(int(x), int(y), int(w), int(h))
        cx = x + w // 2
        cy = y + h // 2
        self.last_match_box = (x, y, w, h)
        return True, (cx, cy)

    # -----------------------------

    def _copy_detector_debug(self):
        self.dbg_best = getattr(self.detector, "dbg_best", None)
        self.dbg_second = getattr(self.detector, "dbg_second", None)
        self.dbg_margin = getattr(self.detector, "dbg_margin", None)
        self.dbg_red = getattr(self.detector, "dbg_red", None)
        self.dbg_white = getattr(self.detector, "dbg_white", None)
        self.dbg_reject = getattr(self.detector, "dbg_reject", None)
        self.dbg_candidate_center = getattr(self.detector, "dbg_candidate_center", None)
        self.dbg_confirm = getattr(self.detector, "dbg_confirm", None)
        self.dbg_votes = getattr(self.detector, "dbg_votes", None)

        self.dbg_similarity = getattr(self.detector, "dbg_similarity", None)
        self.dbg_similarity_pct = getattr(self.detector, "dbg_similarity_pct", None)
        self.dbg_ignore_hit = getattr(self.detector, "dbg_ignore_hit", None)

        self.miss_count = getattr(self.detector, "miss_count", 0)

        lb = getattr(self.detector, "last_match_box", None)
        if lb is not None:
            self.last_match_box = lb

    def _roi_from_last(self, img_bgr):
        return self.detector._roi_from_last(img_bgr)

    # -----------------------------

    def _warp_to_respawn(self):
        rx, ry = self.respawn_xy
        self.last_x, self.last_y = int(rx), int(ry)

        self.detector.last_x, self.detector.last_y = self.last_x, self.last_y
        self.detector.smooth_x, self.detector.smooth_y = float(self.last_x), float(self.last_y)

    def on_player_death(self):
        """
        ✅ 핵심:
        - 즉시 tracker 폐기
        - respawn 쪽으로 워프(ROI 유도)
        - death_cooldown_sec 동안 update()는 hold(추적 정지)
        - 쿨다운 끝나면 N프레임 동안 detect 강제 + ROI 반경 키움
        """
        self._detect_fail_streak = 0
        self._track_fail_streak = 0

        # tracker 폐기
        self._cv_tracker = None
        self._track_ok = False

        # 탐색 중심을 부활 지점으로 워프
        self._warp_to_respawn()

        # ===== NEW: hold / post-death detect 강화 =====
        now = time.time()
        self._hold_until = now + self.death_cooldown_sec
        self._post_death_frames_left = self.post_death_force_detect_frames

        # detector miss_count를 올려서(간접) ROI 반경 확대 유도
        try:
            if hasattr(self.detector, "miss_count"):
                self.detector.miss_count = max(int(getattr(self.detector, "miss_count", 0)), self.post_death_boost_miss)
        except Exception:
            pass

    # -----------------------------

    def update(self, frame_bgr: np.ndarray) -> TrackResult:
        self._frame_i += 1
        now = time.time()

        # ===== death 직후 hold =====
        if now < self._hold_until:
            # ROI 중심은 계속 respawn 쪽으로 유지(드리프트 방지)
            self._warp_to_respawn()
            return TrackResult(self.last_x, self.last_y, 0.0, False, "hold")

        # 주기적 재탐지
        force_redetect = (self.redetect_every > 0) and (self._frame_i % self.redetect_every == 0)

        # ✅ 이 프레임에서 "post-death detect 강제"가 활성인지 스냅샷으로 고정
        post_death_active = (self._post_death_frames_left > 0)

        # 1) 기본은 tracker로 시도 (단, post-death 강제 detect 구간이면 tracker 사용 금지)
        if self._track_ok and (not force_redetect) and (not post_death_active):
            ok, center = self._update_tracker(frame_bgr)
            if ok and center is not None:
                cx, cy = center
                self.last_x, self.last_y = int(cx), int(cy)
                self._track_fail_streak = 0
                self._detect_fail_streak = 0

                self.detector.last_x, self.detector.last_y = self.last_x, self.last_y
                self.detector.smooth_x, self.detector.smooth_y = float(self.last_x), float(self.last_y)

                return TrackResult(self.last_x, self.last_y, 1.0, True, "track")

            # tracker 실패
            self._track_fail_streak += 1
            if self._track_fail_streak < self.track_fail_to_detect:
                return TrackResult(self.last_x, self.last_y, 0.0, False, "track")

            self._track_ok = False
            self._cv_tracker = None

        # 2) detector로 재획득
        if post_death_active:
            # detect 구간 동안은 tracker를 강제로 비활성(갈아타기 방지)
            self._cv_tracker = None
            self._track_ok = False

            # ROI 반경 넓히기 유도(가능한 경우)
            try:
                if hasattr(self.detector, "miss_count"):
                    self.detector.miss_count = max(
                        int(getattr(self.detector, "miss_count", 0)),
                        self.post_death_boost_miss
                    )
            except Exception:
                pass

        det = self.detector.update(frame_bgr)
        self._copy_detector_debug()

        cx, cy = int(det.x), int(det.y)

        votes = int(getattr(self.detector, "dbg_votes", 0) or 0)
        cand = getattr(self.detector, "dbg_candidate_center", None)
        if cand is not None and votes >= getattr(self.detector, "vote_min", 2):
            cx, cy = int(cand[0]), int(cand[1])

        if det.found:
            self._detect_fail_streak = 0
            self.last_x, self.last_y = cx, cy

            # ✅ "post-death detect 강제 구간"에서만 재획득을 까다롭게
            if post_death_active:
                votes = int(getattr(self.detector, "dbg_votes", 0) or 0)
                best = getattr(self.detector, "dbg_best", None)
                best = float(best) if best is not None else 0.0

                # (너가 쓰던 컷 유지. 필요하면 0.50/2/0.50으로 완화 추천)
                if (float(det.conf) < 0.55) or (votes < 3) or (best < 0.55):
                    # ✅ 프레임 카운트는 "이 프레임을 소비"했으니 여기서 감소
                    self._post_death_frames_left = max(0, self._post_death_frames_left - 1)
                    # tracker init 금지: detect-only로 더 찾게 둠
                    return TrackResult(self.last_x, self.last_y, float(det.conf), False, "detect")

            # 통과(or 평상시)이면 tracker init
            self._init_tracker(frame_bgr, self.last_x, self.last_y)

            # ✅ post-death 구간이면 이 프레임 소비 처리
            if post_death_active:
                self._post_death_frames_left = max(0, self._post_death_frames_left - 1)

            return TrackResult(self.last_x, self.last_y, float(det.conf), True, "detect")

        # detector 실패
        self._detect_fail_streak += 1

        # ✅ post-death 구간이면 실패 프레임도 소비
        if post_death_active:
            self._post_death_frames_left = max(0, self._post_death_frames_left - 1)

        if self._detect_fail_streak >= self.lost_fail_to_respawn:
            self._warp_to_respawn()
            return TrackResult(self.last_x, self.last_y, float(det.conf), False, "detect")

        return TrackResult(self.last_x, self.last_y, float(det.conf), False, "detect")
