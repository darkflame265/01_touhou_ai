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
    name = str(prefer).upper()
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
    SUPER-RELAXED acquire gate:
    - Lock-in happens easily (streak=2, votes can be 1, margin ignored).
    - Goal: grab something quickly near Reimu and let CSRT hold it.
    """

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        template_paths: list[str],
        init_xy: tuple[int, int] | None = None,
        tracker_prefer: str = "CSRT",
        init_box: int = 56,
        respawn_xy: tuple[int, int] | None = None,

        death_cooldown_sec: float = 1.2,

        max_detect_events: int = 3,
        acquire_window_frames: int = 420,
        acquire_boost_miss: int = 10,

        # ===== SUPER RELAXED =====
        acquire_streak_needed: int = 2,
        acquire_pos_tol: int = 60,

        acquire_min_best: float = 0.36,
        acquire_min_margin: float = 0.0,
        acquire_min_votes: int = 1,

        acquire_allow_votes1_if_best_ge: float = 0.40,

        acquire_roi_start_r: int = 140,
        acquire_roi_expand_per_frame: float = 3.0,
        acquire_roi_max_r: int = 520,

        **detector_kwargs,
    ):
        self.w = int(frame_w)
        self.h = int(frame_h)

        self.init_box = int(init_box)
        self._tracker_prefer = str(tracker_prefer)

        if respawn_xy is None:
            respawn_xy = (self.w // 2, int(self.h * 0.78))
        self.respawn_xy = (int(respawn_xy[0]), int(respawn_xy[1]))

        if init_xy is None:
            init_xy = (self.w // 2, int(self.h * 0.78))
        self.last_x, self.last_y = int(init_xy[0]), int(init_xy[1])

        self.death_cooldown_sec = float(death_cooldown_sec)
        self._hold_until = 0.0
        self._need_acquire_after_hold = False

        self.detector = MultiTemplateTracker(
            frame_w=self.w,
            frame_h=self.h,
            template_paths=template_paths,
            init_xy=init_xy,
            **detector_kwargs,
        )

        self.max_detect_events = int(max_detect_events)
        self.acquire_window_frames = int(acquire_window_frames)
        self.acquire_boost_miss = int(acquire_boost_miss)
        self._detect_events_used = 0
        self._acquire_left = 0

        self.acquire_streak_needed = int(max(1, acquire_streak_needed))
        self.acquire_pos_tol = int(max(1, acquire_pos_tol))
        self.acquire_min_best = float(acquire_min_best)
        self.acquire_min_margin = float(acquire_min_margin)
        self.acquire_min_votes = int(acquire_min_votes)
        self.acquire_allow_votes1_if_best_ge = float(acquire_allow_votes1_if_best_ge)

        self.acquire_roi_start_r = int(max(30, acquire_roi_start_r))
        self.acquire_roi_expand_per_frame = float(max(0.1, acquire_roi_expand_per_frame))
        self.acquire_roi_max_r = int(max(self.acquire_roi_start_r, acquire_roi_max_r))

        self._acquire_elapsed = 0
        self._streak = 0
        self._streak_x = None
        self._streak_y = None
        self._streak_best = 0.0

        self._orig_base_r = getattr(self.detector, "base_r", None)

        self._cv_tracker = None
        self._track_ok = False

        # debug passthrough
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

        self._start_acquire_if_budget()

    # -----------------------------
    # helpers
    # -----------------------------

    def _clamp_box(self, x, y, w, h):
        x = int(max(0, min(self.w - 1, x)))
        y = int(max(0, min(self.h - 1, y)))
        w = int(max(8, min(self.w - x, w)))
        h = int(max(8, min(self.h - y, h)))
        return x, y, w, h

    def _box_from_center(self, cx: int, cy: int, size: int):
        half = int(size) // 2
        x = int(cx) - half
        y = int(cy) - half
        return self._clamp_box(x, y, int(size), int(size))

    def _init_tracker(self, frame_bgr: np.ndarray, cx: int, cy: int):
        self._cv_tracker = _create_cv_tracker(self._tracker_prefer)
        x, y, w, h = self._box_from_center(cx, cy, self.init_box)
        ok = bool(self._cv_tracker.init(frame_bgr, (x, y, w, h)))
        self._track_ok = ok
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

    def _copy_detector_debug(self):
        self.dbg_best = getattr(self.detector, "dbg_best", None)
        self.dbg_second = getattr(self.detector, "dbg_second", None)
        self.dbg_margin = getattr(self.detector, "dbg_margin", None)
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

    def _warp_to_respawn(self):
        rx, ry = self.respawn_xy
        self.last_x, self.last_y = int(rx), int(ry)
        self.detector.last_x, self.detector.last_y = self.last_x, self.last_y
        self.detector.smooth_x, self.detector.smooth_y = float(self.last_x), float(self.last_y)

    def _restore_detector_base_r(self):
        try:
            if self._orig_base_r is not None:
                self.detector.base_r = int(self._orig_base_r)
        except Exception:
            pass

    def _apply_acquire_base_r(self):
        try:
            start = float(self.acquire_roi_start_r)
            inc = float(self.acquire_roi_expand_per_frame) * float(max(0, self._acquire_elapsed))
            r = int(min(float(self.acquire_roi_max_r), start + inc))
            if hasattr(self.detector, "base_r"):
                self.detector.base_r = int(max(30, r))
        except Exception:
            pass

    def _reset_streak(self):
        self._streak = 0
        self._streak_x = None
        self._streak_y = None
        self._streak_best = 0.0

    def _start_acquire_if_budget(self):
        if self._detect_events_used >= self.max_detect_events:
            self._acquire_left = 0
            return False

        self._detect_events_used += 1
        self._acquire_left = max(1, self.acquire_window_frames)
        self._acquire_elapsed = 0
        self._reset_streak()

        try:
            if hasattr(self.detector, "miss_count"):
                self.detector.miss_count = max(int(getattr(self.detector, "miss_count", 0)), self.acquire_boost_miss)
        except Exception:
            pass

        self._apply_acquire_base_r()
        return True

    def on_player_death(self):
        self._cv_tracker = None
        self._track_ok = False

        self._warp_to_respawn()

        now = time.time()
        self._hold_until = now + self.death_cooldown_sec
        self._need_acquire_after_hold = True

        self._acquire_left = 0
        self._reset_streak()
        self._restore_detector_base_r()

    # -----------------------------
    # main update
    # -----------------------------

    def update(self, frame_bgr: np.ndarray) -> TrackResult:
        now = time.time()

        if now < self._hold_until:
            self._warp_to_respawn()
            return TrackResult(self.last_x, self.last_y, 0.0, False, "hold")

        if self._need_acquire_after_hold:
            self._need_acquire_after_hold = False
            self._start_acquire_if_budget()

        if self._track_ok:
            ok, center = self._update_tracker(frame_bgr)
            if ok and center is not None:
                cx, cy = center
                self.last_x, self.last_y = int(cx), int(cy)
                self.detector.last_x, self.detector.last_y = self.last_x, self.last_y
                self.detector.smooth_x, self.detector.smooth_y = float(self.last_x), float(self.last_y)
                return TrackResult(self.last_x, self.last_y, 1.0, True, "track")

            self._track_ok = False
            self._cv_tracker = None
            return TrackResult(self.last_x, self.last_y, 0.0, False, "track")

        if self._acquire_left > 0:
            self._acquire_left -= 1
            self._acquire_elapsed += 1
            self._apply_acquire_base_r()

            det = self.detector.update(frame_bgr)
            self._copy_detector_debug()

            cx, cy = int(det.x), int(det.y)
            try:
                votes = int(getattr(self.detector, "dbg_votes", 0) or 0)
            except Exception:
                votes = 0
            cand = getattr(self.detector, "dbg_candidate_center", None)
            if cand is not None:
                cx, cy = int(cand[0]), int(cand[1])

            best = float(getattr(self.detector, "dbg_best", 0.0) or 0.0)
            margin = float(getattr(self.detector, "dbg_margin", 0.0) or 0.0)

            # ✅ SUPER RELAXED quality_ok:
            # - votes 거의 안 봄 (1도 OK)
            # - margin 무시 수준
            votes_ok = (votes >= self.acquire_min_votes) or (votes == 1 and best >= self.acquire_allow_votes1_if_best_ge)
            quality_ok = (best >= self.acquire_min_best) and votes_ok and (margin >= self.acquire_min_margin)

            if quality_ok:
                if self._streak_x is None:
                    self._streak_x, self._streak_y = int(cx), int(cy)
                    self._streak = 1
                    self._streak_best = best
                else:
                    dx = int(cx) - int(self._streak_x)
                    dy = int(cy) - int(self._streak_y)
                    if (dx * dx + dy * dy) <= float(self.acquire_pos_tol * self.acquire_pos_tol):
                        self._streak += 1
                        self._streak_x = int(round(0.7 * self._streak_x + 0.3 * cx))
                        self._streak_y = int(round(0.7 * self._streak_y + 0.3 * cy))
                        self._streak_best = max(self._streak_best, best)
                    else:
                        # 너무 튀어도 리셋하지 말고 거의 유지
                        self._streak = max(0, self._streak - 0)  # 유지
                        self._streak_x = int(round(0.5 * self._streak_x + 0.5 * cx))
                        self._streak_y = int(round(0.5 * self._streak_y + 0.5 * cy))
            else:
                # 불안정해도 천천히만 감소
                self._streak = max(0, self._streak - 0)

            if self._streak >= self.acquire_streak_needed:
                lock_x = int(self._streak_x if self._streak_x is not None else cx)
                lock_y = int(self._streak_y if self._streak_y is not None else cy)

                self.last_x, self.last_y = lock_x, lock_y
                self._init_tracker(frame_bgr, self.last_x, self.last_y)

                self._acquire_left = 0
                self._restore_detector_base_r()
                self._reset_streak()

                return TrackResult(self.last_x, self.last_y, float(det.conf), True, "detect")

            return TrackResult(self.last_x, self.last_y, float(det.conf), False, "detect")

        self._restore_detector_base_r()
        return TrackResult(self.last_x, self.last_y, 0.0, False, "track")
