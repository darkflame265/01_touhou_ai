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
    """Create an OpenCV tracker instance across cv2 / cv2.legacy variations."""
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

    # fallback: KCF
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
    Template detector + OpenCV tracker hybrid.

    ✅ POLICY:
    - detector.update() is called ONLY inside "acquire windows" and only for limited events.
      Default: 3 events total (start + after 1st respawn + after 2nd respawn)
    - outside acquire windows: tracker-only (NO periodic redetect, NO failover redetect)
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

        # ===== death hold =====
        death_cooldown_sec: float = 1.2,

        # ===== detection budget policy =====
        max_detect_events: int = 3,
        acquire_window_frames: int = 60,
        acquire_boost_miss: int = 10,

        # legacy args that some callers may pass (we ignore them safely)
        track_fail_to_detect: int | None = None,
        redetect_every: int | None = None,

        # detector kwargs (MultiTemplateTracker)
        **detector_kwargs,
    ):
        self.w = int(frame_w)
        self.h = int(frame_h)

        self.init_box = int(init_box)
        self._tracker_prefer = str(tracker_prefer)

        # respawn
        if respawn_xy is None:
            respawn_xy = (self.w // 2, int(self.h * 0.78))
        self.respawn_xy = (int(respawn_xy[0]), int(respawn_xy[1]))

        if init_xy is None:
            init_xy = (self.w // 2, int(self.h * 0.78))
        self.last_x, self.last_y = int(init_xy[0]), int(init_xy[1])

        # hold
        self.death_cooldown_sec = float(death_cooldown_sec)
        self._hold_until = 0.0
        self._need_acquire_after_hold = False

        # ===== 핵심: detector_kwargs를 MultiTemplateTracker가 받는 키만 남긴다 =====
        detector_kwargs = self._filter_detector_kwargs(detector_kwargs)

        self.detector = MultiTemplateTracker(
            frame_w=self.w,
            frame_h=self.h,
            template_paths=template_paths,
            init_xy=init_xy,
            **detector_kwargs,
        )

        # detection budget
        self.max_detect_events = int(max_detect_events)
        self.acquire_window_frames = int(acquire_window_frames)
        self.acquire_boost_miss = int(acquire_boost_miss)
        self._detect_events_used = 0
        self._acquire_left = 0

        # tracker
        self._cv_tracker = None
        self._track_ok = False

        # Debug passthrough (for DebugViz)
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

        # Start: first (1/3) detect event
        self._start_acquire_if_budget()

    # -----------------------------
    # detector kwargs filtering
    # -----------------------------
    def _filter_detector_kwargs(self, kw: dict) -> dict:
        """
        MultiTemplateTracker.__init__()가 실제로 받는 키만 통과시킨다.
        (여기서 빠지면 'unexpected keyword argument'를 원천 차단)
        """
        allowed = {
            "ema_alpha",
            "base_search_radius",
            "scales",
            "min_score",
            "min_margin",
            "method",
            "red_min_ratio",
            "white_min_ratio",
            "soft_update_score",
            "soft_alpha",
            "strong_score",
            "max_jump",
            "vote_radius",
            "vote_min",
            "vote_min_score",
            "ignore_template_paths",
            "ignore_min_score",
            "ignore_block_radius",
            "enable_ignore_block",
            "enable_full_search",
            "full_search_after_miss",
            "full_search_frames",
            "require_confirm_to_accept",
        }
        return {k: v for k, v in kw.items() if k in allowed}

    # -----------------------------
    # Tracker helpers
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

    # -----------------------------
    # Debug passthrough
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
    # Acquire policy
    # -----------------------------
    def _warp_to_respawn(self):
        rx, ry = self.respawn_xy
        self.last_x, self.last_y = int(rx), int(ry)
        self.detector.last_x, self.detector.last_y = self.last_x, self.last_y
        self.detector.smooth_x, self.detector.smooth_y = float(self.last_x), float(self.last_y)

    def _start_acquire_if_budget(self):
        if self._detect_events_used >= self.max_detect_events:
            self._acquire_left = 0
            return False

        self._detect_events_used += 1
        self._acquire_left = max(1, self.acquire_window_frames)

        # Boost detector search radius indirectly
        try:
            if hasattr(self.detector, "miss_count"):
                self.detector.miss_count = max(int(getattr(self.detector, "miss_count", 0)), self.acquire_boost_miss)
        except Exception:
            pass

        return True

    # -----------------------------
    # Public hooks
    # -----------------------------
    def on_player_death(self):
        self._cv_tracker = None
        self._track_ok = False

        self._warp_to_respawn()

        now = time.time()
        self._hold_until = now + self.death_cooldown_sec
        self._need_acquire_after_hold = True

        self._acquire_left = 0

    # -----------------------------
    # Main update
    # -----------------------------
    def update(self, frame_bgr: np.ndarray) -> TrackResult:
        now = time.time()

        # 0) HOLD right after death
        if now < self._hold_until:
            self._warp_to_respawn()
            return TrackResult(self.last_x, self.last_y, 0.0, False, "hold")

        # 0.5) First frame after hold ends: start acquire (if budget remains)
        if self._need_acquire_after_hold:
            self._need_acquire_after_hold = False
            self._start_acquire_if_budget()

        # 1) If tracker is running, ONLY track (no detector refresh)
        if self._track_ok:
            ok, center = self._update_tracker(frame_bgr)
            if ok and center is not None:
                cx, cy = center
                self.last_x, self.last_y = int(cx), int(cy)

                # keep detector ROI centered (but do NOT call detector.update)
                self.detector.last_x, self.detector.last_y = self.last_x, self.last_y
                self.detector.smooth_x, self.detector.smooth_y = float(self.last_x), float(self.last_y)

                return TrackResult(self.last_x, self.last_y, 1.0, True, "track")

            # tracker failed: stop tracking; do NOT auto-detect
            self._track_ok = False
            self._cv_tracker = None
            return TrackResult(self.last_x, self.last_y, 0.0, False, "track")

        # 2) No tracker: only detect if we're inside an acquire window
        if self._acquire_left > 0:
            self._acquire_left -= 1

            det = self.detector.update(frame_bgr)
            self._copy_detector_debug()

            cx, cy = int(det.x), int(det.y)

            # use candidate center when votes are enough
            try:
                votes = int(getattr(self.detector, "dbg_votes", 0) or 0)
            except Exception:
                votes = 0
            cand = getattr(self.detector, "dbg_candidate_center", None)
            if cand is not None and votes >= int(getattr(self.detector, "vote_min", 2)):
                cx, cy = int(cand[0]), int(cand[1])

            if bool(det.found):
                self.last_x, self.last_y = int(cx), int(cy)
                self._init_tracker(frame_bgr, self.last_x, self.last_y)
                return TrackResult(self.last_x, self.last_y, float(det.conf), True, "detect")

            return TrackResult(self.last_x, self.last_y, float(det.conf), False, "detect")

        # 3) No tracker + no acquire window: freeze
        return TrackResult(self.last_x, self.last_y, 0.0, False, "track")
