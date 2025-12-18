# env/debug_viz.py
import cv2
import numpy as np


class DebugViz:
    def __init__(self):
        self.show_tracker_debug = True
        self.tracker_show_every = 1
        self._tracker_dbg_i = 0

        self.FULL_POS = (50, 50)

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.62
        self.thickness = 2
        self.line_gap = 6
        self.text_color = (255, 255, 255)
        self.text_shadow = (0, 0, 0)

        self.smooth_alpha = 0.35
        self._s_tr = None
        self._s_cand = None

        self._windows_inited = False
        self._last_full_wh = None  # (w,h)

    def _ensure_windows(self):
        if self._windows_inited:
            return
        try:
            cv2.namedWindow("TRACK_FULL", cv2.WINDOW_NORMAL)
        except Exception:
            pass
        self._windows_inited = True

    def _ensure_window_sizes(self, w, h):
        wh = (int(w), int(h))
        if self._last_full_wh == wh:
            return
        self._last_full_wh = wh
        try:
            cv2.resizeWindow("TRACK_FULL", wh[0], wh[1])
        except Exception:
            pass

    def _fnum(self, v, nd=3, none="-"):
        if v is None:
            return none
        try:
            return f"{float(v):.{nd}f}"
        except Exception:
            return str(v)

    def _fint(self, v, none="-"):
        if v is None:
            return none
        try:
            return str(int(v))
        except Exception:
            return str(v)

    def _summ_reject(self, rej):
        if rej is None:
            return "-"
        try:
            if isinstance(rej, dict):
                true_keys = [k for k, v in rej.items() if bool(v)]
                return "{" + ",".join(map(str, true_keys)) + "}" if true_keys else "{none}"
            if isinstance(rej, (list, tuple, set)):
                lst = list(rej)
                return "[" + ",".join(map(str, lst[:10])) + ("..." if len(lst) > 10 else "") + "]"
            return str(rej)
        except Exception:
            return str(rej)

    def _ema_point(self, prev, x, y):
        if prev is None:
            return (float(x), float(y))
        a = float(self.smooth_alpha)
        px, py = prev
        sx = (1 - a) * float(px) + a * float(x)
        sy = (1 - a) * float(py) + a * float(y)
        return (sx, sy)

    def _draw_cross(self, img, x, y, size=10, thickness=2, color=(0, 255, 0)):
        x, y = int(x), int(y)
        cv2.line(img, (x - size, y), (x + size, y), color, thickness)
        cv2.line(img, (x, y - size), (x, y + size), color, thickness)

    def _put_lines(self, img, lines, x=10, y=10):
        cy = int(y)
        for line in lines:
            if line is None:
                continue
            line = str(line)
            (tw, th), _ = cv2.getTextSize(line, self.font, self.font_scale, self.thickness)
            cv2.putText(img, line, (int(x) + 1, cy + th + 1), self.font,
                        self.font_scale, self.text_shadow, self.thickness + 1, cv2.LINE_AA)
            cv2.putText(img, line, (int(x), cy + th), self.font,
                        self.font_scale, self.text_color, self.thickness, cv2.LINE_AA)
            cy += th + self.line_gap

    def _safe_roi_from_tracker(self, tracker, img_bgr):
        fn = getattr(tracker, "_roi_from_last", None)
        if callable(fn):
            try:
                roi, ox, oy = fn(img_bgr)
                if roi is not None and roi.size > 0:
                    return roi, int(ox), int(oy)
            except Exception:
                pass
        return img_bgr, 0, 0

    def _get_detector(self, tracker):
        det = getattr(tracker, "detector", None)
        return det if det is not None else tracker

    def show_tracker(self, img_bgr, tracker, tr, crop_size):
        if not self.show_tracker_debug:
            return

        self._ensure_windows()

        self._tracker_dbg_i += 1
        if (self._tracker_dbg_i % max(1, self.tracker_show_every)) != 0:
            return

        H, W = img_bgr.shape[:2]
        self._ensure_window_sizes(W, H)

        # ROI 박스(파랑)용
        roi_det, ox_det, oy_det = self._safe_roi_from_tracker(tracker, img_bgr)
        rh_det, rw_det = roi_det.shape[:2] if roi_det is not None else (0, 0)

        # tr
        tr_x = getattr(tr, "x", None)
        tr_y = getattr(tr, "y", None)
        tr_conf = float(getattr(tr, "conf", 0.0) or 0.0)
        tr_method = getattr(tr, "method", "trk")
        tr_found = bool(getattr(tr, "found", False))

        cx = int(tr_x) if tr_x is not None else 0
        cy = int(tr_y) if tr_y is not None else 0

        # cand
        cand = getattr(tracker, "dbg_candidate_center", None)

        # smoothing
        self._s_tr = self._ema_point(self._s_tr, cx, cy)
        scx, scy = int(round(self._s_tr[0])), int(round(self._s_tr[1]))

        if cand is not None:
            try:
                cdx, cdy = int(cand[0]), int(cand[1])
                self._s_cand = self._ema_point(self._s_cand, cdx, cdy)
            except Exception:
                self._s_cand = self._ema_point(self._s_cand, scx, scy)
        else:
            self._s_cand = self._ema_point(self._s_cand, scx, scy)

        sdcx, sdcy = int(round(self._s_cand[0])), int(round(self._s_cand[1]))

        # detector debug 핵심
        det = self._get_detector(tracker)
        best = getattr(det, "dbg_best", None)
        second = getattr(det, "dbg_second", None)
        margin = getattr(det, "dbg_margin", None)
        reject = getattr(det, "dbg_reject", None)
        confirm = getattr(det, "dbg_confirm", None)

        votes = getattr(det, "dbg_votes", getattr(tracker, "dbg_votes", None))
        vote_min = getattr(det, "vote_min", None)

        # draw
        full = img_bgr.copy()

        # ROI 박스
        if roi_det is not None:
            if not (ox_det == 0 and oy_det == 0 and rw_det == full.shape[1] and rh_det == full.shape[0]):
                cv2.rectangle(full, (ox_det, oy_det), (ox_det + rw_det, oy_det + rh_det), (255, 0, 0), 2)

        # 확정이면 crop 박스 + 초록 십자
        if tr_found:
            self._draw_cross(full, scx, scy, size=10, thickness=2, color=(0, 255, 0))
            try:
                half = int(crop_size) // 2
                cv2.rectangle(full, (scx - half, scy - half), (scx + half, scy + half), (0, 255, 255), 2)
            except Exception:
                pass

        # 후보는 항상 노랑 십자
        self._draw_cross(full, sdcx, sdcy, size=8, thickness=2, color=(255, 255, 0))

        # 텍스트(짧게)
        lines = [
            f"[TR] {tr_method} found={int(tr_found)} conf={tr_conf:.3f} tr=({cx},{cy})",
            f"[DET] best={self._fnum(best)} second={self._fnum(second)} margin={self._fnum(margin)}",
            f"[DET] votes={self._fint(votes)} vote_min={self._fint(vote_min)} confirm={confirm}",
            f"[DET] reject={self._summ_reject(reject)}",
        ]
        self._put_lines(full, lines, x=10, y=10)

        cv2.imshow("TRACK_FULL", full)
        cv2.moveWindow("TRACK_FULL", *self.FULL_POS)
        cv2.waitKey(1)
