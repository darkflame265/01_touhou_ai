# env/obs_builder.py
import cv2
import numpy as np
from env.det_track_tracker import DetTrackTracker


class ObsBuilder:
    def __init__(self, screen, debug_viz=None, obs_out_size=84, crop_size=160, use_fallback_full_preprocess=True):
        self.screen = screen
        self.debug = debug_viz

        self.obs_out_size = obs_out_size
        self.crop_size = crop_size
        self.use_fallback_full_preprocess = use_fallback_full_preprocess

        img0 = self.screen.capture()
        h0, w0 = img0.shape[:2]
        self.H, self.W = h0, w0

        self.tracker = DetTrackTracker(
            frame_w=w0,
            frame_h=h0,
            template_paths=[
                "assets/reimu_still.png",
                "assets/reimu_left.png",
                "assets/reimu_right.png",
                "assets/reimu_still_tight.png",
                "assets/reimu_left_tight.png",
                "assets/reimu_right_tight.png",
                "assets/reimu_black_still.png",
                "assets/reimu_1.png",
            ],
            init_xy=(w0 // 2, int(h0 * 0.78)),
            respawn_xy=(int(w0 * 0.35), int(h0 * 0.85)),

            tracker_prefer="CSRT",
            init_box=56,

            max_detect_events=3,

            # ✅ SUPER RELAXED acquire
            acquire_window_frames=520,
            acquire_streak_needed=2,
            acquire_pos_tol=60,
            acquire_min_best=0.36,
            acquire_min_margin=0.0,
            acquire_min_votes=1,
            acquire_allow_votes1_if_best_ge=0.40,

            acquire_roi_start_r=140,
            acquire_roi_expand_per_frame=3.0,
            acquire_roi_max_r=520,

            # detector side (그대로 두되, 필요하면 더 완화 가능)
            ema_alpha=0.35,
            base_search_radius=260,
            scales=(0.97, 1.0, 1.03),
            min_score=0.36,
            min_margin=0.01,

            vote_radius=18,
            vote_min=2,
            vote_min_score=0.25,

            ignore_template_paths=[
                "assets/item_black_1.png",
                "assets/item_black_2.png",
                "assets/item_black_3.png",
                "assets/item_black_4.png",
                "assets/item_black_5.png",
            ],
            ignore_min_score=0.60,
            ignore_block_radius=36,
            enable_ignore_block=True,
        )

        self.player_center = None

        self.ui_cut_ratio = 0.66
        self.ui_cut_bottom_ratio = 1.00

    def _crop_square_bgr(self, img_bgr, cx, cy, size):
        h, w = img_bgr.shape[:2]
        size = int(size)
        half = size // 2

        x1 = int(round(cx - half))
        y1 = int(round(cy - half))
        x2 = x1 + size
        y2 = y1 + size

        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - w)
        pad_b = max(0, y2 - h)

        if pad_l or pad_t or pad_r or pad_b:
            img_bgr = cv2.copyMakeBorder(
                img_bgr,
                pad_t, pad_b, pad_l, pad_r,
                borderType=cv2.BORDER_REFLECT_101
            )
            x1 += pad_l
            y1 += pad_t
            x2 += pad_l
            y2 += pad_t

        crop = img_bgr[y1:y2, x1:x2]
        if crop.shape[0] != size or crop.shape[1] != size:
            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
        return crop

    def _mask_out_ui_region(self, img_bgr: np.ndarray) -> np.ndarray:
        out = img_bgr.copy()
        x_ui = int(self.W * self.ui_cut_ratio)
        out[:, x_ui:, :] = 0
        if self.ui_cut_bottom_ratio < 1.0:
            y0 = int(self.H * self.ui_cut_bottom_ratio)
            out[y0:, x_ui:, :] = 0
        return out

    def make_state(self, img_bgr):
        img_for_track = self._mask_out_ui_region(img_bgr)
        tr = self.tracker.update(img_for_track)

        cx, cy = int(tr.x), int(tr.y)
        self.player_center = (cx, cy)

        if self.debug is not None:
            self.debug.show_tracker(img_for_track, self.tracker, tr, self.crop_size)

        if self.use_fallback_full_preprocess:
            if (not tr.found) and (float(tr.conf) <= 0.01):
                return self.screen.preprocess(img_bgr)

        crop_bgr = self._crop_square_bgr(img_bgr, cx, cy, self.crop_size)
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        obs = cv2.resize(crop_gray, (self.obs_out_size, self.obs_out_size), interpolation=cv2.INTER_AREA)
        return obs.astype(np.float32) / 255.0
