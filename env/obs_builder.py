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

        # ✅ Detector(템플릿) + Tracker(OpenCV) 하이브리드
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
            respawn_xy=(int(w0*0.35), int(h0*0.85)),

            # ---- tracking side ----
            tracker_prefer="CSRT",      # 정확도 우선(느리면 "MOSSE"로)
            init_box=56,                # 레이무 bbox 크기(너 캡쳐 기준 튜닝 가능)
            track_fail_to_detect=2,     # tracker 2번 연속 실패하면 detector로 재획득
            redetect_every=15,          # 15프레임마다 detector로 재동기화(드리프트 방지)

            # ---- detector side (MultiTemplateTracker 파라미터) ----
            ema_alpha=0.35,
            base_search_radius=260,
            scales=(0.97, 1.0, 1.03),
            min_score=0.40,
            min_margin=0.08,
            red_min_ratio=0.06,
            white_min_ratio=0.09,

            # detector 내부 vote 기준
            vote_radius=18,
            vote_min=2,
            vote_min_score=0.30,

            # ✅ NEW: 아이템 2개 블랙리스트(템플릿 매칭 초점 뺏김 방지)
            ignore_template_paths=[
                "assets/item_black_1.png",   # <- 아이템1 템플릿 파일명으로 교체
                "assets/item_black_2.png",   # <- 아이템2 템플릿 파일명으로 교체
                "assets/item_black_3.png",
                "assets/item_black_4.png",
                "assets/item_black_5.png",
            ],
            ignore_min_score=0.65,      # 아이템이라고 확정할 최소 점수(필요시 0.45~0.60 튜닝)
            ignore_block_radius=32,     # 아이템 중심 근처 후보 차단 반경(24~40 튜닝)
            enable_ignore_block=True,
            
        )

        self.player_center = None  # (x,y)

        # =========================
        # ✅ UI 영역 제외 설정
        # =========================
        self.ui_cut_ratio = 0.66
        self.ui_cut_bottom_ratio = 1.00



    def _crop_square_bgr(self, img_bgr, cx, cy, size):
        """
        (중요) 코너/벽에서 정보량이 줄어드는 '치팅'을 막기 위해
        화면 밖은 검정 패딩이 아니라 REFLECT 패딩으로 채운다.

        - 항상 (size x size) 반환
        - cx,cy가 화면 밖으로 조금 나가도 안전
        """
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
            # ✅ 검정(0) 대신 반사 패딩: 코너에서도 '검은 힌트'가 생기지 않게 함
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

        # 혹시라도 안전장치: 항상 size 보장
        if crop.shape[0] != size or crop.shape[1] != size:
            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)

        return crop


    def _mask_out_ui_region(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        트래커 입력에서 우측 UI 영역을 "검정"으로 지워버림.
        (주의) crop 관측은 원본 img_bgr에서 하고, 트래커만 이걸로 돌림.
        """
        out = img_bgr.copy()

        x_ui = int(self.W * self.ui_cut_ratio)
        # 우측 UI 제거
        out[:, x_ui:, :] = 0

        # 필요 시 우측 UI 하단만 추가로 제거(옵션)
        if self.ui_cut_bottom_ratio < 1.0:
            y0 = int(self.H * self.ui_cut_bottom_ratio)
            out[y0:, x_ui:, :] = 0

        return out

    def make_state(self, img_bgr):
        # ✅ 트래커 입력만 UI 제거한 프레임 사용
        img_for_track = self._mask_out_ui_region(img_bgr)
        tr = self.tracker.update(img_for_track)

        # 하이브리드는 tr.x,tr.y가 “최종” 중심값
        cx, cy = int(tr.x), int(tr.y)
        self.player_center = (cx, cy)

        # 디버그는 트래커가 실제로 본 화면 기준
        if self.debug is not None:
            self.debug.show_tracker(img_for_track, self.tracker, tr, self.crop_size)

        # ✅ tracker/detector 둘 다 실패해서 붕괴한 경우에만 fallback
        # (conf가 0에 가깝고 found=False일 때)
        if self.use_fallback_full_preprocess:
            if (not tr.found) and (float(tr.conf) <= 0.01):
                return self.screen.preprocess(img_bgr)

        # ✅ crop은 원본에서
        crop_bgr = self._crop_square_bgr(img_bgr, cx, cy, self.crop_size)
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        obs = cv2.resize(crop_gray, (self.obs_out_size, self.obs_out_size), interpolation=cv2.INTER_AREA)
        return obs.astype(np.float32) / 255.0

