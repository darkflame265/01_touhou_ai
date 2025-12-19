# env/reimu_debug_viz.py
import cv2
import numpy as np
import torch


class ReimuDebugViz:
    def __init__(
        self,
        win_main="REIMU_HEATMAP",
        win_hm="REIMU_HEATMAP_RAW",
        hm_scale=4,          # RAW 히트맵 확대 배율
        gap_px=20,           # 창 사이 간격
        anchor=(50, 50),     # 메인 창 좌상단 위치
    ):
        self.win_main = win_main
        self.win_hm = win_hm
        self.hm_scale = int(hm_scale)
        self.gap_px = int(gap_px)
        self.anchor = anchor  # (x,y)

        self._inited = False
        self._last_main_size = None

    def _ensure_window(self):
        if self._inited:
            return

        cv2.namedWindow(self.win_main, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.win_hm, cv2.WINDOW_NORMAL)

        # 초기 위치(메인만 먼저)
        cv2.moveWindow(self.win_main, self.anchor[0], self.anchor[1])

        self._inited = True

    def show(
        self,
        play_gray: np.ndarray,
        heatmap_logits: torch.Tensor,
        xy_norm: tuple[float, float],
        conf: float,
    ):
        """
        play_gray: (H,W) uint8 or float
        heatmap_logits: (1,1,h,w) torch tensor
        xy_norm: (x,y) normalized 0..1 on playfield
        """
        self._ensure_window()

        # =========================================================
        # Main view: playfield (실제 게임 크기)
        # =========================================================
        disp = play_gray
        if disp.dtype != np.uint8:
            disp = (disp * 255).astype(np.uint8)
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

        H, W = disp.shape[:2]
        px = int(xy_norm[0] * W)
        py = int(xy_norm[1] * H)

        cv2.drawMarker(disp, (px, py), (0, 255, 0),
                       cv2.MARKER_CROSS, 28, 2)
        cv2.circle(disp, (px, py), 12, (0, 255, 0), 2)

        txt = f"x={xy_norm[0]:.3f} y={xy_norm[1]:.3f} conf={conf:.4f}"
        cv2.putText(disp, txt, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, txt, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow(self.win_main, disp)

        # 메인 창 크기 기억 (RAW 창 배치용)
        self._last_main_size = (W, H)

        # =========================================================
        # RAW heatmap view (아래쪽 배치)
        # =========================================================
        hm = torch.sigmoid(heatmap_logits)[0, 0].detach().cpu().numpy()
        hm_u8 = (np.clip(hm, 0, 1) * 255).astype(np.uint8)

        hm_show = cv2.resize(
            hm_u8,
            None,
            fx=self.hm_scale,
            fy=self.hm_scale,
            interpolation=cv2.INTER_NEAREST
        )

        cv2.imshow(self.win_hm, hm_show)

        # =========================================================
        # 창 위치 조정 (겹치지 않게)
        # =========================================================
        if self._last_main_size is not None:
            main_x, main_y = self.anchor
            main_w, main_h = self._last_main_size

            hm_x = main_x
            hm_y = main_y + main_h + self.gap_px

            cv2.moveWindow(self.win_hm, hm_x, hm_y)

        cv2.waitKey(1)
