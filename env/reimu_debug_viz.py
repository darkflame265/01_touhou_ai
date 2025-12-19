# env/reimu_debug_viz.py
import cv2
import numpy as np
import torch


class ReimuDebugViz:
    def __init__(self, win_main="REIMU_HEATMAP", win_hm="REIMU_HEATMAP_RAW"):
        self.win_main = win_main
        self.win_hm = win_hm
        self._inited = False

    def _ensure_window(self):
        if self._inited:
            return
        cv2.namedWindow(self.win_main, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.win_hm, cv2.WINDOW_NORMAL)
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

        # --- main view (playfield + cross) ---
        disp = play_gray
        if disp.dtype != np.uint8:
            disp = (disp * 255).astype(np.uint8)
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

        H, W = disp.shape[:2]
        px = int(xy_norm[0] * W)
        py = int(xy_norm[1] * H)

        cv2.drawMarker(disp, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
        cv2.circle(disp, (px, py), 10, (0, 255, 0), 2)

        txt = f"x={xy_norm[0]:.3f} y={xy_norm[1]:.3f} conf={conf:.4f}"
        cv2.putText(disp, txt, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, txt, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow(self.win_main, disp)

        # --- heatmap view ---
        hm = torch.sigmoid(heatmap_logits)[0, 0].detach().cpu().numpy()
        hm_u8 = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
        hm_show = cv2.resize(hm_u8, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(self.win_hm, hm_show)

        cv2.waitKey(1)
