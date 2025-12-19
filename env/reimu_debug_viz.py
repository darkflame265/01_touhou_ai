# env/reimu_debug_viz.py
import cv2
import numpy as np
import torch


class ReimuDebugViz:
    def __init__(self, win_main="REIMU_HEATMAP", win_hm="REIMU_HEATMAP_RAW"):
        self.win_main = win_main
        self.win_hm = win_hm
        self._inited = False

        # ✅ 한 번만 창 크기 강제 설정(프레임마다 resizeWindow 하면 깜빡/불편 가능)
        self._last_size = None  # (W, H)

    def _ensure_window(self):
        if self._inited:
            return
        cv2.namedWindow(self.win_main, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.win_hm, cv2.WINDOW_NORMAL)
        self._inited = True

    def _ensure_size(self, W: int, H: int):
        """창을 play_gray 크기와 동일하게 맞춘다(크기 변경 시에만)."""
        size = (int(W), int(H))
        if self._last_size == size:
            return
        self._last_size = size

        # ✅ 창 크기 강제
        cv2.resizeWindow(self.win_main, W, H)
        cv2.resizeWindow(self.win_hm, W, H)

    def show(
        self,
        play_gray: np.ndarray,
        heatmap_logits: torch.Tensor,
        xy_norm: tuple[float, float],
        conf: float,
        reward: float | None = None,
        total_reward: float | None = None,
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

        # ✅ 창 크기를 playfield(=게임 화면으로 쓰는 프레임) 크기와 동일하게 맞춤
        self._ensure_size(W, H)

        px = int(xy_norm[0] * W)
        py = int(xy_norm[1] * H)

        # 마커도 화면 크기에 비례해서 조금 키움
        marker_size = max(24, int(min(W, H) * 0.05))
        circle_r = max(10, int(min(W, H) * 0.02))
        thickness = 2

        cv2.drawMarker(disp, (px, py), (0, 255, 0), cv2.MARKER_CROSS, marker_size, thickness)
        cv2.circle(disp, (px, py), circle_r, (0, 255, 0), thickness)

        # ✅ 글자 크기도 화면 크기에 비례해서 키움
        font_scale = max(0.7, min(1.6, H / 450.0))  # H=450이면 1.0 정도
        line_h = int(30 * font_scale)

        txt1 = f"x={xy_norm[0]:.3f} y={xy_norm[1]:.3f} conf={conf:.4f}"
        cv2.putText(disp, txt1, (10, 10 + line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(disp, txt1, (10, 10 + line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), 1, cv2.LINE_AA)

        y2 = 10 + line_h * 2
        if reward is not None:
            txt2 = f"reward={float(reward):+.3f}"
            cv2.putText(disp, txt2, (10, y2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(disp, txt2, (10, y2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), 1, cv2.LINE_AA)

        y3 = 10 + line_h * 3
        if total_reward is not None:
            txt3 = f"total={float(total_reward):+.3f}"
            cv2.putText(disp, txt3, (10, y3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(disp, txt3, (10, y3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow(self.win_main, disp)

        # --- heatmap view ---
        hm = torch.sigmoid(heatmap_logits)[0, 0].detach().cpu().numpy()
        hm_u8 = (np.clip(hm, 0, 1) * 255).astype(np.uint8)

        # ✅ heatmap도 playfield 크기(W,H)에 맞춰서 크게 보여주기
        hm_show = cv2.resize(hm_u8, (W, H), interpolation=cv2.INTER_NEAREST)

        # (옵션) heatmap 창에도 reward 표시
        if reward is not None:
            cv2.putText(hm_show, f"r={float(reward):+.3f}", (10, 10 + line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        255, 3, cv2.LINE_AA)
            cv2.putText(hm_show, f"r={float(reward):+.3f}", (10, 10 + line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        0, 1, cv2.LINE_AA)

        cv2.imshow(self.win_hm, hm_show)
        cv2.waitKey(1)
