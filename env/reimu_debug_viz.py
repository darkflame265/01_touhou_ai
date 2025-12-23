# env/reimu_debug_viz.py
import cv2
import numpy as np
import torch


class ReimuDebugViz:
    def __init__(self, win_main="REIMU_HEATMAP", win_hm="REIMU_HEATMAP_RAW"):
        self.win_main = win_main
        self.win_hm = win_hm
        self._inited = False
        self._last_size = None  # (W, H)

    def _ensure_window(self):
        if self._inited:
            return
        cv2.namedWindow(self.win_main, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.win_hm, cv2.WINDOW_NORMAL)
        self._inited = True

    def _ensure_size(self, W: int, H: int):
        size = (int(W), int(H))
        if self._last_size == size:
            return
        self._last_size = size
        cv2.resizeWindow(self.win_main, W, H)
        cv2.resizeWindow(self.win_hm, W, H)

    @staticmethod
    def _clamp_int(v, lo, hi):
        return int(max(lo, min(hi, int(v))))

    def show(
        self,
        play_gray: np.ndarray,
        heatmap_logits: torch.Tensor,
        xy_norm: tuple[float, float],
        conf: float,
        reward: float | None = None,
        total_reward: float | None = None,
        crop_size: int | None = None,          # ✅ NEW: 현재 obs crop 크기(원본 픽셀 기준)
    ):
        """
        play_gray:
          - 보통 screen.get_playfield_gray()의 결과(플레이 영역 gray, 원본 해상도에 가까움)를 넣는다고 가정.
          - 만약 play_gray가 이미 84x84 같은 "리사이즈된 관측"이면 crop_size 사각형은 의미가 약해짐.

        xy_norm:
          - playfield 기준 0..1 정규화 좌표라고 가정(ObsBuilder.last_xy_norm과 동일).
        """
        self._ensure_window()

        disp = play_gray
        if disp.dtype != np.uint8:
            disp = (np.clip(disp, 0, 1) * 255).astype(np.uint8)
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

        H, W = disp.shape[:2]
        self._ensure_size(W, H)

        # ---- marker position ----
        x_n = float(np.clip(xy_norm[0], 0.0, 1.0))
        y_n = float(np.clip(xy_norm[1], 0.0, 1.0))
        px = int(x_n * W)
        py = int(y_n * H)

        marker_size = max(24, int(min(W, H) * 0.05))
        circle_r = max(10, int(min(W, H) * 0.02))
        thickness = 2

        # ✅ (NEW) crop rectangle
        # crop_size는 "playfield 좌표계(픽셀)" 기준이라고 가정.
        # play_gray가 원본 플레이필드 크기면 그대로 표시하면 됨.
        if crop_size is not None:
            try:
                cs = int(crop_size)
                half = cs // 2
                x1 = self._clamp_int(px - half, 0, W - 1)
                y1 = self._clamp_int(py - half, 0, H - 1)
                x2 = self._clamp_int(px + half, 0, W - 1)
                y2 = self._clamp_int(py + half, 0, H - 1)
                # 너무 작은 사각형이면 안 그리기(의미 없음 방지)
                if (x2 - x1) >= 1 and (y2 - y1) >= 1:
                    # 외곽선 + 코너 강조 느낌
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 200, 0), 2)
                    cv2.rectangle(disp, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (0, 80, 255), 1)

                    # crop_size 텍스트
                    font_scale_cs = max(0.55, min(1.2, H / 700.0))
                    cv2.putText(
                        disp, f"crop={cs}",
                        (x1 + 6, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_cs,
                        (255, 255, 255), 3, cv2.LINE_AA
                    )
                    cv2.putText(
                        disp, f"crop={cs}",
                        (x1 + 6, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_cs,
                        (0, 0, 0), 1, cv2.LINE_AA
                    )
            except Exception:
                pass

        # player marker
        cv2.drawMarker(disp, (px, py), (0, 255, 0), cv2.MARKER_CROSS, marker_size, thickness)
        cv2.circle(disp, (px, py), circle_r, (0, 255, 0), thickness)

        font_scale = max(0.7, min(1.6, H / 450.0))
        line_h = int(30 * font_scale)

        txt1 = f"x={x_n:.3f} y={y_n:.3f} conf={float(conf):.4f}"
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
            txt3 = f"episode_total={float(total_reward):+.1f}"
            cv2.putText(disp, txt3, (10, y3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(disp, txt3, (10, y3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow(self.win_main, disp)

        # --- heatmap window ---
        hm = torch.sigmoid(heatmap_logits)[0, 0].detach().cpu().numpy()
        hm_u8 = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
        hm_show = cv2.resize(hm_u8, (W, H), interpolation=cv2.INTER_NEAREST)

        if reward is not None:
            cv2.putText(hm_show, f"r={float(reward):+.3f}", (10, 10 + line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        255, 3, cv2.LINE_AA)
            cv2.putText(hm_show, f"r={float(reward):+.3f}", (10, 10 + line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        0, 1, cv2.LINE_AA)

        cv2.imshow(self.win_hm, hm_show)
        cv2.waitKey(1)
