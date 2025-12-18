# env/ui_guard.py
import numpy as np
from env.ui_lives import count_lives_from_img

class UIGuard:
    def __init__(self, screen, state):
        self.screen = screen
        self.s = state

    def ui_panel_present(self, img_bgr) -> bool:
        return bool(self.screen.ui_panel_present(img_bgr))

    def update_ui_absent(self, ui_ok: bool):
        if not ui_ok:
            self.s.ui_absent_count += 1
        else:
            self.s.ui_absent_count = 0

    def ui_lives_safe(self, img_bgr, ui_panel_ok: bool):
        if not ui_panel_ok:
            return None
        try:
            v = count_lives_from_img(img_bgr)
        except Exception:
            return None
        if v is None:
            return None
        if not isinstance(v, (int, np.integer)):
            return None
        if v < 0 or v > 12:
            return None
        return int(v)
