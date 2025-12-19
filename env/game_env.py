# env/game_env.py
import time
import numpy as np

from env.screen import Screen
from env.controller import press_keys, set_attack_hold, release_all
from env.actions import ACTIONS

from env.env_state import EnvState
from env.episode_guard import EpisodeGuard
from env.ui_guard import UIGuard
from env.reward_engine import RewardEngine
from env.debug_viz import DebugViz
from env.obs_builder import ObsBuilder
from env.reimu_debug_viz import ReimuDebugViz

from env.action_masking import ActionMasker, MaskingConfig


class GameEnv:
    """
    ✅ 초간단 회피 학습 버전 + 반응속도/관측 개선
    - 보상: 생존 시 매 프레임 아주 소량(+alive_reward)
    - 감점: 피격 시 -50, 마지막(게임오버) -100
    - danger / delta reward 제거
    - shaping 제거
    - ✅ action_repeat 줄이고 frame_sleep 줄여서 반응속도 올림
    - ✅ obs 해상도/크롭 키워서 총알이 보이게 함
    """

    def __init__(self, screen_mode="low"):
        self.screen = Screen(mode=screen_mode)

        self.s = EnvState()
        self.guard = EpisodeGuard(self.s)
        self.ui = UIGuard(self.screen, self.s)
        self.reward_engine = RewardEngine(self.s)

        # =========================
        # ✅ 반응속도 튜닝 (중요)
        # =========================
        # EnvState 기본값을 여기서 강제로 덮어씀
        self.s.action_repeat = 1          # ✅ 1이 가장 빠름 (추천 시작값)
        self.s.frame_sleep = 0.012        # ✅ 0.010~0.016 권장(너무 낮으면 불안정 가능)

        # =========================
        # ✅ 관측 튜닝 (총알이 보이게)
        # =========================
        self.debug = DebugViz()
        self.obs = ObsBuilder(
            self.screen,
            debug_viz=self.debug,
            obs_out_size=84,          # ✅ 다시 84로 (shape 고정)
            crop_size=256,            # ✅ 총알 보이게 크롭만 키움
            use_fallback_full_preprocess=True
        )

        # =========================
        # ✅ 최소 reward 구성
        # =========================
        self.alive_reward = 0.005   # 0.002~0.01 사이에서 조절
        self.hit_pen = -50.0
        self.gameover_pen = -100.0
        self.lobby_pen = -100.0

        # =========================
        # ✅ Action Masking (벽 박기 정도만 방지)
        # =========================
        self.mask_cfg = MaskingConfig(
            margin_px=90,     # 60~120 권장
            use_flip=True,
            top_limit_px=None,
            top_limit_fudge_px=10,
        )
        self.masker = ActionMasker(self.screen, self.obs, self.mask_cfg)

        # 실행 기록(학습/디버그용)
        self.s.exec_action_idx = 0
        self.s.exec_was_masked = False

        # 에피소드 누적(디버그용)
        self.s.ep_total_reward = 0.0

        self.show_reimu_debug = True
        self.reimu_debug = ReimuDebugViz()

    # -------------------------
    # Utils
    # -------------------------
    def _ep_add(self, x: float):
        try:
            self.s.ep_total_reward += float(x)
        except Exception:
            pass

    def _end_episode(self, pen: float):
        self.guard.set_terminated()
        self._ep_add(pen)
        self.s.frame_stack.append(self.s.prev_state)
        stacked_state = np.stack(self.s.frame_stack, axis=0)
        return stacked_state, float(pen), True

    # -------------------------
    # Gym API
    # -------------------------
    def reset(self):
        release_all()
        time.sleep(0.5)

        self.s.lives = 3
        self.s.last_hit_time = 0.0
        self.s.slow_streak = 0
        self.s.step_i = 0
        self.s.ui_absent_count = 0
        self.s.episode_terminated = False
        self.s.terminate_until = 0.0
        self.s.prev_action_idx = None
        self.s.same_action_count = 0

        img = self.screen.capture()

        state = self.obs.make_state(img)
        self.s.prev_state = state

        ui_ok = self.ui.ui_panel_present(img)
        self.s.prev_ui_lives = self.ui.ui_lives_safe(img, ui_ok)

        self.s.frame_stack.clear()
        for _ in range(self.s.frame_stack_size):
            self.s.frame_stack.append(state)

        # masking 실행 기록
        self.s.exec_action_idx = 0
        self.s.exec_was_masked = False

        # 에피소드 누적 점수 리셋
        self.s.ep_total_reward = 0.0

        if hasattr(self.obs, "reset"):
            self.obs.reset()

        release_all()
        set_attack_hold(True)
        return np.stack(self.s.frame_stack, axis=0)

    def step(self, action_idx):
        if self.s.episode_terminated:
            self.guard.terminated_step_return()
            set_attack_hold(False)
            for _ in range(6):
                release_all()
                time.sleep(0.02)
            return np.stack(self.s.frame_stack, axis=0), 0.0, True

        # 로비/타이틀 검사
        pre_img = self.screen.capture()
        ui_ok = self.ui.ui_panel_present(pre_img)
        self.ui.update_ui_absent(ui_ok)
        if self.s.ui_absent_count >= self.s.ui_absent_needed:
            return self._end_episode(self.lobby_pen)

        # step 시작 마스킹 + 입력
        masked_idx, was_masked, _ = self.masker.apply_action_mask(action_idx, pre_img)
        action = ACTIONS[masked_idx]
        self.s.exec_action_idx = int(masked_idx)
        self.s.exec_was_masked = bool(was_masked)

        release_all()
        press_keys(action.value)

        total_reward = 0.0

        for _ in range(self.s.action_repeat):
            time.sleep(self.s.frame_sleep)

            img = self.screen.capture()
            ui_ok = self.ui.ui_panel_present(img)
            self.ui.update_ui_absent(ui_ok)
            if self.s.ui_absent_count >= self.s.ui_absent_needed:
                self.guard.set_terminated()
                for _ in range(5):
                    release_all()
                    time.sleep(0.02)
                pen = float(self.lobby_pen)
                total_reward += pen
                self._ep_add(pen)
                self.s.frame_stack.append(self.s.prev_state)
                return np.stack(self.s.frame_stack, axis=0), float(total_reward), True

            # 관측 업데이트
            state = self.obs.make_state(img)

            # 매 프레임 마스킹 재적용
            cur_idx, _, _ = self.masker.apply_action_mask(masked_idx, img)
            if cur_idx != masked_idx:
                masked_idx = cur_idx
                action = ACTIONS[masked_idx]
                self.s.exec_action_idx = int(masked_idx)
                self.s.exec_was_masked = True
                release_all()
                press_keys(action.value)

            reward = float(self.alive_reward)
            now = time.time()

            # flash gameover (-100)
            _, gameover_fx = self.screen.detect_death(img)
            if gameover_fx:
                for _ in range(3):
                    release_all()
                    time.sleep(0.02)
                pen = float(self.gameover_pen)
                total_reward += pen
                self._ep_add(pen)
                self.guard.set_terminated()
                self.s.frame_stack.append(self.s.prev_state)
                return np.stack(self.s.frame_stack, axis=0), float(total_reward), True

            # UI 피격 감지
            ui_now = self.ui.ui_lives_safe(img, ui_ok)
            if (ui_now is not None) and (now - self.s.last_hit_time) > self.s.hit_cooldown:
                if self.s.prev_ui_lives is not None and ui_now < self.s.prev_ui_lives:
                    self.s.lives -= 1
                    self.s.last_hit_time = now

                    if self.s.lives <= 0:
                        for _ in range(3):
                            release_all()
                            time.sleep(0.02)
                        pen = float(self.gameover_pen)
                        total_reward += pen
                        self._ep_add(pen)
                        self.guard.set_terminated()
                        self.s.frame_stack.append(self.s.prev_state)
                        return np.stack(self.s.frame_stack, axis=0), float(total_reward), True

                    # 1~2번째 -50
                    try:
                        trk = getattr(self.obs, "tracker", None)
                        if trk is not None and hasattr(trk, "on_player_death"):
                            trk.on_player_death()
                    except Exception as e:
                        print(f"[WARN] tracker.on_player_death failed: {e}")

                    reward = float(self.hit_pen)

            self.s.prev_ui_lives = ui_now

            # 디버그 표시
            if self.show_reimu_debug:
                dbg = getattr(self.obs, "_dbg_last", None)
                if dbg is not None:
                    x_n, y_n, conf, logits = dbg
                    play_dbg = self.screen.get_playfield_gray(img)
                    self.reimu_debug.show(
                        play_gray=play_dbg,
                        heatmap_logits=logits,
                        xy_norm=(x_n, y_n),
                        conf=conf,
                        reward=reward,
                        total_reward=self.s.ep_total_reward,
                    )

            self.s.prev_state = state
            total_reward += reward
            self._ep_add(reward)

        self.s.step_i += 1
        self.s.frame_stack.append(self.s.prev_state)
        return np.stack(self.s.frame_stack, axis=0), float(total_reward), False
