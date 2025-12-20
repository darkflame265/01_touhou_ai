import time
import numpy as np

from env.screen import Screen
from env.controller import press_keys, set_attack_hold, release_all, set_always_slow

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
    ✅ 루나틱 회피 학습용 (점수 안정화 버전)

    목표:
    - 에피소드 종료 패널티는 "딱 1개만" 적용 (불확실성 제거)
      * death_pen: 죽음으로 종료(피격 포함, flash 포함) -> 동일 패널티
      * abort_pen: 로비/타이틀/ABORTED 등 비정상 종료 -> 동일 패널티

    - hit_pen은 "목숨 감소가 확실할 때"만 1회 적용
    - alive + shaping은 매 프레임 누적
    """

    def __init__(self, screen_mode="low"):
        self.screen = Screen(mode=screen_mode)

        self.s = EnvState()
        self.guard = EpisodeGuard(self.s)
        self.ui = UIGuard(self.screen, self.s)
        self.reward_engine = RewardEngine(self.s)

        # 반응속도
        self.s.action_repeat = 1
        self.s.frame_sleep = 0.012

        # 관측
        self.debug = DebugViz()
        self.obs = ObsBuilder(
            self.screen,
            debug_viz=self.debug,
            obs_out_size=128,
            crop_size=256,
            use_fallback_full_preprocess=True,
        )

        # =========================
        # ✅ Reward (안정화)
        # =========================
        self.alive_reward = 0.03

        # "목숨 감소 1회" 페널티 (에피소드 종료와 별개)
        self.hit_pen = -15.0

        # ✅ 에피소드 종료 패널티는 2개로 단순화
        self.death_pen = -60.0   # 죽음 종료(최종)
        self.abort_pen = -80.0   # 로비/타이틀/중단

        # =========================
        # ✅ 위치 shaping (아래쪽 유지)
        # =========================
        self.use_position_shaping = True
        self.y_floor = 0.60
        self.y_floor_pen_k = 0.05

        # (선택) 기존 우상단 억제
        self.top_soft_y = 0.20
        self.right_soft_x = 0.80
        self.top_pen_k = 0.020
        self.right_pen_k = 0.010
        self.corner_bonus_pen = 0.015

        # Action Masking
        self.mask_cfg = MaskingConfig(
            margin_px=90,
            use_flip=True,
            top_limit_px=None,
            top_limit_fudge_px=10,
        )
        self.masker = ActionMasker(self.screen, self.obs, self.mask_cfg)

        # 기록/디버그
        self.s.exec_action_idx = 0
        self.s.exec_was_masked = False
        self._masked_count = 0
        self._step_count = 0

        self.s.ep_total_reward = 0.0
        self.s.episode_end_reason = ""
        self.s.episode_end_pen = 0.0

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

    def _end_episode(self, pen: float, reason: str):
        # ✅ 종료 패널티 단 1회만!
        self.guard.set_terminated()
        self.s.episode_end_reason = str(reason)
        self.s.episode_end_pen = float(pen)

        self._ep_add(pen)

        self.s.frame_stack.append(self.s.prev_state)
        stacked_state = np.stack(self.s.frame_stack, axis=0)
        return stacked_state, float(pen), True

    def _get_playfield_xy_norm_for_debug(self):
        dbg = getattr(self.obs, "_dbg_last", None)
        if dbg is None:
            return None
        try:
            if len(dbg) >= 6:
                x_lock, y_lock, conf, logits, x_raw, y_raw = dbg[:6]
                return float(x_lock), float(y_lock), float(conf), logits, float(x_raw), float(y_raw)
            else:
                x_lock, y_lock, conf, logits = dbg
                return float(x_lock), float(y_lock), float(conf), logits, None, None
        except Exception:
            return None

    def _position_shaping_penalty(self, x_n: float, y_n: float) -> float:
        if not self.use_position_shaping:
            return 0.0

        pen = 0.0

        # 아래쪽 유지
        if y_n < self.y_floor:
            d = (self.y_floor - y_n) / max(1e-6, self.y_floor)
            pen -= self.y_floor_pen_k * float(d)

        # (선택) 우상단 억제
        if y_n < self.top_soft_y:
            d = (self.top_soft_y - y_n) / max(1e-6, self.top_soft_y)
            pen -= self.top_pen_k * float(d)

        if x_n > self.right_soft_x:
            d = (x_n - self.right_soft_x) / max(1e-6, (1.0 - self.right_soft_x))
            pen -= self.right_pen_k * float(d)

        if (y_n < self.top_soft_y) and (x_n > self.right_soft_x):
            pen -= float(self.corner_bonus_pen)

        return float(pen)

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

        self.s.episode_end_reason = ""
        self.s.episode_end_pen = 0.0
        self.s.ep_total_reward = 0.0

        img = self.screen.capture()

        state = self.obs.make_state(img)
        self.s.prev_state = state

        ui_ok = self.ui.ui_panel_present(img)
        self.s.prev_ui_lives = self.ui.ui_lives_safe(img, ui_ok)

        self.s.frame_stack.clear()
        for _ in range(self.s.frame_stack_size):
            self.s.frame_stack.append(state)

        self.s.exec_action_idx = 0
        self.s.exec_was_masked = False
        self._masked_count = 0
        self._step_count = 0

        if hasattr(self.obs, "reset"):
            self.obs.reset()

        release_all()
        set_attack_hold(True)
        set_always_slow(True)   # ✅ 항상 SLOW

        return np.stack(self.s.frame_stack, axis=0)

    def step(self, action_idx):
        if self.s.episode_terminated:
            self.guard.terminated_step_return()
            set_attack_hold(False)
            for _ in range(6):
                release_all()
                time.sleep(0.02)
            return np.stack(self.s.frame_stack, axis=0), 0.0, True

        # ---------
        # 로비/타이틀(Abort)
        # ---------
        pre_img = self.screen.capture()
        ui_ok = self.ui.ui_panel_present(pre_img)
        self.ui.update_ui_absent(ui_ok)
        if self.s.ui_absent_count >= self.s.ui_absent_needed:
            return self._end_episode(self.abort_pen, "ABORT:UI_ABSENT(pre)")

        # 입력 + 초기 마스킹
        masked_idx, was_masked, _ = self.masker.apply_action_mask(action_idx, pre_img)
        action = ACTIONS[masked_idx]
        self.s.exec_action_idx = int(masked_idx)
        self.s.exec_was_masked = bool(was_masked)
        if was_masked:
            self._masked_count += 1

        release_all()
        press_keys(action.value)

        total_reward = 0.0

        for _ in range(self.s.action_repeat):
            time.sleep(self.s.frame_sleep)

            img = self.screen.capture()
            ui_ok = self.ui.ui_panel_present(img)
            self.ui.update_ui_absent(ui_ok)
            if self.s.ui_absent_count >= self.s.ui_absent_needed:

                for _ in range(5):
                    release_all()
                    time.sleep(0.02)
                # ✅ abort로 종료 패널티 1회
                pen_state, pen_reward, done = self._end_episode(self.abort_pen, "ABORT:UI_ABSENT(loop)")
                total_reward += pen_reward
                return pen_state, float(total_reward), True

            # 관측 업데이트
            state = self.obs.make_state(img)

            # 마스킹 재적용
            cur_idx, cur_was_masked, _ = self.masker.apply_action_mask(masked_idx, img)
            if cur_idx != masked_idx:
                masked_idx = cur_idx
                action = ACTIONS[masked_idx]
                self.s.exec_action_idx = int(masked_idx)
                self.s.exec_was_masked = True
                self._masked_count += 1
                release_all()
                press_keys(action.value)
            elif cur_was_masked:
                self.s.exec_was_masked = True
                self._masked_count += 1

            # ----------
            # 매 프레임 reward
            # ----------
            reward = float(self.alive_reward)
            now = time.time()

            pos_dbg = self._get_playfield_xy_norm_for_debug()
            if pos_dbg is not None:
                x_lock, y_lock, conf, logits, x_raw, y_raw = pos_dbg
                reward += self._position_shaping_penalty(x_lock, y_lock)

            # ----------
            # ✅ death 판정 1: flash gameover
            # ----------
            _, gameover_fx = self.screen.detect_death(img)
            if gameover_fx:
                for _ in range(3):
                    release_all()
                    time.sleep(0.02)
                # ✅ death로 종료 패널티 1회
                pen_state, pen_reward, done = self._end_episode(self.death_pen, "DEATH:FLASH")
                total_reward += (reward + pen_reward)  # 마지막 프레임 보상 + 종료 패널티
                self.s.prev_state = state
                self.s.frame_stack.append(self.s.prev_state)
                return np.stack(self.s.frame_stack, axis=0), float(total_reward), True

            # ----------
            # ✅ hit 판정: UI lives 감소
            # ----------
            ui_now = self.ui.ui_lives_safe(img, ui_ok)
            if (ui_now is not None) and (now - self.s.last_hit_time) > self.s.hit_cooldown:
                if self.s.prev_ui_lives is not None and ui_now < self.s.prev_ui_lives:
                    self.s.lives -= 1
                    self.s.last_hit_time = now

                    # ✅ 목숨 감소는 항상 hit_pen 1회만
                    reward = float(self.hit_pen)

                    # 마지막 목숨이면 죽음 종료 (death_pen 1회)
                    if self.s.lives <= 0:
                        for _ in range(3):
                            release_all()
                            time.sleep(0.02)
                        pen_state, pen_reward, done = self._end_episode(self.death_pen, "DEATH:LIVES0")
                        total_reward += (reward + pen_reward)
                        self.s.prev_state = state
                        self.s.frame_stack.append(self.s.prev_state)
                        return np.stack(self.s.frame_stack, axis=0), float(total_reward), True

                    # 피격 후 detector lock 해제
                    try:
                        if hasattr(self.obs, "on_player_death"):
                            self.obs.on_player_death()
                    except Exception as e:
                        print(f"[WARN] obs.on_player_death failed: {e}")

            self.s.prev_ui_lives = ui_now

            # 디버그 표시
            if self.show_reimu_debug:
                dbg = getattr(self.obs, "_dbg_last", None)
                if dbg is not None:
                    if len(dbg) >= 6:
                        x_n, y_n, conf, logits, x_raw, y_raw = dbg[:6]
                        xy_for_viz = (x_raw, y_raw)
                    else:
                        x_n, y_n, conf, logits = dbg
                        xy_for_viz = (x_n, y_n)

                    play_dbg = self.screen.get_playfield_gray(img)
                    self.reimu_debug.show(
                        play_gray=play_dbg,
                        heatmap_logits=logits,
                        xy_norm=xy_for_viz,
                        conf=conf,
                        reward=reward,
                        total_reward=self.s.ep_total_reward,
                    )

            self.s.prev_state = state
            total_reward += float(reward)
            self._ep_add(float(reward))
            self._step_count += 1

        self.s.step_i += 1
        self.s.frame_stack.append(self.s.prev_state)
        return np.stack(self.s.frame_stack, axis=0), float(total_reward), False
