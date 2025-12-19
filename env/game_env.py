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
from env.position_shaping import PositionShaper, ShapingConfig


class GameEnv:
    """
    ✅ 회피 우선 버전 (단순/빠른 학습 목적)
    - 마지막 죽음은 -100만 적용(추가 -50 없음)
    - danger 기반 dense reward 강화:
        r -= k_abs * danger
        r += k_delta * clip(prev_danger - danger)
    - shaping은 기본 OFF(원하면 나중에 다시 켜기)
    """

    def __init__(self, screen_mode="low"):
        self.screen = Screen(mode=screen_mode)

        self.s = EnvState()
        self.guard = EpisodeGuard(self.s)
        self.ui = UIGuard(self.screen, self.s)
        self.reward_engine = RewardEngine(self.s)

        self.debug = DebugViz()
        self.obs = ObsBuilder(
            self.screen,
            debug_viz=self.debug,
            obs_out_size=84,
            crop_size=160,
            use_fallback_full_preprocess=True
        )

        # =========================
        # ✅ 회피 우선 reward 파라미터
        # =========================
        self.alive_reward = 0.01            # 너무 크면 '가만히 있어도 이득'이 됨
        self.k_danger_abs = 0.35            # danger 자체 패널티(즉시 위험하면 손해)
        self.k_danger_delta = 1.50          # danger 감소 보상(회피 학습 핵심, 눈에 띄게!)
        self.danger_delta_clip = 0.30       # delta 튀는 것 방지

        self.motion_th = 0.0018
        self.motion_pen = 0.01

        self.hit_pen = -50.0               # 1~2번째 피격 패널티
        self.gameover_pen = -100.0         # ✅ 마지막은 -100만
        self.lobby_pen = -100.0

        # =========================
        # shaping: 기본 OFF (나중에 켜기)
        # =========================
        self.shaping_cfg = ShapingConfig(
            target_y_ratio=0.78,
            shaping_k=0.0,
            shaping_clip=0.0,
            stuck_dist_px=2,
            stuck_need=999999,
            stuck_pen=0.0,
            edge_guard_px=0,
            edge_guard_pen=0.0,
            abs_y_penalty_k=0.0,
            top_limit_px=None,
            top_soft_band_px=0,
            top_soft_pen=0.0,
        )
        self.pos_shaper = PositionShaper(self.screen, self.s, self.shaping_cfg)

        # =========================
        # Action Masking: 회피 방해 줄이기
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

        # danger delta용
        self.s.prev_danger = None

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

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return lo if x < lo else (hi if x > hi else x)

    def _end_episode(self, pen: float):
        """공통 종료 처리(누적 반영 + 스택 + done 리턴 준비)"""
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

        self.pos_shaper.reset()
        self.s.exec_action_idx = 0
        self.s.exec_was_masked = False

        self.s.ep_total_reward = 0.0
        self.s.prev_danger = None

        if hasattr(self.obs, "reset"):
            self.obs.reset()

        release_all()
        set_attack_hold(True)
        return np.stack(self.s.frame_stack, axis=0)

    def step(self, action_idx):
        # 종료 후 입력 차단
        if self.s.episode_terminated:
            self.guard.terminated_step_return()
            set_attack_hold(False)
            for _ in range(6):
                release_all()
                time.sleep(0.02)
            stacked_state = np.stack(self.s.frame_stack, axis=0)
            return stacked_state, 0.0, True

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
        danger_sum = 0.0
        is_slow = action.name.startswith("SLOW")

        # action_repeat 루프
        for _ in range(self.s.action_repeat):
            time.sleep(self.s.frame_sleep)

            img = self.screen.capture()
            ui_ok = self.ui.ui_panel_present(img)
            self.ui.update_ui_absent(ui_ok)
            if self.s.ui_absent_count >= self.s.ui_absent_needed:
                # 즉시 종료(-100)
                self.guard.set_terminated()
                for _ in range(5):
                    release_all()
                    time.sleep(0.02)
                total_reward += float(self.lobby_pen)
                self._ep_add(self.lobby_pen)
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

            # -------------------------
            # reward 계산(회피 우선)
            # -------------------------
            reward = float(self.alive_reward)
            now = time.time()

            # flash gameover (-100)
            _, gameover_fx = self.screen.detect_death(img)
            if gameover_fx:
                for _ in range(3):
                    release_all()
                    time.sleep(0.02)
                total_reward += float(self.gameover_pen)
                self._ep_add(self.gameover_pen)
                self.guard.set_terminated()
                self.s.frame_stack.append(self.s.prev_state)
                return np.stack(self.s.frame_stack, axis=0), float(total_reward), True

            # danger + delta reward
            play = self.screen.get_playfield_gray(img)
            danger, _, _, _ = self.screen.danger_from_playfield(play, return_parts=True)
            danger = float(danger)
            danger_sum += danger

            # (1) 위험 자체 패널티
            reward -= float(self.k_danger_abs * danger)

            # (2) 위험 감소 보상 (더 눈에 띄게!)
            d_delta = 0.0
            if self.s.prev_danger is not None:
                d_delta = float(self.s.prev_danger - danger)  # 위험 줄이면 +
                d_delta = self._clip(d_delta, -self.danger_delta_clip, self.danger_delta_clip)
                reward += float(self.k_danger_delta * d_delta)
            self.s.prev_danger = danger

            # 정지 패널티(약하게)
            motion_energy = float(np.abs(state - self.s.prev_state).mean())
            if motion_energy < self.motion_th:
                reward -= float(self.motion_pen)

            # (선택) shaping은 지금 OFF지만 구조는 유지
            # reward += self.pos_shaper.step_reward(img, self.obs.player_center)

            # UI 피격 감지
            ui_now = self.ui.ui_lives_safe(img, ui_ok)
            if (ui_now is not None) and (now - self.s.last_hit_time) > self.s.hit_cooldown:
                if self.s.prev_ui_lives is not None and ui_now < self.s.prev_ui_lives:
                    self.s.lives -= 1
                    self.s.last_hit_time = now

                    if self.s.lives <= 0:
                        # ✅ 마지막은 -100만 (단순)
                        for _ in range(3):
                            release_all()
                            time.sleep(0.02)
                        total_reward += float(self.gameover_pen)
                        self._ep_add(self.gameover_pen)
                        self.guard.set_terminated()
                        self.s.frame_stack.append(self.s.prev_state)
                        return np.stack(self.s.frame_stack, axis=0), float(total_reward), True

                    # 1~2번째는 -50
                    try:
                        trk = getattr(self.obs, "tracker", None)
                        if trk is not None and hasattr(trk, "on_player_death"):
                            trk.on_player_death()
                    except Exception as e:
                        print(f"[WARN] tracker.on_player_death failed: {e}")

                    reward = float(self.hit_pen)

            self.s.prev_ui_lives = ui_now

            # 디버그 표시 (reward/total + danger/d_delta도 같이 넘김)
            if self.show_reimu_debug:
                dbg = getattr(self.obs, "_dbg_last", None)
                if dbg is not None:
                    x_n, y_n, conf, logits = dbg
                    play_dbg = self.screen.get_playfield_gray(img)
                    # ReimuDebugViz가 extra 인자를 아직 안 받으면, 아래 두 줄(danger=..., d_delta=...)은 지워도 됨
                    self.reimu_debug.show(
                        play_gray=play_dbg,
                        heatmap_logits=logits,
                        xy_norm=(x_n, y_n),
                        conf=conf,
                        reward=reward,
                        total_reward=self.s.ep_total_reward,
                        # danger=danger,
                        # d_delta=d_delta,
                    )

            # 누적
            self.s.prev_state = state
            total_reward += reward
            self._ep_add(reward)

        # 후처리: 회피만 보고 싶으면 꺼도 됨(현재는 원본 유지)
        avg_danger = danger_sum / max(1, self.s.action_repeat)
        total_reward = self.reward_engine.postprocess(total_reward, avg_danger, is_slow)

        self.s.step_i += 1
        self.s.frame_stack.append(self.s.prev_state)
        return np.stack(self.s.frame_stack, axis=0), float(total_reward), False
