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
        # Shaping / penalty 설정
        # =========================
        self.shaping_cfg = ShapingConfig(
            target_y_ratio=0.78,
            shaping_k=0.35,
            shaping_clip=0.25,
            stuck_dist_px=2,
            stuck_need=10,
            stuck_pen=0.20,
            edge_guard_px=24,
            edge_guard_pen=0.08,
            abs_y_penalty_k=0.06,
            top_limit_px=None,
            top_soft_band_px=160,
            top_soft_pen=0.08,
        )
        self.pos_shaper = PositionShaper(self.screen, self.s, self.shaping_cfg)

        # =========================
        # Action Masking 설정
        # =========================
        self.mask_cfg = MaskingConfig(
            margin_px=200,
            use_flip=True,
            top_limit_px=None,
            top_limit_fudge_px=10,
        )
        self.masker = ActionMasker(self.screen, self.obs, self.mask_cfg)

        # 실행 기록(학습/디버그용)
        self.s.exec_action_idx = 0
        self.s.exec_was_masked = False

        # 에피소드 누적 점수(디버그 표시용)
        self.s.ep_total_reward = 0.0

        self.show_reimu_debug = True
        self.reimu_debug = ReimuDebugViz()

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
        ui0 = self.ui.ui_lives_safe(img, ui_ok)
        self.s.prev_ui_lives = ui0

        self.s.frame_stack.clear()
        for _ in range(self.s.frame_stack_size):
            self.s.frame_stack.append(state)

        stacked = np.stack(self.s.frame_stack, axis=0)

        # shaping state reset
        self.pos_shaper.reset()

        # masking 실행 기록
        self.s.exec_action_idx = 0
        self.s.exec_was_masked = False

        # ✅ 에피소드 누적 점수 리셋
        self.s.ep_total_reward = 0.0

        if hasattr(self, "obs") and hasattr(self.obs, "reset"):
            self.obs.reset()

        release_all()
        set_attack_hold(True)
        return stacked

    def _ep_add(self, x: float):
        """에피소드 누적 점수(디버그용) 안전 누적."""
        try:
            self.s.ep_total_reward += float(x)
        except Exception:
            pass

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

        # 로비/타이틀 선검사
        pre_img = self.screen.capture()
        ui_ok = self.ui.ui_panel_present(pre_img)
        self.ui.update_ui_absent(ui_ok)

        if self.s.ui_absent_count >= self.s.ui_absent_needed:
            print("[DEBUG] EPISODE END: UI panel absent -> lobby/title detected")
            self.guard.set_terminated()
            self.s.frame_stack.append(self.s.prev_state)
            stacked_state = np.stack(self.s.frame_stack, axis=0)

            pen = -100.0
            self._ep_add(pen)  # ✅ 에피소드 누적에도 반영
            return stacked_state, pen, True

        # step 시작 마스킹
        masked_idx, was_masked, _ = self.masker.apply_action_mask(action_idx, pre_img)

        action = ACTIONS[masked_idx]
        self.s.exec_action_idx = int(masked_idx)
        self.s.exec_was_masked = bool(was_masked)

        release_all()
        press_keys(action.value)

        total_reward = 0.0
        danger_sum = 0.0
        is_slow = action.name.startswith("SLOW")
        force_debug = False

        if self.s.prev_action_idx == action_idx:
            self.s.same_action_count += 1
        else:
            self.s.same_action_count = 0
        self.s.prev_action_idx = action_idx

        # action_repeat 루프
        for _ in range(self.s.action_repeat):
            time.sleep(self.s.frame_sleep)

            img = self.screen.capture()
            ui_ok = self.ui.ui_panel_present(img)
            self.ui.update_ui_absent(ui_ok)

            if self.s.ui_absent_count >= self.s.ui_absent_needed:
                print("[DEBUG] EPISODE END: UI panel absent -> lobby/title detected")
                self.guard.set_terminated()
                for _ in range(5):
                    release_all()
                    time.sleep(0.02)

                pen = -100.0
                total_reward += pen
                self._ep_add(pen)  # ✅ 에피소드 누적에도 반영
                force_debug = True

                self.s.frame_stack.append(self.s.prev_state)
                stacked_state = np.stack(self.s.frame_stack, axis=0)
                return stacked_state, float(total_reward), True

            # 관측/트래킹 업데이트
            state = self.obs.make_state(img)

            # 매 프레임 마스킹 재적용
            cur_idx, cur_was_masked, _ = self.masker.apply_action_mask(masked_idx, img)
            if cur_idx != masked_idx:
                masked_idx = cur_idx
                action = ACTIONS[masked_idx]
                self.s.exec_action_idx = int(masked_idx)
                self.s.exec_was_masked = True
                release_all()
                press_keys(action.value)

            # ===== 여기부터 reward 계산 =====
            reward = 0.1
            now = time.time()

            # 게임오버 flash 감지
            hit_fx, gameover_fx = self.screen.detect_death(img)
            if gameover_fx:
                print("[DEBUG] GAME OVER! (flash detected)")
                self.s.lives = 0
                self.guard.set_terminated()
                for _ in range(3):
                    release_all()
                    time.sleep(0.02)

                pen = -100.0
                total_reward += pen
                self._ep_add(pen)  # ✅ 에피소드 누적에도 반영
                force_debug = True

                self.s.frame_stack.append(self.s.prev_state)
                stacked_state = np.stack(self.s.frame_stack, axis=0)
                return stacked_state, float(total_reward), True

            ui_now = self.ui.ui_lives_safe(img, ui_ok)

            play = self.screen.get_playfield_gray(img)
            danger, edge_r, bright_r, std_n = self.screen.danger_from_playfield(play, return_parts=True)
            danger_sum += danger

            # 정지 화면 패널티
            motion_energy = float(np.abs(state - self.s.prev_state).mean())
            if motion_energy < 0.002:
                reward -= 0.03

            # 위치 기반 shaping/패널티
            reward += self.pos_shaper.step_reward(img, self.obs.player_center)

            # UI 기반 피격 감지
            if (ui_now is not None) and (now - self.s.last_hit_time) > self.s.hit_cooldown:
                if self.s.prev_ui_lives is not None and ui_now < self.s.prev_ui_lives:
                    self.s.lives -= 1
                    self.s.last_hit_time = now

                    if self.s.lives <= 0:
                        print("[DEBUG] GAME OVER! (ui last life lost)")
                        self.guard.set_terminated()
                        for _ in range(3):
                            release_all()
                            time.sleep(0.02)

                        pen = -100.0
                        total_reward += pen
                        self._ep_add(pen)  # ✅ 에피소드 누적에도 반영
                        force_debug = True

                        self.s.frame_stack.append(self.s.prev_state)
                        stacked_state = np.stack(self.s.frame_stack, axis=0)
                        return stacked_state, float(total_reward), True

                    try:
                        trk = getattr(self.obs, "tracker", None)
                        if trk is not None and hasattr(trk, "on_player_death"):
                            trk.on_player_death()
                    except Exception as e:
                        print(f"[WARN] tracker.on_player_death failed: {e}")

                    # ✅ 피격 프레임 reward는 강제 -50
                    reward = -50.0
                    print(f"[DEBUG] HIT! (ui) internal lives={self.s.lives}")
                    force_debug = True

            self.s.prev_ui_lives = ui_now

            # ✅ 디버그 표시: "프레임 reward 최종값" + "에피소드 누적(이번 프레임 더하기 전 값)"
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

            # 내부 lives가 0이면 종료 패널티(-50) 추가 후 break
            if self.s.lives <= 0:
                print("[DEBUG] GAME OVER! internal lives=0")
                self.guard.set_terminated()

                pen = -50.0
                total_reward += pen
                self._ep_add(pen)  # ✅ 에피소드 누적에도 반영
                force_debug = True
                break

            # 정상 프레임 누적
            self.s.prev_state = state
            total_reward += reward
            self._ep_add(reward)  # ✅ 매 프레임 에피소드 누적

        avg_danger = danger_sum / max(1, self.s.action_repeat)
        total_reward = self.reward_engine.postprocess(total_reward, avg_danger, is_slow)

        self.s.step_i += 1
        # (print 디버그는 네가 주석 처리해둔 상태 유지)

        self.s.frame_stack.append(self.s.prev_state)
        stacked_state = np.stack(self.s.frame_stack, axis=0)
        return stacked_state, float(total_reward), False
