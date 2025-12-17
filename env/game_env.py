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

        release_all()
        set_attack_hold(True)

        return stacked

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
            return stacked_state, -100.0, True

        action = ACTIONS[action_idx]

        release_all()
        press_keys(action.value)

        total_reward = 0.0
        danger_sum = 0.0
        is_slow = action.name.startswith("SLOW")
        force_debug = False

        # (추가) 에피소드 카운트 변수 없으면 생성 (reset 안 건드려도 되게)
        if not hasattr(self.s, "edge60_cnt"):
            self.s.edge60_cnt = 0
        if not hasattr(self.s, "top270_cnt"):
            self.s.top270_cnt = 0

        # 같은 액션 반복 카운트
        if self.s.prev_action_idx == action_idx:
            self.s.same_action_count += 1
        else:
            self.s.same_action_count = 0
        self.s.prev_action_idx = action_idx

        for _ in range(self.s.action_repeat):
            time.sleep(self.s.frame_sleep)

            img = self.screen.capture()
            ui_ok = self.ui.ui_panel_present(img)
            self.ui.update_ui_absent(ui_ok)

            if self.s.ui_absent_count >= self.s.ui_absent_needed:
                print("[DEBUG] EPISODE END: UI panel absent -> lobby/title detected")
                self.guard.set_terminated()

                # 로비/타이틀로 이미 나갔으니 키가 남지 않게 확실히 해제
                for _ in range(5):
                    release_all()
                    time.sleep(0.02)

                total_reward += -100.0
                force_debug = True

                # done=True로 즉시 종료 (break 금지)
                self.s.frame_stack.append(self.s.prev_state)
                stacked_state = np.stack(self.s.frame_stack, axis=0)
                return stacked_state, float(total_reward), True


            state = self.obs.make_state(img)

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
                total_reward += -100.0
                force_debug = True
                self.s.frame_stack.append(self.s.prev_state)
                stacked_state = np.stack(self.s.frame_stack, axis=0)
                return stacked_state, float(total_reward), True

            ui_now = self.ui.ui_lives_safe(img, ui_ok)

            play = self.screen.get_playfield_gray(img)
            danger, edge_r, bright_r, std_n = self.screen.danger_from_playfield(play, return_parts=True)
            danger_sum += danger

            motion_energy = float(np.abs(state - self.s.prev_state).mean())
            if motion_energy < 0.002:
                reward -= 0.03

            # ===== Edge / Corner shaping (anti-wall policy) =====
            pc = self.obs.player_center
            edge_pen = 0.0

            if pc is not None:
                px, py = float(pc[0]), float(pc[1])

                pf = getattr(self.screen, "playfield_rect", None) \
                    or getattr(self.screen, "PLAYFIELD_RECT", None) \
                    or getattr(self.screen, "pf_rect", None)

                if pf is not None:
                    l, t, r, b = pf
                else:
                    l, t, r, b = self.screen.win_rect

                w = max(1.0, float(r - l))
                h = max(1.0, float(b - t))

                nx = (px - l) / w
                ny = (py - t) / h
                nx = 0.0 if nx < 0.0 else (1.0 if nx > 1.0 else nx)
                ny = 0.0 if ny < 0.0 else (1.0 if ny > 1.0 else ny)

                d_edge = min(nx, 1.0 - nx, ny, 1.0 - ny)

                margin = 0.12
                if d_edge < margin:
                    x = (margin - d_edge) / margin
                    edge_pen = -0.12 * (x * x)

                    if self.s.same_action_count >= 3:
                        edge_pen += -0.05
                        print(f"[EDGE] 현재 벽에 박고 있습니다. edge_pen={edge_pen}")

            reward += edge_pen

            # ===== edge_px <= 60 패널티 + 카운트 + print =====
            if pc is not None:
                px_i, py_i = int(pc[0]), int(pc[1])
                H, W = img.shape[:2]
                pf_r = int(W * float(self.screen.PLAYFIELD_RIGHT_RATIO))
                edge_px = min(px_i - 0, pf_r - px_i, py_i - 0, H - py_i)

                if edge_px <= 60:
                    edge60_pen = 0.12
                    reward -= edge60_pen
                    self.s.edge60_cnt += 1  # (추가) 카운트
                    print(f"[EDGE60] edge_px={edge_px}px <= 60 -> -{edge60_pen:.2f} (pos=({px_i},{py_i}), pf_r={pf_r}, H={H})")

                # ===== cy < 270 패널티 + 카운트 + print =====
                if py_i < 270:
                    top_pen = 0.10
                    reward -= top_pen
                    self.s.top270_cnt += 1  # (추가) 카운트
                    print(f"[TOP270] cy={py_i} < 270 -> -{top_pen:.2f} (pos=({px_i},{py_i}))")

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
                        total_reward += -100.0
                        force_debug = True
                        self.s.frame_stack.append(self.s.prev_state)
                        stacked_state = np.stack(self.s.frame_stack, axis=0)
                        return stacked_state, float(total_reward), True

                    reward = -10.0
                    print(f"[DEBUG] HIT! (ui) internal lives={self.s.lives}")
                    force_debug = True

            self.s.prev_ui_lives = ui_now

            if self.s.lives <= 0:
                print("[DEBUG] GAME OVER! internal lives=0")
                self.guard.set_terminated()
                total_reward += -100.0
                force_debug = True
                break

            self.s.prev_state = state
            total_reward += reward

        avg_danger = danger_sum / max(1, self.s.action_repeat)
        total_reward = self.reward_engine.postprocess(total_reward, avg_danger, is_slow)

        self.s.step_i += 1
        if force_debug or (self.s.step_i % self.s.debug_every == 0):
            pc = self.obs.player_center
            if pc is not None:
                print(f"[DEBUG] action={action.name} ui_ok={ui_ok} danger={avg_danger:.4f} player=({pc[0]},{pc[1]})")
            else:
                print(f"[DEBUG] action={action.name} ui_ok={ui_ok} danger={avg_danger:.4f}")

        self.s.frame_stack.append(self.s.prev_state)
        stacked_state = np.stack(self.s.frame_stack, axis=0)
        return stacked_state, float(total_reward), False
