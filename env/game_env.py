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

        # ===== (기존) shaping 파라미터 =====
        self.target_y_ratio = 0.78
        self.shaping_k = 0.35
        self.shaping_clip = 0.25

        self.stuck_dist_px = 2
        self.stuck_need = 10
        self.stuck_pen = 0.20

        self.edge_guard_px = 24
        self.edge_guard_pen = 0.08

        # ===== ✅ Action Masking 파라미터 =====
        # "벽에서 이 거리 이내면 그 벽 방향 입력을 금지" (너가 원한 200px)
        self.mask_margin_px = 200
        
        # 마스킹이 걸렸을 때, NONE으로 떨구지 말고
        # 가능하면 "안쪽"으로 자동 치환(반대 방향) 시도
        self.mask_use_flip = True

        # 디버그용: step마다 마스킹이 걸렸는지 기록
        self.s.exec_action_idx = 0
        self.s.exec_was_masked = False

        self.show_reimu_debug = True   # ← 토글 가능
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

        # shaping용 상태 리셋
        self.s.prev_dist_norm = None
        self.s.prev_pc = None
        self.s.stuck_run = 0

        # 통계 카운트
        self.s.edge60_cnt = 0
        self.s.top270_cnt = 0

        # 마스킹 실행 기록
        self.s.exec_action_idx = 0
        self.s.exec_was_masked = False

        if hasattr(self, "obs") and hasattr(self.obs, "reset"):
            self.obs.reset()

        release_all()
        set_attack_hold(True)

        return stacked

    # =========================================================
    # ✅ 플레이필드 경계(= UI 제외) 좌표를 "항상" 얻는 함수
    # - screen.py에 playfield_rect가 없으니 win_rect + PLAYFIELD_RIGHT_RATIO로 만든다
    # - 좌표는 "캡처 이미지 좌표계" (0~W, 0~H) 로 반환
    # =========================================================
    def _get_playfield_rect_safe(self, img_bgr):
        H, W = img_bgr.shape[:2]

        # 좌측/상단/하단은 전체를 쓰되,
        # 우측은 UI 시작 지점으로 제한
        r = int(W * float(self.screen.PLAYFIELD_RIGHT_RATIO))
        r = max(1, min(W, r))

        l = 0
        t = 0
        b = H

        # (옵션) Screen에 PLAYFIELD_*_CROP이 있으면 반영
        try:
            l = int(W * float(self.screen.PLAYFIELD_LEFT_CROP))
            t = int(H * float(self.screen.PLAYFIELD_TOP_CROP))
            r = int(r * float(self.screen.PLAYFIELD_RIGHT_CROP))
            b = int(H * float(self.screen.PLAYFIELD_BOTTOM_CROP))
        except Exception:
            pass

        # 안전 클램프
        l = max(0, min(W - 1, l))
        t = max(0, min(H - 1, t))
        r = max(l + 1, min(W, r))
        b = max(t + 1, min(H, b))

        return (l, t, r, b)

    def _get_target_point(self, img):
        l, t, r, b = self._get_playfield_rect_safe(img)
        w = max(1, int(r - l))
        h = max(1, int(b - t))
        tx = int(l + w * 0.5)
        ty = int(t + h * float(self.target_y_ratio))
        return tx, ty, (l, t, r, b)

    # =========================
    # Action Masking helpers
    # =========================
    def _action_dir(self, action_enum):
        keys = set(action_enum.value)
        dx = (-1 if "LEFT" in keys else (1 if "RIGHT" in keys else 0))
        dy = (-1 if "UP" in keys else (1 if "DOWN" in keys else 0))
        is_slow = ("SLOW" in keys)
        return dx, dy, is_slow

    def _flip_action(self, action_enum, flip_x=False, flip_y=False):
        keys = list(action_enum.value)

        def repl(k):
            if flip_x:
                if k == "LEFT": return "RIGHT"
                if k == "RIGHT": return "LEFT"
            if flip_y:
                if k == "UP": return "DOWN"
                if k == "DOWN": return "UP"
            return k

        new_keys = [repl(k) for k in keys]

        for a in ACTIONS:
            if list(a.value) == new_keys:
                return a
        return action_enum

    def _get_action_mask(self, img_bgr, margin_px):
        mask = np.ones((len(ACTIONS),), dtype=np.bool_)

        pc = getattr(self.obs, "player_center", None)
        if pc is None:
            return mask

        px, py = int(pc[0]), int(pc[1])
        l, t, r, b = self._get_playfield_rect_safe(img_bgr)

        left_d = px - l
        right_d = r - px
        top_d = py - t
        bot_d = b - py

        near_left = (left_d <= margin_px)
        near_right = (right_d <= margin_px)
        near_top = (top_d <= margin_px)
        near_bot = (bot_d <= margin_px)
       
        for i, a in enumerate(ACTIONS):
            dx, dy, _ = self._action_dir(a)
            if a.name == "NONE":
                continue

            # 가까운 벽 "방향"으로 더 가는 입력만 금지
            if near_left and dx < 0:
                mask[i] = False
            if near_right and dx > 0:
                mask[i] = False
            if near_top and dy < 0:
                mask[i] = False
            if near_bot and dy > 0:
                mask[i] = False

        return mask

    def _apply_action_mask(self, action_idx, img_bgr, margin_px):
        mask = self._get_action_mask(img_bgr, margin_px=margin_px)

        # 이미 허용이면 그대로
        if 0 <= int(action_idx) < len(ACTIONS) and bool(mask[int(action_idx)]):
            return int(action_idx), False, mask

        orig = ACTIONS[int(action_idx)]

        pc = getattr(self.obs, "player_center", None)
        flip_x = False
        flip_y = False

        if pc is not None:
            px, py = int(pc[0]), int(pc[1])
            l, t, r, b = self._get_playfield_rect_safe(img_bgr)

            left_d = px - l
            right_d = r - px
            top_d = py - t
            bot_d = b - py

            dx, dy, _ = self._action_dir(orig)

            # ===== 기존 벽 마스킹 기반 flip =====
            if (left_d <= margin_px and dx < 0) or (right_d <= margin_px and dx > 0):
                flip_x = True
            if (top_d <= margin_px and dy < 0) or (bot_d <= margin_px and dy > 0):
                flip_y = True

        alt = orig
        if self.mask_use_flip:
            alt = self._flip_action(orig, flip_x=flip_x, flip_y=flip_y)

        try:
            alt_idx = ACTIONS.index(alt)
        except ValueError:
            alt_idx = 0  # NONE

        # alt도 금지면: NONE
        if not bool(mask[alt_idx]):
            alt_idx = 0

        return int(alt_idx), True, mask

    # =========================================================
    # STEP
    # =========================================================
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

        # ===== ✅ (중요) step 시작에서도 한 번 마스킹 =====
        masked_idx, was_masked, _ = self._apply_action_mask(
            action_idx, pre_img, margin_px=self.mask_margin_px
        )

        action = ACTIONS[masked_idx]

        # ✅ 학습용으로 "실제 실행 action" 기록 (C)
        self.s.exec_action_idx = int(masked_idx)
        self.s.exec_was_masked = bool(was_masked)

        release_all()
        press_keys(action.value)

        total_reward = 0.0
        danger_sum = 0.0
        is_slow = action.name.startswith("SLOW")
        force_debug = False

        # 같은 액션 반복 카운트(디버그용 유지)
        if self.s.prev_action_idx == action_idx:
            self.s.same_action_count += 1
        else:
            self.s.same_action_count = 0
        self.s.prev_action_idx = action_idx

        # =========================================================
        # ✅ (B) action_repeat 루프 안에서도 "매 프레임" 마스킹 재적용
        # =========================================================
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

                total_reward += -100.0
                force_debug = True

                self.s.frame_stack.append(self.s.prev_state)
                stacked_state = np.stack(self.s.frame_stack, axis=0)
                return stacked_state, float(total_reward), True

            # 관측/트래킹 업데이트 (player_center 최신화)
            state = self.obs.make_state(img)

            if self.show_reimu_debug:
                dbg = getattr(self.obs, "_dbg_last", None)
                if dbg is not None:
                    x_n, y_n, conf, logits = dbg

                    # ✅ 여기서 step 루프의 현재 프레임은 img
                    play = self.screen.get_playfield_gray(img)  # (H,W) gray playfield

                    self.reimu_debug.show(
                        play_gray=play,
                        heatmap_logits=logits,
                        xy_norm=(x_n, y_n),
                        conf=conf,
                    )


            # ===== ✅ 매 프레임 입력 재검사: 벽 쪽이면 즉시 다른 입력으로 갈아끼움 =====
            # (이게 없으면 action_repeat 동안 계속 벽으로 밀어붙이는 문제가 남아있음)
            cur_idx, cur_was_masked, _ = self._apply_action_mask(
                masked_idx, img, margin_px=self.mask_margin_px
            )
            if cur_idx != masked_idx:
                masked_idx = cur_idx
                action = ACTIONS[masked_idx]

                # 실행 action 기록을 최신으로 갱신
                self.s.exec_action_idx = int(masked_idx)
                self.s.exec_was_masked = True

                release_all()
                press_keys(action.value)

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

            # 정지 화면 패널티
            motion_energy = float(np.abs(state - self.s.prev_state).mean())
            if motion_energy < 0.002:
                reward -= 0.03

            # ===== shaping =====
            pc = self.obs.player_center
            if pc is not None:
                px, py = int(pc[0]), int(pc[1])

                tx, ty, (l, t, r, b) = self._get_target_point(img)
                w = max(1.0, float(r - l))
                h = max(1.0, float(b - t))

                dx = (px - tx) / w
                dy = (py - ty) / h
                dist_norm = float((dx * dx + dy * dy) ** 0.5)

                if self.s.prev_dist_norm is not None:
                    delta = float(self.s.prev_dist_norm - dist_norm)
                    shape = float(self.shaping_k * delta)
                    if shape > self.shaping_clip:
                        shape = self.shaping_clip
                    elif shape < -self.shaping_clip:
                        shape = -self.shaping_clip
                    reward += shape

                self.s.prev_dist_norm = dist_norm

                # stuck 패널티
                if self.s.prev_pc is not None:
                    dxp = px - int(self.s.prev_pc[0])
                    dyp = py - int(self.s.prev_pc[1])
                    d2 = dxp * dxp + dyp * dyp
                    if d2 <= (self.stuck_dist_px * self.stuck_dist_px):
                        self.s.stuck_run += 1
                    else:
                        self.s.stuck_run = 0

                    if self.s.stuck_run >= self.stuck_need:
                        reward -= float(self.stuck_pen)
                        self.s.stuck_run = int(self.stuck_need * 0.6)

                self.s.prev_pc = (px, py)

                # 진짜 벽 초근접 패널티(약하게)
                edge_px = min(px - l, r - px, py - t, b - py)
                if edge_px <= self.edge_guard_px:
                    x = float((self.edge_guard_px - edge_px) / max(1, self.edge_guard_px))
                    reward -= float(self.edge_guard_pen * (x * x))

                # 통계 카운트
                if edge_px <= 60:
                    self.s.edge60_cnt += 1
                if py < (t + 270):
                    self.s.top270_cnt += 1

            # UI 기반 피격 감지 + 부활 워프
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

                    try:
                        trk = getattr(self.obs, "tracker", None)
                        if trk is not None and hasattr(trk, "on_player_death"):
                            trk.on_player_death()
                    except Exception as e:
                        print(f"[WARN] tracker.on_player_death failed: {e}")


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
                print(
                    f"[DEBUG] action={action.name} ui_ok={ui_ok} danger={avg_danger:.4f} "
                    f"player=({pc[0]},{pc[1]}) edge60={getattr(self.s,'edge60_cnt',0)} "
                    f"stuck={getattr(self.s,'stuck_run',0)} masked={getattr(self.s,'exec_was_masked',False)}"
                )
            else:
                print(f"[DEBUG] action={action.name} ui_ok={ui_ok} danger={avg_danger:.4f}")

        self.s.frame_stack.append(self.s.prev_state)
        stacked_state = np.stack(self.s.frame_stack, axis=0)
        return stacked_state, float(total_reward), False
