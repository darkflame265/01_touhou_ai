# env/game_env.py
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
    ✅ 루나틱 회피 학습용
    + ✅ 성능 프로파일링(중복 프레임 누적, 구간별 ms)
    + ✅ (NEW) 중복 프레임 스킵/재캡처 로직
       - 같은 프레임이 잡히면: (1) 짧게 sleep 후 재캡처 몇 번 시도
       - 그래도 같은 프레임이면: "관측/판정/마스킹 재적용"을 스킵하고
         prev_state 유지 + reward를 0 처리(기본)
    """

    def __init__(self, screen_mode="low"):
        self.screen = Screen(mode=screen_mode)

        self.s = EnvState()
        self.guard = EpisodeGuard(self.s)
        self.ui = UIGuard(self.screen, self.s)
        self.reward_engine = RewardEngine(self.s)

        # 반응속도
        self.s.action_repeat = 2
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

        # Reward
        self.alive_reward = 0.1
        self.hit_pen = -5.0
        self.death_pen = -5.0
        self.abort_pen = -5.0

        # 위치 shaping
        self.use_position_shaping = True
        self.y_floor = 0.60
        self.y_zone_enter_pen = 1.5
        self.y_zone_stay_pen_k = 0.08
        self.y_pen_conf_thr = 0.02

        self.top_soft_y = 0.20
        self.right_soft_x = 0.80
        self.top_pen_k = 0.020
        self.right_pen_k = 0.010
        self.corner_bonus_pen = 0.015

        self._in_y_bad_zone = False
        self._last_y_pen = 0.0
        self._last_pos_pen = 0.0

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

        # =========================
        # ✅ DUP FRAME SKIP (NEW)
        # =========================
        self.skip_dup_frames = True
        self.dup_retry = 2           # dup면 추가로 몇 번 더 캡처해볼지
        self.dup_sleep = 0.012       # 재캡처 사이 sleep(초)
        self.dup_reward_zero = True  # True면 dup 프레임에서 reward=0, False면 alive_reward 유지
        self.dup_thr_mean_abs = 0.05 # dup 판정 mean_abs 기준(너의 기존값)
        self.dup_sample_stride = 8   # 샘플 다운샘플 간격(너의 기존값)

        # PROFILING
        self._prof_enable = True
        self._prof_every_steps = 200
        self._prof_t0 = time.perf_counter()
        self._prof_last_print_t = self._prof_t0

        self._prof_steps = 0
        self._prof_dup_count = 0
        self._prof_prev_sample = None

        self._prof_sum_capture = 0.0
        self._prof_sum_ui = 0.0
        self._prof_sum_obs = 0.0
        self._prof_sum_mask = 0.0
        self._prof_sum_ctrl = 0.0
        self._prof_sum_dbg = 0.0

        self._prof_last_mean_abs = None
        self._prof_last_max_abs = None

    def _as_chw(self, obs: np.ndarray) -> np.ndarray:
        """
        ObsBuilder가 (H,W) 또는 (C,H,W)를 반환한다고 가정.
        - (H,W)면 (1,H,W)로 변환
        - (H,W,C) 같은 형태는 여기선 지원 안 함(필요하면 추가 가능)
        """
        if obs is None:
            return None
        obs = np.asarray(obs)
        if obs.ndim == 2:
            return obs[None, :, :]  # (1,H,W)
        if obs.ndim == 3:
            return obs  # (C,H,W)
        raise ValueError(f"Unexpected obs shape: {obs.shape}")

    def _pack_frames_concat(self) -> np.ndarray:
        """
        frame_stack(list of obs)를 채널 축으로 concat해서 (C_total,H,W)로 반환.
        """
        if len(self.s.frame_stack) == 0:
            # 안전장치
            return self._as_chw(self.s.prev_state)

        frames = [self._as_chw(x) for x in list(self.s.frame_stack)]
        # frames: [(C,H,W), (C,H,W), ...]
        return np.concatenate(frames, axis=0)  # (T*C, H, W)

    # -------------------------
    # Utils
    # -------------------------
    def _ep_add(self, x: float):
        try:
            self.s.ep_total_reward += float(x)
        except Exception:
            pass

    def _end_episode(self, pen: float, reason: str):
            self.guard.set_terminated()
            self.s.episode_end_reason = str(reason)
            self.s.episode_end_pen = float(pen)
            self._ep_add(pen)

            # ✅ prev_state를 frame_stack에 넣고 concat 반환
            self.s.frame_stack.append(self.s.prev_state)
            packed = self._pack_frames_concat()
            return packed, float(pen), True

    def _get_playfield_xy_norm_for_shaping(self):
        x_n, y_n = getattr(self.obs, "last_xy_norm", (None, None))
        conf = float(getattr(self.obs, "last_conf", 0.0))
        if x_n is None or y_n is None:
            return None
        return float(x_n), float(y_n), conf

    def _position_shaping_penalty(self, x_n: float, y_n: float) -> float:
        if not self.use_position_shaping:
            self._last_pos_pen = 0.0
            return 0.0

        pen = 0.0
        if y_n < self.top_soft_y:
            d = (self.top_soft_y - y_n) / max(1e-6, self.top_soft_y)
            pen -= self.top_pen_k * float(d)

        if x_n > self.right_soft_x:
            d = (x_n - self.right_soft_x) / max(1e-6, (1.0 - self.right_soft_x))
            pen -= self.right_pen_k * float(d)

        if (y_n < self.top_soft_y) and (x_n > self.right_soft_x):
            pen -= float(self.corner_bonus_pen)

        self._last_pos_pen = float(pen)
        return float(pen)

    def _y_zone_penalty(self, y_n: float, conf: float) -> float:
        self._last_y_pen = 0.0

        # (기본 OFF 유지)
        # if conf < self.y_pen_conf_thr:
        #     return 0.0

        bad = (y_n < self.y_floor)

        if bad and (not self._in_y_bad_zone):
            self._in_y_bad_zone = True
            self._last_y_pen -= float(self.y_zone_enter_pen)

        if bad:
            d = (self.y_floor - y_n) / max(1e-6, self.y_floor)
            self._last_y_pen -= float(self.y_zone_stay_pen_k) * float(d)

        if (not bad) and self._in_y_bad_zone:
            self._in_y_bad_zone = False

        return float(self._last_y_pen)

    # =========================
    # PROFILING helpers
    # =========================
    def _prof_reset_episode(self):
        self._prof_t0 = time.perf_counter()
        self._prof_last_print_t = self._prof_t0

        self._prof_steps = 0
        self._prof_dup_count = 0
        self._prof_prev_sample = None

        self._prof_sum_capture = 0.0
        self._prof_sum_ui = 0.0
        self._prof_sum_obs = 0.0
        self._prof_sum_mask = 0.0
        self._prof_sum_ctrl = 0.0
        self._prof_sum_dbg = 0.0

        self._prof_last_mean_abs = None
        self._prof_last_max_abs = None

    def _prof_sample_frame(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            return None
        if img.ndim == 3:
            ch0 = img[:, :, 0]
        else:
            ch0 = img
        s = int(self.dup_sample_stride)
        return ch0[::s, ::s].astype(np.uint8, copy=False)

    def _prof_update_frame_dup(self, img: np.ndarray) -> bool:
        """
        returns: is_dup (bool)
        """
        if not self._prof_enable:
            return False

        sample = self._prof_sample_frame(img)
        if sample is None:
            return False

        if self._prof_prev_sample is None:
            self._prof_prev_sample = sample
            self._prof_last_mean_abs = None
            self._prof_last_max_abs = None
            return False

        diff = np.abs(sample.astype(np.int16) - self._prof_prev_sample.astype(np.int16))
        mean_abs = float(diff.mean())
        max_abs = int(diff.max())

        self._prof_last_mean_abs = mean_abs
        self._prof_last_max_abs = max_abs

        thr = float(self.dup_thr_mean_abs)
        is_dup = (max_abs == 0) or (mean_abs < thr)
        if is_dup:
            self._prof_dup_count += 1

        self._prof_prev_sample = sample
        return bool(is_dup)

    def _prof_maybe_print(self):
        if not self._prof_enable:
            return

        self._prof_steps += 1
        if (self._prof_steps % self._prof_every_steps) != 0:
            return

        now = time.perf_counter()
        dt = max(1e-9, now - self._prof_last_print_t)
        fps = self._prof_every_steps / dt
        self._prof_last_print_t = now

        if self._prof_last_mean_abs is None:
            print(f"[FRAMEDBG] step={self._prof_steps} mean_abs_diff=N/A")
        else:
            print(
                f"[FRAMEDBG] step={self._prof_steps} "
                f"mean_abs_diff={self._prof_last_mean_abs:.3f} max_abs={self._prof_last_max_abs} fps~{fps:.1f}"
            )
            if self._prof_last_max_abs == 0 or (self._prof_last_mean_abs < float(self.dup_thr_mean_abs)):
                print("  [FRAMEDBG][HINT] mean_abs_diff 매우 낮음 -> 같은 프레임 중복 캡처 가능성↑ (frame_sleep 너무 짧을 수 있음)")

        dup_ratio = self._prof_dup_count / max(1, self._prof_steps)
        print(f"  [FRAMEDBG] dup_frames={self._prof_dup_count}/{self._prof_steps} ({dup_ratio*100:.2f}%)")

        denom = max(1, self._prof_steps)
        cap_ms = (self._prof_sum_capture / denom) * 1000.0
        ui_ms = (self._prof_sum_ui / denom) * 1000.0
        obs_ms = (self._prof_sum_obs / denom) * 1000.0
        mask_ms = (self._prof_sum_mask / denom) * 1000.0
        ctrl_ms = (self._prof_sum_ctrl / denom) * 1000.0
        dbg_ms = (self._prof_sum_dbg / denom) * 1000.0

        print(
            "  [PROF] avg_ms/step | "
            f"capture={cap_ms:.2f} ui={ui_ms:.2f} obs={obs_ms:.2f} mask={mask_ms:.2f} ctrl={ctrl_ms:.2f} dbg={dbg_ms:.2f}"
        )

    # =========================
    # DUP frame handling
    # =========================
    def _capture_with_dup_retry(self):
        """
        returns: (img_bgr, is_dup_final)
        - 첫 캡처가 dup이면 짧게 대기 후 재캡처를 dup_retry 만큼 시도
        - 최종적으로도 dup이면 is_dup_final=True
        """
        t0 = time.perf_counter()
        img = self.screen.capture()
        self._prof_sum_capture += (time.perf_counter() - t0)

        is_dup = self._prof_update_frame_dup(img)
        if (not self.skip_dup_frames) or (not is_dup):
            return img, bool(is_dup)

        for _ in range(int(self.dup_retry)):
            if self.dup_sleep > 0:
                time.sleep(float(self.dup_sleep))

            t1 = time.perf_counter()
            img2 = self.screen.capture()
            self._prof_sum_capture += (time.perf_counter() - t1)

            is_dup2 = self._prof_update_frame_dup(img2)
            if not is_dup2:
                return img2, False

            img = img2
            is_dup = True

        return img, True

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

        self._in_y_bad_zone = False
        self._last_y_pen = 0.0
        self._last_pos_pen = 0.0

        self._prof_reset_episode()

        img, _ = self._capture_with_dup_retry()
        g = self.screen.gray(img)

        t1 = time.perf_counter()
        state = self.obs.make_state(img)
        self._prof_sum_obs += (time.perf_counter() - t1)

        # ✅ prev_state는 (C,H,W) 형태로 저장
        self.s.prev_state = self._as_chw(state)

        t2 = time.perf_counter()
        ui_ok = self.screen.ui_panel_present(img, gray=g)
        self.s.prev_ui_lives = self.ui.ui_lives_safe(img, ui_ok)
        self._prof_sum_ui += (time.perf_counter() - t2)

        # frame_stack 초기화
        self.s.frame_stack.clear()
        for _ in range(self.s.frame_stack_size):
            self.s.frame_stack.append(self.s.prev_state)

        self.s.exec_action_idx = 0
        self.s.exec_was_masked = False
        self._masked_count = 0
        self._step_count = 0

        if hasattr(self.obs, "reset"):
            self.obs.reset()

        release_all()
        set_attack_hold(True)
        set_always_slow(True)

        # ✅ concat 반환 (C_total,H,W)
        return self._pack_frames_concat()

    def step(self, action_idx):
        if self.s.episode_terminated:
            self.guard.terminated_step_return()
            set_attack_hold(False)
            for _ in range(6):
                release_all()
                time.sleep(0.02)
            # ✅ concat 반환
            return self._pack_frames_concat(), 0.0, True

        # ---------
        # Abort 사전 체크 (pre_img)
        # ---------
        pre_img, pre_is_dup = self._capture_with_dup_retry()

        if self.skip_dup_frames and pre_is_dup:
            ui_ok = True
        else:
            pre_g = self.screen.gray(pre_img)
            t_ui = time.perf_counter()
            ui_ok = self.screen.ui_panel_present(pre_img, gray=pre_g)
            self.ui.update_ui_absent(ui_ok)
            self._prof_sum_ui += (time.perf_counter() - t_ui)

            if self.s.ui_absent_count >= self.s.ui_absent_needed:
                self._prof_maybe_print()
                return self._end_episode(self.abort_pen, "ABORT:UI_ABSENT(pre)")

        # 입력 + 초기 마스킹
        t2 = time.perf_counter()
        masked_idx, was_masked, _ = self.masker.apply_action_mask(action_idx, pre_img)
        self._prof_sum_mask += (time.perf_counter() - t2)

        action = ACTIONS[masked_idx]
        self.s.exec_action_idx = int(masked_idx)
        self.s.exec_was_masked = bool(was_masked)
        if was_masked:
            self._masked_count += 1

        t3 = time.perf_counter()
        release_all()
        press_keys(action.value)
        self._prof_sum_ctrl += (time.perf_counter() - t3)

        total_reward = 0.0

        # ---------
        # action_repeat loop
        # ---------
        for _ in range(self.s.action_repeat):
            if self.s.frame_sleep > 0:
                time.sleep(self.s.frame_sleep)

            img, is_dup = self._capture_with_dup_retry()

            # -------------------------
            # DUP FRAME: heavy parts skip
            # -------------------------
            if self.skip_dup_frames and is_dup:
                reward = 0.0 if self.dup_reward_zero else float(self.alive_reward)

                # ✅ prev_state는 이미 (C,H,W), 그대로 쌓기
                self.s.frame_stack.append(self.s.prev_state)

                total_reward += float(reward)
                self._ep_add(float(reward))
                self._step_count += 1

                self._prof_maybe_print()
                continue

            # fresh frame
            g = self.screen.gray(img)

            # UI 체크
            t5 = time.perf_counter()
            ui_ok = self.screen.ui_panel_present(img, gray=g)
            self.ui.update_ui_absent(ui_ok)
            self._prof_sum_ui += (time.perf_counter() - t5)

            if self.s.ui_absent_count >= self.s.ui_absent_needed:
                for _ in range(5):
                    release_all()
                    time.sleep(0.02)
                _, pen_reward, _ = self._end_episode(self.abort_pen, "ABORT:UI_ABSENT(loop)")
                total_reward += pen_reward
                self._prof_maybe_print()
                return self._pack_frames_concat(), float(total_reward), True

            # 관측 업데이트
            t6 = time.perf_counter()
            state = self.obs.make_state(img)
            self._prof_sum_obs += (time.perf_counter() - t6)

            # ✅ prev_state는 (C,H,W)로 고정
            state_chw = self._as_chw(state)

            # 마스킹 재적용
            t7 = time.perf_counter()
            cur_idx, cur_was_masked, _ = self.masker.apply_action_mask(masked_idx, img)
            self._prof_sum_mask += (time.perf_counter() - t7)

            if cur_idx != masked_idx:
                masked_idx = cur_idx
                action = ACTIONS[masked_idx]
                self.s.exec_action_idx = int(masked_idx)
                self.s.exec_was_masked = True
                self._masked_count += 1

                t8 = time.perf_counter()
                release_all()
                press_keys(action.value)
                self._prof_sum_ctrl += (time.perf_counter() - t8)

            elif cur_was_masked:
                self.s.exec_was_masked = True
                self._masked_count += 1

            # reward
            reward = float(self.alive_reward)
            now = time.time()

            pos = self._get_playfield_xy_norm_for_shaping()
            if pos is not None:
                x_n, y_n, conf = pos
                reward += self._y_zone_penalty(y_n, conf)
                reward += self._position_shaping_penalty(x_n, y_n)

            # death 판정
            _, gameover_fx = self.screen.detect_death(img, gray=g)
            if gameover_fx:
                for _ in range(3):
                    release_all()
                    time.sleep(0.02)

                _, pen_reward, _ = self._end_episode(self.death_pen, "DEATH:FLASH")
                total_reward += (reward + pen_reward)

                self.s.prev_state = state_chw
                self.s.frame_stack.append(self.s.prev_state)

                self._prof_maybe_print()
                return self._pack_frames_concat(), float(total_reward), True

            # hit 판정
            t_ui_lives = time.perf_counter()
            ui_now = self.ui.ui_lives_safe(img, ui_ok)
            self._prof_sum_ui += (time.perf_counter() - t_ui_lives)

            if (ui_now is not None) and (now - self.s.last_hit_time) > self.s.hit_cooldown:
                if self.s.prev_ui_lives is not None and ui_now < self.s.prev_ui_lives:
                    self.s.lives -= 1
                    self.s.last_hit_time = now
                    reward = float(self.hit_pen)

                    if self.s.lives <= 0:
                        for _ in range(3):
                            release_all()
                            time.sleep(0.02)

                        _, pen_reward, _ = self._end_episode(self.death_pen, "DEATH:LIVES0")
                        total_reward += (reward + pen_reward)

                        self.s.prev_state = state_chw
                        self.s.frame_stack.append(self.s.prev_state)

                        self._prof_maybe_print()
                        return self._pack_frames_concat(), float(total_reward), True

                    try:
                        if hasattr(self.obs, "on_player_death"):
                            self.obs.on_player_death()
                    except Exception as e:
                        print(f"[WARN] obs.on_player_death failed: {e}")

            self.s.prev_ui_lives = ui_now

            # (디버그는 그대로 두되, return은 concat 기준)
            # ---- (중간 디버그 show_reimu_debug 부분은 기존 그대로 둬도 OK) ----

            self.s.prev_state = state_chw
            self.s.frame_stack.append(self.s.prev_state)

            total_reward += float(reward)
            self._ep_add(float(reward))
            self._step_count += 1

            self._prof_maybe_print()

        self.s.step_i += 1
        return self._pack_frames_concat(), float(total_reward), False
