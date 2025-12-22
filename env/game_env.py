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
    ✅ 루나틱 회피 학습용 (점수 안정화 + y-존 강제 + 좌표 안정화 버전)
    + ✅ 성능 프로파일링(중복 프레임 누적, 구간별 ms)

    목표:
    - 에피소드 종료 패널티는 "딱 1개만" 적용
      * death_pen: 죽음으로 종료(피격 포함, flash 포함) -> 동일 패널티
      * abort_pen: 로비/타이틀/ABORTED 등 비정상 종료 -> 동일 패널티

    - hit_pen은 "목숨 감소가 확실할 때"만 1회 적용
    - alive + shaping은 매 프레임 누적

    ✅ y-zone 강제:
    - y < y_floor(위로 올라감) 구간은 강한 패널티
      1) 진입 순간 큰 패널티(one-shot)
      2) 체류 시 매 프레임 누적 패널티

    ✅ 매우 중요:
    - shaping/y존 패널티는 ObsBuilder.last_xy_norm/last_conf 기반 (끊김 방지)
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
        # ✅ Reward (현재 네 코드 기준)
        # =========================
        self.alive_reward = 0.1
        self.hit_pen = -5.0
        self.death_pen = -5.0
        self.abort_pen = -5.0

        # =========================
        # ✅ 위치 shaping (아래쪽 유지)
        # =========================
        self.use_position_shaping = True

        self.y_floor = 0.60
        self.y_zone_enter_pen = 1.5
        self.y_zone_stay_pen_k = 0.08

        # conf threshold (네가 주석처리한 상태 유지하고 싶으면 아래 사용 안 함)
        self.y_pen_conf_thr = 0.02

        # (선택) 기존 우상단 억제
        self.top_soft_y = 0.20
        self.right_soft_x = 0.80
        self.top_pen_k = 0.020
        self.right_pen_k = 0.010
        self.corner_bonus_pen = 0.015

        # --- y존 상태 머신(진입 감지용) ---
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
        # ✅ PROFILING (NEW)
        # =========================
        self._prof_enable = True
        self._prof_every_steps = 30  # 30 step마다 1번 출력
        self._prof_t0 = time.perf_counter()
        self._prof_last_print_t = self._prof_t0

        # 누적(에피소드 단위)
        self._prof_steps = 0
        self._prof_dup_count = 0
        self._prof_prev_sample = None  # (H,W) uint8 sample

        # 구간별 누적 시간(에피소드 단위)
        self._prof_sum_capture = 0.0
        self._prof_sum_ui = 0.0
        self._prof_sum_obs = 0.0
        self._prof_sum_mask = 0.0
        self._prof_sum_ctrl = 0.0
        self._prof_sum_dbg = 0.0

        # 프레임 변화량 로그용 (최근값)
        self._prof_last_mean_abs = None
        self._prof_last_max_abs = None

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

        # ✅ 네가 최근에 conf 조건을 껐었으니, 여기서는 기본 "OFF" 상태 유지:
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
    # ✅ PROFILING helpers
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
        """
        비용 적게 프레임 중복 감지하기 위한 샘플.
        - 컬러 전체를 다 쓰지 않고, (채널0)만 다운샘플링해서 uint8로 비교
        """
        if img is None:
            return None
        # (H,W,3) 또는 (H,W) 모두 대응
        if img.ndim == 3:
            ch0 = img[:, :, 0]
        else:
            ch0 = img
        # 다운샘플: 8픽셀 간격
        return ch0[::8, ::8].astype(np.uint8, copy=False)

    def _prof_update_frame_dup(self, img: np.ndarray):
        if not self._prof_enable:
            return

        sample = self._prof_sample_frame(img)
        if sample is None:
            return

        if self._prof_prev_sample is None:
            self._prof_prev_sample = sample
            self._prof_last_mean_abs = None
            self._prof_last_max_abs = None
            return

        # abs diff
        diff = np.abs(sample.astype(np.int16) - self._prof_prev_sample.astype(np.int16))
        mean_abs = float(diff.mean())
        max_abs = int(diff.max())

        self._prof_last_mean_abs = mean_abs
        self._prof_last_max_abs = max_abs

        # "완전 동일" 또는 "거의 동일" 기준
        # - 완전 동일: max_abs == 0
        # - 거의 동일: mean_abs < 0.05 (환경에 따라 조절 가능)
        is_dup = (max_abs == 0) or (mean_abs < 0.05)
        if is_dup:
            self._prof_dup_count += 1

        self._prof_prev_sample = sample

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

        # 최근 프레임 변화량(마지막 측정값)
        if self._prof_last_mean_abs is None:
            print(f"[FRAMEDBG] step={self._prof_steps} mean_abs_diff=N/A")
        else:
            print(
                f"[FRAMEDBG] step={self._prof_steps} "
                f"mean_abs_diff={self._prof_last_mean_abs:.3f} max_abs={self._prof_last_max_abs} fps~{fps:.1f}"
            )
            if self._prof_last_max_abs == 0 or (self._prof_last_mean_abs < 0.05):
                print("  [FRAMEDBG][HINT] mean_abs_diff가 매우 낮음 -> 같은 프레임 중복 캡처 가능성↑ (frame_sleep 너무 짧을 수 있음)")

        # 누적 중복 프레임 비율
        dup_ratio = self._prof_dup_count / max(1, self._prof_steps)
        print(f"  [FRAMEDBG] dup_frames={self._prof_dup_count}/{self._prof_steps} ({dup_ratio*100:.2f}%)")

        # 구간별 평균 ms (step당)
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

        # ✅ 프로파일링 리셋(에피소드 시작마다)
        self._prof_reset_episode()

        t0 = time.perf_counter()
        img = self.screen.capture()
        self._prof_sum_capture += (time.perf_counter() - t0)

        t1 = time.perf_counter()
        state = self.obs.make_state(img)
        self._prof_sum_obs += (time.perf_counter() - t1)

        self.s.prev_state = state

        t2 = time.perf_counter()
        ui_ok = self.ui.ui_panel_present(img)
        self.s.prev_ui_lives = self.ui.ui_lives_safe(img, ui_ok)
        self._prof_sum_ui += (time.perf_counter() - t2)

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
        set_always_slow(True)

        # 첫 프레임 샘플 저장
        self._prof_update_frame_dup(img)
        # reset은 step 카운트에 포함시키지 않음(혼동 방지)

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
        # 로비/타이틀(Abort) 사전 체크
        # ---------
        t0 = time.perf_counter()
        pre_img = self.screen.capture()
        self._prof_sum_capture += (time.perf_counter() - t0)

        self._prof_update_frame_dup(pre_img)

        t1 = time.perf_counter()
        ui_ok = self.ui.ui_panel_present(pre_img)
        self.ui.update_ui_absent(ui_ok)
        self._prof_sum_ui += (time.perf_counter() - t1)

        if self.s.ui_absent_count >= self.s.ui_absent_needed:
            # step 출력(원인 추적 편의)
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

        for _ in range(self.s.action_repeat):
            if self.s.frame_sleep > 0:
                time.sleep(self.s.frame_sleep)

            t4 = time.perf_counter()
            img = self.screen.capture()
            self._prof_sum_capture += (time.perf_counter() - t4)

            self._prof_update_frame_dup(img)

            t5 = time.perf_counter()
            ui_ok = self.ui.ui_panel_present(img)
            self.ui.update_ui_absent(ui_ok)
            self._prof_sum_ui += (time.perf_counter() - t5)

            if self.s.ui_absent_count >= self.s.ui_absent_needed:
                for _ in range(5):
                    release_all()
                    time.sleep(0.02)
                _, pen_reward, _ = self._end_episode(self.abort_pen, "ABORT:UI_ABSENT(loop)")
                total_reward += pen_reward
                self._prof_maybe_print()
                return np.stack(self.s.frame_stack, axis=0), float(total_reward), True

            # 관측 업데이트
            t6 = time.perf_counter()
            state = self.obs.make_state(img)
            self._prof_sum_obs += (time.perf_counter() - t6)

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

            # ----------
            # 매 프레임 reward
            # ----------
            reward = float(self.alive_reward)
            now = time.time()

            pos = self._get_playfield_xy_norm_for_shaping()
            if pos is not None:
                x_n, y_n, conf = pos
                reward += self._y_zone_penalty(y_n, conf)
                reward += self._position_shaping_penalty(x_n, y_n)

            # ----------
            # death 판정: flash gameover
            # ----------
            # (detect_death 자체가 비용이 있을 수 있음. 필요하면 여기 또한 시간 측정 가능)
            _, gameover_fx = self.screen.detect_death(img)
            if gameover_fx:
                for _ in range(3):
                    release_all()
                    time.sleep(0.02)

                _, pen_reward, _ = self._end_episode(self.death_pen, "DEATH:FLASH")
                total_reward += (reward + pen_reward)

                self.s.prev_state = state
                self.s.frame_stack.append(self.s.prev_state)

                self._prof_maybe_print()
                return np.stack(self.s.frame_stack, axis=0), float(total_reward), True

            # ----------
            # hit 판정: UI lives 감소
            # ----------
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

                        self.s.prev_state = state
                        self.s.frame_stack.append(self.s.prev_state)

                        self._prof_maybe_print()
                        return np.stack(self.s.frame_stack, axis=0), float(total_reward), True

                    try:
                        if hasattr(self.obs, "on_player_death"):
                            self.obs.on_player_death()
                    except Exception as e:
                        print(f"[WARN] obs.on_player_death failed: {e}")

            self.s.prev_ui_lives = ui_now

            # 디버그 표시 (옵션)
            if self.show_reimu_debug:
                tdbg = time.perf_counter()

                dbg = getattr(self.obs, "_dbg_last", None)
                if dbg is not None:
                    if len(dbg) >= 6:
                        x_d, y_d, conf_d, logits, x_raw, y_raw = dbg[:6]
                        xy_for_viz = (x_raw, y_raw)
                    else:
                        x_d, y_d, conf_d, logits = dbg
                        xy_for_viz = (x_d, y_d)

                    play_dbg = self.screen.get_playfield_gray(img)
                    self.reimu_debug.show(
                        play_gray=play_dbg,
                        heatmap_logits=logits,
                        xy_norm=xy_for_viz,
                        conf=conf_d,
                        reward=reward,
                        total_reward=self.s.ep_total_reward,
                    )

                self._prof_sum_dbg += (time.perf_counter() - tdbg)

            self.s.prev_state = state
            total_reward += float(reward)
            self._ep_add(float(reward))
            self._step_count += 1

            # ✅ 30 step마다 프로파일 출력
            self._prof_maybe_print()

        self.s.step_i += 1
        self.s.frame_stack.append(self.s.prev_state)
        return np.stack(self.s.frame_stack, axis=0), float(total_reward), False
