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

    목표:
    - 에피소드 종료 패널티는 "딱 1개만" 적용
      * death_pen: 죽음으로 종료(피격 포함, flash 포함) -> 동일 패널티
      * abort_pen: 로비/타이틀/ABORTED 등 비정상 종료 -> 동일 패널티

    - hit_pen은 "목숨 감소가 확실할 때"만 1회 적용
    - alive + shaping은 매 프레임 누적

    ✅ 추가(핵심):
    - y < y_floor(위로 올라감) 구간은 "PPO가 무시 못하게" 강하게 패널티
      1) 존(구간) 진입 순간 큰 패널티 (one-shot)
      2) 존 체류 시 매 프레임 누적 패널티
      3) conf가 충분할 때만 적용 (탐지 흔들림 방지)

    ✅ 매우 중요:
    - shaping/y존 패널티는 _dbg_last(=det None이면 끊김)가 아니라
      ObsBuilder.last_xy_norm/last_conf(=det 흔들려도 유지)로 계산한다.
      -> "패널티 적용이 끊겨서 위로 올라가는" 문제를 크게 줄임.
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
        self.alive_reward = 0.1

        # "목숨 감소 1회" 페널티 (에피소드 종료와 별개)
        self.hit_pen = -5.0

        # ✅ 에피소드 종료 패널티는 2개로 단순화
        self.death_pen = -5.0   # 죽음 종료(최종)
        self.abort_pen = -5.0   # 로비/타이틀/중단

        # =========================
        # ✅ 위치 shaping (아래쪽 유지)
        # =========================
        self.use_position_shaping = True

        # --- 아래쪽 유지 기준 ---
        self.y_floor = 0.60

        # ✅ y존 패널티는 "무시 못하게" 2단 구성
        self.y_zone_enter_pen = 1.5     # 존(위쪽) 진입 순간 1회 큰 패널티 (0.8~3.0)
        self.y_zone_stay_pen_k = 0.08   # 존 체류 프레임당 패널티 (0.04~0.15)

        # ✅ conf가 이 이상일 때만 y존 패널티 적용 (탐지 흔들림 방지)
        self.y_pen_conf_thr = 0.02

        # (선택) 기존 우상단 억제 (원하면 끄면 됨)
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
        """
        디버그 창 표시용. (det None이면 끊길 수 있음)
        _dbg_last: (x_lock,y_lock,conf,logits,x_raw,y_raw) or (x_lock,y_lock,conf,logits)
        """
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
        """
        ✅ shaping/y존 패널티용.
        _dbg_last는 det=None이면 끊기므로 부적합.
        ObsBuilder가 유지하는 last_xy_norm/last_conf를 사용한다.
        """
        x_n, y_n = getattr(self.obs, "last_xy_norm", (None, None))
        conf = float(getattr(self.obs, "last_conf", 0.0))
        if x_n is None or y_n is None:
            return None
        return float(x_n), float(y_n), conf

    def _position_shaping_penalty(self, x_n: float, y_n: float) -> float:
        """
        (선택) 우상단/상단/우측 억제용 미세 shaping
        """
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
        """
        ✅ 핵심: y < y_floor 구간을 강하게 밀어내는 패널티
        - conf 충분할 때만 적용 (안정성)
        - 진입 1회 패널티 + 체류 누적 패널티
        """
        self._last_y_pen = 0.0

        # conf 낮으면 "존 상태"도 업데이트하지 않음 (깜빡임/튐 방지)
        if conf < self.y_pen_conf_thr:
            return 0.0

        bad = (y_n < self.y_floor)

        if bad and (not self._in_y_bad_zone):
            # ✅ 진입 순간 1회 패널티
            self._in_y_bad_zone = True
            self._last_y_pen -= float(self.y_zone_enter_pen)

        if bad:
            # ✅ 체류 패널티 (y가 더 위로 갈수록 더 아프게)
            d = (self.y_floor - y_n) / max(1e-6, self.y_floor)  # 0..1+
            self._last_y_pen -= float(self.y_zone_stay_pen_k) * float(d)

        if (not bad) and self._in_y_bad_zone:
            # 존 탈출
            self._in_y_bad_zone = False

        return float(self._last_y_pen)

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

        # y존 상태 리셋
        self._in_y_bad_zone = False
        self._last_y_pen = 0.0
        self._last_pos_pen = 0.0

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
                pen_state, pen_reward, _ = self._end_episode(self.abort_pen, "ABORT:UI_ABSENT(loop)")
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

            # ✅ shaping/y존은 "끊기지 않는" last_xy_norm 기반으로 계산
            pos = self._get_playfield_xy_norm_for_shaping()
            if pos is not None:
                x_n, y_n, conf = pos

                # y존 강제(핵심)
                reward += self._y_zone_penalty(y_n, conf)

                # (선택) 미세 shaping
                # conf가 너무 낮을 때는 미세 shaping도 꺼도 됨(원하면 아래 줄을 conf 조건으로 감싸도 OK)
                reward += self._position_shaping_penalty(x_n, y_n)

            # ----------
            # ✅ death 판정 1: flash gameover
            # ----------
            _, gameover_fx = self.screen.detect_death(img)
            if gameover_fx:
                for _ in range(3):
                    release_all()
                    time.sleep(0.02)

                # 마지막 프레임 보상 + 종료 패널티
                _, pen_reward, _ = self._end_episode(self.death_pen, "DEATH:FLASH")
                total_reward += (reward + pen_reward)

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

                    # ✅ 목숨 감소는 항상 hit_pen 1회만 (shaping보다 우선)
                    reward = float(self.hit_pen)

                    # 마지막 목숨이면 죽음 종료 (death_pen 1회)
                    if self.s.lives <= 0:
                        for _ in range(3):
                            release_all()
                            time.sleep(0.02)

                        _, pen_reward, _ = self._end_episode(self.death_pen, "DEATH:LIVES0")
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

            # 디버그 표시 (표시는 기존 _dbg_last 기반 유지)
            if self.show_reimu_debug:
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

            self.s.prev_state = state
            total_reward += float(reward)
            self._ep_add(float(reward))
            self._step_count += 1

        self.s.step_i += 1
        self.s.frame_stack.append(self.s.prev_state)
        return np.stack(self.s.frame_stack, axis=0), float(total_reward), False
