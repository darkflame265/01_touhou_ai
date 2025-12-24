# main_ppo.py
import argparse
import os
from datetime import datetime
import time
from collections import Counter
import re
import tempfile

from env.game_env import GameEnv
from env.controller import release_all, set_attack_hold
from env.controller import cleanup_inputs_on_exit
from env.menu import (
    detect_location,
    boot_into_practice,
)
from env.actions import ACTIONS
from agents.ppo_agent import PPOAgent

import ctypes
user32 = ctypes.WinDLL("user32", use_last_error=True)
VK_ESCAPE = 0x1B


def esc_pressed() -> bool:
    return (user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000) != 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--no-render", action="store_true", help="disable debug/obs windows")
    p.add_argument("--eval", action="store_true", help="evaluation mode (no training, no checkpoint, no STATS update)")
    return p.parse_args()


def safe_release_inputs():
    try:
        set_attack_hold(False)
    except Exception:
        pass
    try:
        release_all()
    except Exception:
        pass


def boot_print_state(env):
    print("\n[BOOT] 현재 화면 위치 감지 중...")
    st = detect_location(env.screen)
    print(f"[BOOT] state={st.get('state')} selected={st.get('selected_name')}")

    # 철학: OTHER는 정상임(난이도/옵션/인게임/일러스트 전부 OTHER)
    if st.get("state") == "SCORE":
        print("[BOOT] SCORE 감지됨: boot_into_practice()가 알아서 복구할 것")
    elif st.get("state") == "LOBBY":
        print("[BOOT] LOBBY 감지됨: 바로 practice 진입 가능")
    else:
        print("[BOOT] OTHER 감지됨: X 연타 복귀로 LOBBY 만들 예정")


def ensure_practice_ready_for_episode(env: GameEnv, ep: int) -> bool:
    print(f"[EP_PREP] boot_into_practice (ep={ep})")
    ok = boot_into_practice(env.screen, max_sec_lobby=12.0)
    if not ok:
        print("[EP_PREP][WARN] boot_into_practice failed (will continue and let env/reset try)")
    return ok


def _try_clear_agent_rollout(agent):
    for name in ("clear", "reset_buffer", "reset_storage", "clear_buffer", "clear_rollout"):
        fn = getattr(agent, name, None)
        if callable(fn):
            try:
                fn()
                print(f"[PPO] agent.{name}() called (abort cleanup)")
            except Exception:
                pass
            break


def _safe_save_checkpoint(agent, ckpt_path: str) -> bool:
    try:
        ret = agent.save(ckpt_path)
        if isinstance(ret, bool):
            return ret
        return True
    except Exception as e:
        print(f"[WARN] checkpoint save failed (ignored): {e}")
        return False


# =========================================================
# ✅ 누적 최고기록(STATS) 관리
# =========================================================
STATS_BEGIN = "# === PPO_STATS_BEGIN ==="
STATS_END = "# === PPO_STATS_END ==="


def _default_stats():
    return {
        "total_completed": 0,
        "best_reward": None,
        "best_reward_ts": "",
        "best_reward_ep": "",
        "best_reward_run": "",
        "best_survival": None,
        "best_survival_ts": "",
        "best_survival_ep": "",
        "best_survival_run": "",
    }


def _parse_float_or_none(s: str):
    try:
        return float(s)
    except Exception:
        return None


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""
    except Exception:
        return ""


def _atomic_write(path: str, text: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix="tmp_stats_", suffix=".txt", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _extract_stats_block(text: str):
    if (STATS_BEGIN not in text) or (STATS_END not in text):
        return None, text

    pattern = re.compile(re.escape(STATS_BEGIN) + r"(.*?)" + re.escape(STATS_END), re.DOTALL)
    m = pattern.search(text)
    if not m:
        return None, text

    block = m.group(1)
    stats = _default_stats()

    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()

        if k == "total_completed":
            try:
                stats[k] = int(v)
            except Exception:
                pass
        elif k in ("best_reward", "best_survival"):
            stats[k] = _parse_float_or_none(v)
        else:
            if k in stats:
                stats[k] = v

    return stats, text


def _format_stats_block(stats: dict) -> str:
    br = "" if stats["best_reward"] is None else f"{stats['best_reward']:.6f}"
    bs = "" if stats["best_survival"] is None else f"{stats['best_survival']:.3f}"

    lines = [
        STATS_BEGIN,
        f"total_completed={int(stats['total_completed'])}",
        f"best_reward={br}",
        f"best_reward_ts={stats['best_reward_ts']}",
        f"best_reward_run={stats['best_reward_run']}",
        f"best_reward_ep={stats['best_reward_ep']}",
        f"best_survival={bs}",
        f"best_survival_ts={stats['best_survival_ts']}",
        f"best_survival_run={stats['best_survival_run']}",
        f"best_survival_ep={stats['best_survival_ep']}",
        STATS_END,
        "",
    ]
    return "\n".join(lines)


def _ensure_stats_header(log_path: str) -> dict:
    text = _read_text(log_path)
    stats, _ = _extract_stats_block(text)
    if stats is not None:
        return stats

    stats = _default_stats()
    new_text = _format_stats_block(stats) + text
    _atomic_write(log_path, new_text)
    return stats


def _update_stats_in_file(log_path: str, stats: dict):
    text = _read_text(log_path)
    cur_stats, _ = _extract_stats_block(text)

    new_block = _format_stats_block(stats)
    if cur_stats is None:
        new_text = new_block + text
    else:
        pattern = re.compile(re.escape(STATS_BEGIN) + r".*?" + re.escape(STATS_END) + r"\n?", re.DOTALL)
        new_text = pattern.sub(new_block.strip() + "\n", text, count=1)

    _atomic_write(log_path, new_text)


def _maybe_update_records(stats: dict, reward: float, survival_sec: float, run_ts: str, ep_tag: str):
    stats["total_completed"] = int(stats.get("total_completed", 0)) + 1

    br = stats.get("best_reward", None)
    if (br is None) or (float(reward) > float(br)):
        stats["best_reward"] = float(reward)
        stats["best_reward_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stats["best_reward_run"] = run_ts
        stats["best_reward_ep"] = ep_tag

    bs = stats.get("best_survival", None)
    if (bs is None) or (float(survival_sec) > float(bs)):
        stats["best_survival"] = float(survival_sec)
        stats["best_survival_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stats["best_survival_run"] = run_ts
        stats["best_survival_ep"] = ep_tag


def _stats_one_line(stats: dict) -> str:
    br = stats["best_reward"]
    bs = stats["best_survival"]
    br_s = "N/A" if br is None else f"{br:.3f}"
    bs_s = "N/A" if bs is None else f"{bs:.2f}s"
    return f"[STATS] total_completed={stats['total_completed']} | best_reward={br_s} | best_survival={bs_s}"


def _apply_no_render(env: GameEnv):
    # reimu heatmap/debug 창
    try:
        env.show_reimu_debug = False
    except Exception:
        pass

    # ObsBuilder crop 디버그 창
    try:
        if hasattr(env, "obs") and hasattr(env.obs, "show_obs_debug"):
            env.obs.show_obs_debug = False
    except Exception:
        pass

    # (구 DebugViz가 남아있으면 끄기)
    dbg = getattr(env, "debug", None)
    if dbg is not None:
        for name, val in (
            ("show_tracker_debug", False),
            ("show_roi_window", False),
            ("show_full_window", False),
            ("show_mask_window", False),
        ):
            try:
                if hasattr(dbg, name):
                    setattr(dbg, name, val)
            except Exception:
                pass


def _append_run_header(log_path: str, run_ts: str, episodes: int, is_eval: bool, stats: dict):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n========\n")
        f.write(f"[RUN] {run_ts}  episodes={episodes} eval={str(bool(is_eval))}\n")
        f.write(_stats_one_line(stats) + "\n")
        f.write("idx\treward\tsurvival_sec\tnote\n")


def main():
    args = parse_args()
    is_eval = bool(args.eval)

    CKPT_PATH = "checkpoints/lunatic_v1_ch4.pth"

    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

    pth_name = os.path.splitext(os.path.basename(CKPT_PATH))[0]
    log_path = os.path.join(os.path.dirname(CKPT_PATH), f"{pth_name}_episode_log.txt")

    stats = _ensure_stats_header(log_path)

    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _append_run_header(log_path, run_ts, int(args.episodes), is_eval, stats)

    print(_stats_one_line(stats))

    env = GameEnv(screen_mode="low")
    if args.no_render:
        _apply_no_render(env)

    boot_print_state(env)

    obs_channels = int(getattr(env.obs, "obs_channels", 1))
    stack_size = int(getattr(env.s, "frame_stack_size", 4))
    input_channels = obs_channels * stack_size
    print(f"[PPO] input_channels={input_channels} (obs_channels={obs_channels} * stack={stack_size})")

    agent = PPOAgent(
        input_channels=input_channels,
        num_actions=len(ACTIONS),
        obs_channels_per_frame=obs_channels,   # ✅ 이거 추가!
    )

    # ✅ eval이면 "로드만" 권장, 그래도 파일 있으면 로드하고 없으면 그냥 진행
    if os.path.exists(CKPT_PATH):
        agent.load(CKPT_PATH, load_optimizer=False)
        print(f"[PPO] checkpoint loaded: {CKPT_PATH}")
    else:
        print("[PPO] no checkpoint found, training from scratch" if not is_eval else "[PPO][EVAL] no checkpoint found (evaluating random policy)")

    # =========================================================
    # ✅ 액션공간 전환기: ckpt 로드 후 하이퍼파라미터 강제 재설정
    #    (8방향 고정 + 상시 slow 최적화)
    # =========================================================
    agent.ent_coef = 0.04
    agent.ent_min = 0.005
    agent.ent_decay = 0.9995
    agent.ent_warmup_updates = 30

    agent.clip_eps = 0.15
    agent.rollout_steps = 128
    agent.update_epochs = 5

    print(
        "[PPO][OVERRIDE] hyperparams overridden after ckpt load | "
        f"ent_coef={agent.ent_coef:.3f}, "
        f"ent_min={agent.ent_min:.3f}, "
        f"clip_eps={agent.clip_eps:.2f}, "
        f"rollout_steps={agent.rollout_steps}, "
        f"update_epochs={agent.update_epochs}"
    )


    print("\n[INFO] ESC 중단: Windows 전역 감지(GetAsyncKeyState)")
    print(" - 게임 창이 포커스여도 ESC를 잡고 즉시 종료합니다.\n")
    time.sleep(0.7)

    stop_requested = False

    try:
        for ep in range(1, args.episodes + 1):
            if esc_pressed():
                stop_requested = True
                print("[STOP] ESC pressed before episode start -> stopping.")
                break

            print(f"\n========== EPISODE {ep}/{args.episodes} ==========")

            st = detect_location(env.screen)
            print(f"[BOOT->EP] state={st.get('state')} selected={st.get('selected_name')}")

            print("[MENU] [practice 준비/진입 중...]")
            ensure_practice_ready_for_episode(env, ep)
            print("[MENU] [practice 준비/진입 완료]")

            safe_release_inputs()
            state = env.reset()

            # 디버그(원하면 지워도 됨)
            try:
                print("[DBG] state.shape =", state.shape, "dtype=", state.dtype, "min/max=", float(state.min()), float(state.max()))
            except Exception:
                pass

            ep_t0 = time.time()

            done = False
            total_reward = 0.0
            steps = 0
            slow_count = 0
            action_counter = Counter()
            aborted = False

            while not done:
                if esc_pressed():
                    stop_requested = True
                    aborted = True
                    print("[STOP] ESC pressed -> aborting NOW (release inputs, NO SAVE/NO UPDATE for this episode).")
                    safe_release_inputs()
                    done = True
                    break

                action_idx, log_prob, value = agent.select_action(state)

                action_name = ACTIONS[action_idx].name
                action_counter[action_name] += 1
                if action_name.startswith("SLOW"):
                    slow_count += 1

                next_state, reward, done = env.step(action_idx)

                # ✅ eval 모드면 학습 버퍼에 저장하지 않음
                if not is_eval:
                    exec_idx = getattr(env.s, "exec_action_idx", action_idx)
                    agent.store(state, exec_idx, reward, done, log_prob, value)

                state = next_state
                total_reward += reward
                steps += 1

                # ✅ eval 모드면 update 자체를 하지 않음
                if (not is_eval) and agent.should_update():
                    agent.update(last_state=state, last_done=done)

            survival_sec = time.time() - ep_t0
            slow_ratio = slow_count / max(1, steps)
            top_actions = action_counter.most_common(5)
            top_actions_str = ";".join(f"{k}:{v}" for k, v in top_actions)
            note_parts = []
            if is_eval:
                note_parts.append("EVAL")
            if aborted:
                note_parts.append("ABORTED")
            note = ",".join(note_parts)

            print(
                f"[PPO] episode end | steps={steps} total_reward={total_reward:.1f} "
                f"survival_sec={survival_sec:.2f} slow_ratio={slow_ratio:.3f} "
                f"top_actions={top_actions_str} {note}"
            )

            # ✅ (중요) eval이어도 에피소드 결과 라인은 항상 로그에 남긴다
            ep_tag = f"({ep}/{args.episodes})"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{ep_tag}\t{total_reward:.6f}\t{survival_sec:.3f}\t{note}\n")

            # ✅ eval이면 STATS/체크포인트 갱신은 절대 안 함
            if aborted:
                if not is_eval:
                    _try_clear_agent_rollout(agent)
                print("[STOP] Episode aborted -> stopping.")
                break

            if not is_eval:
                _maybe_update_records(stats, total_reward, survival_sec, run_ts, ep_tag)
                _update_stats_in_file(log_path, stats)
                print(_stats_one_line(stats))

                agent.update(last_state=state, last_done=True)

                ok = _safe_save_checkpoint(agent, CKPT_PATH)
                if ok:
                    print("[PPO] checkpoint saved")
                else:
                    print("[WARN] checkpoint save failed -> continue training without stopping")
            else:
                # eval은 stats 고정 출력(변화 없음)
                print(_stats_one_line(stats))

            if stop_requested:
                print("[STOP] Training stopped by ESC. Exiting main_ppo.py.")
                break

            if ep < args.episodes:
                time.sleep(0.3)

    finally:
        cleanup_inputs_on_exit()

    print("\n[PPO] Finished.")


if __name__ == "__main__":
    main()
