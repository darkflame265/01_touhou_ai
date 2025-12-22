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
from env.menu import (
    enter_practice_from_cursor,
    recover_to_practice_from_lobby,
    recover_from_score_to_lobby,
    detect_location,
    ensure_practice_cursor_from_lobby,
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
    return p.parse_args()


def safe_release_inputs():
    # 어떤 예외가 나도 입력이 남지 않게 "무조건" 정리
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

    if st.get("state") in ("ILLUST", "LOBBY"):
        ok = ensure_practice_cursor_from_lobby(env.screen, verify=True, max_try=3)
        if ok:
            print("[BOOT] [practice 커서 정렬 완료]")
        else:
            print("[BOOT] [practice 커서 정렬 실패] (감지가 흔들릴 수 있음. 그래도 시퀀스는 시도함)")
    elif st.get("state") == "SCORE":
        print("[BOOT] [SCORE] 감지됨 -> recover_from_score_to_lobby 후 다시 시도 추천")
    elif st.get("state") == "IN_GAME":
        print("[BOOT] [IN_GAME] 감지됨 (이미 플레이 중일 수 있음)")
    else:
        print("[BOOT] [UNKNOWN] 감지 실패 (창 크기/밝기/텍스처에 따라 흔들릴 수 있음)")


def _try_clear_agent_rollout(agent):
    # 에피소드 중단(abort) 시, 롤아웃 버퍼 찌꺼기 때문에 다음 update가 꼬이는 걸 방지
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
        "total_completed": 0,          # ABORT 제외 누적 에피소드 수
        "best_reward": None,           # float
        "best_reward_ts": "",          # "YYYY-mm-dd HH:MM:SS"
        "best_reward_ep": "",          # "(ep/episodes)"
        "best_reward_run": "",         # run_ts

        "best_survival": None,         # float seconds
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
    # 같은 폴더에 임시파일 -> replace 로 원자적 교체
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
    """
    로그 파일에서 STATS 블록을 찾아 dict로 파싱.
    없으면 (None, 원문) 반환.
    """
    if (STATS_BEGIN not in text) or (STATS_END not in text):
        return None, text

    pattern = re.compile(
        re.escape(STATS_BEGIN) + r"(.*?)" + re.escape(STATS_END),
        re.DOTALL
    )
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
    """
    로그 파일 상단에 STATS 블록이 없으면 생성.
    있으면 읽어서 stats 반환.
    """
    text = _read_text(log_path)
    stats, _ = _extract_stats_block(text)

    if stats is not None:
        return stats

    stats = _default_stats()
    new_text = _format_stats_block(stats) + text
    _atomic_write(log_path, new_text)
    return stats


def _update_stats_in_file(log_path: str, stats: dict):
    """
    log_path 파일의 STATS 블록을 stats로 교체(없으면 상단에 삽입).
    """
    text = _read_text(log_path)
    cur_stats, _ = _extract_stats_block(text)

    new_block = _format_stats_block(stats)

    if cur_stats is None:
        new_text = new_block + text
    else:
        pattern = re.compile(
            re.escape(STATS_BEGIN) + r".*?" + re.escape(STATS_END) + r"\n?",
            re.DOTALL
        )
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
    """
    --no-render 일 때, 디버그/시각화 창을 확실히 꺼서
    성능 + 안정성(윈도우 핸들/리소스) 확보.
    (프로젝트 내부 클래스마다 필드명이 다를 수 있어 try로 안전 처리)
    """
    # 1) reimu heatmap/debug 창
    try:
        env.show_reimu_debug = False
    except Exception:
        pass

    # 2) DebugViz 창들
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


def main():
    args = parse_args()

    CKPT_PATH = "checkpoints/lunatic_v1.pth"
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

    pth_name = os.path.splitext(os.path.basename(CKPT_PATH))[0]
    log_path = os.path.join(os.path.dirname(CKPT_PATH), f"{pth_name}_episode_log.txt")

    stats = _ensure_stats_header(log_path)

    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n========\n")
        f.write(f"[RUN] {run_ts}  episodes={args.episodes}\n")
        f.write(_stats_one_line(stats) + "\n")
        f.write("idx\treward\tsurvival_sec\tnote\n")

    print(_stats_one_line(stats))

    env = GameEnv(screen_mode="low")
    if args.no_render:
        _apply_no_render(env)

    boot_print_state(env)

    agent = PPOAgent(
        input_channels=4,
        num_actions=len(ACTIONS),
    )

    if os.path.exists(CKPT_PATH):
        agent.load(CKPT_PATH, load_optimizer=False)
        print(f"[PPO] checkpoint loaded: {CKPT_PATH}")
    else:
        print("[PPO] no checkpoint found, training from scratch")

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

            if ep == 1:
                print("[MENU] [practice 모드 진입 중...]")
                enter_practice_from_cursor()
                print("[MENU] [practice 모드 진입 완료(시퀀스 수행)]")
            else:
                recover_from_score_to_lobby(env.screen, max_sec=3.0)
                recover_to_practice_from_lobby()

            state = env.reset()
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

                # ✅ 마스킹 후 실제 실행된 액션 인덱스로 store
                exec_idx = getattr(env.s, "exec_action_idx", action_idx)
                agent.store(state, exec_idx, reward, done, log_prob, value)

                state = next_state
                total_reward += reward
                steps += 1

                if agent.should_update():
                    agent.update(last_state=state, last_done=done)

            survival_sec = time.time() - ep_t0
            slow_ratio = slow_count / max(1, steps)
            top_actions = action_counter.most_common(5)
            top_actions_str = ";".join(f"{k}:{v}" for k, v in top_actions)
            note = "ABORTED" if aborted else ""

            print(
                f"[PPO] episode end | steps={steps} total_reward={total_reward:.1f} "
                f"survival_sec={survival_sec:.2f} slow_ratio={slow_ratio:.3f} "
                f"top_actions={top_actions_str} {note}"
            )

            ep_tag = f"({ep}/{args.episodes})"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{ep_tag}\t{total_reward:.6f}\t{survival_sec:.3f}\t{note}\n")

            if aborted:
                _try_clear_agent_rollout(agent)
                print("[STOP] Episode aborted -> skip final update & checkpoint save.")
                break

            # ✅ 정상 종료만 stats 갱신 + 파일 반영
            _maybe_update_records(stats, total_reward, survival_sec, run_ts, ep_tag)
            _update_stats_in_file(log_path, stats)
            print(_stats_one_line(stats))

            # ✅ 정상 종료만 마지막 update + 저장
            agent.update(last_state=state, last_done=True)

            ok = _safe_save_checkpoint(agent, CKPT_PATH)
            if ok:
                print("[PPO] checkpoint saved")
            else:
                print("[WARN] checkpoint save failed -> continue training without stopping")

            if stop_requested:
                print("[STOP] Training stopped by ESC. Exiting main_ppo.py.")
                break

            if ep < args.episodes:
                time.sleep(0.3)

    finally:
        safe_release_inputs()

    print("\n[PPO] Finished.")


if __name__ == "__main__":
    main()
