import argparse
import os
from datetime import datetime
import time
from collections import Counter

from env.game_env import GameEnv
from env.controller import release_all, set_attack_hold   # ✅ 추가: 중단 시 입력 즉시 해제
from env.menu import enter_practice_from_cursor, recover_to_practice_from_lobby, recover_from_score_to_lobby
from env.actions import ACTIONS
from agents.ppo_agent import PPOAgent


# =========================
# ✅ ESC stop helper (Windows global)
# - 외부 모듈 없이, 게임 창 포커스여도 ESC를 감지
# =========================
import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
VK_ESCAPE = 0x1B

def esc_pressed() -> bool:
    # GetAsyncKeyState: 최상위 비트(0x8000)가 눌림 상태
    return (user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000) != 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--no-render", action="store_true", help="disable OBS window")
    return p.parse_args()


def main():
    args = parse_args()

    CKPT_PATH = "checkpoints/ppo_hard_reimuheat_crop_v1.pth"
    os.makedirs("checkpoints", exist_ok=True)

    pth_name = os.path.splitext(os.path.basename(CKPT_PATH))[0]
    log_path = os.path.join("checkpoints", f"{pth_name}_episode_log.txt")

    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n========\n")
        f.write(f"[RUN] {run_ts}  episodes={args.episodes}\n")
        f.write("idx\treward\tedge60\ttop270\tnote\n")

    env = GameEnv(screen_mode="low")
    if args.no_render:
        env.show_obs = False

    agent = PPOAgent(
        input_channels=4,
        num_actions=len(ACTIONS),
    )

    if os.path.exists(CKPT_PATH):
        agent.load(CKPT_PATH, load_optimizer=True)
        print(f"[PPO] checkpoint loaded: {CKPT_PATH}")
    else:
        print("[PPO] no checkpoint found, training from scratch")

    print("\n[INFO] ESC 중단: Windows 전역 감지(GetAsyncKeyState)")
    print(" - 게임 창이 포커스여도 ESC를 잡고 즉시 종료/저장합니다.\n")
    time.sleep(1.0)

    stop_requested = False

    for ep in range(1, args.episodes + 1):
        if esc_pressed():
            stop_requested = True
            print("[STOP] ESC pressed before episode start -> stopping.")
            break

        print(f"\n========== EPISODE {ep}/{args.episodes} ==========")

        if ep == 1:
            enter_practice_from_cursor()
        else:
            recover_from_score_to_lobby(env.screen, max_sec=3.0)
            recover_to_practice_from_lobby()

        state = env.reset()
        done = False

        total_reward = 0.0
        steps = 0
        slow_count = 0
        action_counter = Counter()
        aborted = False

        while not done:
            # ✅ ESC 즉시 중단
            if esc_pressed():
                stop_requested = True
                aborted = True
                print("[STOP] ESC pressed -> aborting NOW (release inputs, save, exit).")

                # ✅ 게임 입력 즉시 해제(일시정지/오작동 방지)
                try:
                    set_attack_hold(False)
                except Exception:
                    pass
                try:
                    release_all()
                except Exception:
                    pass

                done = True
                break

            action_idx, log_prob, value = agent.select_action(state)
            action_name = ACTIONS[action_idx].name
            action_counter[action_name] += 1
            if action_name.startswith("SLOW"):
                slow_count += 1

            next_state, reward, done = env.step(action_idx)

            exec_idx = getattr(env.s, "exec_action_idx", action_idx)
            agent.store(state, exec_idx, reward, done, log_prob, value)

            state = next_state
            total_reward += reward
            steps += 1

            if agent.should_update():
                agent.update(last_state=state, last_done=done)

        # ✅ 에피소드 끝(정상/중단 모두) 마지막 업데이트
        agent.update(last_state=state, last_done=True)

        slow_ratio = slow_count / max(1, steps)
        top_actions = action_counter.most_common(5)
        top_actions_str = ";".join(f"{k}:{v}" for k, v in top_actions)
        note = "ABORTED" if aborted else ""

        print(
            f"[PPO] episode end | steps={steps} total_reward={total_reward:.1f} "
            f"slow_ratio={slow_ratio:.3f} top_actions={top_actions_str} {note}"
        )

        edge60_cnt = getattr(env.s, "edge60_cnt", 0)
        top270_cnt = getattr(env.s, "top270_cnt", 0)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"({ep}/{args.episodes})\t{total_reward:.6f}\t{edge60_cnt}\t{top270_cnt}\t{note}\n")

        # ✅ 저장
        agent.save(CKPT_PATH)
        print("[PPO] checkpoint saved")

        # ✅ ESC면 즉시 종료
        if stop_requested:
            print("[STOP] Training stopped by ESC. Exiting main_ppo.py.")
            break

        if ep < args.episodes:
            time.sleep(0.3)

    # 마지막으로도 한 번 안전하게 입력 해제
    try:
        set_attack_hold(False)
    except Exception:
        pass
    try:
        release_all()
    except Exception:
        pass

    print("\n[PPO] Finished.")


if __name__ == "__main__":
    main()
