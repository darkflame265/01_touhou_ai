# main_ppo.py
import argparse
import os
from datetime import datetime
import time
from collections import Counter

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
    p.add_argument("--no-render", action="store_true", help="disable OBS window")
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
    print(f"[BOOT] state={st.get('state')} selected={st.get('selected_name')} scores={st.get('scores')}")

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
    """
    ABORTED 때, 아직 update() 되지 않은 rollout/버퍼가 남아있으면 버리는 게 안전함.
    PPOAgent 구현이 다양해서 '있으면 호출' 방식으로만 처리.
    """
    for name in ("clear", "reset_buffer", "reset_storage", "clear_buffer", "clear_rollout"):
        fn = getattr(agent, name, None)
        if callable(fn):
            try:
                fn()
                print(f"[PPO] agent.{name}() called (abort cleanup)")
            except Exception:
                pass
            break


def main():
    args = parse_args()

    CKPT_PATH = "checkpoints/lunatic_v1.pth"
    os.makedirs("checkpoints", exist_ok=True)

    pth_name = os.path.splitext(os.path.basename(CKPT_PATH))[0]
    log_path = os.path.join("checkpoints", f"{pth_name}_episode_log.txt")

    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n========\n")
        f.write(f"[RUN] {run_ts}  episodes={args.episodes}\n")
        f.write("idx\treward\tsurvival_sec\tnote\n")

    env = GameEnv(screen_mode="low")
    if args.no_render:
        env.show_obs = False

    # ✅ 부팅 상태 출력 + 로비/일러스트 처리
    boot_print_state(env)

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

            # ✅ 에피소드 시작 전에 현재 화면 위치 한번 찍기
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

            # ✅ 로그는 ABORTED여도 남긴다
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"({ep}/{args.episodes})\t{total_reward:.6f}\t{survival_sec:.3f}\t{note}\n")

            if aborted:
                # ✅ ABORTED 에피소드는 학습 반영/저장 금지
                _try_clear_agent_rollout(agent)  # 있으면 버퍼 비우기
                print("[STOP] Episode aborted -> skip final update & checkpoint save.")
                break  # 보통은 바로 종료하는 게 안전(다음 ep 진행 X)

            # ✅ 정상 종료 에피소드만 마지막 업데이트 + 저장
            agent.update(last_state=state, last_done=True)

            agent.save(CKPT_PATH)
            print("[PPO] checkpoint saved")

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
