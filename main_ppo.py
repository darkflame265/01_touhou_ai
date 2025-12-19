import argparse
import os
from datetime import datetime
import time
from collections import Counter

from env.game_env import GameEnv
from env.menu import enter_practice_from_cursor, recover_to_practice_from_lobby, recover_from_score_to_lobby
from env.actions import ACTIONS
from agents.ppo_agent import PPOAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--no-render", action="store_true", help="disable OBS window")
    return p.parse_args()


def main():
    args = parse_args()

    #CKPT_PATH = "checkpoints/ppo_hard_v1.pth"
    CKPT_PATH = "checkpoints/ppo_hard_reimuheat_crop_v1.pth"
    os.makedirs("checkpoints", exist_ok=True)

    # ==== 학습 로그 파일 저장 (ckpt 이름과 연동) ====
    pth_name = os.path.splitext(os.path.basename(CKPT_PATH))[0]  # e.g. ppo_hard_v1
    log_path = os.path.join("checkpoints", f"{pth_name}_episode_log.txt")

    # run 헤더 기록 (실행 단위로 구분)
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("========\n")
        f.write(f"[RUN] {run_ts}  episodes={args.episodes}\n")
        f.write("idx\treward\tedge60\ttop270\n")

    # ===== Env =====
    env = GameEnv(screen_mode="low")
    if args.no_render:
        env.show_obs = False

    agent = PPOAgent(
        input_channels=4,   # 🔴 7 → 4 로 되돌리기
        num_actions=len(ACTIONS),
    )

    if os.path.exists(CKPT_PATH):
        agent.load(CKPT_PATH, load_optimizer=True)
        print(f"[PPO] checkpoint loaded: {CKPT_PATH}")
    else:
        print("[PPO] no checkpoint found, training from scratch")

    print("\n[INFO] 실행 전 확인:")
    print(" - 게임이 로비 화면에 있음")
    print(" - 커서가 Practice Mode에 위치해 있음")
    print(" - 지금부터 키보드/마우스 건들지 마세요\n")
    time.sleep(1.0)

    for ep in range(1, args.episodes + 1):
        print(f"\n========== EPISODE {ep}/{args.episodes} ==========")

        if ep == 1:
            enter_practice_from_cursor()
        else:
            # 1) 혹시 Score 화면이면 먼저 빠져나오기
            recover_from_score_to_lobby(env.screen, max_sec=3.0)

            # 2) 그 다음 네 기존 로비 복구 루틴
            recover_to_practice_from_lobby()

        state = env.reset()
        done = False

        total_reward = 0.0
        steps = 0
        slow_count = 0
        action_counter = Counter()

        while not done:
            action_idx, log_prob, value = agent.select_action(state)
            action_name = ACTIONS[action_idx].name
            action_counter[action_name] += 1
            if action_name.startswith("SLOW"):
                slow_count += 1

            next_state, reward, done = env.step(action_idx)

            exec_idx = getattr(env.s, "exec_action_idx", action_idx)  # ✅ 실제 실행된 action
            agent.store(state, exec_idx, reward, done, log_prob, value)


            state = next_state
            total_reward += reward
            steps += 1

            # 롤아웃 쌓이면 중간 업데이트
            if agent.should_update():
                agent.update(last_state=state, last_done=done)

        # 에피소드 끝났으면 마지막 업데이트
        agent.update(last_state=state, last_done=True)

        slow_ratio = slow_count / max(1, steps)
        top_actions = action_counter.most_common(5)
        top_actions_str = ";".join(f"{k}:{v}" for k, v in top_actions)

        print(
            f"[PPO] episode done | steps={steps} total_reward={total_reward:.1f} "
            f"slow_ratio={slow_ratio:.3f} top_actions={top_actions_str}"
        )

        # ===== penalty summary (per-episode) =====
        edge60_cnt = getattr(env.s, "edge60_cnt", 0)
        top270_cnt = getattr(env.s, "top270_cnt", 0)
        print(f"[PENALTY] edge60_cnt={edge60_cnt}  top270_cnt={top270_cnt}")

        # ===== write per-episode log line (run-local idx format) =====
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"({ep}/{args.episodes})\t{total_reward:.6f}\t{edge60_cnt}\t{top270_cnt}\n")

        # 다음 에피소드용으로 카운터 리셋
        env.s.edge60_cnt = 0
        env.s.top270_cnt = 0

        agent.save(CKPT_PATH)
        print("[PPO] checkpoint saved")

        # 다음 에피소드 전에 로비 안정 대기
        if ep < args.episodes:
            time.sleep(0.3)

    print("\n[PPO] All episodes finished. Exiting.")


if __name__ == "__main__":
    main()
