# env/episode_guard.py
import time
from env.controller import release_all, set_attack_hold


class EpisodeGuard:
    def __init__(self, state):
        self.s = state

    def set_terminated(self):
        self.s.episode_terminated = True
        self.s.terminate_until = time.time() + self.s.terminate_cooldown_sec
        
        set_attack_hold(False)  # 종료 후 공격키 자동 유지 OFF

        # 강제 키업 여러 번
        for _ in range(5):
            release_all()
            time.sleep(0.03)

    def terminated_step_return(self):
        # 종료 후 입력 차단
        for _ in range(2):
            release_all()
            time.sleep(0.01)
        if time.time() < self.s.terminate_until:
            time.sleep(0.02)
