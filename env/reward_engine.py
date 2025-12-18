# env/reward_engine.py
class RewardEngine:
    def __init__(self, state):
        self.s = state

    def postprocess(self, total_reward, avg_danger, is_slow):
        if is_slow:
            self.s.slow_streak += 1
            if avg_danger > 0.12:
                total_reward += 0.20
            elif avg_danger < 0.05:
                total_reward -= 0.10
            if self.s.slow_streak > self.s.slow_streak_max:
                total_reward -= 0.05
        else:
            self.s.slow_streak = 0
            if avg_danger > 0.12:
                total_reward -= 0.07
            elif avg_danger < 0.05:
                total_reward += 0.02
        return total_reward
