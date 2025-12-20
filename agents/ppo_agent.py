# agents/ppo_agent.py
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import os

from models.shared.cnn_actor_critic import ActorCriticCNN


class PPOAgent:
    def __init__(
        self,
        input_channels,
        num_actions,
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        ent_min=0.0,
        ent_decay=0.999,
        rollout_steps=256,
        update_epochs=4,
        mini_batch_size=64,
        device=None,
        max_grad_norm=0.5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCriticCNN(
            input_channels=input_channels,
            num_actions=num_actions,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef

        self.ent_coef = ent_coef
        self.ent_min = ent_min
        self.ent_decay = ent_decay

        self.rollout_steps = rollout_steps
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm

        self.global_step = 0

        self.reset_buffer()

    def reset_buffer(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def select_action(self, state):
        s = torch.from_numpy(state[None].astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits, value = self.model(s)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))

        self.global_step += 1

    def should_update(self):
        return len(self.rewards) >= self.rollout_steps

    def _compute_gae(self, last_value: float = 0.0):
        """
        last_value:
          - rollout이 에피소드 중간에서 끊겼을 때(done=False로 끝났을 때) 마지막 상태의 V(s)를 넣어 부트스트랩
          - 에피소드 종료(done=True)로 끝났으면 0.0을 넣으면 됨
        """
        advantages = []
        returns = []

        gae = 0.0
        next_value = float(last_value)

        for t in reversed(range(len(self.rewards))):
            mask = 1.0 - float(self.dones[t])  # done이면 0, 아니면 1
            delta = self.rewards[t] + self.gamma * next_value * mask - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])

            next_value = self.values[t]

        returns = np.asarray(returns, dtype=np.float32)
        advantages = np.asarray(advantages, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self, last_state=None, last_done: bool = False):
        if len(self.rewards) < 2:
            self.reset_buffer()
            return None

        # ✅ rollout이 중간에서 끊긴 경우 마지막 상태 가치로 부트스트랩
        last_value = 0.0
        if (last_state is not None) and (not last_done):
            with torch.no_grad():
                s = torch.from_numpy(last_state[None].astype(np.float32)).to(self.device)
                _, v = self.model(s)
                last_value = float(v.item())

        returns, advantages = self._compute_gae(last_value=last_value)

        states = torch.from_numpy(np.asarray(self.states, dtype=np.float32)).to(self.device)
        actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.from_numpy(returns).to(self.device)
        adv_t = torch.from_numpy(advantages).to(self.device)

        n = states.size(0)
        idxs = np.arange(n)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        steps = 0

        for _ in range(self.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = idxs[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_adv = adv_t[mb_idx]

                logits, values = self.model(mb_states)
                dist = Categorical(logits=logits)

                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values.squeeze(-1), mb_returns)

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += float(loss.item())
                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                steps += 1

        # 엔트로피 계수 decay
        self.ent_coef = max(self.ent_min, self.ent_coef * self.ent_decay)

        self.reset_buffer()

        if steps == 0:
            return None

        return {
            "loss": total_loss / steps,
            "policy_loss": total_policy_loss / steps,
            "value_loss": total_value_loss / steps,
            "entropy": total_entropy / steps,
            "entropy_coef": float(self.ent_coef),
            "rollout_steps": int(n),
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        print("[LOAD] partial-load loader active")

        # 1) state_dict 가져오기
        sd = ckpt.get("model", ckpt)

        # 2) 현재 모델의 state_dict
        cur = self.model.state_dict()

        # 3) 호환되는 키만 골라서 로드 (shape까지 일치해야 함)
        filtered = {}
        skipped = []
        for k, v in sd.items():
            if (k in cur) and (cur[k].shape == v.shape):
                filtered[k] = v
            else:
                skipped.append(k)

        # 4) 부분 로드 (strict=False)
        msg = self.model.load_state_dict(filtered, strict=False)

        # 5) 옵티마이저는 구조 바뀌면 거의 항상 깨지니 기본은 끄는 게 안전
        if load_optimizer:
            try:
                if "optimizer" in ckpt:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[WARN] optimizer state not loaded (model changed): {e}")

        self.global_step = int(ckpt.get("global_step", self.global_step))

        # 로드 결과 로그 (원인 추적에 매우 도움)
        try:
            print("[LOAD] loaded keys:", len(filtered))
            print("[LOAD] missing keys:", msg.missing_keys)
            print("[LOAD] unexpected keys:", msg.unexpected_keys)
            if skipped:
                print("[LOAD] skipped incompatible keys(sample):", skipped[:10], "...")
        except Exception:
            pass

