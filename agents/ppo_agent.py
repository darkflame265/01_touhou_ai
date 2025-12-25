# agents/ppo_agent.py
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import os

from models.shared.two_stream_actor_critic import TwoStreamActorCriticCNN


class PPOAgent:
    """
    ✅ 마스킹(exec_idx) 때문에 실제 실행 액션이 바뀌어도 PPO가 꼬이지 않게:
      - rollout에 저장되는 (action, log_prob)는 '실행된 액션(exec_action)' 기준이어야 함
      - store()에서 log_prob/value가 None이면, 현재 모델로 state를 다시 평가해서 저장
        (rollout 중간에는 모델이 업데이트되지 않으니 on-policy로 취급 가능)

    ✅ 또한 main_ppo.py가 예전 인자(two_stream 등)를 넘겨도 죽지 않게 **kwargs 흡수
    """

    # agents/ppo_agent.py (PPOAgent.__init__ 시그니처에 추가)
    def __init__(
        self,
        input_channels,
        num_actions,
        obs_channels_per_frame=8,

        # ✅ 추가: main에서 실측한 g_ch / l_ch를 받는다
        global_channels=4,
        local_channels=4,

        lr=3.0e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.15,
        vf_coef=0.5,
        ent_coef=0.04,
        ent_min=0.005,
        ent_decay=0.9995,
        rollout_steps=128,
        update_epochs=5,
        mini_batch_size=64,
        device=None,
        max_grad_norm=0.5,
        ent_warmup_updates=30,
        **kwargs,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ✅ 여기서 assert를 "모델 내부"가 아니라 "agent 쪽"에서 먼저 터뜨려서 원인 바로 보이게
        if int(global_channels) + int(local_channels) != int(obs_channels_per_frame):
            raise ValueError(
                f"Channel mismatch: global_channels({global_channels}) + local_channels({local_channels}) "
                f"!= obs_channels_per_frame({obs_channels_per_frame}). "
                f"Fix main_ppo.py or PPOAgent init args."
            )

        self.model = TwoStreamActorCriticCNN(
            input_channels=int(input_channels),
            num_actions=int(num_actions),
            obs_channels_per_frame=int(obs_channels_per_frame),

            # ✅ 하드코딩 제거하고 전달값 사용
            global_channels=int(global_channels),
            local_channels=int(local_channels),

            meta_patch=4,
            meta_local_channel_offset=0,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr))


        # ✅ (중요) GAE에 쓰는 gamma 누락 수정
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.vf_coef = float(vf_coef)

        self.ent_coef = float(ent_coef)
        self.ent_min = float(ent_min)
        self.ent_decay = float(ent_decay)

        self.rollout_steps = int(rollout_steps)
        self.update_epochs = int(update_epochs)
        self.mini_batch_size = int(mini_batch_size)
        self.max_grad_norm = float(max_grad_norm)

        self.global_step = 0
        self.update_step = 0
        self.ent_warmup_updates = int(ent_warmup_updates)

        self.reset_buffer()

    def reset_buffer(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    # -------------------------
    # Policy helpers
    # -------------------------
    def _forward(self, state_np: np.ndarray):
        s = torch.from_numpy(state_np[None].astype(np.float32)).to(self.device)
        logits, value = self.model(s)  # logits: (1,A), value:(1,1)
        return logits[0], value[0, 0]

    @torch.no_grad()
    def select_action(self, state: np.ndarray):
        """
        반환:
          action_idx: policy가 샘플한 액션 (env에서 마스킹될 수 있음)
          log_prob: 그 action_idx의 log_prob (참고용)
          value: V(s)

        ✅ 학습 rollout에는 '실행된 exec_action' 기준 log_prob가 들어가야 한다.
          -> store()에서 exec_action에 대해 다시 계산한다.
        """
        logits, value = self._forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    @torch.no_grad()
    def evaluate_action(self, state: np.ndarray, action_idx: int):
        """
        state에서 action_idx의 log_prob와 value를 계산 (exec_idx용)
        """
        logits, value = self._forward(state)
        dist = Categorical(logits=logits)
        a = torch.tensor(int(action_idx), device=logits.device, dtype=torch.long)
        log_prob = dist.log_prob(a)
        return float(log_prob.item()), float(value.item())

    # -------------------------
    # Rollout storage
    # -------------------------
    def store(self, state, action, reward, done, log_prob=None, value=None):
        """
        ✅ action은 '실행된 액션(exec_action_idx)'을 넣어야 한다.
        - log_prob/value를 넘기지 않으면 여기서 exec_action 기준으로 재평가한다.
        """
        exec_action = int(action)

        if (log_prob is None) or (value is None):
            lp, v = self.evaluate_action(state, exec_action)
            log_prob = lp
            value = v

        self.states.append(state)
        self.actions.append(exec_action)
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.global_step += 1

    def should_update(self):
        return len(self.rewards) >= self.rollout_steps

    # -------------------------
    # GAE + PPO update
    # -------------------------
    def _compute_gae(self, last_value: float = 0.0):
        advantages = []
        returns = []

        gae = 0.0
        next_value = float(last_value)

        for t in reversed(range(len(self.rewards))):
            mask = 1.0 - float(self.dones[t])
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

        total_loss = total_policy_loss = total_value_loss = total_entropy = 0.0
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

        self.update_step += 1
        if self.update_step > self.ent_warmup_updates:
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
            "update_step": int(self.update_step),
        }

    # -------------------------
    # Save / Load
    # -------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "update_step": self.update_step,
                "ent_coef": float(self.ent_coef),
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        print("[LOAD] partial-load loader active")

        sd = ckpt.get("model", ckpt)
        cur = self.model.state_dict()

        # -------------------------
        # Helper: safe tensor copy
        # -------------------------
        def _copy_param(dst: torch.Tensor, src: torch.Tensor) -> bool:
            if (dst is None) or (src is None):
                return False
            if dst.shape != src.shape:
                return False
            dst.copy_(src)
            return True

        def _copy_conv_in_channels(dst_w: torch.Tensor, src_w: torch.Tensor) -> bool:
            """
            conv weight: (out_ch, in_ch, kH, kW)
            in_ch mismatch 시: 겹치는 채널만 복사하고 나머지는 0(또는 평균)로 채움.
            """
            if (dst_w.ndim != 4) or (src_w.ndim != 4):
                return False
            if (dst_w.shape[0] != src_w.shape[0]) or (dst_w.shape[2:] != src_w.shape[2:]):
                return False

            dst_w.zero_()
            c = min(dst_w.shape[1], src_w.shape[1])
            dst_w[:, :c].copy_(src_w[:, :c])

            # dst의 in_ch가 더 큰 경우: 남는 채널은 src 평균으로 채우면 학습 초반 안정적
            if dst_w.shape[1] > src_w.shape[1]:
                mean = src_w.mean(dim=1, keepdim=True)  # (out,1,kH,kW)
                dst_w[:, c:].copy_(mean.repeat(1, dst_w.shape[1] - c, 1, 1))
            return True

        def _copy_linear_partial(dst_w: torch.Tensor, dst_b: torch.Tensor,
                                src_w: torch.Tensor, src_b: torch.Tensor,
                                dst_offset: int):
            """
            dst_w: (out, dst_in)
            src_w: (out, src_in)
            dst_offset부터 src_in만큼 부분 복사.
            out 차원도 겹치는 만큼만.
            """
            if dst_w.ndim != 2 or src_w.ndim != 2:
                return 0
            out = min(dst_w.shape[0], src_w.shape[0])
            src_in = src_w.shape[1]
            if dst_offset >= dst_w.shape[1]:
                return 0
            copy_in = min(src_in, dst_w.shape[1] - dst_offset)

            # weight
            dst_w[:out, dst_offset:dst_offset + copy_in].copy_(src_w[:out, :copy_in])

            # bias (out 차원만 맞으면 복사)
            if (dst_b is not None) and (src_b is not None) and (dst_b.ndim == 1) and (src_b.ndim == 1):
                bb = min(dst_b.shape[0], src_b.shape[0])
                dst_b[:bb].copy_(src_b[:bb])

            return copy_in

        # -------------------------
        # 1) 먼저 "이식 가능한지" 체크
        # -------------------------
        has_old_single = any(k.startswith("conv.") for k in sd.keys()) and any(k.startswith("fc_img.") for k in sd.keys())
        has_new_two = any(k.startswith("conv_g.") for k in cur.keys()) and any(k.startswith("conv_l.") for k in cur.keys())

        transplanted = False
        filtered = {}
        skipped = []

        # -------------------------
        # 2) 기본 partial-load (이식 전에 일단 같은-shape 키는 로드 후보)
        # -------------------------
        for k, v in sd.items():
            if (k in cur) and (cur[k].shape == v.shape):
                filtered[k] = v
            else:
                skipped.append(k)

        # 우선 같은 키/shape는 로드
        msg = self.model.load_state_dict(filtered, strict=False)

        # -------------------------
        # 3) 단일 -> 2stream 이식
        # -------------------------
        if has_old_single and has_new_two:
            try:
                with torch.no_grad():
                    # ---- conv trunk 복사 ----
                    # old: conv.0/2/4.*  -> new: conv_g.0/2/4.*, conv_l.0/2/4.*
                    for layer in [0, 2, 4]:
                        ok_g = False
                        ok_l = False

                        old_w = sd.get(f"conv.{layer}.weight", None)
                        old_b = sd.get(f"conv.{layer}.bias", None)

                        new_g_w = cur.get(f"conv_g.{layer}.weight", None)
                        new_g_b = cur.get(f"conv_g.{layer}.bias", None)
                        new_l_w = cur.get(f"conv_l.{layer}.weight", None)
                        new_l_b = cur.get(f"conv_l.{layer}.bias", None)

                        if (old_w is not None) and (new_g_w is not None):
                            ok_g = _copy_conv_in_channels(cur[f"conv_g.{layer}.weight"], old_w)
                            if ok_g and (old_b is not None) and (new_g_b is not None) and (cur[f"conv_g.{layer}.bias"].shape == old_b.shape):
                                cur[f"conv_g.{layer}.bias"].copy_(old_b)

                        if (old_w is not None) and (new_l_w is not None):
                            ok_l = _copy_conv_in_channels(cur[f"conv_l.{layer}.weight"], old_w)
                            if ok_l and (old_b is not None) and (new_l_b is not None) and (cur[f"conv_l.{layer}.bias"].shape == old_b.shape):
                                cur[f"conv_l.{layer}.bias"].copy_(old_b)

                    # ---- fc_img 복사 ----
                    # old: fc_img.0.* -> new: fc_g.0.*, fc_l.0.*
                    old_fc_w = sd.get("fc_img.0.weight", None)
                    old_fc_b = sd.get("fc_img.0.bias", None)

                    if old_fc_w is not None and "fc_g.0.weight" in cur:
                        if cur["fc_g.0.weight"].shape == old_fc_w.shape:
                            cur["fc_g.0.weight"].copy_(old_fc_w)
                            if old_fc_b is not None and cur["fc_g.0.bias"].shape == old_fc_b.shape:
                                cur["fc_g.0.bias"].copy_(old_fc_b)

                    if old_fc_w is not None and "fc_l.0.weight" in cur:
                        if cur["fc_l.0.weight"].shape == old_fc_w.shape:
                            cur["fc_l.0.weight"].copy_(old_fc_w)
                            if old_fc_b is not None and cur["fc_l.0.bias"].shape == old_fc_b.shape:
                                cur["fc_l.0.bias"].copy_(old_fc_b)

                    # ---- meta 복사 ----
                    # old: fc_meta.0.* -> new도 동일 키가 있다면 복사
                    old_m_w = sd.get("fc_meta.0.weight", None)
                    old_m_b = sd.get("fc_meta.0.bias", None)
                    if old_m_w is not None and "fc_meta.0.weight" in cur:
                        if cur["fc_meta.0.weight"].shape == old_m_w.shape:
                            cur["fc_meta.0.weight"].copy_(old_m_w)
                            if old_m_b is not None and cur["fc_meta.0.bias"].shape == old_m_b.shape:
                                cur["fc_meta.0.bias"].copy_(old_m_b)

                    # ---- policy/value head 부분 이식 ----
                    # old head는 (A, 512+32)일 확률이 높고
                    # new head는 (A, 512(g)+512(l)+32(meta)) 같은 구조일 확률이 큼.
                    # => global(앞 512) + meta(맨끝 32)만 채우고 local(중간 512)은 0으로 둠.
                    old_pi_w = sd.get("policy_head.weight", None)
                    old_pi_b = sd.get("policy_head.bias", None)
                    old_v_w = sd.get("value_head.weight", None)
                    old_v_b = sd.get("value_head.bias", None)

                    if (old_pi_w is not None) and ("policy_head.weight" in cur):
                        dst_w = cur["policy_head.weight"]
                        dst_b = cur.get("policy_head.bias", None)

                        dst_w.zero_()
                        if dst_b is not None:
                            dst_b.zero_()

                        # 1) global slice (offset 0)
                        _copy_linear_partial(dst_w, dst_b, old_pi_w, old_pi_b, dst_offset=0)

                        # 2) meta slice (끝 32칸이 meta라고 가정)
                        meta_dim = 32
                        if dst_w.shape[1] >= meta_dim and old_pi_w.shape[1] >= meta_dim:
                            dst_meta_off = dst_w.shape[1] - meta_dim
                            src_meta_off = old_pi_w.shape[1] - meta_dim
                            # meta 부분만 따로 복사
                            dst_w[:min(dst_w.shape[0], old_pi_w.shape[0]), dst_meta_off:dst_meta_off + meta_dim] = \
                                old_pi_w[:min(dst_w.shape[0], old_pi_w.shape[0]), src_meta_off:src_meta_off + meta_dim]
                            if (dst_b is not None) and (old_pi_b is not None):
                                bb = min(dst_b.shape[0], old_pi_b.shape[0])
                                dst_b[:bb] = old_pi_b[:bb]

                    if (old_v_w is not None) and ("value_head.weight" in cur):
                        dst_w = cur["value_head.weight"]
                        dst_b = cur.get("value_head.bias", None)

                        dst_w.zero_()
                        if dst_b is not None:
                            dst_b.zero_()

                        _copy_linear_partial(dst_w, dst_b, old_v_w, old_v_b, dst_offset=0)

                        meta_dim = 32
                        if dst_w.shape[1] >= meta_dim and old_v_w.shape[1] >= meta_dim:
                            dst_meta_off = dst_w.shape[1] - meta_dim
                            src_meta_off = old_v_w.shape[1] - meta_dim
                            dst_w[:min(dst_w.shape[0], old_v_w.shape[0]), dst_meta_off:dst_meta_off + meta_dim] = \
                                old_v_w[:min(dst_w.shape[0], old_v_w.shape[0]), src_meta_off:src_meta_off + meta_dim]
                            if (dst_b is not None) and (old_v_b is not None):
                                bb = min(dst_b.shape[0], old_v_b.shape[0])
                                dst_b[:bb] = old_v_b[:bb]

                # 이식한 cur를 모델에 반영
                self.model.load_state_dict(cur, strict=False)
                transplanted = True
            except Exception as e:
                print(f"[LOAD][WARN] transplant failed (will keep partial-load only): {e}")
                transplanted = False

        # -------------------------
        # 4) optimizer / counters
        # -------------------------
        if load_optimizer:
            try:
                if "optimizer" in ckpt:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[WARN] optimizer state not loaded (model changed): {e}")

        self.global_step = int(ckpt.get("global_step", self.global_step))
        self.update_step = int(ckpt.get("update_step", self.update_step))
        if "ent_coef" in ckpt:
            self.ent_coef = float(ckpt["ent_coef"])

        # -------------------------
        # 5) print summary
        # -------------------------
        try:
            print("[LOAD] loaded keys:", len(filtered))
            print("[LOAD] missing keys(sample):", msg.missing_keys[:12], "..." if len(msg.missing_keys) > 12 else "")
            print("[LOAD] unexpected keys:", msg.unexpected_keys)
            if skipped:
                print("[LOAD] skipped incompatible keys(sample):", skipped[:10], "...")
            print(f"[LOAD] global_step={self.global_step} update_step={self.update_step} ent_coef={self.ent_coef:.6f}")
            if transplanted:
                print("[LOAD][TRANSPLANT] single->two_stream weights transplanted (conv/fc/meta + partial heads).")
            else:
                print("[LOAD][TRANSPLANT] not applied.")
        except Exception:
            pass
