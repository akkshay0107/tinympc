import numpy as np
import torch
import torch.nn as nn
from gym import PyEnvironment
from torch.distributions import Normal
from torch.optim import Adam


def _init_layer(linear: nn.Linear, gain: float):
    nn.init.orthogonal_(linear.weight, gain)  # type: ignore
    nn.init.constant_(linear.bias, 0.0)


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # std is learnable

        gain = nn.init.calculate_gain("relu")
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                _init_layer(layer, gain)
        _init_layer(self.mean_layer, 0.01)

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean_layer(x)
        std = self.log_std.clamp(-20, 2).exp()
        return mean, std

    def get_dist(self, obs):
        mean, std = self.forward(obs)
        return Normal(mean, std)


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        gain = nn.init.calculate_gain("relu")
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                _init_layer(layer, gain)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class PPOAgent:
    def __init__(
        self,
        env,
        gamma=0.995,
        lam=0.95,
        clip_eps=0.2,
        lr=1e-4,
        epochs=4,
        batch_size=256,
        ent_coef=0.01,
        vf_coef=0.5,
        device="cpu",
    ):
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        self.device = torch.device(device)
        self.policy = PolicyNet(self.env.obs_dim, self.env.act_dim).to(self.device)
        self.value = ValueNet(self.env.obs_dim).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.value_optim = Adam(self.value.parameters(), lr=lr)

    @staticmethod
    def _unsquash_action(bounded_action: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(bounded_action.dtype).eps
        a = torch.clamp(bounded_action, -1.0 + eps, 1.0 - eps)
        return torch.atanh(a)

    @staticmethod
    def _tanh_log_prob(
        dist: Normal, raw_action: torch.Tensor, tanh_action: torch.Tensor
    ) -> torch.Tensor:
        log_prob = dist.log_prob(raw_action) - torch.log(1 - tanh_action.pow(2) + 1e-6)
        return log_prob.sum(dim=-1)

    def _compute_gae(self, rewards, values, next_values, bootstrap_mask, dones):
        T = rewards.shape[0]
        adv = torch.zeros((T,), device=rewards.device, dtype=rewards.dtype)
        gae = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
        for t in reversed(range(T)):
            delta = (
                rewards[t] + self.gamma * next_values[t] * bootstrap_mask[t] - values[t]
            )
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            adv[t] = gae
        return adv

    @torch.no_grad()
    def collect_trajectories(self, horizon=8192):
        obs = self.env.reset()

        obs_buf = np.zeros((horizon, self.env.obs_dim), dtype=np.float32)
        act_buf = np.zeros((horizon, self.env.act_dim), dtype=np.float32)
        logp_buf = np.zeros((horizon,), dtype=np.float32)
        rew_buf = np.zeros((horizon,), dtype=np.float32)
        done_buf = np.zeros((horizon,), dtype=np.bool_)
        trunc_buf = np.zeros((horizon,), dtype=np.bool_)
        val_buf = np.zeros((horizon,), dtype=np.float32)
        val_next_buf = np.zeros((horizon,), dtype=np.float32)

        for t in range(horizon):
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            dist = self.policy.get_dist(obs_t)
            v = self.value(obs_t).item()

            raw_action = dist.rsample()
            tanh_action = torch.tanh(raw_action)
            logp = self._tanh_log_prob(dist, raw_action, tanh_action).item()

            action_np = tanh_action.squeeze(0).cpu().numpy().astype(np.float32)
            next_obs, reward, done, reason = self.env.step(action_np)
            is_trunc = bool(done and (reason == "timeout"))

            next_obs_t = torch.as_tensor(
                next_obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            v_next = self.value(next_obs_t).item()

            obs_buf[t] = obs
            act_buf[t] = action_np
            logp_buf[t] = logp
            rew_buf[t] = reward
            done_buf[t] = done
            trunc_buf[t] = is_trunc
            val_buf[t] = v
            val_next_buf[t] = v_next

            obs = next_obs
            if done:
                obs = self.env.reset()

        return (
            obs_buf,
            act_buf,
            logp_buf,
            rew_buf,
            done_buf,
            trunc_buf,
            val_buf,
            val_next_buf,
        )

    def train(self, num_rollouts, horizon=8192):
        time = 0
        while time < num_rollouts:
            obs_b, act_b, logp_old_b, rew_b, done_b, trunc_b, val_old_b, val_next_b = (
                self.collect_trajectories(horizon=horizon)
            )
            time += 1

            obs_t = torch.as_tensor(obs_b, dtype=torch.float32, device=self.device)
            act_t = torch.as_tensor(act_b, dtype=torch.float32, device=self.device)
            logp_old_t = torch.as_tensor(
                logp_old_b, dtype=torch.float32, device=self.device
            )
            rew_t = torch.as_tensor(rew_b, dtype=torch.float32, device=self.device)
            val_old_t = torch.as_tensor(
                val_old_b, dtype=torch.float32, device=self.device
            )
            val_next_t = torch.as_tensor(
                val_next_b, dtype=torch.float32, device=self.device
            )

            terminated_t = torch.as_tensor(
                done_b & (~trunc_b), dtype=torch.float32, device=self.device
            )
            done_t = torch.as_tensor(done_b, dtype=torch.float32, device=self.device)
            bootstrap_mask = 1.0 - terminated_t

            adv_t = self._compute_gae(
                rew_t, val_old_t, val_next_t, bootstrap_mask, done_t
            )
            ret_t = adv_t + val_old_t

            adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

            n = obs_t.shape[0]
            avg_pi_loss, avg_v_loss, num_batches = 0.0, 0.0, 0

            for _ in range(self.epochs):
                idx = torch.randperm(n, device=self.device)
                for start in range(0, n, self.batch_size):
                    b = idx[start : start + self.batch_size]
                    obs_mb = obs_t[b]
                    act_mb = act_t[b]
                    logp_old_mb = logp_old_t[b]
                    adv_mb = adv_t[b]
                    ret_mb = ret_t[b]

                    raw_action = self._unsquash_action(act_mb)
                    dist = self.policy.get_dist(obs_mb)
                    tanh_action = torch.tanh(raw_action)
                    logp = self._tanh_log_prob(dist, raw_action, tanh_action)
                    entropy_proxy = dist.entropy().sum(
                        dim=-1
                    )  # entropy from the original normal dist

                    ratio = torch.exp(logp - logp_old_mb)
                    surr1 = ratio * adv_mb
                    surr2 = (
                        torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                        * adv_mb
                    )
                    pi_loss = (
                        -(torch.min(surr1, surr2)).mean()
                        - self.ent_coef * entropy_proxy.mean()
                    )

                    self.policy_optim.zero_grad(set_to_none=True)
                    pi_loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    self.policy_optim.step()

                    value = self.value(obs_mb)
                    value_loss = self.vf_coef * nn.MSELoss()(value, ret_mb)

                    self.value_optim.zero_grad(set_to_none=True)
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                    self.value_optim.step()

                    avg_pi_loss += pi_loss.item()
                    avg_v_loss += value_loss.item()
                    num_batches += 1

            avg_pi_loss /= max(num_batches, 1)
            avg_v_loss /= max(num_batches, 1)
            print(
                f"Rollout: {time}, Policy Loss: {avg_pi_loss:.3f}, Value Loss: {avg_v_loss:.3f}"
            )

            if time % 100 == 0:
                torch.save(self.policy.state_dict(), "./models/policy_net.pth")
                torch.save(self.value.state_dict(), "./models/value_net.pth")


def main():
    import os

    os.makedirs("./models", exist_ok=True)

    # constants
    MAX_STEPS = 3000
    NUM_ROLLOUTS = 1000

    env = PyEnvironment(MAX_STEPS)
    agent = PPOAgent(env)

    # optionally load old dicts before training
    # state_dict = torch.load("./models/policy_net.pth", map_location=agent.device)
    # agent.policy.load_state_dict(state_dict)

    # state_dict = torch.load("./models/value_net.pth", map_location=agent.device)
    # agent.value.load_state_dict(state_dict)

    print("Training started.")
    agent.train(NUM_ROLLOUTS)
    print("Training completed.")


if __name__ == "__main__":
    main()
