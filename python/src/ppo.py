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
        std = torch.clamp(torch.exp(self.log_std), min=None, max=1)
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
        self, env, gamma=0.99, lam=0.95, clip_eps=0.2, lr=5e-5, epochs=10, batch_size=64
    ):
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        self.policy = PolicyNet(self.env.obs_dim, self.env.act_dim)
        self.value = ValueNet(self.env.obs_dim)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.value_optim = Adam(self.value.parameters(), lr=lr)

    @staticmethod
    def _score(rewards, dones):
        if not rewards:
            return float("-inf")

        num_ep = sum(dones)
        tot_rew = sum(rewards)

        if num_ep == 0:
            return float("-inf")
        return tot_rew / num_ep

    @staticmethod
    def _unsquash_action(bounded_action):
        # action from buffer, already in [-1,1]
        eps = torch.finfo(bounded_action.dtype).eps
        tanh_action = torch.clamp(bounded_action, -1.0 + eps, 1.0 - eps)
        return torch.atanh(tanh_action)

    @staticmethod
    def _tanh_log_prob(dist, raw_action, tanh_action):
        log_prob = dist.log_prob(raw_action) - torch.log(1 - tanh_action.pow(2) + 1e-6)
        return log_prob.sum(dim=-1)

    def _compute_gae(self, rewards, values, dones):
        advantages = torch.zeros((len(rewards),))
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        return advantages

    @torch.no_grad()
    def select_action(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        dist = self.policy.get_dist(obs_t)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        action_np = action.detach().flatten().numpy()
        log_prob = self._tanh_log_prob(dist, raw_action, torch.tanh(raw_action))
        return action_np, log_prob

    def evaluate_action(self, obs, action):
        raw_action = self._unsquash_action(action)
        dist = self.policy.get_dist(obs)
        tanh_action = torch.tanh(raw_action)
        log_prob = self._tanh_log_prob(dist, raw_action, tanh_action)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(obs)
        return log_prob, entropy, value

    @torch.no_grad()
    def collect_trajectories(self, horizon=4096):
        obs = self.env.reset()
        observations, actions, rewards, dones, log_probs, values = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for _ in range(horizon):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            dist = self.policy.get_dist(obs_t)
            value = self.value(obs_t).item()

            raw_action = dist.rsample()
            action = torch.tanh(raw_action)
            action_np = action.detach().flatten().numpy()
            log_prob = self._tanh_log_prob(dist, raw_action, torch.tanh(raw_action))

            next_obs, reward, done, _ = self.env.step(action_np)

            observations.append(obs)
            actions.append(action.detach().numpy().flatten())
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.item())
            values.append(value)

            obs = next_obs

            if done:
                obs = self.env.reset()

        # bootstrap value of last state
        last_val = 0
        if horizon > 0 and not dones[-1]:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            last_val = self.value(obs_t).item()

        return observations, actions, rewards, dones, log_probs, values, last_val

    def train(self, total_timesteps):
        timestep = 0
        best_score = float("-inf")
        while timestep < total_timesteps:
            (
                obs_batch,
                act_batch,
                rew_batch,
                done_batch,
                logp_batch,
                val_batch,
                last_val,
            ) = self.collect_trajectories(horizon=8192)
            timestep += len(obs_batch)

            score = self._score(rew_batch, done_batch)
            if score > best_score:
                best_score = score
                torch.save(self.policy.state_dict(), "./models/policy_net.pth")
                torch.save(self.value.state_dict(), "./models/value_net.pth")
                print(f"New best score {score:.2f} - Checkpoint saved")

            advantages = self._compute_gae(
                rew_batch, val_batch + [last_val], done_batch
            )
            ret_tensor = advantages + torch.tensor(val_batch)

            obs_tensor = torch.tensor(np.array(obs_batch), dtype=torch.float32)
            act_tensor = torch.tensor(np.array(act_batch), dtype=torch.float32)
            old_logp_tensor = torch.tensor(np.array(logp_batch), dtype=torch.float32)

            # Normalize advantages
            adv_tensor = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )  # eps for DBZ

            # PPO update
            num_samples = len(obs_batch)
            avg_policy_loss, avg_value_loss = 0.0, 0.0
            processed_batches = 0
            for _ in range(self.epochs):
                indices = torch.randperm(num_samples)
                for idx in range(0, num_samples, self.batch_size):
                    batch_indices = indices[idx : idx + self.batch_size]
                    obs_b = obs_tensor[batch_indices]
                    act_b = act_tensor[batch_indices]
                    ret_b = ret_tensor[batch_indices]
                    adv_b = adv_tensor[batch_indices]
                    old_logp_b = old_logp_tensor[batch_indices]

                    new_logp, entropy, value = self.evaluate_action(obs_b, act_b)
                    ratio = torch.exp(new_logp - old_logp_b)

                    # PPO clipped objective
                    clip_adv = (
                        torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                    )

                    policy_loss = (
                        -torch.min(ratio * adv_b, clip_adv).mean()
                        - 0.01 * entropy.mean()
                    )
                    self.policy_optim.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), max_norm=0.2
                    )
                    self.policy_optim.step()
                    avg_policy_loss += policy_loss.item()

                    value_loss = nn.MSELoss()(value, ret_b)
                    self.value_optim.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.value.parameters(), max_norm=0.2
                    )
                    self.value_optim.step()
                    avg_value_loss += value_loss.item()

                    processed_batches += 1

            avg_policy_loss /= processed_batches
            avg_value_loss /= processed_batches
            print(
                f"Timesteps: {timestep}, Policy Loss: {avg_policy_loss:.3f}, Value Loss: {avg_value_loss:.3f}"
            )


def main():
    # constants
    MAX_STEPS = 3000
    TOTAL_TIMESTEPS = 250_000

    env = PyEnvironment(MAX_STEPS)
    agent = PPOAgent(
        env, gamma=0.99, lam=0.95, clip_eps=0.2, lr=3e-4, epochs=4, batch_size=256
    )

    print("Training started.")
    agent.train(TOTAL_TIMESTEPS)
    print("Training completed.")


if __name__ == "__main__":
    main()
