import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
import numpy as np
from gym import PyEnvironment

MIN_LOG_STD = -7 # Prevent collapse of std under 1e-3

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.mean_layer = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # std is learnable

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std)
        return mean, std

    def get_dist(self, obs):
        mean, std = self.forward(obs)
        return Normal(mean, std)

class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages

class PPOAgent:
    def __init__(self, env, gamma=0.99, lam=0.95, clip_eps=0.2, lr=3e-4, epochs=10, batch_size=64):
        self.env = env
        self.obs_dim = 6
        self.act_dim = 2
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        self.policy = PolicyNet(self.obs_dim, self.act_dim)
        self.value = ValueNet(self.obs_dim)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.value_optim = Adam(self.value.parameters(), lr=lr)

    @staticmethod
    def _squash_action(raw_action):
        squashed = torch.tanh(raw_action)
        return (squashed + 1) / 2  # [-1, 1] to [0, 1]

    @staticmethod
    def _unsquash_action(bounded_action):
        # action from buffer, already in [0,1]
        tanh_action = bounded_action * 2.0 - 1.0
        eps = torch.finfo(tanh_action.dtype).eps
        tanh_action = torch.clamp(tanh_action, -1.0 + eps, 1.0 - eps)
        # Stable atanh calc
        return 0.5 * (torch.log1p(tanh_action) - torch.log1p(-tanh_action))

    @staticmethod
    def _tanh_log_prob(dist, raw_action, squashed_action):
        log_prob = dist.log_prob(raw_action) - torch.log(1 - squashed_action.pow(2) + 1e-6)
        return log_prob.sum(dim=-1)

    def select_action(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        dist = self.policy.get_dist(obs_t)
        raw_action = dist.rsample()
        action = self._squash_action(raw_action)
        log_prob = self._tanh_log_prob(dist, raw_action, torch.tanh(raw_action))
        return action.detach().numpy()[0], log_prob.detach()

    def evaluate_action(self, obs, action):
        raw_action = self._unsquash_action(action)
        dist = self.policy.get_dist(obs)
        squashed = torch.tanh(raw_action)
        log_prob = self._tanh_log_prob(dist, raw_action, squashed)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(obs)
        return log_prob, entropy, value

    def collect_trajectories(self, horizon=2048):
        obs = self.env.reset()
        observations, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        for _ in range(horizon):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            dist = self.policy.get_dist(obs_t)
            value = self.value(obs_t).item()

            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1).item()

            next_obs, reward, done = self.env.step(action.detach().numpy().flatten())

            observations.append(obs)
            actions.append(action.detach().numpy()[0])
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value)

            obs = next_obs

            if done:
                obs = self.env.reset()

        return observations, actions, rewards, dones, log_probs, values

    def train(self, total_timesteps):
        timestep = 0
        while timestep < total_timesteps:
            obs_batch, act_batch, rew_batch, done_batch, logp_batch, val_batch = self.collect_trajectories()
            timestep += len(rew_batch)

            advantages = compute_gae(rew_batch, val_batch, done_batch, self.gamma, self.lam)
            returns = [adv + val for adv, val in zip(advantages, val_batch)]

            obs_tensor = torch.as_tensor(np.array(obs_batch), dtype=torch.float32)
            act_tensor = torch.as_tensor(np.array(act_batch), dtype=torch.float32)
            ret_tensor = torch.as_tensor(np.array(returns), dtype=torch.float32)
            adv_tensor = torch.as_tensor(np.array(advantages), dtype=torch.float32)
            old_logp_tensor = torch.as_tensor(np.array(logp_batch), dtype=torch.float32)


            # Normalize advantages
            adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8) # eps for DBZ

            # PPO update for given epochs
            for _ in range(self.epochs):
                for idx in range(0, len(obs_batch), self.batch_size):
                    slice_idx = slice(idx, idx + self.batch_size)
                    obs_b = obs_tensor[slice_idx]
                    act_b = act_tensor[slice_idx]
                    ret_b = ret_tensor[slice_idx]
                    adv_b = adv_tensor[slice_idx]
                    old_logp_b = old_logp_tensor[slice_idx]

                    new_logp, entropy, value = self.evaluate_action(obs_b, act_b)
                    ratio = torch.exp(new_logp - old_logp_b)

                    # PPO clipped objective
                    clip_adv = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                    policy_loss = -torch.min(ratio * adv_b, clip_adv).mean()
                    value_loss = nn.MSELoss()(value, ret_b)
                    entropy_bonus = entropy.mean()

                    total_loss = policy_loss + 0.5 * value_loss - 0.1 * entropy_bonus

                    self.policy_optim.zero_grad()
                    self.value_optim.zero_grad()
                    total_loss.backward()
                    self.policy_optim.step()
                    self.value_optim.step()

            print(f'Timesteps: {timestep}, Policy Loss: {policy_loss.item():.3f}, Value Loss: {value_loss.item():.3f}')

def main():
    env = PyEnvironment(1000)
    agent = PPOAgent(
        env,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        lr=3e-4,
        epochs=10,
        batch_size=64
    )

    total_timesteps = 10_000

    print("Training started.")
    agent.train(total_timesteps)
    print("Training completed.")

    # torch.save(agent.policy.state_dict(), "ppo_policy.pth")
    # torch.save(agent.value.state_dict(), "ppo_value.pth")
    # print("Model weights saved to pth files.")

    # Test episode
    obs = env.reset()
    done = False
    step = 1
    while not done:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        mean, _ = agent.policy.forward(obs_t)
        mean = mean[0]  # shape: (2,)
        squashed = torch.tanh(mean)
        action = ((squashed + 1) / 2).detach().numpy()  # ensure [0, 1]
        obs, reward, done = env.step(action)
        thrust = action.tolist()
        print(f"{step=} {obs=} {thrust=} {reward=}")
        step += 1

    print("Test episode completed.")

if __name__ == "__main__":
    main()
