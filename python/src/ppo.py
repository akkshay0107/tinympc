import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from gym import PyEnvironment

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Shared network layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Actor head - outputs mean and log_std for each action dimension
        self.actor_mean = nn.Linear(128, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic head - outputs state value
        self.critic = nn.Linear(128, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1)
            nn.init.constant_(module.bias, 0)

    def forward(self, state):
        shared = self.shared_layers(state)
        return self.actor_mean(shared), self.critic(shared)

    def act(self, state, action=None):
        mean, value = self.forward(state)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        if action is None:  # During rollout
            raw_action = dist.sample()
            env_action = torch.tanh(raw_action)
            env_action = (env_action + 1.0) / 2.0
        else:  # During update, action is from buffer
            env_action = action
            # Need to get the pre_tanh_action that would produce it
            tanh_action = action * 2.0 - 1.0
            # Clamp to prevent NaNs in log
            eps = torch.finfo(tanh_action.dtype).eps
            tanh_action = torch.clamp(tanh_action, -1.0 + eps, 1.0 - eps)
            # Stable atanh using log1p
            raw_action = 0.5 * (torch.log1p(tanh_action) - torch.log1p(-tanh_action))

        log_prob = dist.log_prob(raw_action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return env_action, log_prob, entropy, value

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_param=0.2, ppo_epochs=10, batch_size=64):
        self.device = torch.device("cpu") # only torch-cpu installed in project
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def compute_gae(self, next_value, rewards, masks, values):
        values = values + [next_value]
        gae = 0
        returns = []

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            returns.insert(0, gae + values[step])

        return returns

    def ppo_iter(self, states, actions, log_probs, returns, advantages):
        batch_size = states.size(0)
        indices = np.arange(batch_size)
        np.random.shuffle(indices)

        for start in range(0, batch_size, self.batch_size):
            end = start + self.batch_size
            batch_indices = indices[start:end]

            yield (
                states[batch_indices],
                actions[batch_indices],
                log_probs[batch_indices],
                returns[batch_indices],
                advantages[batch_indices]
            )

    def update(self, rollouts):
        states, actions, old_log_probs, returns, advantages = rollouts

        for _ in range(self.ppo_epochs):
            for state, action, old_log_prob, return_, advantage in self.ppo_iter(
                states, actions, old_log_probs, returns, advantages
            ):
                action_new, log_prob, entropy, value = self.policy.act(state, action)

                # pi_theta / pi_theta_old
                ratio = (log_prob - old_log_prob).exp()

                # Surrogate loss
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

                # Policy loss
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(return_, value.squeeze(-1))

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

    def get_value(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, value = self.policy(state)
            return value.cpu().numpy()[0]

    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _, _, _ = self.policy.act(state)
            return action.squeeze(0).cpu().numpy()

def gather_trajectory(agent, env, horizon, device):
    state = env.reset()
    states = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    masks = []

    for _ in range(horizon):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_tensor, log_prob, _, value = agent.policy.act(state_tensor)
            action = action_tensor.squeeze(0).cpu().numpy()
            log_prob = log_prob.cpu().numpy()
            value = value.item()

        next_state, reward, done = env.step(action)

        # All must be arrays or lists for later stacking
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
        masks.append(1.0 - float(done))  # mask = !done

        state = next_state
        if done:
            state = env.reset()

    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        _, next_value = agent.policy(state_tensor)
    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.float32),
        np.array(log_probs, dtype=np.float32),
        np.array(rewards, dtype=np.float32),
        np.array(values, dtype=np.float32),
        np.array(masks, dtype=np.float32),
        next_value.item()
    )

def train():
    state_dim = 6
    action_dim = 2
    agent = PPO(state_dim, action_dim)
    env = PyEnvironment(max_steps=1000)
    device = agent.device

    NUM_UPDATES = 1000
    TIMESTEPS_PER_ROLLOUT = 2 * env.max_steps
    EVAL_EVERY = 50

    for update in range(NUM_UPDATES):
        (
            states, actions, log_probs, rewards, values, masks, next_value
        ) = gather_trajectory(agent, env, TIMESTEPS_PER_ROLLOUT, device)

        returns = agent.compute_gae(next_value, rewards.tolist(), masks.tolist(), values.tolist())
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = returns - torch.tensor(values, dtype=torch.float32)

        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        log_probs = torch.tensor(log_probs, dtype=torch.float32)
        # returns, advantages already set to tensors

        # PPO update
        agent.update((states, actions, log_probs, returns, advantages))

        if (update + 1) % EVAL_EVERY == 0:
            reward = evaluate(agent, env, episodes=5)
            print(f"Update {update+1}: Mean Eval Reward {reward:.3f}")

def evaluate(agent, env, episodes=5):
    total_reward = 0.0
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0.0
        done = False
        for _ in range(env.max_steps):
            action = agent.act(state)
            state, reward, done = env.step(action)
            ep_reward += reward
            if done:
                break
        total_reward += ep_reward
    return total_reward / episodes

if __name__ == "__main__":
    train()
