import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

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

        # Initialize weights
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

        if action is None:
            action = dist.sample()

        # Apply tanh squashing to [-1,1] and normalize to [0,1]
        action = torch.tanh(action)
        action = (action + 1.0) / 2.0
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return action, log_prob, entropy, value

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

                # Calculate ratio (pi_theta / pi_theta_old)
                ratio = (log_prob - old_log_prob).exp()

                # Surrogate loss
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

                # Policy loss
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(return_, value)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

                # Optimize
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

def main():
    state_dim = 6 # [x, y, theta, x dot, y dot, theta dot]
    action_dim = 2 # [T_L, T_R] (normalized to [0,1])

    agent = PPO(state_dim, action_dim)
    pass

if __name__ == "__main__":
    main()
