import sys
import torch
from pathlib import Path
from collections import Counter
from gym import PyEnvironment

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from src.ppo import PPOAgent

def print_counter_table(counter: Counter):
    if not counter:
        print("(empty Counter)")
        return

    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    # Determine col width
    max_key_len = max(len(str(k)) for k, _ in sorted_items)
    max_val_len = max(len(str(v)) for _, v in sorted_items)

    header_item = "Reason"
    header_count = "Count"
    print(f"{header_item:<{max_key_len}}  {header_count:>{max_val_len}}")
    print(f"{'-' * max_key_len}  {'-' * max_val_len}")

    for key, count in sorted_items:
        print(f"{key:<{max_key_len}}  {count:>{max_val_len}}")

def run_test_episodes(
    max_steps: int = 3000,
    test_episodes: int = 100,
):
    env = PyEnvironment(max_steps)
    agent = PPOAgent(
        env,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.1,
        lr=5e-5,
        epochs=10,
        batch_size=64,
    )

    model_dir = "./models"
    policy_net_path = Path(model_dir) / "policy_net.pth"
    value_net_path = Path(model_dir) / "value_net.pth"

    if not policy_net_path.exists() or not value_net_path.exists():
        raise FileNotFoundError(
            f"Saved models not found. Expected: {policy_net_path} and {value_net_path}"
        )

    agent.policy.load_state_dict(torch.load(policy_net_path))
    agent.value.load_state_dict(torch.load(value_net_path))
    print("Loaded saved policy and value networks. Running test episodes...")

    rewards = []
    num_steps = []
    reasons = Counter()

    for i in range(test_episodes):
        obs = env.reset()
        start_state = [x for x in obs]  # deep-ish copy for printing
        done = False
        total_reward = 0.0
        steps = 0
        reason = "N/A"

        while not done:
            action, _ = agent.select_action(obs)
            obs, reward, done, reason = env.step(action)
            total_reward += reward
            steps += 1

        end_state = obs

        print(f"\nTest Episode {i + 1}:")
        print(f"Start State: {start_state}")
        print(f"End State:   {end_state}")
        print(f"Total Steps: {steps}")
        print(f"Total Reward: {total_reward:.2f}")

        rewards.append(total_reward)
        num_steps.append(steps)
        reasons[reason] += 1

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    avg_steps = sum(num_steps) / len(num_steps) if num_steps else 0.0

    print("\nOverall Metrics")
    print(f"Average Cumulative Reward: {avg_reward}")
    print(f"Average number of timesteps: {avg_steps}")
    print_counter_table(reasons)

    return {
        "rewards": rewards,
        "num_steps": num_steps,
        "reasons": reasons,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
    }


if __name__ == "__main__":
    run_test_episodes(test_episodes=1000)
