import sys
import subprocess
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict

# Add project root to PATH
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tinympc.dynamics_model import DynamicsModel, INPUT_DIMS, OUTPUT_DIMS
from src.utils.load_toml_config import load_toml_config

# TODO: add support for normalized inputs to the model
class TrajectoryRolloutTester:
    def __init__(self):
        self.rust_bin_path = PROJECT_ROOT / "target" / "debug" / "simulate_trajectory"
        self.setup_model()

    def setup_model(self):
        training_config_path = str(PROJECT_ROOT / "dynamics_model_training_config.toml")
        config = load_toml_config(training_config_path)
        self.model = DynamicsModel(
            input_dim=INPUT_DIMS,
            output_dim=OUTPUT_DIMS,
            hidden_dims=config["hidden_dims"],
            dropout_rate=config["dropout_rate"]
        )

        model_path = config["model_save_path"]
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        checkpoint = torch.load(str(model_path), map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Loaded pre-trained model from {model_path}")

    def generate_initial_conditions(self) -> Dict[str, float]:
        # TODO: rewrite this using constants instead of raw values
        return {
            'pos_x': np.random.uniform(30.0, 50.0),  # plus minus 10m from center
            'pos_y': np.random.uniform(25.0, 40.0),  # Upper half
            'angle': np.random.uniform(-np.pi/4, np.pi/4)  # plus minus 45 degrees
        }

    def run_rust_simulation(self, initial_conditions: Dict[str, float]) -> pd.DataFrame:
        if not self.rust_bin_path.exists():
            self.build_rust_binary()

        cmd = [
            str(self.rust_bin_path),
            str(initial_conditions['pos_x']),
            str(initial_conditions['pos_y']),
            str(initial_conditions['angle'])
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            from io import StringIO
            df = pd.read_csv(StringIO(result.stdout))
            return df
        except subprocess.CalledProcessError as e:
            print(f"Error running Rust simulation: {e}")
            print(f"Stderr: {e.stderr}")
            raise

    def build_rust_binary(self):
        print("Building Rust binary...")
        try:
            subprocess.run(
                ["cargo", "build", "--bin", "simulate_trajectory"],
                cwd=PROJECT_ROOT,
                check=True
            )
            print("Successfully built Rust binary")
        except subprocess.CalledProcessError as e:
            print(f"Error building Rust binary: {e}")
            raise

    def simulate_with_model(self, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        states = []

        for action in actions:
            states.append(state.copy())
            model_input = np.concatenate([state, action])
            input_tensor = torch.FloatTensor(model_input).unsqueeze(0)

            with torch.no_grad():
                deltas = self.model(input_tensor).numpy()[0]

            state += deltas

        return np.array(states)

    def calculate_metrics(self, rust_states: np.ndarray, model_states: np.ndarray) -> Dict[str, float]:
        # Force same length
        min_len = min(len(rust_states), len(model_states))
        rust_states = rust_states[:min_len]
        model_states = model_states[:min_len]

        errors = rust_states - model_states
        mse = np.mean(errors**2, axis=0)

        return {
            'mse_pos': np.mean(mse[:2]),  # Position MSE (x,y)
            'mse_angle': mse[2],          # Angle MSE
            'mse_vel': np.mean(mse[3:5]),  # Velocity MSE (vx, vy)
            'mse_angular_vel': mse[5],     # Angular velocity MSE
        }

    def plot_trajectories(self, rust_positions: np.ndarray, model_positions: np.ndarray, save_path: str):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))

        # Plot rust and model trajectories (only position)
        plt.scatter(
            rust_positions[:, 0],
            rust_positions[:, 1],
            c='blue',
            label='Actual Trajectory',
            alpha=0.6,
            s=10
        )
        plt.scatter(
            model_positions[:, 0],
            model_positions[:, 1],
            c='red',
            label='Predicted Trajectory',
            alpha=0.6,
            s=10,
            marker='x'
        )

        # Add start marker
        plt.scatter(
            rust_positions[0, 0],
            rust_positions[0, 1],
            c='green',
            marker='o',
            s=100,
            label='Start'
        )

        # Setting limits slightly around the actual x and y limits
        # TODO: use constants
        plt.xlim(0, 80)
        plt.ylim(0, 50)

        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Rocket Trajectory comparison')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Saved trajectory comparison plot to {save_path}")
        plt.close()


def test_trajectory_rollout():
    tester = TrajectoryRolloutTester()

    initial_conditions = tester.generate_initial_conditions()
    print(f"Initial conditions: {initial_conditions}")

    print("Running Rust simulation...")
    rust_traj = tester.run_rust_simulation(initial_conditions)
    initial_state = np.array([
        initial_conditions['pos_x'],
        initial_conditions['pos_y'],
        initial_conditions['angle'],
        0.0,  # Initial velocity x
        0.0,  # Initial velocity y
        0.0   # Initial angular velocity
    ])

    actions = rust_traj[['left_thrust', 'right_thrust']].to_numpy()

    print("Running model simulation...")
    model_states = tester.simulate_with_model(initial_state, actions)

    metrics = tester.calculate_metrics(
        rust_traj[['pos_x', 'pos_y', 'angle', 'vel_x', 'vel_y', 'angular_vel']].to_numpy(),
        model_states
    )

    print("\n=== Trajectory rollout results ===")
    print(f"Position MSE: {metrics['mse_pos']:.6f}")
    print(f"Angle MSE: {metrics['mse_angle']:.6f}")
    print(f"Velocity MSE: {metrics['mse_vel']:.6f}")
    print(f"Angular Velocity MSE: {metrics['mse_angular_vel']:.6f}")

    tester.plot_trajectories(
        rust_traj[['pos_x', 'pos_y']].to_numpy(),
        model_states[:, :2],
        save_path=str(PROJECT_ROOT / 'plots' / 'trajectory_comparison.png')
    )

if __name__ == "__main__":
    test_trajectory_rollout()
