# tinympc

A Proximal Policy Optimization (PPO) agent trained to emulate an MPC in landing scenarios. The rocket landing simulation is built in Rust using `rapier2d` and `macroquad`, with FFI bindings to python for training the RL model.

## PPO Model

The Proximal Policy Optimization (PPO) model is trained to control the rocket based on the following observation and action spaces:

### Observation Space

The observation space is a 6-dimensional vector (since the sim is only in 2D) containing the following in order:

-	$x, y$: The x and y coordinates of the rocket in the world.
-	$\theta$: The angle of the rocket from the vertical.
-	$v_x, v_y$: The linear velocity of the rocket in the x and y directions.
-	$\omega$: The angular velocity of the rocket.

### Action Space

The action space is a 2-dimensional vector representing standard thrust vector controls:

-	$F_{\text{thrust}}$: The amount of thrust to apply, normalized to the range [-1, 1].
-	$\theta_{\text{gimbal}}$: The angle of the gimbal, normalized to the range [-1, 1].

See `base/src/constants.rs` for the true min and max values for the action and observation space.


## Project Structure

The project is divided into three main parts:

-   `base`: A Rust crate that contains the simulation (including the game engine, physics, and rendering).
-   `gym`: A Rust crate that provides a Python binding to the simulation for a "gym" style reinforcement learning environment.
-   `python`: Source code for training the PPO agent using PyTorch.


## Prerequisites

-   [Rust](https://www.rust-lang.org/tools/install)
-   [Python 3.12+](https://www.python.org/downloads/)
-   [Poetry](https://python-poetry.org/docs/#installation)

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/akkshay0107/tinympc.git
     cd tinympc
    ```

2.  **Set up python venv inside poetry and install dependencies:**

    ```bash
    cd python
     poetry install
    ```

## Usage

### Training the Model

To train the PPO agent, run the following commands from the `python` directory:

```bash
poetry run maturin develop # builds and installs the gym crate as a wheel in the venv
poetry run python ./src/ppo.py
```

The trained model will be saved to `python/models/policy_net.pth`.

Additionally to run test episodes using the trained model, run the following command:

```bash
poetry run python ./tests/ppo_test_episodes.py
```

### Running the Simulation

To run the simulation with the PPO agent providing controls, you first need to export the trained model to ONNX format. From the `python` directory, run:

```bash
poetry run python ./utils/export_to_onnx.py
```

Then, run the simulation with the following command from the project root:

```bash
cargo run --bin controlled_sim --release
```

Additionally, the simulation can also be run with keyboard inputs (instead of the model providing controls). For this, from the project root, run:

```bash
cargo run --bin base --release --features="keyinput"
```
