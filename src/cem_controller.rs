use crate::dynamics_model::DynamicsModel;
use ndarray::{Array1, Array2};
use rand::rng;
use rand_distr::{Distribution, Normal};

pub struct CEMController {
    dynamics_model: DynamicsModel,
    horizon: usize,
    num_samples: usize,
    num_elite: usize,
    max_iterations: usize,
    action_dim: usize,
    state_dim: usize,
    action_bounds: (f32, f32), // (min, max) for thrust
    convergence_threshold: f32,
    alpha: f32, // smoothing factor for mean/std updates
}

impl CEMController {
    pub fn new(
        model_path: &str,
        horizon: usize,
        num_samples: usize,
        num_elite: usize,
        max_iterations: usize,
        action_bounds: (f32, f32),
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let dynamics_model = DynamicsModel::new(model_path)?;

        Ok(CEMController {
            dynamics_model,
            horizon,
            num_samples,
            num_elite,
            max_iterations,
            action_dim: 2,
            state_dim: 6,
            action_bounds,
            convergence_threshold: 0.1,
            alpha: 0.1,
        })
    }

    pub fn control(
        &mut self,
        initial_state: &Array1<f32>,
        goal_state: &Array1<f32>,
        q_weights: &Array1<f32>, // Weights for state cost
        r_weights: &Array1<f32>, // Weights for action cost
    ) -> Result<Array1<f32>, Box<dyn std::error::Error>> {
        let mut action_mean = Array2::zeros((self.horizon, self.action_dim));
        let mut action_std = Array2::from_elem((self.horizon, self.action_dim), 1.0);

        let mut rng = rng();
        let mut best_cost = f32::INFINITY;
        let mut best_actions = Array2::zeros((self.horizon, self.action_dim));

        for i in 0..self.max_iterations {
            let mut action_samples = Vec::new();
            let mut costs = Vec::new();

            for _ in 0..self.num_samples {
                let mut actions = Array2::zeros((self.horizon, self.action_dim));

                // Sample actions from current distribution
                for t in 0..self.horizon {
                    for a in 0..self.action_dim {
                        let normal = Normal::new(action_mean[[t, a]], action_std[[t, a]])
                            .map_err(|e| format!("Failed to create normal distribution: {}", e))?;
                        let mut action: f32 = normal.sample(&mut rng);

                        // Clip to bounds
                        action = action.clamp(self.action_bounds.0, self.action_bounds.1);
                        actions[[t, a]] = action;
                    }
                }

                let cost = self.evaluate_trajectory(
                    initial_state,
                    &actions,
                    goal_state,
                    q_weights,
                    r_weights,
                )?;

                action_samples.push(actions);
                costs.push(cost);

                // Track best solution
                if cost < best_cost {
                    best_cost = cost;
                    best_actions = action_samples.last().unwrap().clone();
                }
            }

            let mut indices: Vec<usize> = (0..self.num_samples).collect();
            indices.sort_by(|&a, &b| costs[a].partial_cmp(&costs[b]).unwrap());

            let elite_indices = &indices[..self.num_elite];

            let mut new_mean = Array2::zeros((self.horizon, self.action_dim));
            let mut new_std = Array2::zeros((self.horizon, self.action_dim));

            // Calculate new mean and std
            for &idx in elite_indices {
                new_mean = new_mean + &action_samples[idx];
            }
            new_mean = new_mean / (self.num_elite as f32);

            for &idx in elite_indices {
                let diff = &action_samples[idx] - &new_mean;
                new_std = new_std + &diff.mapv(|x| x * x);
            }
            new_std = new_std / (self.num_elite as f32);
            let eps: f32 = 1e-6; // For preventing instability
            new_std = new_std.mapv(|x: f32| x.sqrt() + eps);

            action_mean = (1.0 - self.alpha) * &action_mean + self.alpha * &new_mean;
            action_std = (1.0 - self.alpha) * &action_std + self.alpha * &new_std;

            let std_sum: f32 = action_std.sum();
            if std_sum < self.convergence_threshold {
                println!("CEM converged after {} iterations", i + 1);
                break;
            }

            #[cfg(feature = "logging")]
            {
                if (i + 1) % 10 == 0 {
                    println!(
                        "Iteration {}: Best cost = {:.6}, Avg std = {:.6}",
                        i,
                        best_cost,
                        std_sum / (self.horizon * self.action_dim) as f32
                    );
                }
            }
        }

        println!("CEM completed. Best cost: {:.6}", best_cost);
        Ok(best_actions.row(0).to_owned())
    }

    fn evaluate_trajectory(
        &mut self,
        initial_state: &Array1<f32>,
        actions: &Array2<f32>,
        goal_state: &Array1<f32>,
        q_weights: &Array1<f32>,
        r_weights: &Array1<f32>,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let mut state = initial_state.clone();
        let mut total_cost = 0.0;

        for t in 0..self.horizon {
            let action = actions.row(t);
            let action_cost: f32 = action
                .iter()
                .zip(r_weights.iter())
                .map(|(a, r)| r * a * a)
                .sum();

            let mut model_input = Array2::zeros((1, 8));
            for i in 0..self.state_dim {
                model_input[[0, i]] = state[i];
            }
            for i in 0..self.action_dim {
                model_input[[0, self.state_dim + i]] = action[i];
            }

            let next_state_array = self.dynamics_model.predict(&model_input)?;
            let next_state = next_state_array
                .into_dimensionality::<ndarray::Ix2>()?
                .row(0)
                .to_owned();

            let state_diff = &next_state - goal_state;
            let state_cost: f32 = state_diff
                .iter()
                .zip(q_weights.iter())
                .map(|(s, q)| q * s * s)
                .sum();

            total_cost += state_cost + action_cost;
            state = next_state;
        }

        let terminal_diff = &state - goal_state;
        let terminal_cost: f32 = terminal_diff
            .iter()
            .zip(q_weights.iter())
            .map(|(s, q)| q * s * s * 10.0) // Higher weight for terminal state
            .sum();

        total_cost += terminal_cost;

        Ok(total_cost)
    }

    pub fn get_trajectory(
        &mut self,
        initial_state: &Array1<f32>,
        actions: &Array2<f32>,
    ) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        let mut trajectory = Array2::zeros((self.horizon + 1, self.state_dim));
        let mut state = initial_state.clone();

        for i in 0..self.state_dim {
            trajectory[[0, i]] = state[i];
        }

        for t in 0..self.horizon {
            let action = actions.row(t);

            let mut model_input = Array2::zeros((1, 8));
            for i in 0..self.state_dim {
                model_input[[0, i]] = state[i];
            }
            for i in 0..self.action_dim {
                model_input[[0, self.state_dim + i]] = action[i];
            }

            let next_state_array = self.dynamics_model.predict(&model_input)?;
            let next_state = next_state_array
                .into_dimensionality::<ndarray::Ix2>()?
                .row(0)
                .to_owned();

            // Store in trajectory
            for i in 0..self.state_dim {
                trajectory[[t + 1, i]] = next_state[i];
            }

            state = next_state;
        }

        Ok(trajectory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    const MODEL_PATH: &str = "models/dynamics_model.onnx";

    fn ensure_model_exists() {
        assert!(
            Path::new(MODEL_PATH).exists(),
            "Model file not found at {}.",
            MODEL_PATH
        );
    }

    #[test]
    fn test_cem_controller_initialization() {
        ensure_model_exists();

        let controller = CEMController::new(
            MODEL_PATH,
            10,          // horizon
            100,         // num_samples
            10,          // num_elite
            50,          // max_iterations
            (-1.0, 1.0), // action_bounds
        );

        assert!(controller.is_ok(), "Failed to initialize CEMController");
    }

    #[test]
    fn test_cem_control() {
        ensure_model_exists();

        let mut controller = CEMController::new(
            MODEL_PATH,
            5,          // horizon
            50,         // num_samples
            5,          // num_elite
            20,         // max_iterations
            (0.0, 5.0), // action_bounds
        )
        .expect("Failed to create controller");

        let initial_state = Array1::zeros(6);
        let goal_state = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let q_weights = Array1::from_vec(vec![1.0, 1.0, 1.0, 0.1, 0.1, 0.1]);
        let r_weights = Array1::from_vec(vec![0.1, 0.1]);

        let result = controller.control(&initial_state, &goal_state, &q_weights, &r_weights);
        assert!(result.is_ok(), "CEM control failed");

        let action = result.unwrap();
        assert_eq!(action.len(), 2, "Action should have 2 elements");

        for &a in action.iter() {
            assert!(a >= 0.0 && a <= 5.0, "Action {} out of bounds", a);
        }

        println!("Computed action: {:?}", action);
    }

    #[test]
    fn test_trajectory_generation() {
        ensure_model_exists();

        let mut controller = CEMController::new(
            MODEL_PATH,
            3,          // horizon
            10,         // num_samples
            2,          // num_elite
            5,          // max_iterations
            (0.0, 5.0), // action_bounds
        )
        .expect("Failed to create controller");

        let initial_state = Array1::zeros(6);
        let actions = Array2::zeros((3, 2)); // 3 time steps, 2 actions each

        let result = controller.get_trajectory(&initial_state, &actions);
        assert!(result.is_ok(), "Trajectory generation failed");

        let trajectory = result.unwrap();
        assert_eq!(trajectory.shape(), [4, 6], "Unexpected trajectory shape"); // horizon+1 x state_dim

        println!("Generated trajectory shape: {:?}", trajectory.shape());
    }
}
