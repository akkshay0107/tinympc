use std::f32::consts::PI;

use base::constants::*;
use base::world::World;
use pyo3::prelude::*;
use rand::Rng;
use rapier2d::na::Isometry2;
use rapier2d::prelude::*;

const MAX_SAFE_Y: f32 = MAX_POS_Y - 10.0; // Prevent rocket going out of bounds immediately

#[derive(Debug, Clone, Copy, PartialEq)]
enum EpisodeStatus {
    InProgress,
    Success,
    Crash,
    OutOfBounds,
    Timeout,
}

impl EpisodeStatus {
    fn as_str(&self) -> &'static str {
        match self {
            EpisodeStatus::InProgress => "in_progress",
            EpisodeStatus::Success => "success",
            EpisodeStatus::Crash => "crash",
            EpisodeStatus::OutOfBounds => "out_of_bounds",
            EpisodeStatus::Timeout => "timeout",
        }
    }
}

#[pyclass]
pub struct PyEnvironment {
    world: World,
    prev_potential: f32,
    steps: u32,
    max_steps: u32,
    obs_dim: u32,
    act_dim: u32,
}

#[pymethods]
impl PyEnvironment {
    #[new]
    pub fn new(max_steps: u32) -> Self {
        let world = World::new();
        Self {
            world,
            prev_potential: 0.0,
            steps: 0,
            max_steps,
            obs_dim: 6,
            act_dim: 2,
        }
    }

    // world doesn't need a getter
    #[getter]
    pub fn get_steps(&self) -> PyResult<u32> {
        Ok(self.steps)
    }

    #[getter]
    pub fn get_max_steps(&self) -> PyResult<u32> {
        Ok(self.max_steps)
    }

    #[getter]
    pub fn get_prev_potential(&self) -> PyResult<f32> {
        Ok(self.prev_potential)
    }

    #[getter]
    pub fn get_obs_dim(&self) -> PyResult<u32> {
        Ok(self.obs_dim)
    }

    #[getter]
    pub fn get_act_dim(&self) -> PyResult<u32> {
        Ok(self.act_dim)
    }

    pub fn reset(&mut self) -> PyResult<[f32; 6]> {
        self.world = World::new();
        self.steps = 0;

        let init_state = self._sample(); // Random valid state
        self.prev_potential = self.calculate_potential(
            init_state[0],
            init_state[1],
            init_state[2],
            init_state[3],
            init_state[4],
            init_state[5],
        );

        // Set the state to the rocket in the world
        let rocket = self
            .world
            .rigid_body_set
            .get_mut(self.world.rocket_body_handle)
            .unwrap();
        rocket.set_position(
            Isometry2::new(vector![init_state[0], init_state[1]], init_state[2]),
            true,
        );
        rocket.set_linvel(vector![init_state[3], init_state[4]], true);
        rocket.set_angvel(init_state[5], true);

        Ok(init_state)
    }

    pub fn step(&mut self, action: [f32; 2]) -> PyResult<([f32; 6], f32, bool, &'static str)> {
        // Run the physics step in rapier for the next state
        let [thrust, gimbal_angle] = action;
        self.world.apply_thruster_forces(thrust, gimbal_angle);
        self.world.step();
        self.steps += 1;

        let (x, y, theta) = self.world.get_rocket_state();
        let (vx, vy, omega) = self.world.get_rocket_dynamics();
        let next_state = [x, y, theta, vx, vy, omega];

        let (reason, done) = self._episode_status(x, y, theta, vx, vy, omega);
        let reward = self.calculate_reward(x, y, theta, vx, vy, omega, thrust, gimbal_angle);

        Ok((next_state, reward, done, reason))
    }

    fn calculate_potential(&self, x: f32, y: f32, theta: f32, vx: f32, vy: f32, omega: f32) -> f32 {
        // center is (max_x/2, 0) => potential should be min there
        let ndy = (y - _MIN_POS_Y) / (MAX_POS_Y - _MIN_POS_Y); // [0, 1]
        let ndx = (2.0 * x - MAX_POS_X) / MAX_POS_X; // [-1, 1]
        let sq_dist = ndx.powi(2) + ndy.powi(2); // [0, 2]
        let dist_score = 1.0 - (sq_dist / 2.0);

        // slow velocity preferred
        let vel = vx.powi(2) + vy.powi(2);
        let sq_max_vel = 400.0; // estimate
        let vel_score = 1.0 - (vel / sq_max_vel).clamp(0.0, 1.0);

        // reward being upright and not spinning too much
        let ntheta = theta / PI;
        let angle_norm = ntheta.powi(2) + omega.clamp(-1.0, 1.0).powi(2); // [0, 2]
        let angle_score = 1.0 - (angle_norm / 2.0);

        100.0 * (0.5 * dist_score + 0.15 * vel_score + 0.35 * angle_score) // scaling it to match magnitude of terminal reward
    }

    fn calculate_reward(
        &mut self,
        x: f32,
        y: f32,
        theta: f32,
        vx: f32,
        vy: f32,
        omega: f32,
        thrust: f32,
        gimbal: f32,
    ) -> f32 {
        let current_potential = self.calculate_potential(x, y, theta, vx, vy, omega);
        // F = gamma * Phi(s') - Phi(s)
        // gamma = 0.99 in this case
        let shaping_reward = 0.99 * current_potential - self.prev_potential;
        self.prev_potential = current_potential;

        let action_penalty = -0.1 * (thrust.powi(2) + gimbal.powi(2));

        let mut terminal_reward = 0.0;
        let base_success = 200.0;

        if self._is_crash_landing(x, y, theta, vx, vy, omega) || self._is_oob(x, y) {
            terminal_reward = -base_success;
        } else if self._is_successful_landing(x, y, theta, vx, vy, omega) {
            let ndx = (2.0 * x - MAX_POS_X) / MAX_POS_X;
            if ndx.abs() > 0.1 {
                terminal_reward = -base_success / 2.0;
            } else {
                terminal_reward = base_success;
            }
        }

        let time_penalty = 0.1;
        shaping_reward + action_penalty + terminal_reward - time_penalty
    }

    fn _sample(&self) -> [f32; 6] {
        let mut rng = rand::rng();

        let start_x: f32 = rng.random_range(10.0..=(MAX_POS_X - 10.0));
        let start_y: f32 = rng.random_range(20.0..=MAX_SAFE_Y);
        let start_angle: f32 = rng.random_range(-MAX_ANGLE_DEFLECTION..=MAX_ANGLE_DEFLECTION);

        // From the implememtation in base/src/world.rs
        // rotation is prevented when dragging, and velocities
        // is forcefully set to 0 when the drag ends
        // All starting sequences must have v & omega = 0

        [start_x, start_y, start_angle, 0.0, 0.0, 0.0]
    }

    pub fn sample(&self) -> PyResult<[f32; 6]> {
        Ok(self._sample())
    }

    fn _is_crash_landing(&self, _x: f32, y: f32, theta: f32, vx: f32, vy: f32, omega: f32) -> bool {
        let landed = y <= _MIN_POS_Y;
        let bad_angle = theta.abs() > MAX_LANDING_ANGLE;
        let fast_land = vy.abs() > MAX_LANDING_VY;
        let fast_horiz = vx.abs() > MAX_LANDING_VX;
        let fast_spin = omega.abs() > MAX_LANDING_ANGULAR_VELOCITY;

        landed && (bad_angle || fast_land || fast_horiz || fast_spin)
    }

    fn _is_successful_landing(
        &self,
        x: f32,
        y: f32,
        theta: f32,
        vx: f32,
        vy: f32,
        omega: f32,
    ) -> bool {
        let landed = y <= _MIN_POS_Y;
        let in_x_range = x > 0.0 && x < MAX_POS_X;
        let gentle_angle = theta.abs() <= MAX_LANDING_ANGLE;
        let gentle_vy = vy.abs() <= MAX_LANDING_VY;
        let gentle_vx = vx.abs() <= MAX_LANDING_VX;
        let gentle_omega = omega.abs() <= MAX_LANDING_ANGULAR_VELOCITY;

        landed && in_x_range && gentle_angle && gentle_vy && gentle_vx && gentle_omega
    }

    fn _is_oob(&self, x: f32, y: f32) -> bool {
        x < 0.0 || x > MAX_POS_X || y > MAX_POS_Y
    }

    fn _episode_status(
        &self,
        x: f32,
        y: f32,
        theta: f32,
        vx: f32,
        vy: f32,
        omega: f32,
    ) -> (&'static str, bool) {
        let status = if self._is_successful_landing(x, y, theta, vx, vy, omega) {
            EpisodeStatus::Success
        } else if self._is_crash_landing(x, y, theta, vx, vy, omega) {
            EpisodeStatus::Crash
        } else if self._is_oob(x, y) {
            EpisodeStatus::OutOfBounds
        } else if self.steps >= self.max_steps {
            EpisodeStatus::Timeout
        } else {
            EpisodeStatus::InProgress
        };
        (status.as_str(), status != EpisodeStatus::InProgress)
    }
}

#[pymodule]
fn gym(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEnvironment>()?;
    Ok(())
}
