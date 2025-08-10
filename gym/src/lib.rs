use base::constants::*;
use base::world::World;
use pyo3::prelude::*;
use rand::Rng;
use rapier2d::na::Isometry2;
use rapier2d::prelude::*;

const MAX_SAFE_Y: f32 = MAX_POS_Y - 3.0; // Prevent rocket going out of bounds immediately

#[pyclass]
pub struct PyEnvironment {
    world: World,
    prev_y: f32,
    steps: u32,
    max_steps: u32,
}

#[pymethods]
impl PyEnvironment {
    #[new]
    pub fn new(max_steps: u32) -> Self {
        let world = World::new();
        Self {
            world,
            prev_y: MAX_POS_Y,
            steps: 0,
            max_steps,
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
    pub fn get_prev_y(&self) -> PyResult<f32> {
        Ok(self.prev_y)
    }

    pub fn reset(&mut self) -> PyResult<[f32; 6]> {
        self.world = World::new();
        self.steps = 0;

        let init_state = self._sample(); // Random valid state
        self.prev_y = init_state[1];

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

    pub fn step(&mut self, action: [f32; 2]) -> PyResult<([f32; 6], f32, bool)> {
        // Run the physics step in rapier for the next state
        let [left_thrust, right_thrust] = action;
        self.world.apply_thruster_forces(left_thrust, right_thrust);
        self.world.step();
        self.steps += 1;

        let (x, y, theta) = self.world.get_rocket_state();
        let (vx, vy, omega) = self.world.get_rocket_dynamics();
        let next_state = [x, y, theta, vx, vy, omega];

        let done = self.is_done(x, y);
        let reward = self.calculate_reward(x, y, theta, vx, vy, omega, left_thrust, right_thrust);

        self.prev_y = y;

        Ok((next_state, reward, done))
    }

    fn calculate_reward(
        &self,
        x: f32,
        y: f32,
        theta: f32,
        vx: f32,
        vy: f32,
        omega: f32,
        left_thrust: f32,
        right_thrust: f32,
    ) -> f32 {
        // Landing at any valid x is fine (No x based rewards for now)
        let dy = y - _MIN_POS_Y; // dy in [0,40]
        let dist_penalty = -0.2 * dy.powi(2); // dist_penalty in [-480, 0]
        let descent_penalty = -0.5 * vy.powi(2);

        let angle_penalty = -20.0 * theta.powi(2); // angle_penalty in [-200, 0]
        let ang_vel_penalty = -10.0 * omega.powi(2);

        let downward_progress_reward = if self.steps > 1 && self.prev_y > y {
            10.0 * (self.prev_y - y)
        } else {
            0.0
        };

        let upper_exit_penalty = if self.steps > 1 && y > MAX_POS_Y {
            -1e3
        } else {
            0.0
        };

        let landing_bonus = if self._is_crash_landing(x, y, theta, vx, vy, omega) {
            -1e3
        } else if self._is_successful_landing(x, y, theta, vx, vy, omega) {
            1e3
        } else {
            0.0
        };

        let thrust_penalty = -(left_thrust.powi(2) + right_thrust.powi(2)); // Deter model from using full thrust at all times

        let time_penalty = -0.01;

        dist_penalty
            + descent_penalty
            + angle_penalty
            + ang_vel_penalty
            + downward_progress_reward
            + landing_bonus
            + upper_exit_penalty
            + thrust_penalty
            + time_penalty
    }

    fn is_done(&self, x: f32, y: f32) -> bool {
        let out_of_bounds = x < 0.0 || x > MAX_POS_X || y > MAX_POS_Y;
        let max_steps_reached = self.steps >= self.max_steps;
        let landed = y <= GROUND_THRESHOLD;

        out_of_bounds || max_steps_reached || landed
    }

    fn _sample(&self) -> [f32; 6] {
        let mut rng = rand::rng();

        let start_x: f32 = rng.random_range(10.0..=(MAX_POS_X - 10.0));
        let start_y: f32 = rng.random_range(10.0..=MAX_SAFE_Y);
        let start_vx: f32 = rng.random_range(-MAX_VX..=MAX_VX);
        let start_vy: f32 = rng.random_range(MIN_VY..=MAX_VY);
        let start_angle: f32 = rng.random_range(-MAX_ANGLE_DEFLECTION..=MAX_ANGLE_DEFLECTION);

        // From the implememtation in base/src/world.rs
        // rotation is prevented when dragging, and angvel
        // is forcefully set to 0 when the drag ends
        // All starting sequences must have omega = 0

        [start_x, start_y, start_angle, start_vx, start_vy, 0.0]
    }

    pub fn sample(&self) -> PyResult<[f32; 6]> {
        Ok(self._sample())
    }

    fn _is_crash_landing(&self, x: f32, y: f32, theta: f32, vx: f32, vy: f32, omega: f32) -> bool {
        let landed = y <= _MIN_POS_Y;
        let horizontal_out = x <= 0.0 || x >= MAX_POS_X;
        let vertical_out = y >= MAX_POS_Y;
        let bad_angle = theta.abs() > MAX_LANDING_ANGLE;
        let fast_land = vy.abs() > MAX_LANDING_VY;
        let fast_horiz = vx.abs() > MAX_LANDING_VX;
        let fast_spin = omega.abs() > MAX_LANDING_ANGULAR_VELOCITY;

        (landed && (bad_angle || fast_land || fast_horiz || fast_spin))
            || horizontal_out
            || vertical_out
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
}

#[pymodule]
fn gym(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEnvironment>()?;
    Ok(())
}
