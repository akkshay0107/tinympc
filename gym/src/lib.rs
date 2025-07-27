use base::constants::*;
use base::world::World;
use pyo3::prelude::*;
use rand::Rng;
use rapier2d::na::Isometry2;
use rapier2d::prelude::*;

#[pyclass]
pub struct PyEnvironment {
    world: World,
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
            steps: 0,
            max_steps,
        }
    }

    // world doesn't need a getter and max steps defined in python
    #[getter]
    pub fn steps(&self) -> PyResult<u32> {
        Ok(self.steps)
    }

    pub fn reset(&mut self) -> PyResult<[f32; 6]> {
        self.world = World::new();
        self.steps = 0;

        let init_state = self._sample(); // Random valid state

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

        let reward = self.calculate_reward(x, y, theta, vx, vy, omega);

        let done = self.is_done(x, y);

        Ok((next_state, reward, done))
    }

    fn calculate_reward(&self, x: f32, y: f32, theta: f32, vx: f32, vy: f32, omega: f32) -> f32 {
        let dy = y - _MIN_POS_Y;

        let dist_penalty = -2.0 * dy.powi(2); // Allowing landing on any valid x
        let descent_penalty = -vy.powi(2);
        let horizontal_movement_penalty = -vx.powi(4); // Want to highly discourage haivng a large horizontal velocity
        let angle_penalty = -5.0 * theta.powi(2);
        let ang_vel_penalty = -0.1 * omega.powi(2); // Enforcing larger penalty on angle to allow rocket to correct angle using angvel
        let time_penalty = -0.02;

        let crash_landing_penalty = if self._is_crash_landing(x, y, theta, vx, vy, omega) {
            -1e4 // Needs to be larger than the largest valid penalty
        } else {
            0.0
        };

        let landing_bonus = if self._is_successful_landing(x, y, theta, vx, vy, omega) {
            1e4 // Arbitrarily picked
        } else {
            0.0
        };

        dist_penalty
            + descent_penalty
            + horizontal_movement_penalty
            + angle_penalty
            + ang_vel_penalty
            + time_penalty
            + crash_landing_penalty
            + landing_bonus
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
        let start_y: f32 = rng.random_range(10.0..=MAX_POS_Y);
        let start_vx: f32 = rng.random_range(-MAX_VX..=MAX_VX);
        let start_vy: f32 = rng.random_range(MIN_VY..=MAX_VY);
        let start_angle: f32 = rng.random_range(-MAX_ANGLE_DEFLECTION..=MAX_ANGLE_DEFLECTION);

        // From the implememtation in base/src/world.rs
        // rotation is prevented when dragging, and angvel
        // is forcefully set to 0 when the drag ends
        // All starting sequences have omega = 0

        [start_x, start_y, start_angle, start_vx, start_vy, 0.0]
    }

    pub fn sample(&self) -> PyResult<[f32; 6]> {
        Ok(self._sample())
    }

    fn _is_crash_landing(&self, x: f32, y: f32, theta: f32, vx: f32, vy: f32, omega: f32) -> bool {
        let landed = y <= _MIN_POS_Y;
        let horizontal_out = x <= 0.0 || x >= MAX_POS_X;
        let vertical_out = y <= GROUND_THRESHOLD || y >= MAX_POS_Y;
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
