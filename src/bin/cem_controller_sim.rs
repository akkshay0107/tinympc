use std::time::{self, UNIX_EPOCH};

use macroquad::prelude::*;
use ndarray::Array1;
use rapier2d::na::Isometry2;
use rapier2d::prelude::*;
use tinympc::cem_controller::CEMController;
use tinympc::game::Game;
use tinympc::world::{MAX_THRUST, World, world_to_pixel};

pub const START_BOX: [f32; 4] = [25.0, 40.0, 55.0, 30.0]; // (xy top left, xy bottom right)
const CONTROL_HORIZON: usize = 20;
const ELITE_SAMPLES: usize = 10;
const MAX_ITER: usize = 10;
const ALPHA: f32 = 0.1;
const TARGET_ARRAY: [f32; 6] = [40.0, 2.0, 0.0, 0.0, 0.0, 0.0];
const Q_ARRAY: [f32; 6] = [10.0, 10.0, 0.0, 0.0, 0.0, 0.0];
const R_ARRAY: [f32; 2] = [0.0, 0.0];

fn get_new_start() -> (f32, f32) {
    let start_x = rand::gen_range(START_BOX[0], START_BOX[2]);
    let start_y = rand::gen_range(START_BOX[1], START_BOX[3]);
    (start_x, start_y)
}

#[macroquad::main("CEM Controller Simulation")]
async fn main() {
    rand::srand(
        time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    );

    let mut world = World::new();
    let mut game = Game::new();
    let mut cem_controller = CEMController::new(
        "models/dynamics_model.onnx",
        CONTROL_HORIZON,
        20 * ELITE_SAMPLES,
        ELITE_SAMPLES,
        MAX_ITER,
        (0.0, MAX_THRUST),
        ALPHA,
    )
    .unwrap();

    // Initialize arrays once before the loop
    let target = Array1::from_vec(TARGET_ARRAY.to_vec());
    let q = Array1::from_vec(Q_ARRAY.to_vec());
    let r = Array1::from_vec(R_ARRAY.to_vec());

    // Top right corner
    // let reset_button = ResetButton::new(700.0, 20.0, 70.0, 20.0);

    let (start_x, start_y) = get_new_start();
    let rocket = world
        .rigid_body_set
        .get_mut(world.rocket_body_handle)
        .unwrap();
    rocket.set_position(Isometry2::new(vector![start_x, start_y], 0.0), true);

    loop {
        game.update();

        let positions = world.get_rocket_state();
        let velocities = world.get_rocket_dynamics();

        let mut curr_state = Array1::zeros((6,));
        curr_state[0] = positions.0;
        curr_state[1] = positions.1;
        curr_state[2] = positions.2;
        curr_state[3] = velocities.0;
        curr_state[4] = velocities.1;
        curr_state[5] = velocities.2;

        let action = cem_controller
            .control(&curr_state, &target, &q, &r)
            .unwrap();

        let left_thruster = action[0];
        let right_thruster = action[1];

        world.apply_thruster_forces(left_thruster, right_thruster);
        world.step();

        let (rocket_x, rocket_y, rocket_angle) = world.get_rocket_state();
        let (px_rocket_x, px_rocket_y) = world_to_pixel(rocket_x, rocket_y);

        #[cfg(feature = "logging")]
        {
            let (vel_x, vel_y, ang_vel) = world.get_rocket_dynamics();

            println!("Rocket State:");
            println!("Position: ({:.2}, {:.2})", rocket_x, rocket_y);
            println!("Velocity: ({:.2}, {:.2})", vel_x, vel_y);
            println!("Angular Velocity: {:.2}", ang_vel);
            println!("Angle: {:.2}", rocket_angle);
            println!("Action: ({:.2}, {:.2})", left_thruster, right_thruster);
        }

        game.rocket.set_state(
            px_rocket_x,
            px_rocket_y,
            rocket_angle,
            (left_thruster, right_thruster),
        );
        game.draw();
        game.rocket.draw();
        // reset_button.draw();

        next_frame().await;
    }
}
