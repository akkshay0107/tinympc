use std::f32::consts::PI;

use macroquad::prelude::*;
use rapier2d::prelude::*;
use std::env;
use tinympc::world::{MAX_THRUST, World};

// TODO: get the constants from generate_physics_data.rs
const MAX_POS_X: f32 = 80.0; // Min pos x is 0
const _MIN_POS_Y: f32 = 2.0; // COM of vertical rocket is at 2.0 when it touches the ground
const MAX_POS_Y: f32 = 45.0;
const GROUND_THRESHOLD: f32 = 1.99; // Slightly under min possible y

// TODO: get the helper fns from generate_physics_data.rs instead of rewrite
fn get_balanced_thrusters(state: (f32, f32, f32)) -> (f32, f32) {
    let x_center = MAX_POS_X / 2.0;
    let x_bias = (x_center - state.0) / x_center;
    let angle_bias = state.2 / PI;
    let bias = (x_bias + angle_bias) / 2.0;
    let noise = rand::gen_range(-0.3, 0.3);

    let total_thrust = 1.2 * rand::gen_range(0.0, MAX_THRUST); // Lower likelihood of thrust exceeding gravity
    let left_thrust = total_thrust * (1.0 + bias + noise);
    let right_thrust = total_thrust * (1.0 - bias - noise);

    (left_thrust, right_thrust)
}

fn in_bounds(state: (f32, f32, f32)) -> bool {
    if state.0 < 0.0 || state.1 > MAX_POS_X {
        return false;
    }
    if state.1 < GROUND_THRESHOLD || state.1 > MAX_POS_Y {
        return false;
    }
    if state.2 < -PI / 2.0 || state.2 > PI / 2.0 {
        return false;
    }
    true
}

fn main() {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!(
            "Usage: {} <initial_x> <initial_y> <initial_angle_rad>",
            args[0]
        );
        std::process::exit(1);
    }

    let initial_x = args[1].parse::<f32>().unwrap_or_else(|_| {
        eprintln!("Error: initial_x must be a number");
        std::process::exit(1);
    });

    let initial_y = args[2].parse::<f32>().unwrap_or_else(|_| {
        eprintln!("Error: initial_y must be a number");
        std::process::exit(1);
    });

    let initial_angle = args[3].parse::<f32>().unwrap_or_else(|_| {
        eprintln!("Error: initial_angle must be a number (in radians)");
        std::process::exit(1);
    });

    // Initialize the physics world with command line args
    let mut world = World::new();
    if let Some(rocket_body) = world.rigid_body_set.get_mut(world.rocket_body_handle) {
        rocket_body.set_translation(vector![initial_x, initial_y], true);
        rocket_body.set_rotation(Rotation::new(initial_angle), true);
        rocket_body.set_linvel(vector![0.0, 0.0], true);
        rocket_body.set_angvel(0.0, true);
    }

    // CSV output header
    println!("pos_x,pos_y,angle,vel_x,vel_y,angular_vel,left_thrust,right_thrust");

    loop {
        let original_state = world.get_rocket_state();
        let original_dynamics = world.get_rocket_dynamics();
        let (left_thrust, right_thrust) = get_balanced_thrusters(original_state);

        if in_bounds(original_state) {
            // CSV row [S_t, A_t]
            println!(
                " {:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                original_state.0,
                original_state.1,
                original_state.2,
                original_dynamics.0,
                original_dynamics.1,
                original_dynamics.2,
                left_thrust,
                right_thrust
            );
        } else {
            break;
        }

        world.apply_thruster_forces(left_thrust, right_thrust);
        world.step();
    }
}
