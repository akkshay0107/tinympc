use std::f32::consts::PI;
// TODO: fix imports to not be hardcoded to crate name
use core::record::Record;
use core::world::{MAX_THRUST, World};
use macroquad::prelude::*;
use macroquad::rand::gen_range;
use rapier2d::na::Isometry2;
use rapier2d::prelude::*;

const EXPECTED_TRAJ_SIZE: usize = 200; // roughly 3s per trajectory at 60 fps
const NUM_TRAJECTORIES: usize = 4000;

const MAX_POS_X: f32 = 80.0; // Min pos x is 0
const _MIN_POS_Y: f32 = 2.0; // COM of vertical rocket is at 2.0 when it touches the ground
const MAX_POS_Y: f32 = 45.0;
const MAX_ANGLE_DEFLECTION: f32 = PI / 12.0;
const MAX_VX: f32 = 5.0;
const MIN_VY: f32 = -2.0;
const MAX_VY: f32 = 5.0;
const MAX_ANGULAR_VELOCITY: f32 = 0.3;

const GROUND_THRESHOLD: f32 = 1.99; // Slightly under min possible y

fn get_balanced_velocity(x: f32) -> ((f32, f32), f32) {
    let angvel: f32 = rand::gen_range(-MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY);
    let x_center = MAX_POS_X / 2.0;
    let x_bias = (x_center - x) / x_center; // bias velocity to head towards center;
    let vx = rand::gen_range(-MAX_VX, MAX_VX) + x_bias * MAX_VX;
    let vy = rand::gen_range(MIN_VY, MAX_VY);
    ((vx, vy), angvel)
}

fn get_balanced_thrusters(state: (f32, f32, f32), dominant: bool) -> (f32, f32) {
    let x_center = MAX_POS_X / 2.0;
    let x_bias = (x_center - state.0) / x_center;
    let angle_bias = state.2 / PI;
    let bias = (x_bias + angle_bias) / 2.0;
    let noise = rand::gen_range(-0.3, 0.3);

    let total_thrust = if dominant {
        1.5 * gen_range(0.0, MAX_THRUST) // 34% chance of thrust exceeding gravity
    } else {
        gen_range(0.0, MAX_THRUST) // 2% chance of thrust exceeding gravity
    };

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

fn generate_physics_data(
    output_path: &str,
    num_traj: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut records: Vec<Record> = Vec::with_capacity(num_traj * EXPECTED_TRAJ_SIZE);

    let five_percent = NUM_TRAJECTORIES / 20;
    let mut world = World::new();
    for i in 0..num_traj {
        let rocket_handle = world.rocket_body_handle;
        let rocket = world.rigid_body_set.get_mut(rocket_handle).unwrap();

        let start_x: f32 = rand::gen_range(MAX_POS_X / 4.0, 3.0 * MAX_POS_X / 4.0); // Middle half
        let start_y: f32 = rand::gen_range(MAX_POS_Y / 2.0, MAX_POS_Y); // Upper half
        let start_angle: f32 = rand::gen_range(-MAX_ANGLE_DEFLECTION, MAX_ANGLE_DEFLECTION);
        let (init_linvel, init_angvel) = get_balanced_velocity(start_x);

        rocket.set_position(Isometry2::new(vector![start_x, start_y], start_angle), true);
        rocket.set_linvel(vector![init_linvel.0, init_linvel.1], true);
        rocket.set_angvel(init_angvel, true);

        let dominant = (i % 2) == 0;
        loop {
            let original_state = world.get_rocket_state();
            let original_dynamics = world.get_rocket_dynamics();
            let (left_thruster, right_thruster) = get_balanced_thrusters(original_state, dominant);
            world.apply_thruster_forces(left_thruster, right_thruster);
            world.step();
            let resultant_state = world.get_rocket_state();
            let resultant_dynamics = world.get_rocket_dynamics();
            if in_bounds(resultant_state) {
                let delta_pos = (
                    resultant_state.0 - original_state.0,
                    resultant_state.1 - original_state.1,
                );
                let delta_angle = resultant_state.2 - original_state.2;
                let delta_vel = (
                    resultant_dynamics.0 - original_dynamics.0,
                    resultant_dynamics.1 - original_dynamics.1,
                );
                let delta_angular_vel = resultant_dynamics.2 - original_dynamics.2;
                let obs = Record::new(
                    (original_state.0, original_state.1),
                    original_state.2,
                    (original_dynamics.0, original_dynamics.1),
                    original_dynamics.2,
                    left_thruster,
                    right_thruster,
                    delta_pos,
                    delta_angle,
                    delta_vel,
                    delta_angular_vel,
                );
                records.push(obs);
            } else {
                break;
            }
        }
        // Progress indicator
        if (i + 1) % five_percent == 0 {
            println!(
                "{}/{} trajectories completed [{}%]",
                i + 1,
                NUM_TRAJECTORIES,
                5 * (i + 1) / five_percent
            );
        }
    }

    Record::write_to_csv(&records, output_path)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create data directory if it doesn't exist
    std::fs::create_dir_all("data")?;
    let output_path = "data/physics_data.csv";
    // roughly 60 fps, estimating 3s of trajectory on average
    generate_physics_data(output_path, NUM_TRAJECTORIES)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    #[test]
    fn test_generate_physics_data() {
        let test_path = "test_physics_data.csv";
        let test_samples = 10;
        if Path::new(test_path).exists() {
            fs::remove_file(test_path).unwrap();
        }

        generate_physics_data(test_path, test_samples).unwrap();
        assert!(Path::new(test_path).exists());
        let content = fs::read_to_string(test_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        // 1 for header, variable count of rows due to filtering procedure
        assert!(lines.len() > 1, "Expected at least header + some data");

        fs::remove_file(test_path).unwrap();
    }
}
