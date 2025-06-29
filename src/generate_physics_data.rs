use macroquad::prelude::*;
use rapier2d::prelude::*;

use crate::record::Record;
use crate::world::World;

// TODO: fix velocity constants
const NUM_SAMPLES: usize = 1000;
const MIN_POS_X: f32 = 0.0;
const MAX_POS_X: f32 = 100.0;
const MIN_POS_Y: f32 = 2.0;
const MAX_POS_Y: f32 = 45.0;
const MAX_VELOCITY: f32 = 5.0;
const MAX_ANGULAR_VELOCITY: f32 = 2.0;

fn generate_random_position() -> (f32, f32) {
    let x = rand::gen_range(MIN_POS_X, MAX_POS_X);
    let y = rand::gen_range(MIN_POS_Y, MAX_POS_Y);
    (x, y)
}

fn generate_random_velocity() -> (f32, f32) {
    let vx = rand::gen_range(-MAX_VELOCITY, MAX_VELOCITY);
    let vy = rand::gen_range(-MAX_VELOCITY, MAX_VELOCITY);
    (vx, vy)
}

pub fn generate_physics_data(output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    rand::srand(macroquad::miniquad::date::now() as _);
    let mut records = Vec::with_capacity(NUM_SAMPLES);
    let mut world = World::new();

    for _ in 0..NUM_SAMPLES {
        // Generate random initial state
        let (pos_x, pos_y) = generate_random_position();
        let (vel_x, vel_y) = generate_random_velocity();
        let initial_pos = (pos_x, pos_y);
        let initial_vel = (vel_x, vel_y);
        let angular_vel = rand::gen_range(-MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY);
        let left_thruster = rand::gen_range(0.0, 1.0);
        let right_thruster = rand::gen_range(0.0, 1.0);

        let rocket_handle = world.rocket_body_handle;
        let rocket_body = world.rigid_body_set.get_mut(rocket_handle).unwrap();

        // Reset forces from previous iteration
        // and set the state to the rand values from current iteration
        rocket_body.set_translation(vector![pos_x, pos_y], true);
        rocket_body.set_linvel(vector![vel_x, vel_y], true);
        rocket_body.set_angvel(angular_vel, true);
        rocket_body.wake_up(true);
        rocket_body.reset_forces(true);
        rocket_body.reset_torques(true);

        // Apply thrusters and step physics
        world.apply_thruster_forces(left_thruster, right_thruster);
        world.step();

        // Get the rocket body again after physics step
        let rocket_body = world.rigid_body_set.get(rocket_handle).unwrap();
        let (result_pos_x, result_pos_y, _) = world.get_rocket_state();
        let result_vel = rocket_body.linvel();
        let result_angular_vel = rocket_body.angvel();

        let record = Record::new(
            initial_pos,
            initial_vel,
            angular_vel,
            left_thruster,
            right_thruster,
            (result_pos_x, result_pos_y),
            (result_vel.x, result_vel.y),
            result_angular_vel,
        );

        records.push(record);
    }

    Record::write_to_csv(&records, output_path)?;
    println!(
        "Generated {} physics samples to {}",
        NUM_SAMPLES, output_path
    );

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
        if Path::new(test_path).exists() {
            fs::remove_file(test_path).unwrap();
        }

        generate_physics_data(test_path).unwrap();
        assert!(Path::new(test_path).exists());
        let content = fs::read_to_string(test_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        // +1 for header
        assert_eq!(
            lines.len(),
            NUM_SAMPLES + 1,
            "Expected {} lines ({} samples + header), got {}",
            NUM_SAMPLES + 1,
            NUM_SAMPLES,
            lines.len()
        );

        fs::remove_file(test_path).unwrap();
    }
}
