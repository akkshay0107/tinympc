use macroquad::prelude::*;
use rapier2d::na::UnitComplex;
use rapier2d::prelude::*;
use tinympc::record::Record;
use tinympc::world::World;

const NUM_SAMPLES: usize = 200_000;
const NUM_STEPS: usize = 2; // steps between each control action input

const MIN_POS_X: f32 = 0.0;
const MAX_POS_X: f32 = 100.0;
const MIN_POS_Y: f32 = 2.0;
const MAX_POS_Y: f32 = 45.0;
const MAX_VELOCITY: f32 = 30.0;
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

fn generate_physics_data(
    output_path: &str,
    num_samples: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    rand::srand(macroquad::miniquad::date::now() as _);
    let mut records = Vec::with_capacity(num_samples);
    let mut world = World::new();

    for _ in 0..num_samples {
        // Generate random initial state
        let (pos_x, pos_y) = generate_random_position();
        let (vel_x, vel_y) = generate_random_velocity();
        let initial_angle = rand::gen_range(-std::f32::consts::PI, std::f32::consts::PI);
        let initial_pos = (pos_x, pos_y);
        let initial_vel = (vel_x, vel_y);
        let angular_vel = rand::gen_range(-MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY);
        let left_thruster = rand::gen_range(0.0, 5.0);
        let right_thruster = rand::gen_range(0.0, 5.0);

        let rocket_handle = world.rocket_body_handle;
        let rocket_body = world.rigid_body_set.get_mut(rocket_handle).unwrap();

        // Reset forces from previous iteration
        // and set the state to the rand values from current iteration
        rocket_body.set_translation(vector![pos_x, pos_y], true);
        rocket_body.set_rotation(UnitComplex::new(initial_angle), true);
        rocket_body.set_linvel(vector![vel_x, vel_y], true);
        rocket_body.set_angvel(angular_vel, true);
        rocket_body.wake_up(true);
        rocket_body.reset_forces(true);
        rocket_body.reset_torques(true);

        // Data now stores result when action is applied for NUM_STEPS (>= 1)
        // With a larger NUM_STEPS value, the deltas are slightly larger
        // which reduces the probability of model predicing 0 for all deltas
        // TODO: fix sampling strategy to prevent collisions with ground
        // messing with the physics data
        world.apply_thruster_forces(left_thruster, right_thruster);
        for _ in 0..NUM_STEPS {
            world.step();
        }

        let rocket_body = world.rigid_body_set.get(rocket_handle).unwrap();
        let (result_pos_x, result_pos_y, result_angle) = world.get_rocket_state();
        let result_vel = rocket_body.linvel();
        let result_angular_vel = rocket_body.angvel();

        // Record now storing delta states
        let delta_pos = (result_pos_x - pos_x, result_pos_y - pos_y);
        let delta_angle = result_angle - initial_angle;
        let delta_vel = (result_vel.x - vel_x, result_vel.y - vel_y);
        let delta_angular_vel = result_angular_vel - angular_vel;

        let record = Record::new(
            initial_pos,
            initial_angle,
            initial_vel,
            angular_vel,
            left_thruster,
            right_thruster,
            delta_pos,
            delta_angle,
            delta_vel,
            delta_angular_vel,
        );

        records.push(record);
    }

    Record::write_to_csv(&records, output_path)?;
    println!(
        "Generated {} physics samples to {}",
        num_samples, output_path
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create data directory if it doesn't exist
    std::fs::create_dir_all("data")?;
    let output_path = "data/physics_data.csv";
    generate_physics_data(output_path, NUM_SAMPLES)?;

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
        let test_samples = 100;
        if Path::new(test_path).exists() {
            fs::remove_file(test_path).unwrap();
        }

        generate_physics_data(test_path, test_samples).unwrap();
        assert!(Path::new(test_path).exists());
        let content = fs::read_to_string(test_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        // +1 for header
        assert_eq!(
            lines.len(),
            test_samples + 1,
            "Expected {} lines ({} samples + header), got {}",
            test_samples + 1,
            test_samples,
            lines.len()
        );

        fs::remove_file(test_path).unwrap();
    }
}
