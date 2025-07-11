use macroquad::prelude::*;
use rapier2d::na::UnitComplex;
use rapier2d::prelude::*;
use tinympc::record::Record;
use tinympc::world::World;

const NUM_SAMPLES: usize = 200_000;
const NUM_STEPS: usize = 1; // Number of steps of physics with the same action
const MAX_RETRIES: usize = 5;

const MIN_POS_X: f32 = 0.0;
const MAX_POS_X: f32 = 80.0;
const MIN_POS_Y: f32 = 2.0; // COM of vertical rocket is at 2.0 when it touches the ground
const MAX_POS_Y: f32 = 45.0;
const MAX_VELOCITY: f32 = 25.0;
const MAX_ANGULAR_VELOCITY: f32 = 1.0;

const GROUND_THRESHOLD: f32 = 1.9; // slightly under min possible y
// Limit on delta magnitude to filter out chaotic movement of rocket
// which would be unreasonable
const MAX_DELTA_MAGNITUDE: f32 = 5.0;

#[derive(Debug, Clone, Copy)]
enum SamplingStrategy {
    Random,
    DescentTrajectory,
    NearGround,
    LowVelocity,
}

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

fn generate_realistic_thrusters() -> (f32, f32) {
    // More balanced thruster patterns
    let total_thrust: f32 = rand::gen_range(0.0, 8.0);
    let balance: f32 = rand::gen_range(-0.8, 0.8);
    let left = (total_thrust * (1.0 + balance) / 2.0).clamp(0.0, 5.0);
    let right = (total_thrust * (1.0 - balance) / 2.0).clamp(0.0, 5.0);
    (left, right)
}

fn generate_descent_trajectory_sample() -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    let altitude = rand::gen_range(5.0, 45.0);
    let progress = (45.0 - altitude) / 40.0; // 0 to 1, where 1 is near ground

    let pos_x = rand::gen_range(10.0, 70.0);
    let pos_y = altitude;

    // Velocity profile that gets more controlled near ground
    let vel_x = rand::gen_range(-6.0, 6.0) * (1.0 - progress * 0.5);
    let vel_y = -rand::gen_range(2.0, 10.0) * (1.0 + progress * 0.3);

    // Angle gets more upright near ground
    let angle = rand::gen_range(-0.4, 0.4) * (1.0 - progress * 0.6);
    let angular_vel = rand::gen_range(-0.3, 0.3) * (1.0 - progress * 0.4);

    // More aggressive control near ground
    let thrust_intensity = 0.4 + progress * 0.6;
    let (left_base, right_base) = generate_realistic_thrusters();

    (
        pos_x,
        pos_y,
        angle,
        vel_x,
        vel_y,
        angular_vel,
        left_base * thrust_intensity,
        right_base * thrust_intensity,
    )
}

fn generate_near_ground_sample() -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    let pos_x = rand::gen_range(20.0, 60.0);
    let pos_y = rand::gen_range(2.0, 8.0); // Low altitude

    // Extremely slow velocities
    let vel_x = rand::gen_range(-1.0, 1.0);
    let vel_y = rand::gen_range(-3.0, 0.5);

    // Nearly upright
    let angle = rand::gen_range(-0.1, 0.1);
    let angular_vel = rand::gen_range(-0.1, 0.1);

    // High thrust for landing(counter gravity)
    let (left_thruster, right_thruster) = generate_realistic_thrusters();
    let thrust_boost = rand::gen_range(0.8, 1.0);

    (
        pos_x,
        pos_y,
        angle,
        vel_x,
        vel_y,
        angular_vel,
        left_thruster * thrust_boost,
        right_thruster * thrust_boost,
    )
}

fn generate_low_velocity_sample() -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    let (pos_x, pos_y) = generate_random_position();

    // Very low velocities
    let vel_x = rand::gen_range(-2.0, 2.0);
    let vel_y = rand::gen_range(-3.0, 1.0);

    let angle = rand::gen_range(-0.3, 0.3);
    let angular_vel = rand::gen_range(-0.2, 0.2);

    // Low to moderate thrust
    let left_thruster = rand::gen_range(0.0, 3.0);
    let right_thruster = rand::gen_range(0.0, 3.0);

    (
        pos_x,
        pos_y,
        angle,
        vel_x,
        vel_y,
        angular_vel,
        left_thruster,
        right_thruster,
    )
}

fn generate_sample_by_strategy(
    strategy: SamplingStrategy,
) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    match strategy {
        SamplingStrategy::Random => {
            let (pos_x, pos_y) = generate_random_position();
            let (vel_x, vel_y) = generate_random_velocity();
            let angle = rand::gen_range(-std::f32::consts::PI, std::f32::consts::PI);
            let angular_vel = rand::gen_range(-MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY);
            let left_thruster = rand::gen_range(0.0, 5.0);
            let right_thruster = rand::gen_range(0.0, 5.0);
            (
                pos_x,
                pos_y,
                angle,
                vel_x,
                vel_y,
                angular_vel,
                left_thruster,
                right_thruster,
            )
        }
        SamplingStrategy::DescentTrajectory => generate_descent_trajectory_sample(),
        SamplingStrategy::NearGround => generate_near_ground_sample(),
        SamplingStrategy::LowVelocity => generate_low_velocity_sample(),
    }
}

fn is_sample_valid(final_pos: (f32, f32), delta_pos: (f32, f32), delta_vel: (f32, f32)) -> bool {
    // Rocket clipping into ground check
    if final_pos.1 <= GROUND_THRESHOLD {
        return false;
    }

    // Rocket out of bounds check
    if final_pos.0 < MIN_POS_X || final_pos.0 > MAX_POS_X || final_pos.1 > MAX_POS_Y {
        return false;
    }

    // Filter out extreme cases
    let delta_magnitude =
        (delta_pos.0.powi(2) + delta_pos.1.powi(2) + delta_vel.0.powi(2) + delta_vel.1.powi(2))
            .sqrt();
    if delta_magnitude > MAX_DELTA_MAGNITUDE {
        return false;
    }

    true
}

fn generate_physics_data(
    output_path: &str,
    num_samples: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    rand::srand(macroquad::miniquad::date::now() as _);
    let mut records = Vec::with_capacity(num_samples);
    let mut world = World::new();

    // Define sampling strategy distribution
    let strategies = [
        SamplingStrategy::DescentTrajectory,
        SamplingStrategy::NearGround,
        SamplingStrategy::LowVelocity,
        SamplingStrategy::Random,
    ];
    // Arbitrarily picked as of rn
    let strategy_weights = [0.4, 0.25, 0.20, 0.15];

    let mut generated_samples = 0;
    let mut total_attempts = 0;
    let mut skipped_collisions = 0;
    let mut skipped_outliers = 0;

    while generated_samples < num_samples && total_attempts < num_samples * MAX_RETRIES {
        total_attempts += 1;

        // Select sampling strategy based on weights
        let rand_val = rand::gen_range(0.0, 1.0);
        let mut cumulative_weight = 0.0;
        let mut selected_strategy = SamplingStrategy::Random;

        for (i, &weight) in strategy_weights.iter().enumerate() {
            cumulative_weight += weight;
            if rand_val <= cumulative_weight {
                selected_strategy = strategies[i];
                break;
            }
        }

        let (pos_x, pos_y, initial_angle, vel_x, vel_y, angular_vel, left_thruster, right_thruster) =
            generate_sample_by_strategy(selected_strategy);

        let rocket_handle = world.rocket_body_handle;
        let rocket_body = world.rigid_body_set.get_mut(rocket_handle).unwrap();

        // Reset state from previous iteration
        // and set it to the values for this iteration
        rocket_body.set_translation(vector![pos_x, pos_y], true);
        rocket_body.set_rotation(UnitComplex::new(initial_angle), true);
        rocket_body.set_linvel(vector![vel_x, vel_y], true);
        rocket_body.set_angvel(angular_vel, true);
        rocket_body.wake_up(true);
        rocket_body.reset_forces(true);
        rocket_body.reset_torques(true);

        // Simulate resultant state
        world.apply_thruster_forces(left_thruster, right_thruster);
        for _ in 0..NUM_STEPS {
            world.step();
        }

        let rocket_body = world.rigid_body_set.get(rocket_handle).unwrap();
        let (result_pos_x, result_pos_y, result_angle) = world.get_rocket_state();
        let result_vel = rocket_body.linvel();
        let result_angular_vel = rocket_body.angvel();

        // Calculate deltas
        let delta_pos = (result_pos_x - pos_x, result_pos_y - pos_y);
        let delta_angle = result_angle - initial_angle;
        let delta_vel = (result_vel.x - vel_x, result_vel.y - vel_y);
        let delta_angular_vel = result_angular_vel - angular_vel;

        // Validate sample
        if !is_sample_valid((result_pos_x, result_pos_y), delta_pos, delta_vel) {
            if result_pos_y <= GROUND_THRESHOLD {
                skipped_collisions += 1;
            } else {
                skipped_outliers += 1;
            }
            continue;
        }

        let record = Record::new(
            (pos_x, pos_y),
            initial_angle,
            (vel_x, vel_y),
            angular_vel,
            left_thruster,
            right_thruster,
            delta_pos,
            delta_angle,
            delta_vel,
            delta_angular_vel,
        );

        records.push(record);
        generated_samples += 1;

        // Progress tracker
        if generated_samples % 10000 == 0 {
            println!(
                "Generated {}/{} samples (attempts: {}, collisions skipped: {}, outliers skipped: {})",
                generated_samples,
                num_samples,
                total_attempts,
                skipped_collisions,
                skipped_outliers
            );
        }
    }

    Record::write_to_csv(&records, output_path)?;
    println!(
        "Generated {} physics samples to {} (total attempts: {}, collisions skipped: {}, outliers skipped: {})",
        generated_samples, output_path, total_attempts, skipped_collisions, skipped_outliers
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

        // 1 for header, variable count of rows due to filtering procedure
        assert!(lines.len() > 1, "Expected at least header + some data");

        fs::remove_file(test_path).unwrap();
    }
}
