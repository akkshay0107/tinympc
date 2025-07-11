use macroquad::prelude::*;
use rapier2d::na::UnitComplex;
use rapier2d::prelude::*;
use tinympc::record::Record;
use tinympc::world::World;

const NUM_TRAJECTORIES: usize = 200;
const MAX_RETRIES: usize = 5;

const MIN_POS_X: f32 = 0.0;
const MAX_POS_X: f32 = 80.0;
const MIN_POS_Y: f32 = 2.0; // COM of vertical rocket is at 2.0 when it touches the ground
const MAX_POS_Y: f32 = 45.0;
const MAX_VELOCITY: f32 = 25.0;
const MAX_ANGULAR_VELOCITY: f32 = 1.0;

const GROUND_THRESHOLD: f32 = 1.9; // slightly under min possible y

fn generate_physics_data(
    output_path: &str,
    num_samples: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let records: Vec<Record> = Vec::with_capacity(num_samples);
    // TODO: sample using trajectories instead of random points
    Record::write_to_csv(&records, output_path)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create data directory if it doesn't exist
    std::fs::create_dir_all("data")?;
    let output_path = "data/physics_data.csv";
    generate_physics_data(output_path, NUM_TRAJECTORIES * 1000)?;
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
