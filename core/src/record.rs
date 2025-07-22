use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    // Current state
    pub current_pos_x: f32,
    pub current_pos_y: f32,
    pub current_angle: f32,
    pub current_vel_x: f32,
    pub current_vel_y: f32,
    pub current_angular_vel: f32,

    // Action
    pub left_thruster: f32,
    pub right_thruster: f32,

    // Deltas from current to resulting state
    pub delta_pos_x: f32,
    pub delta_pos_y: f32,
    pub delta_angle: f32,
    pub delta_vel_x: f32,
    pub delta_vel_y: f32,
    pub delta_angular_vel: f32,
}

impl Record {
    pub fn new(
        current_pos: (f32, f32),
        current_angle: f32,
        current_vel: (f32, f32),
        current_angular_vel: f32,
        left_thruster: f32,
        right_thruster: f32,
        delta_pos: (f32, f32),
        delta_angle: f32,
        delta_vel: (f32, f32),
        delta_angular_vel: f32,
    ) -> Self {
        Self {
            current_pos_x: current_pos.0,
            current_pos_y: current_pos.1,
            current_angle,
            current_vel_x: current_vel.0,
            current_vel_y: current_vel.1,
            current_angular_vel,
            left_thruster,
            right_thruster,
            delta_pos_x: delta_pos.0,
            delta_pos_y: delta_pos.1,
            delta_angle,
            delta_vel_x: delta_vel.0,
            delta_vel_y: delta_vel.1,
            delta_angular_vel,
        }
    }

    pub fn write_to_csv(records: &[Self], path: &str) -> Result<(), csv::Error> {
        let mut writer = csv::Writer::from_path(path)?;
        for record in records {
            writer.serialize(record)?;
        }
        writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_record_write_to_csv() {
        let record = Record::new(
            (1.0, 2.0),
            0.1,
            (0.1, 0.2),
            0.05,
            1.0,
            0.5,
            (0.1, 0.05),
            0.05,
            (0.01, 0.01),
            -0.01,
        );

        let temp_path = "test_record.csv";
        Record::write_to_csv(&[record], temp_path).unwrap();
        
        let content = fs::read_to_string(temp_path).unwrap();
        fs::remove_file(temp_path).unwrap();
        
        // Check if the content contains the expected values (ignoring headers)
        let expected_values = "1.0,2.0,0.1,0.1,0.2,0.05,1.0,0.5,0.1,0.05,0.05,0.01,0.01,-0.01";
        if !content.contains(expected_values) {
            panic!("Content does not contain expected values.\nExpected: {}\nActual: {}", expected_values, content);
        }
    }
}