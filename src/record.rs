use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    // Current state
    pub current_pos_x: f32,
    pub current_pos_y: f32,
    pub current_vel_x: f32,
    pub current_vel_y: f32,
    pub current_angular_vel: f32,

    // Action
    pub left_thruster: f32,
    pub right_thruster: f32,

    // Resulting state
    pub resulting_pos_x: f32,
    pub resulting_pos_y: f32,
    pub resulting_vel_x: f32,
    pub resulting_vel_y: f32,
    pub resulting_angular_vel: f32,
}

impl Record {
    pub fn new(
        current_pos: (f32, f32),
        current_vel: (f32, f32),
        current_angular_vel: f32,
        left_thruster: f32,
        right_thruster: f32,
        resulting_pos: (f32, f32),
        resulting_vel: (f32, f32),
        resulting_angular_vel: f32,
    ) -> Self {
        Self {
            current_pos_x: current_pos.0,
            current_pos_y: current_pos.1,
            current_vel_x: current_vel.0,
            current_vel_y: current_vel.1,
            current_angular_vel,
            left_thruster,
            right_thruster,
            resulting_pos_x: resulting_pos.0,
            resulting_pos_y: resulting_pos.1,
            resulting_vel_x: resulting_vel.0,
            resulting_vel_y: resulting_vel.1,
            resulting_angular_vel,
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
            (0.1, 0.2),
            0.05,
            1.0,
            0.5,
            (1.1, 2.05),
            (0.11, 0.21),
            0.04,
        );

        let temp_path = "test_record.csv";
        Record::write_to_csv(&[record], temp_path).unwrap();
        
        let content = fs::read_to_string(temp_path).unwrap();
        fs::remove_file(temp_path).unwrap();
        
        // Check if the content contains the expected values (ignoring headers)
        let expected_values = "1.0,2.0,0.1,0.2,0.05,1.0,0.5,1.1,2.05,0.11,0.21,0.04";
        if !content.contains(expected_values) {
            panic!("Content does not contain expected values.\nExpected: {}\nActual: {}", expected_values, content);
        }
    }
}