use std::f32::consts::PI;

pub const MAX_POS_X: f32 = 80.0; // Min pos x is 0
pub const _MIN_POS_Y: f32 = 2.0; // COM of vertical rocket is at 2.0 when it touches the ground
pub const MAX_POS_Y: f32 = 45.0;
pub const MAX_ANGLE_DEFLECTION: f32 = PI / 12.0; // 15 degrees
pub const MAX_ANGULAR_VELOCITY: f32 = 0.3;
pub const GROUND_THRESHOLD: f32 = 1.99; // Slightly under min possible y

pub const MAX_LANDING_ANGLE: f32 = PI / 36.0; // 5 degrees tolerance when landing
pub const MAX_LANDING_VX: f32 = 0.5;
pub const MAX_LANDING_VY: f32 = 1.0;
pub const MAX_LANDING_ANGULAR_VELOCITY: f32 = 0.05;

pub const MAX_GIMBAL_ANGLE: f32 = PI / 8.0; // 22.5 degress
