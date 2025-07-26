use std::f32::consts::PI;

pub const MAX_POS_X: f32 = 80.0; // Min pos x is 0
pub const _MIN_POS_Y: f32 = 2.0; // COM of vertical rocket is at 2.0 when it touches the ground
pub const MAX_POS_Y: f32 = 45.0;
pub const MAX_ANGLE_DEFLECTION: f32 = PI / 12.0;
pub const MAX_VX: f32 = 5.0;
pub const MIN_VY: f32 = -2.0;
pub const MAX_VY: f32 = 5.0;
pub const MAX_ANGULAR_VELOCITY: f32 = 0.3;
pub const GROUND_THRESHOLD: f32 = 1.99; // Slightly under min possible y
