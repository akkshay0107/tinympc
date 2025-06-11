//! Module for managing the rocket sprite
//!
//! This module contains the implementation to draw the (now simplified)
//! rocket sprite based on its internal state and the functions to
//! update its internal state

use macroquad::prelude::*;

pub const ROCKET_WIDTH: f32 = 10.0;
pub const ROCKET_HEIGHT: f32 = 40.0;
const THRUSTER_WIDTH_RATIO: f32 = 0.4;
const THRUSTER_HEIGHT_RATIO: f32 = 0.4;

pub struct Rocket {
    x: f32,
    y: f32,
    angle: f32,  // angle from vertical (radians)
    thrust: f32, // thrust (normalized)
}

impl Rocket {
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            angle: 0.0,
            thrust: 0.0,
        }
    }

    pub fn get_state(&self) -> (f32, f32, f32, f32) {
        (self.x, self.y, self.angle, self.thrust)
    }

    pub fn set_state(&mut self, x: f32, y: f32, angle: f32, thrust: f32) {
        self.x = x;
        self.y = y;
        self.angle = angle;
        self.thrust = thrust.min(1.0);
    }

    pub fn draw(&self) {
        let half_width = ROCKET_WIDTH / 2.0;
        let half_height = ROCKET_HEIGHT / 2.0;

        // Main body (central cuboid)
        let top_left = Vec2::new(self.x - half_width, self.y - half_height);
        draw_rectangle(top_left.x, top_left.y, ROCKET_WIDTH, ROCKET_HEIGHT, GRAY);

        // Thrusters
        let thruster_width = ROCKET_WIDTH * THRUSTER_WIDTH_RATIO;
        let thruster_height = ROCKET_HEIGHT * THRUSTER_HEIGHT_RATIO;

        let left_thruster_pos = Vec2::new(
            self.x - half_width - thruster_width,
            self.y + half_height - thruster_height,
        );

        draw_rectangle(
            left_thruster_pos.x,
            left_thruster_pos.y,
            thruster_width,
            thruster_height,
            GRAY,
        );

        let right_thruster_pos =
            Vec2::new(self.x + half_width, self.y + half_height - thruster_height);

        draw_rectangle(
            right_thruster_pos.x,
            right_thruster_pos.y,
            thruster_width,
            thruster_height,
            GRAY,
        );
    }
}
