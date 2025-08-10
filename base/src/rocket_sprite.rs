//! Module for managing the rocket sprite
//!
//! This module contains the implementation to draw the rocket sprite
//! based on its internal state and the functions to update its internal
//! state
use macroquad::prelude::*;

pub const ROCKET_WIDTH: f32 = 10.0;
pub const ROCKET_HEIGHT: f32 = 40.0;
const THRUSTER_WIDTH_RATIO: f32 = 0.4;
const THRUSTER_HEIGHT_RATIO: f32 = 0.4;

pub struct RocketSprite {
    x: f32,
    y: f32,
    angle: f32,          // angle from vertical (radians)
    left_thruster: f32,  // left thruster power (normalized)
    right_thruster: f32, // right thruster power (normalized)
}

impl RocketSprite {
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            angle: 0.0,
            left_thruster: 0.0,
            right_thruster: 0.0,
        }
    }

    pub fn get_state(&self) -> (f32, f32, f32, (f32, f32)) {
        (
            self.x,
            self.y,
            self.angle,
            (self.left_thruster, self.right_thruster),
        )
    }

    pub fn set_state(&mut self, x: f32, y: f32, angle: f32, thrust: (f32, f32)) {
        self.x = x;
        self.y = y;
        self.angle = angle;
        self.left_thruster = thrust.0.clamp(0.0, 1.0);
        self.right_thruster = thrust.1.clamp(0.0, 1.0);
    }

    pub fn draw(&self) {
        let half_width = ROCKET_WIDTH / 2.0;
        let half_height = ROCKET_HEIGHT / 2.0;
        let thruster_width = ROCKET_WIDTH * THRUSTER_WIDTH_RATIO;
        let thruster_height = ROCKET_HEIGHT * THRUSTER_HEIGHT_RATIO;

        let center = vec2(self.x, self.y);

        // Define all polygon vertices relative to center
        let body_vertices = [
            vec2(-half_width, -half_height),
            vec2(half_width, -half_height),
            vec2(half_width, half_height),
            vec2(-half_width, half_height),
        ];

        let left_thruster_vertices = [
            vec2(-half_width - thruster_width, half_height - thruster_height),
            vec2(-half_width, half_height - thruster_height),
            vec2(-half_width, half_height),
            vec2(-half_width - thruster_width, half_height),
        ];

        let right_thruster_vertices = [
            vec2(half_width, half_height - thruster_height),
            vec2(half_width + thruster_width, half_height - thruster_height),
            vec2(half_width + thruster_width, half_height),
            vec2(half_width, half_height),
        ];

        self.draw_rotated_polygon(&body_vertices, center, GRAY);
        self.draw_rotated_polygon(&left_thruster_vertices, center, GRAY);
        self.draw_rotated_polygon(&right_thruster_vertices, center, GRAY);
    }

    fn draw_rotated_polygon(&self, vertices: &[Vec2], center: Vec2, color: Color) {
        let cos_a = self.angle.cos();
        let sin_a = self.angle.sin();

        // Transform vertices to world coordinates
        let transformed_vertices: Vec<Vec2> = vertices
            .iter()
            .map(|v| {
                vec2(
                    v.x * cos_a - v.y * sin_a + center.x,
                    v.x * sin_a + v.y * cos_a + center.y,
                )
            })
            .collect();

        // Draw polygon outline
        for i in 0..transformed_vertices.len() {
            let current = transformed_vertices[i];
            let next = transformed_vertices[(i + 1) % transformed_vertices.len()];
            draw_line(current.x, current.y, next.x, next.y, 2.0, color);
        }

        // Fill polygon using triangle fan from first vertex
        if transformed_vertices.len() >= 3 {
            let first_vertex = transformed_vertices[0];

            for i in 1..transformed_vertices.len() - 1 {
                draw_triangle(
                    first_vertex,
                    transformed_vertices[i],
                    transformed_vertices[i + 1],
                    color,
                );
            }
        }
    }
}
