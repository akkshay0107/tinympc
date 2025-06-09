//! Module for managing the rocket sprite
//!
//! This module contains the implementation to draw the rocket sprite based on
//! its internal state and the functions to update its internal state

use macroquad::prelude::*;

pub const ROCKET_WIDTH: f32 = 10.0;
pub const ROCKET_HEIGHT: f32 = 40.0;
const ROCKET_BASE: Color = Color::new(0.6, 0.6, 0.65, 1.0); // Blue-gray
const ROCKET_HIGHLIGHT: Color = Color::new(0.9, 0.9, 0.95, 1.0); // Chrome
const ROCKET_SHADOW: Color = Color::new(0.4, 0.4, 0.45, 1.0); // Dark-gray

// Flame colors
const FLAME_CORE: Color = Color::new(1.0, 1.0, 0.8, 1.0); // Yellow
const FLAME_MID: Color = Color::new(1.0, 0.6, 0.2, 1.0); // Orange
const FLAME_OUTER: Color = Color::new(0.8, 0.2, 0.1, 1.0); // Red

pub struct Rocket {
    x: f32,
    y: f32,
    angle: f32, // angle from vertical (radians)
    thrust: f32, // thrust (normalized)
}

impl Rocket {
    pub fn new(x: f32, y: f32) -> Self {
        Self { 
            x, 
            y, 
            angle: 0.0, 
            thrust: 0.0 
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

    fn rotate_point(&self, local_x: f32, local_y: f32) -> Vec2 {
        // Rotates a point around the rocket's center by the body angle
        let cos_a = self.angle.cos();
        let sin_a = self.angle.sin();
        
        Vec2::new(
            self.x + local_x * cos_a - local_y * sin_a,
            self.y + local_x * sin_a + local_y * cos_a,
        )
    }

    pub fn draw(&self) {
        if self.thrust > 0.0 {
            self.draw_flame();
        }

        // Rocket nose cone (base)
        let nose_tip = self.rotate_point(0.0, -ROCKET_HEIGHT);
        let nose_left = self.rotate_point(-ROCKET_WIDTH / 2.0, -ROCKET_HEIGHT + 10.0);
        let nose_right = self.rotate_point(ROCKET_WIDTH / 2.0, -ROCKET_HEIGHT + 10.0);

        draw_triangle(nose_tip, nose_left, nose_right, ROCKET_BASE);

        // Nose cone highlight (left side)
        let nose_highlight = self.rotate_point(-1.0, -ROCKET_HEIGHT + 8.0);
        draw_triangle(nose_tip, nose_left, nose_highlight, ROCKET_HIGHLIGHT);

        // Rocket body (base)
        let body_tl = self.rotate_point(-ROCKET_WIDTH / 2.0, -ROCKET_HEIGHT + 10.0);
        let body_tr = self.rotate_point(ROCKET_WIDTH / 2.0, -ROCKET_HEIGHT + 10.0);
        let body_bl = self.rotate_point(-ROCKET_WIDTH / 2.0, 0.0);
        let body_br = self.rotate_point(ROCKET_WIDTH / 2.0, 0.0);
        
        // Draw body as two triangles
        draw_triangle(body_tl, body_tr, body_bl, ROCKET_BASE);
        draw_triangle(body_tr, body_bl, body_br, ROCKET_BASE);

        // Body highlight strip (left side)
        let highlight_tl = self.rotate_point(-ROCKET_WIDTH / 2.0, -ROCKET_HEIGHT + 10.0);
        let highlight_tr = self.rotate_point(-ROCKET_WIDTH / 2.0 + 2.0, -ROCKET_HEIGHT + 10.0);
        let highlight_bl = self.rotate_point(-ROCKET_WIDTH / 2.0, 0.0);
        let highlight_br = self.rotate_point(-ROCKET_WIDTH / 2.0 + 2.0, 0.0);
        
        draw_triangle(highlight_tl, highlight_tr, highlight_bl, ROCKET_HIGHLIGHT);
        draw_triangle(highlight_tr, highlight_bl, highlight_br, ROCKET_HIGHLIGHT);

        // Body shadow strip (right side)
        let shadow_tl = self.rotate_point(ROCKET_WIDTH / 2.0 - 2.0, -ROCKET_HEIGHT + 10.0);
        let shadow_tr = self.rotate_point(ROCKET_WIDTH / 2.0, -ROCKET_HEIGHT + 10.0);
        let shadow_bl = self.rotate_point(ROCKET_WIDTH / 2.0 - 2.0, 0.0);
        let shadow_br = self.rotate_point(ROCKET_WIDTH / 2.0, 0.0);
        
        draw_triangle(shadow_tl, shadow_tr, shadow_bl, ROCKET_SHADOW);
        draw_triangle(shadow_tr, shadow_bl, shadow_br, ROCKET_SHADOW);

        // Left rocket fin
        let left_fin_tip = self.rotate_point(-ROCKET_WIDTH / 2.0 - 5.0, -10.0);
        let left_fin_top = self.rotate_point(-ROCKET_WIDTH / 2.0, -20.0);
        let left_fin_bottom = self.rotate_point(-ROCKET_WIDTH / 2.0, 0.0);
        
        draw_triangle(left_fin_tip, left_fin_top, left_fin_bottom, ROCKET_BASE);

        // Left fin highlight
        let left_highlight_tip = self.rotate_point(-ROCKET_WIDTH / 2.0 - 4.0, -12.0);
        let left_highlight_top = self.rotate_point(-ROCKET_WIDTH / 2.0, -18.0);
        let left_highlight_bottom = self.rotate_point(-ROCKET_WIDTH / 2.0, -15.0);
        
        draw_triangle(left_highlight_tip, left_highlight_top, left_highlight_bottom, ROCKET_HIGHLIGHT);

        // Right rocket fin
        let right_fin_tip = self.rotate_point(ROCKET_WIDTH / 2.0 + 5.0, -10.0);
        let right_fin_top = self.rotate_point(ROCKET_WIDTH / 2.0, -20.0);
        let right_fin_bottom = self.rotate_point(ROCKET_WIDTH / 2.0, 0.0);
        
        draw_triangle(right_fin_tip, right_fin_top, right_fin_bottom, ROCKET_BASE);

        // Right fin shadow
        let right_shadow_tip = self.rotate_point(ROCKET_WIDTH / 2.0 + 4.0, -12.0);
        let right_shadow_top = self.rotate_point(ROCKET_WIDTH / 2.0, -18.0);
        let right_shadow_bottom = self.rotate_point(ROCKET_WIDTH / 2.0, -15.0);
        
        draw_triangle(right_shadow_tip, right_shadow_top, right_shadow_bottom, ROCKET_SHADOW);
    }

    fn draw_flame(&self) {
        let flame_length = self.thrust * 25.0; // Scale flame based on thrust
        let flame_width = ROCKET_WIDTH * 0.8;
        
        if flame_length > 0.0 {
            // Outer flame (red)
            let outer_left = self.rotate_point(-flame_width / 2.0, 0.0);
            let outer_right = self.rotate_point(flame_width / 2.0, 0.0);
            let outer_tip = self.rotate_point(0.0, flame_length);
            
            draw_triangle(outer_left, outer_right, outer_tip, FLAME_OUTER);
            
            // Middle flame (orange)
            let mid_width = flame_width * 0.7;
            let mid_length = flame_length * 0.8;
            let mid_left = self.rotate_point(-mid_width / 2.0, 0.0);
            let mid_right = self.rotate_point(mid_width / 2.0, 0.0);
            let mid_tip = self.rotate_point(0.0, mid_length);
            
            draw_triangle(mid_left, mid_right, mid_tip, FLAME_MID);
            
            // Core flame (yellow)
            let core_width = flame_width * 0.4;
            let core_length = flame_length * 0.6;
            let core_left = self.rotate_point(-core_width / 2.0, 0.0);
            let core_right = self.rotate_point(core_width / 2.0, 0.0);
            let core_tip = self.rotate_point(0.0, core_length);
            
            draw_triangle(core_left, core_right, core_tip, FLAME_CORE);
        }
    }
}