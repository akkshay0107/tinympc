use crate::world::pixels_per_meter;
use macroquad::prelude::*;

pub const PARALLAX_SCROLL_SPEED: f32 = 0.08;

pub const DRAG_POINTER_COLOR: Color = Color::new(1.0, 1.0, 0.0, 1.0);
pub const DRAG_POINTER_RADIUS: f32 = 4.0;

pub struct Game {
    stars: Vec<(f32, f32, f32)>,
    pub rocket: crate::rocket_sprite::RocketSprite,
}

impl Game {
    pub fn new() -> Self {
        let mut stars = Vec::new();
        for _ in 0..50 {
            let x = rand::gen_range(0.0, screen_width());
            let y = rand::gen_range(0.0, screen_height() * 0.5); // Stars only in upper 50% of screen
            let radius = rand::gen_range(1.0, 2.0);
            stars.push((x, y, radius));
        }

        let rocket_x = screen_width() / 2.0;
        let ground_y = screen_height() * 0.8;
        let rocket = crate::rocket_sprite::RocketSprite::new(rocket_x, ground_y);
        Self { stars, rocket }
    }

    pub fn update(&mut self) {
        // Parallax effect implementation
        for star in self.stars.iter_mut() {
            star.0 -= PARALLAX_SCROLL_SPEED;
            if star.0 < 0.0 {
                star.0 = screen_width();
                star.1 = rand::gen_range(0.0, screen_height() * 0.5);
            }
        }
    }

    pub fn draw_space_atmos(&self) {
        let screen_w = screen_width();
        let screen_h = screen_height();
        let num_steps = 50;
        let strip_height = screen_h / num_steps as f32;

        let top_color = Color::new(0.05, 0.02, 0.15, 1.0); // Deep purple
        let mid_color = Color::new(0.02, 0.05, 0.25, 1.0); // Dark blue
        let bottom_color = Color::new(0.01, 0.02, 0.08, 1.0); // Very dark blue

        for i in 0..num_steps {
            let y = i as f32 * strip_height;
            let t = i as f32 / (num_steps - 1) as f32;

            // Linear interpolation of colors to get a gradient
            // Fades from top to mid then mid to bottom in a 60:40 ratio
            let color = if t < 0.6 {
                // Top to mid transition
                let local_t = t / 0.6;
                Color::new(
                    top_color.r + (mid_color.r - top_color.r) * local_t,
                    top_color.g + (mid_color.g - top_color.g) * local_t,
                    top_color.b + (mid_color.b - top_color.b) * local_t,
                    1.0,
                )
            } else {
                // Mid to bottom transition
                let local_t = (t - 0.6) / 0.4;
                Color::new(
                    mid_color.r + (bottom_color.r - mid_color.r) * local_t,
                    mid_color.g + (bottom_color.g - mid_color.g) * local_t,
                    mid_color.b + (bottom_color.b - mid_color.b) * local_t,
                    1.0,
                )
            };

            draw_rectangle(0.0, y, screen_w, strip_height + 1.0, color);
        }
    }

    pub fn draw_ground(&self) {
        let screen_w = screen_width();
        let screen_h = screen_height();
        let ground_y = screen_h * 0.8;

        let base_color = Color::from_rgba(74, 48, 30, 255);
        let shadow_color = Color::from_rgba(57, 36, 23, 255);
        let grass_color = Color::from_rgba(43, 86, 29, 255);
        let grass_edge_color = Color::from_rgba(34, 69, 20, 255);

        let ppm = pixels_per_meter();
        let block_size = 1.6 * ppm;

        for x in (0..screen_w as i32).step_by(block_size as usize) {
            let top_y = ground_y;

            // Draw the top grass block
            draw_rectangle(x as f32, top_y, block_size, block_size, grass_color);
            draw_rectangle(
                x as f32,
                top_y + block_size - 4.0,
                block_size,
                4.0,
                grass_edge_color,
            );

            for y in ((top_y + block_size) as i32..screen_h as i32).step_by(block_size as usize) {
                let noise_val = (x as f32 * 0.1).sin() + (y as f32 * 0.07).cos();
                let color = if noise_val > 0.0 {
                    base_color
                } else {
                    shadow_color
                };
                draw_rectangle(x as f32, y as f32, block_size, block_size, color);
            }
        }

        let ground_height = screen_h - ground_y;
        let num_steps = 20;
        let step_height = ground_height / num_steps as f32;
        let shine_top = Color::new(0.1, 0.1, 0.3, 0.2); // More shine at the top
        let shine_bottom = Color::new(0.05, 0.05, 0.15, 0.0); // Fades out

        for i in 0..num_steps {
            let y_pos = ground_y + i as f32 * step_height;
            let t = i as f32 / (num_steps - 1) as f32;
            let interpolated_color = Color {
                r: shine_top.r * (1.0 - t) + shine_bottom.r * t,
                g: shine_top.g * (1.0 - t) + shine_bottom.g * t,
                b: shine_top.b * (1.0 - t) + shine_bottom.b * t,
                a: shine_top.a * (1.0 - t) + shine_bottom.a * t,
            };
            // 1.0 + step_height to avoid gaps
            draw_rectangle(0.0, y_pos, screen_w, step_height + 1.0, interpolated_color);
        }
    }

    pub fn draw_landing_flags(&self, dist_from_center_m: f32) {
        let ppm = pixels_per_meter();
        let dist_from_center = dist_from_center_m * ppm;

        let screen_w = screen_width();
        let screen_h = screen_height();
        let ground_y = screen_h * 0.8;
        let center_x = screen_w / 2.0;

        let pole_height = 4.0 * ppm;
        let pole_thickness = 0.2 * ppm;
        let flag_width = 2.5 * ppm;
        let flag_height = 1.5 * ppm;

        let pole_color = WHITE;
        let flag_color = RED;

        // The landing pad is between the two flags.
        // One flag is at `center_x - dist_from_center` and the other is at `center_x + dist_from_center`.
        let positions = [center_x - dist_from_center, center_x + dist_from_center];

        for &x_pos in &positions {
            // Draw pole
            draw_line(
                x_pos,
                ground_y,
                x_pos,
                ground_y - pole_height,
                pole_thickness,
                pole_color,
            );

            // Flags should point inwards to frame the landing zone
            let direction = if x_pos < center_x { 1.0 } else { -1.0 };

            // Draw flag triangle
            let v1 = Vec2::new(x_pos, ground_y - pole_height);
            let v2 = Vec2::new(
                x_pos + direction * flag_width,
                ground_y - pole_height + flag_height / 2.0,
            );
            let v3 = Vec2::new(x_pos, ground_y - pole_height + flag_height);

            draw_triangle(v1, v2, v3, flag_color);
        }
    }

    pub fn draw(&self) {
        self.draw_space_atmos();

        // Draw stars
        for (x, y, radius) in &self.stars {
            draw_circle(*x, *y, *radius, WHITE);
        }

        self.draw_ground();
        self.draw_landing_flags(10.0);
    }
}
