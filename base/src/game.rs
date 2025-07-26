use macroquad::prelude::*;

pub const GROUND_CRUST_COLOR: Color = Color::new(0.4, 0.2, 0.1, 1.0); // Darker than interior
pub const GROUND_INTERIOR_COLOR: Color = Color::new(0.6, 0.4, 0.2, 1.0); // Tan color
pub const GROUND_CRUST_THICKNESS: f32 = 10.0;
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

        draw_rectangle(
            0.0,
            ground_y,
            screen_w,
            GROUND_CRUST_THICKNESS,
            GROUND_CRUST_COLOR,
        );
        let base_color = GROUND_INTERIOR_COLOR;

        // Draw in texture in the interior of the ground
        for y in ((ground_y + GROUND_CRUST_THICKNESS) as i32..screen_h as i32).step_by(4) {
            for x in (0..screen_w as i32).step_by(4) {
                // Simple noise function based on position
                // Combines three different sine waves and a stable pattern
                // to create a more complex noise pattern
                let n1 = (x as f32 * 0.1).sin();
                let n2 = (y as f32 * 0.07).cos();
                let n3 = ((x as f32 * 0.05 + y as f32 * 0.05).sin() * 2.0).sin();
                let noise = (n1 * 0.3 + n2 * 0.2 + n3 * 0.5).abs() * 0.03;

                let stable = ((x as i32 + y * 12345 as i32) % 100) as f32 / 600.0;
                let noise = (noise + stable).min(0.06);

                let color = Color::new(
                    (base_color.r - noise * 1.0).max(0.0),
                    (base_color.g - noise * 0.7).max(0.0),
                    (base_color.b - noise * 0.4).max(0.0),
                    base_color.a,
                );

                draw_rectangle(x as f32, y as f32, 4.0, 4.0, color);
            }
        }
    }

    pub fn draw(&self) {
        self.draw_space_atmos();

        // Draw stars
        for (x, y, radius) in &self.stars {
            draw_circle(*x, *y, *radius, WHITE);
        }

        self.draw_ground();

        // Draw bounding box
        let ground_y = screen_height() * 0.8;
        let box_width = 40.0;
        let box_height = 50.0;
        let box_x = screen_width() / 2.0 - box_width / 2.0;
        let box_color = Color::new(1.0, 0.0, 0.0, 0.7);

        draw_rectangle_lines(
            box_x,
            ground_y - box_height,
            box_width,
            box_height,
            2.0,
            box_color,
        );
    }
}
