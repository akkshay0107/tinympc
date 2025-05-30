use macroquad::prelude::*;
use tinympc::path_drawer::PathDrawer;
use tinympc::rocket::*;

const GROUND_CRUST_COLOR: Color = Color::new(0.4, 0.2, 0.1, 1.0); // Darker than interior
const GROUND_INTERIOR_COLOR: Color = Color::new(0.6, 0.4, 0.2, 1.0); // Tan color
const GROUND_CRUST_THICKNESS: f32 = 10.0;
const PARALLAX_SCROLL_SPEED: f32 = 0.08;

struct Game {
    stars: Vec<(f32, f32, f32)>,
    ground_height: f32,
    rocket: Rocket,
}

impl Game {
    fn new() -> Self {
        let mut stars = Vec::new();
        for _ in 0..50 {
            let x = rand::gen_range(0.0, screen_width());
            let y = rand::gen_range(0.0, screen_height() * 0.5); // Stars only in upper 50% of screen
            let radius = rand::gen_range(1.0, 2.0);
            stars.push((x, y, radius));
        }

        let rocket = Rocket::new(screen_width() / 2.0, screen_height() * 0.8);
        Self {
            stars,
            ground_height: screen_height() * 0.8,
            rocket,
        }
    }

    fn update(&mut self) {
        // Parallax effect implementation
        for star in self.stars.iter_mut() {
            star.0 -= PARALLAX_SCROLL_SPEED;
            if star.0 < 0.0 {
                star.0 = screen_width();
                star.1 = rand::gen_range(0.0, screen_height() * 0.5);
            }
        }
    }

    fn draw_space_atmos(&self) {
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

    fn draw(&self) {
        // Draw the space background
        self.draw_space_atmos();

        // Stars
        for (x, y, radius) in &self.stars {
            draw_circle(*x, *y, *radius, WHITE);
        }

        // Ground crust
        let screen_w = screen_width();
        let screen_h = screen_height();
        let ground_y = self.ground_height;

        draw_rectangle(
            0.0,
            ground_y - GROUND_CRUST_THICKNESS,
            screen_w,
            GROUND_CRUST_THICKNESS,
            GROUND_CRUST_COLOR,
        );

        // Ground interior
        draw_rectangle(
            0.0,
            ground_y,
            screen_w,
            screen_h - ground_y,
            GROUND_INTERIOR_COLOR,
        );

        // Draw rocket
        self.rocket.draw();
    }
}

// #[macroquad::main("Scene Testing")]
// async fn main() {
//     let mut game = Game::new();
//     loop {
//         game.update();
//         game.draw();
//         next_frame().await
//     }
// }

#[macroquad::main("Draw Testing")]
async fn main() {
    let mut path_test = PathDrawer::new();
    loop {
        clear_background(WHITE);
        path_test.update();
        path_test.draw();
        next_frame().await
    }
}
