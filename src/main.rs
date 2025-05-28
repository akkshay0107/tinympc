use macroquad::prelude::*;
use tinympc::rocket::*;

// Constants for colors and sizes
const SPACE_BACKGROUND: Color = Color::new(0.05, 0.05, 0.2, 1.0);
const GROUND_CRUST_COLOR: Color = Color::new(0.1, 0.05, 0.05, 1.0);
const GROUND_INTERIOR_COLOR: Color = Color::new(0.25, 0.2, 0.15, 1.0);
const GROUND_CRUST_THICKNESS: f32 = 10.0;

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
            let radius = rand::gen_range(1.0, 2.0); // Random radius between 1 and 2
            stars.push((x, y, radius));
        }

        // Position rocket at center of screen
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
            star.0 -= 0.1;
            if star.0 < 0.0 {
                star.0 = screen_width();
                star.1 = rand::gen_range(0.0, screen_height() * 0.5);
            }
        }
    }

    fn draw(&self) {
        // Draw space background shade
        clear_background(SPACE_BACKGROUND);

        // Stars
        for star in &self.stars {
            draw_circle(star.0, star.1, star.2, WHITE);
        }

        // Ground crust
        draw_rectangle(
            0.0,
            self.ground_height,
            screen_width(),
            GROUND_CRUST_THICKNESS,
            GROUND_CRUST_COLOR,
        );

        // Ground interior
        draw_rectangle(
            0.0,
            self.ground_height + GROUND_CRUST_THICKNESS,
            screen_width(),
            screen_height() - self.ground_height - GROUND_CRUST_THICKNESS,
            GROUND_INTERIOR_COLOR,
        );

        // Draw rocket
        self.rocket.draw();
    }
}

#[macroquad::main("Testing")]
async fn main() {
    let mut game = Game::new();
    loop {
        game.update();
        game.draw();
        next_frame().await
    }
}
