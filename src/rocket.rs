use macroquad::prelude::*;

pub const ROCKET_WIDTH: f32 = 10.0;
pub const ROCKET_HEIGHT: f32 = 40.0;
pub const ROCKET_COLOR: Color = Color::new(0.9, 0.9, 0.9, 1.0);

pub struct Rocket {
    x: f32,
    y: f32,
}

impl Rocket {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn draw(&self) {
        // Rocket nose cone
        draw_triangle(
            Vec2::new(self.x, self.y - ROCKET_HEIGHT),
            Vec2::new(self.x - ROCKET_WIDTH / 2.0, self.y - ROCKET_HEIGHT + 10.0),
            Vec2::new(self.x + ROCKET_WIDTH / 2.0, self.y - ROCKET_HEIGHT + 10.0),
            ROCKET_COLOR,
        );

        // Rocket body
        draw_rectangle(
            self.x - ROCKET_WIDTH / 2.0,
            self.y - ROCKET_HEIGHT + 10.0,
            ROCKET_WIDTH,
            ROCKET_HEIGHT - 10.0,
            ROCKET_COLOR,
        );

        // Rocket fins
        draw_triangle(
            Vec2::new(self.x - ROCKET_WIDTH / 2.0 - 5.0, self.y - 10.0),
            Vec2::new(self.x - ROCKET_WIDTH / 2.0, self.y - 20.0),
            Vec2::new(self.x - ROCKET_WIDTH / 2.0, self.y),
            ROCKET_COLOR,
        );
        draw_triangle(
            Vec2::new(self.x + ROCKET_WIDTH / 2.0 + 5.0, self.y - 10.0),
            Vec2::new(self.x + ROCKET_WIDTH / 2.0, self.y - 20.0),
            Vec2::new(self.x + ROCKET_WIDTH / 2.0, self.y),
            ROCKET_COLOR,
        );
    }
}
