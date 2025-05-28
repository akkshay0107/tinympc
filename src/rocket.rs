use macroquad::prelude::*;

pub const ROCKET_WIDTH: f32 = 10.0;
pub const ROCKET_HEIGHT: f32 = 40.0;
pub const ROCKET_BASE: Color = Color::new(0.6, 0.6, 0.65, 1.0); // Blue-gray
pub const ROCKET_HIGHLIGHT: Color = Color::new(0.9, 0.9, 0.95, 1.0); // Chrome
pub const ROCKET_SHADOW: Color = Color::new(0.4, 0.4, 0.45, 1.0); // Dark gray

pub struct Rocket {
    x: f32,
    y: f32,
}

impl Rocket {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn draw(&self) {
        // Rocket nose cone (base)
        draw_triangle(
            Vec2::new(self.x, self.y - ROCKET_HEIGHT),
            Vec2::new(self.x - ROCKET_WIDTH / 2.0, self.y - ROCKET_HEIGHT + 10.0),
            Vec2::new(self.x + ROCKET_WIDTH / 2.0, self.y - ROCKET_HEIGHT + 10.0),
            ROCKET_BASE,
        );

        // Nose cone highlight (left side)
        draw_triangle(
            Vec2::new(self.x, self.y - ROCKET_HEIGHT),
            Vec2::new(self.x - ROCKET_WIDTH / 2.0, self.y - ROCKET_HEIGHT + 10.0),
            Vec2::new(self.x - 1.0, self.y - ROCKET_HEIGHT + 8.0),
            ROCKET_HIGHLIGHT,
        );

        // Rocket body (base)
        draw_rectangle(
            self.x - ROCKET_WIDTH / 2.0,
            self.y - ROCKET_HEIGHT + 10.0,
            ROCKET_WIDTH,
            ROCKET_HEIGHT - 10.0,
            ROCKET_BASE,
        );

        // Body highlight strip (left side)
        draw_rectangle(
            self.x - ROCKET_WIDTH / 2.0,
            self.y - ROCKET_HEIGHT + 10.0,
            2.0,
            ROCKET_HEIGHT - 10.0,
            ROCKET_HIGHLIGHT,
        );

        // Body shadow strip (right side)
        draw_rectangle(
            self.x + ROCKET_WIDTH / 2.0 - 2.0,
            self.y - ROCKET_HEIGHT + 10.0,
            2.0,
            ROCKET_HEIGHT - 10.0,
            ROCKET_SHADOW,
        );

        // Left rocket fin (base)
        draw_triangle(
            Vec2::new(self.x - ROCKET_WIDTH / 2.0 - 5.0, self.y - 10.0),
            Vec2::new(self.x - ROCKET_WIDTH / 2.0, self.y - 20.0),
            Vec2::new(self.x - ROCKET_WIDTH / 2.0, self.y),
            ROCKET_BASE,
        );

        // Left fin highlight
        draw_triangle(
            Vec2::new(self.x - ROCKET_WIDTH / 2.0 - 4.0, self.y - 12.0),
            Vec2::new(self.x - ROCKET_WIDTH / 2.0, self.y - 18.0),
            Vec2::new(self.x - ROCKET_WIDTH / 2.0, self.y - 15.0),
            ROCKET_HIGHLIGHT,
        );

        // Right rocket fin (base)
        draw_triangle(
            Vec2::new(self.x + ROCKET_WIDTH / 2.0 + 5.0, self.y - 10.0),
            Vec2::new(self.x + ROCKET_WIDTH / 2.0, self.y - 20.0),
            Vec2::new(self.x + ROCKET_WIDTH / 2.0, self.y),
            ROCKET_BASE,
        );

        // Right fin shadow
        draw_triangle(
            Vec2::new(self.x + ROCKET_WIDTH / 2.0 + 4.0, self.y - 12.0),
            Vec2::new(self.x + ROCKET_WIDTH / 2.0, self.y - 18.0),
            Vec2::new(self.x + ROCKET_WIDTH / 2.0, self.y - 15.0),
            ROCKET_SHADOW,
        );
    }
}
