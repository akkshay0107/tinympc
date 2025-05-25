use macroquad::prelude::*;

#[macroquad::main("Test Window")]
async fn main() {
    loop {
        clear_background(DARKBLUE);
        draw_circle(screen_width() / 2.0, screen_height() / 2.0, 30.0, YELLOW);
        next_frame().await
    }
}
