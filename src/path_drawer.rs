//! Module for drawing the rockets path
//!
//! Stores the history of the mouse's path and draws a line following it
//! Only allows for one stroke (since the rocket's path should only be one
//! continuous line) and limits the length of the path (equivalent to fuel
//! on a rocket)

use macroquad::prelude::*;

const MIN_DIST: f32 = 1.0; // Minimum distance between two points in the path (One pixel)
const LINE_THICKNESS: f32 = 2.0; // Thickness of line drawn
const MAX_FUEL_LIMIT: i32 = 1000; // Maximum count of mouse history stored before line breaks

pub struct PathDrawer {
    points: Vec<Vec2>,
    is_drawing: bool,
    last_point: Option<Vec2>,
    stroke_complete: bool,
}

impl PathDrawer {
    pub fn new() -> Self {
        PathDrawer {
            points: Vec::new(),
            is_drawing: false,
            last_point: None,
            stroke_complete: false,
        }
    }

    pub fn update(&mut self) {
        if self.stroke_complete {
            return;
        }

        if is_mouse_button_down(MouseButton::Left) {
            let current_point = mouse_position().into();

            if !self.is_drawing {
                self.points.push(current_point);
                self.last_point = Some(current_point);
                self.is_drawing = true;
            } else {
                // Add new point only if it's far enough from the last point
                if let Some(last) = self.last_point {
                    let distance = (current_point - last).length();
                    if distance > MIN_DIST {
                        self.points.push(current_point);
                        self.last_point = Some(current_point);
                    }
                }
            }
        } else if self.is_drawing {
            // Stroke completed if currently drawing + mouse released
            self.is_drawing = false;
            self.stroke_complete = true;
        }
    }

    pub fn draw(&self) {
        // Mouse path approximated as multiple small straight lines joined
        if self.points.len() > 1 {
            for i in 0..self.points.len() - 1 {
                let start = self.points[i];
                let end = self.points[i + 1];
                draw_line(start.x, start.y, end.x, end.y, LINE_THICKNESS, RED);
            }
        }

        if self.is_drawing {
            if let Some(last) = self.last_point {
                let current: Vec2 = mouse_position().into();
                draw_line(last.x, last.y, current.x, current.y, LINE_THICKNESS, RED);
            }
        }
    }
}
