//! Module for drawing the rockets path
//!
//! Stores the history of the mouse's path and draws a line following it
//! Only allows for one stroke (since the rocket's path should only be one
//! continuous line) and limits the length of the path (equivalent to fuel
//! on a rocket)

use macroquad::prelude::*;

const MIN_DIST: f32 = 1.0; // Minimum distance between points (one pixel)
const LINE_THICKNESS: f32 = 2.0;
const MAX_FUEL: f32 = 100.0;
const FUEL_PER_PIXEL: f32 = 0.1;

pub struct PathDrawer {
    points: Vec<Vec2>,
    last_point: Option<Vec2>,
    trigger_center: Vec2,
    tolerance_radius: f32,
    fuel: f32,
    stroke_complete: bool,
    is_drawing: bool,
}

impl PathDrawer {
    pub fn new(trigger_x: f32, trigger_y: f32, tolerance_radius: f32) -> Self {
        let trigger_center = Vec2::new(trigger_x, trigger_y);
        PathDrawer {
            points: Vec::new(),
            is_drawing: false,
            last_point: None,
            stroke_complete: false,
            fuel: MAX_FUEL,
            trigger_center,
            tolerance_radius,
        }
    }

    pub fn update(&mut self) {
        if self.stroke_complete {
            return;
        }

        if is_mouse_button_down(MouseButton::Left) {
            let current_point: Vec2 = mouse_position().into();
            if self.is_drawing {
                if let Some(last) = self.last_point {
                    let distance = (current_point - last).length();
                    if distance < MIN_DIST {
                        return;
                    }
                    let fuel_needed = FUEL_PER_PIXEL * distance;
                    if self.fuel >= fuel_needed {
                        self.points.push(current_point);
                        self.last_point = Some(current_point);
                        self.fuel -= fuel_needed;
                    } else {
                        // Not enough fuel to continue drawing
                        // Breaks stroke
                        self.is_drawing = false;
                        self.stroke_complete = true;
                    }
                }
            } else {
                self.points.push(current_point);
                self.last_point = Some(current_point);
                self.is_drawing = true;
            }
        } else if self.is_drawing {
            // Stroke complete if currently drawing and then mouse release triggered
            // setting fuel to 0 for consistency with fuel gauge
            self.is_drawing = false;
            self.stroke_complete = true;
            self.fuel = 0.0;
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

        // Fuel gauge
        let screen_w = screen_width();
        let screen_h = screen_height();
        let gauge_width = 100.0;
        let gauge_height = 20.0;
        let gauge_x = screen_w - gauge_width - 10.0;
        let gauge_y = screen_h - gauge_height - 10.0;
        draw_rectangle(
            gauge_x,
            gauge_y,
            gauge_width,
            gauge_height,
            Color::new(0.2, 0.2, 0.2, 1.0),
        );
        draw_rectangle(
            gauge_x,
            gauge_y,
            gauge_width * (self.fuel / MAX_FUEL),
            gauge_height,
            ORANGE,
        );
        // Fuel gauge border
        draw_rectangle_lines(gauge_x, gauge_y, gauge_width, gauge_height, 2.0, WHITE);
    }
}
