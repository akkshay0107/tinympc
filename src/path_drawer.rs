//! Module for drawing the rockets path
//!
//! Stores the history of the mouse's path and draws a line following it
//! Only allows for one stroke (since the rocket's path should only be one
//! continuous line) and limits the length of the path (equivalent to fuel
//! on a rocket). Uses spline interpolation to smooth out the path drawn
//! by the mouse.

use macroquad::prelude::*;

use crate::rocket::ROCKET_HEIGHT;

const MIN_DIST: f32 = 1.0; // Minimum distance between points (one pixel)
const LINE_THICKNESS: f32 = 2.0;
pub const MAX_FUEL: f32 = 50.0;
pub const FUEL_PER_PIXEL: f32 = 0.1;

pub struct PathDrawer {
    points: Vec<Vec2>,
    last_point: Option<Vec2>,
    trigger_center: Vec2,
    tolerance_radius: f32,
    fuel: f32,
    fix_applied: bool,
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
            fix_applied: false,
            fuel: MAX_FUEL,
            trigger_center,
            tolerance_radius,
        }
    }

    /// Function to update fuel and history of points stored, by tracking mouse
    /// movement and whether stroke is complete. Also dispatches path smoothing
    /// after stroke is complete.
    pub fn update(&mut self) {
        if self.stroke_complete {
            if !self.fix_applied {
                self.fix_path();
                self.fix_applied = true;
            }
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
            } else if (current_point - self.trigger_center).length() <= self.tolerance_radius {
                // Only allow stroke to begin when mouse starts click
                // within tolerance around the trigger center
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

    /// Draws the path stored in points vector onto the screen.
    /// Also draws fuel gauge as a bar in the bottom right corner.
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

    /// Function to smooth out mouse path using spline interpolation.
    /// Also makes sure that path is valid by deleting sections of path
    /// that travel underground
    fn fix_path(&mut self) {
        // Prevent rocket path from clipping into the ground
        let y_limit = (screen_height() * 0.8) - (ROCKET_HEIGHT / 2.0);

        // Find first point that clips below the ground limit
        if let Some(clip_index) = self.points.iter().position(|point| point.y > y_limit) {
            // Remove all points from the clipping point onward
            self.points.truncate(clip_index);
        }

        const STEPS: usize = 5;
        // 4 points needed for spline interpolation
        if self.points.len() >= 4 {
            let original_points = std::mem::take(&mut self.points);
            self.points.reserve(original_points.len() * STEPS + 2);
            self.points.push(original_points[0]);

            for window in original_points.windows(4) {
                for step in 0..=STEPS {
                    let t = step as f32 / STEPS as f32;
                    self.points.push(Self::catmull_rom_point(window, t));
                }
            }
            self.points.push(original_points[original_points.len() - 1]);
        }
    }

    /// Function to generate a point as per the Catmull-Rom spline algorithm
    fn catmull_rom_point(control_points: &[Vec2], t: f32) -> Vec2 {
        let [p0, p1, p2, p3] = [
            control_points[0],
            control_points[1],
            control_points[2],
            control_points[3],
        ];
        let t2 = t * t;
        let t3 = t2 * t;

        let x = 0.5
            * ((2.0 * p1.x)
                + (-p0.x + p2.x) * t
                + (2.0 * p0.x - 5.0 * p1.x + 4.0 * p2.x - p3.x) * t2
                + (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) * t3);

        let y = 0.5
            * ((2.0 * p1.y)
                + (-p0.y + p2.y) * t
                + (2.0 * p0.y - 5.0 * p1.y + 4.0 * p2.y - p3.y) * t2
                + (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) * t3);

        Vec2::new(x, y)
    }
}
