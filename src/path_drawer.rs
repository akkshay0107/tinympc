//! Module for drawing the rockets path
//!
//! Stores the history of the mouse's path and draws a line following it
//! Only allows for one stroke (since the rocket's path should only be one
//! continuous line) and limits the length of the path (equivalent to fuel
//! on a rocket)

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

    // TODO:
    // - find better smoothing algorithm (that can handle multiple yvals for one xval)
    fn fix_path(&mut self) {
        // Prevent rocket path from clipping into the ground
        let y_limit = (screen_height() * 0.8) - (ROCKET_HEIGHT / 2.0);

        // Find first point that clips below the ground limit
        if let Some(clip_index) = self.points.iter().position(|point| point.y > y_limit) {
            // Remove all points from the clipping point onward
            self.points.truncate(clip_index);
        }

        // Approximate the path with cubic polynomial
        // coefficients found by optimizing least squares
        if self.points.len() > 3 {
            let mut smoothed_points = Vec::with_capacity(self.points.len());

            let n = self.points.len();
            let mut x = vec![0.0; n];
            let mut y = vec![0.0; n];

            // Normalize x to [0, 1] range
            for (i, point) in self.points.iter().enumerate() {
                x[i] = i as f32 / (n - 1) as f32;
                y[i] = point.y;
            }
            let degree = 3;
            let mut a = vec![0.0; degree + 1]; // stores coefficients of x^k

            let mut sum_x = vec![0.0; 2 * degree + 1];
            let mut sum_xy = vec![0.0; degree + 1];

            for i in 0..n {
                let xi = x[i];
                let yi = y[i];
                let mut x_pow = 1.0;
                for j in 0..sum_x.len() {
                    sum_x[j] += x_pow;
                    x_pow *= xi;
                }
                x_pow = 1.0;
                for j in 0..sum_xy.len() {
                    sum_xy[j] += x_pow * yi;
                    x_pow *= xi;
                }
            }
            // set up matrix equation
            let mut matrix = vec![vec![0.0; degree + 1]; degree + 1];
            for i in 0..=degree {
                for j in 0..=degree {
                    matrix[i][j] = sum_x[i + j];
                }
            }
            // gaussian elimination
            for i in 0..=degree {
                let mut max_row = i;
                for j in i + 1..=degree {
                    if matrix[j][i].abs() > matrix[max_row][i].abs() {
                        max_row = j;
                    }
                }
                matrix.swap(i, max_row);
                sum_xy.swap(i, max_row);
                let pivot = matrix[i][i];
                for j in i..=degree {
                    matrix[i][j] /= pivot;
                }
                sum_xy[i] /= pivot;
                for j in 0..=degree {
                    if j != i {
                        let factor = matrix[j][i];
                        for k in i..=degree {
                            matrix[j][k] -= factor * matrix[i][k];
                        }
                        sum_xy[j] -= factor * sum_xy[i];
                    }
                }
            }

            for i in 0..=degree {
                a[i] = sum_xy[i];
            }

            let num_points = self.points.len();
            for i in 0..num_points {
                let t = i as f32 / (num_points - 1) as f32;
                let mut y = 0.0;
                let mut t_pow = 1.0;
                for j in 0..=degree {
                    y += a[j] * t_pow;
                    t_pow *= t;
                }
                // Calculate x position based on original path
                let x = self.points[0].x + t * (self.points.last().unwrap().x - self.points[0].x);
                smoothed_points.push(Vec2::new(x, y));
            }
            self.points = smoothed_points;
        }
    }
}
