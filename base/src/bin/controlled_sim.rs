use base::constants::{_MIN_POS_Y, MAX_ANGLE_DEFLECTION, MAX_POS_X, MAX_POS_Y};
use base::game::{DRAG_POINTER_COLOR, DRAG_POINTER_RADIUS, Game};
use base::policy_net::PolicyNet;
use base::world::{World, pixel_to_world, world_to_pixel};

use macroquad::prelude::*;
use rapier2d::na::Isometry2;
use rapier2d::prelude::*;

use std::f32::consts::PI;

#[macroquad::main("Controlled Sim")]
async fn main() {
    let mut world = World::new();
    let mut game = Game::new();
    let mut policy_net: PolicyNet;

    let model_bytes = include_bytes!("../../../python/models/policy_net.onnx");
    policy_net = PolicyNet::new(model_bytes).unwrap();

    rand::srand(macroquad::miniquad::date::now() as u64); // get different starts on each run
    let start_x: f32 = rand::gen_range(20.0, MAX_POS_X - 20.0);
    let start_angle: f32 = rand::gen_range(-MAX_ANGLE_DEFLECTION, MAX_ANGLE_DEFLECTION);

    let rocket = world
        .rigid_body_set
        .get_mut(world.rocket_body_handle)
        .unwrap();
    rocket.set_position(Isometry2::new(vector![start_x, 40.0], start_angle), true);

    let obs_dim = 6;
    let input_shape = vec![1, obs_dim];

    #[cfg(feature = "logging")]
    let mut step_count = 0;

    loop {
        game.update();

        // Mouse drag handling
        let (mouse_x, mouse_y) = mouse_position();
        let (world_mouse_x, world_mouse_y) = pixel_to_world(mouse_x, mouse_y);
        let mouse_world_pos = vector![world_mouse_x, world_mouse_y];

        if is_mouse_button_pressed(MouseButton::Left) {
            world.start_drag(mouse_world_pos);
        } else if is_mouse_button_down(MouseButton::Left) && world.is_dragging {
            world.update_drag(mouse_world_pos);
        } else if is_mouse_button_released(MouseButton::Left) && world.is_dragging {
            world.end_drag();
        }

        // Get rocket state to form observation for policy net
        let (rocket_x, rocket_y, rocket_angle) = world.get_rocket_state();
        let (vel_x, vel_y, ang_vel) = world.get_rocket_dynamics();

        let obs = _normalize([rocket_x, rocket_y, rocket_angle, vel_x, vel_y, ang_vel]);

        #[cfg(feature = "logging")]
        {
            use std::f32::consts::SQRT_2;

            println!("Observation state: {:?}", obs.clone());

            // from gym
            let [nx, ny, ntheta, nvx, nvy, nomega] = obs[..] else {
                panic!("Failed unpacking normalized observation");
            };

            // center is (max_x/2, 0) => potential should be min there
            let ndist = (nx.powi(2) + ny.powi(2)).sqrt(); // [0, sqrt2]
            let dist_score = 1.0 - (ndist / SQRT_2);

            // slow velocity preferred
            let speed = (nvx.powi(2) + nvy.powi(2)).sqrt();
            let speed_score = 1.0 - (speed / SQRT_2).min(1.0);

            // reward being upright and not spinning too much
            let angle_norm = (ntheta.powi(2) + nomega.powi(2).min(1.0)).sqrt(); // [0, sqrt2]
            let angle_score = 1.0 - (angle_norm / SQRT_2);

            let potential = 100.0 * (0.5 * dist_score + 0.15 * speed_score + 0.35 * angle_score);
            println!("Potential : {:?}", potential);

            step_count += 1;
            println!("Step Count : {:?}", step_count);
        }

        let (thrust, gimbal_angle) = if world.is_dragging || rocket_y <= _MIN_POS_Y {
            (0.0, 0.0)
        } else {
            let action = policy_net
                .get_action(obs.clone(), input_shape.clone())
                .unwrap();
            (action[0], action[1])
        };

        world.apply_thruster_forces(thrust, gimbal_angle);
        world.step();

        let (rocket_x, rocket_y, rocket_angle) = world.get_rocket_state();
        let (px_rocket_x, px_rocket_y) = world_to_pixel(rocket_x, rocket_y);

        game.rocket.set_state(
            px_rocket_x,
            px_rocket_y,
            rocket_angle,
            (thrust, gimbal_angle),
        );

        game.draw();
        game.rocket.draw();

        // Draw drag indicator when dragging
        if world.is_dragging {
            draw_circle(mouse_x, mouse_y, DRAG_POINTER_RADIUS, DRAG_POINTER_COLOR);
        }

        next_frame().await;
    }
}

fn _normalize(obs: [f32; 6]) -> Vec<f32> {
    let nx = (2.0 * obs[0] - MAX_POS_X) / MAX_POS_X;
    let ny = (obs[1] - _MIN_POS_Y) / (MAX_POS_Y - _MIN_POS_Y);
    let ntheta = obs[2] / PI;

    let nvx = (2.0 * obs[3]) / MAX_POS_X;
    let nvy = obs[4] / (MAX_POS_Y - _MIN_POS_Y);
    let nomega = obs[5] / PI;

    vec![nx, ny, ntheta, nvx, nvy, nomega]
}
