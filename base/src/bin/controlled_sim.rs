use base::game::{DRAG_POINTER_COLOR, DRAG_POINTER_RADIUS, Game};
use base::policy_net::PolicyNet;
use base::world::{World, pixel_to_world, world_to_pixel};
use macroquad::prelude::*;
use rapier2d::prelude::*;

#[macroquad::main("Controlled Sim")]
async fn main() {
    let mut world = World::new();
    let mut game = Game::new();

    let model_path = "./python/models/policy_net.onnx";
    let mut policy_net = PolicyNet::new(model_path).expect("Failed to create policy network");

    let obs_dim = 6;
    let input_shape = vec![1, obs_dim];

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

        let obs = vec![rocket_x, rocket_y, rocket_angle, vel_x, vel_y, ang_vel];

        let (thrust, gimbal_angle) = if world.is_dragging {
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

        #[cfg(feature = "logging")]
        {
            let (vel_x, vel_y, ang_vel) = world.get_rocket_dynamics();

            println!("Rocket State:");
            println!(
                "  Position - World: ({:.2}, {:.2}), Screen: ({:.2}, {:.2})",
                rocket_x, rocket_y, px_rocket_x, px_rocket_y
            );
            println!(
                "  Velocity: ({:.2}, {:.2}), Speed: {:.2}",
                vel_x,
                vel_y,
                (vel_x.powi(2) + vel_y.powi(2)).sqrt()
            );
            println!(
                "  Angle: {:.2}, Angular Velocity: {:.2}",
                rocket_angle, ang_vel
            );
            println!("  Observation: {:?}", obs);
            println!(
                "  Action: Thrust: {:.2}, Gimbal: {:.2}",
                thrust, gimbal_angle
            );
        }

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
