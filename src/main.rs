use macroquad::prelude::*;
use rapier2d::prelude::*;
use tinympc::rocket::*;

const GROUND_CRUST_COLOR: Color = Color::new(0.4, 0.2, 0.1, 1.0); // Darker than interior
const GROUND_INTERIOR_COLOR: Color = Color::new(0.6, 0.4, 0.2, 1.0); // Tan color
const GROUND_CRUST_THICKNESS: f32 = 10.0;
const PARALLAX_SCROLL_SPEED: f32 = 0.08;
const PIXELS_PER_METER: f32 = 10.0;

struct Game {
    stars: Vec<(f32, f32, f32)>,
    rocket: Rocket,
}

impl Game {
    fn new() -> Self {
        let mut stars = Vec::new();
        for _ in 0..50 {
            let x = rand::gen_range(0.0, screen_width());
            let y = rand::gen_range(0.0, screen_height() * 0.5); // Stars only in upper 50% of screen
            let radius = rand::gen_range(1.0, 2.0);
            stars.push((x, y, radius));
        }

        let rocket_x = screen_width() / 2.0;
        let ground_y = screen_height() * 0.8;
        let rocket = Rocket::new(rocket_x, ground_y);
        Self { stars, rocket }
    }

    fn update(&mut self) {
        // Parallax effect implementation
        for star in self.stars.iter_mut() {
            star.0 -= PARALLAX_SCROLL_SPEED;
            if star.0 < 0.0 {
                star.0 = screen_width();
                star.1 = rand::gen_range(0.0, screen_height() * 0.5);
            }
        }
    }

    fn draw_space_atmos(&self) {
        let screen_w = screen_width();
        let screen_h = screen_height();
        let num_steps = 50;
        let strip_height = screen_h / num_steps as f32;

        let top_color = Color::new(0.05, 0.02, 0.15, 1.0); // Deep purple
        let mid_color = Color::new(0.02, 0.05, 0.25, 1.0); // Dark blue
        let bottom_color = Color::new(0.01, 0.02, 0.08, 1.0); // Very dark blue

        for i in 0..num_steps {
            let y = i as f32 * strip_height;
            let t = i as f32 / (num_steps - 1) as f32;

            // Linear interpolation of colors to get a gradient
            // Fades from top to mid then mid to bottom in a 60:40 ratio
            let color = if t < 0.6 {
                // Top to mid transition
                let local_t = t / 0.6;
                Color::new(
                    top_color.r + (mid_color.r - top_color.r) * local_t,
                    top_color.g + (mid_color.g - top_color.g) * local_t,
                    top_color.b + (mid_color.b - top_color.b) * local_t,
                    1.0,
                )
            } else {
                // Mid to bottom transition
                let local_t = (t - 0.6) / 0.4;
                Color::new(
                    mid_color.r + (bottom_color.r - mid_color.r) * local_t,
                    mid_color.g + (bottom_color.g - mid_color.g) * local_t,
                    mid_color.b + (bottom_color.b - mid_color.b) * local_t,
                    1.0,
                )
            };

            draw_rectangle(0.0, y, screen_w, strip_height + 1.0, color);
        }
    }

    /// Function to transform coordinates from the physics engine (real life)
    /// to in game coordinates, in order to display it correctly
    fn transform(x: f32, y: f32) -> (f32, f32) {
        // ground ranges from (0.8 to 1) * screen height
        // = 480 - 600 when screen height 600 px
        // 480 pixels = 0 m & +10 px = -1 m
        let ground_y = screen_height() * 0.8;
        let xx = x * PIXELS_PER_METER;
        let yy = ground_y - y * PIXELS_PER_METER;
        (xx, yy)
    }

    fn draw_ground(&self) {
        let screen_w = screen_width();
        let screen_h = screen_height();
        let ground_y = screen_h * 0.8;

        draw_rectangle(
            0.0,
            ground_y,
            screen_w,
            GROUND_CRUST_THICKNESS,
            GROUND_CRUST_COLOR,
        );
        let base_color = GROUND_INTERIOR_COLOR;

        // Draw in texture in the interior of the ground
        for y in ((ground_y + GROUND_CRUST_THICKNESS) as i32..screen_h as i32).step_by(4) {
            for x in (0..screen_w as i32).step_by(4) {
                // Simple noise function based on position
                // Combines three different sine waves and a stable pattern
                // to create a more complex noise pattern
                let n1 = (x as f32 * 0.1).sin();
                let n2 = (y as f32 * 0.07).cos();
                let n3 = ((x as f32 * 0.05 + y as f32 * 0.05).sin() * 2.0).sin();
                let noise = (n1 * 0.3 + n2 * 0.2 + n3 * 0.5).abs() * 0.03;

                let stable = ((x as i32 + y * 12345 as i32) % 100) as f32 / 600.0;
                let noise = (noise + stable).min(0.06);

                let color = Color::new(
                    (base_color.r - noise * 1.0).max(0.0),
                    (base_color.g - noise * 0.7).max(0.0),
                    (base_color.b - noise * 0.4).max(0.0),
                    base_color.a,
                );

                draw_rectangle(x as f32, y as f32, 4.0, 4.0, color);
            }
        }
    }

    fn draw(&self) {
        self.draw_space_atmos();

        // Draw stars
        for (x, y, radius) in &self.stars {
            draw_circle(*x, *y, *radius, WHITE);
        }

        self.draw_ground();
    }
}

#[macroquad::main("Game")]
async fn main() {
    // Initialize physics structures
    let mut rigid_body_set = RigidBodySet::new();
    let mut collider_set = ColliderSet::new();
    let mut impulse_joint_set = ImpulseJointSet::new();

    // Rigid body for the ground
    let ground = RigidBodyBuilder::fixed()
        .translation(vector![50.0, -6.0])
        .build();
    let ground_handle = rigid_body_set.insert(ground);
    let collider = ColliderBuilder::cuboid(50.0, 6.0).restitution(0.5).build();
    collider_set.insert_with_parent(collider, ground_handle, &mut rigid_body_set);

    // Rocket body as a single cuboid
    let body_width = ROCKET_WIDTH / PIXELS_PER_METER;
    let body_height = ROCKET_HEIGHT / PIXELS_PER_METER;

    let rocket_body = RigidBodyBuilder::dynamic()
        .translation(vector![40.0, 40.0])
        .build();
    let rocket_body_handle = rigid_body_set.insert(rocket_body);

    let body_collider = ColliderBuilder::cuboid(body_width / 2.0, body_height / 2.0)
        .restitution(0.1)
        .build();
    collider_set.insert_with_parent(body_collider, rocket_body_handle, &mut rigid_body_set);

    // Create other structures necessary for the simulation.
    let gravity = vector![0.0, -9.81];
    let integration_parameters = IntegrationParameters::default();
    let mut physics_pipeline = PhysicsPipeline::new();
    let mut island_manager = IslandManager::new();
    let mut broad_phase = DefaultBroadPhase::new();
    let mut narrow_phase = NarrowPhase::new();
    let mut multibody_joint_set = MultibodyJointSet::new();
    let mut ccd_solver = CCDSolver::new();
    let mut query_pipeline = QueryPipeline::new();
    let physics_hooks = ();
    let event_handler = ();

    let mut game = Game::new();
    loop {
        game.update();
        let left_thruster = if is_key_down(KeyCode::Left) { 1.0 } else { 0.0 };
        let right_thruster = if is_key_down(KeyCode::Right) {
            1.0
        } else {
            0.0
        };

        let rocket_body = &mut rigid_body_set[rocket_body_handle];
        let body_angle = rocket_body.rotation().angle();
        const THRUST_FORCE: f32 = 0.1;

        let left_thruster_offset = vector![
            -ROCKET_WIDTH / PIXELS_PER_METER / 2.0,
            ROCKET_HEIGHT / PIXELS_PER_METER / 2.0
        ];
        let right_thruster_offset = vector![
            ROCKET_WIDTH / PIXELS_PER_METER / 2.0,
            ROCKET_HEIGHT / PIXELS_PER_METER / 2.0
        ];
        let mut torque = 0.0;
        let force = vector![0.0, THRUST_FORCE];

        if left_thruster != 0.0 {
            let rxf = left_thruster_offset.x * force.y - left_thruster_offset.y * force.x;
            torque += rxf;
        }

        if right_thruster != 0.0 {
            let rxf = right_thruster_offset.x * force.y - right_thruster_offset.y * force.x;
            torque += rxf;
        }

        if left_thruster + right_thruster != 0.0 {
            // Net F = R_{-body_angle} dot F_body
            // Minus since opposite angle conventions
            let net_force = vector![
                force.y * (left_thruster + right_thruster) * body_angle.sin(),
                force.y * (left_thruster + right_thruster) * body_angle.cos()
            ];
            rocket_body.add_force(net_force, true);
        }

        if torque != 0.0 {
            // Minus added since clockwise taken as positive angle
            // which is opposite of physical convention
            rocket_body.add_torque(-torque, true);
        }

        println!("sin: {} / cos: {}", body_angle.sin(), body_angle.cos());

        // Pass state of rocket body to the rocket sprite manager
        let rocket_body = rigid_body_set.get(rocket_body_handle).unwrap();
        let (rocket_x, rocket_y) =
            Game::transform(rocket_body.translation().x, rocket_body.translation().y);
        let rocket_angle = rocket_body.rotation().angle();
        game.rocket.set_state(
            rocket_x,
            rocket_y,
            rocket_angle,
            (left_thruster, right_thruster),
        );

        physics_pipeline.step(
            &gravity,
            &integration_parameters,
            &mut island_manager,
            &mut broad_phase,
            &mut narrow_phase,
            &mut rigid_body_set,
            &mut collider_set,
            &mut impulse_joint_set,
            &mut multibody_joint_set,
            &mut ccd_solver,
            Some(&mut query_pipeline),
            &physics_hooks,
            &event_handler,
        );

        game.draw();
        game.rocket.draw();
        next_frame().await;
    }
}
