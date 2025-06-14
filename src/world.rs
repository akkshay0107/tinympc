//! A 2D physics simulation world controlling the rocket
//! 
//! This module contains the physics simulation world for the rocket landing scenario
//! Contains the physical rocket body and the ground, and has methods to apply forces
//! to the rocket body using the thrusters


use rapier2d::prelude::*;
use macroquad::prelude::*;

const PIXELS_PER_METER: f32 = 10.0;
const THRUST_FORCE: f32 = 0.1;
const GROUND_RESTITUTION: f32 = 0.5;
const ROCKET_RESTITUTION: f32 = 0.1;
const GROUND_POSITION: Vector<f32> = vector![50.0, -6.0];
const GROUND_SIZE: Vector<f32> = vector![50.0, 6.0];
const ROCKET_START_POSITION: Vector<f32> = vector![40.0, 40.0];

pub const ROCKET_WIDTH: f32 = 20.0;
pub const ROCKET_HEIGHT: f32 = 40.0;

pub struct World {
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    pub impulse_joint_set: ImpulseJointSet,
    pub multibody_joint_set: MultibodyJointSet,
    pub island_manager: IslandManager,
    pub broad_phase: DefaultBroadPhase,
    pub narrow_phase: NarrowPhase,
    pub ccd_solver: CCDSolver,
    pub query_pipeline: QueryPipeline,
    pub gravity: Vector<f32>,
    pub integration_parameters: IntegrationParameters,
    pub physics_pipeline: PhysicsPipeline,
    pub rocket_body_handle: RigidBodyHandle,
}

impl World {
    pub fn new() -> Self {
        let mut rigid_body_set = RigidBodySet::new();
        let mut collider_set = ColliderSet::new();

        let ground_handle = Self::create_ground(&mut rigid_body_set);
        Self::create_ground_collider(&mut collider_set, ground_handle, &mut rigid_body_set);

        let rocket_body_handle = Self::create_rocket(&mut rigid_body_set);
        Self::create_rocket_collider(&mut collider_set, rocket_body_handle, &mut rigid_body_set);

        Self {
            rigid_body_set,
            collider_set,
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            gravity: vector![0.0, -9.81],
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            rocket_body_handle,
        }
    }

    fn create_ground(rigid_body_set: &mut RigidBodySet) -> RigidBodyHandle {
        let ground = RigidBodyBuilder::fixed()
            .translation(GROUND_POSITION)
            .build();
        rigid_body_set.insert(ground)
    }

    fn create_ground_collider(
        collider_set: &mut ColliderSet,
        ground_handle: RigidBodyHandle,
        rigid_body_set: &mut RigidBodySet,
    ) {
        let collider = ColliderBuilder::cuboid(GROUND_SIZE.x, GROUND_SIZE.y)
            .restitution(GROUND_RESTITUTION)
            .build();
        collider_set.insert_with_parent(collider, ground_handle, rigid_body_set);
    }

    fn create_rocket(rigid_body_set: &mut RigidBodySet) -> RigidBodyHandle {
        let rocket_body = RigidBodyBuilder::dynamic()
            .translation(ROCKET_START_POSITION)
            .build();
        rigid_body_set.insert(rocket_body)
    }

    fn create_rocket_collider(
        collider_set: &mut ColliderSet,
        rocket_body_handle: RigidBodyHandle,
        rigid_body_set: &mut RigidBodySet,
    ) {
        let half_width = ROCKET_WIDTH / PIXELS_PER_METER / 2.0;
        let half_height = ROCKET_HEIGHT / PIXELS_PER_METER / 2.0;
        
        let body_collider = ColliderBuilder::cuboid(half_width, half_height)
            .restitution(ROCKET_RESTITUTION)
            .build();
        collider_set.insert_with_parent(body_collider, rocket_body_handle, rigid_body_set);
    }

    pub fn step(&mut self) {
        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            Some(&mut self.query_pipeline),
            &(),
            &(),
        );
    }

    pub fn apply_thruster_forces(&mut self, left_thruster: f32, right_thruster: f32) {
        if left_thruster == 0.0 && right_thruster == 0.0 {
            return;
        }

        let rocket_body = &mut self.rigid_body_set[self.rocket_body_handle];
        let body_angle = rocket_body.rotation().angle();
        
        let half_width = ROCKET_WIDTH / PIXELS_PER_METER / 2.0;
        let half_height = ROCKET_HEIGHT / PIXELS_PER_METER / 2.0;
        
        let left_thruster_offset = vector![-half_width, half_height];
        let right_thruster_offset = vector![half_width, half_height];
        let thrust_vector = vector![0.0, THRUST_FORCE];

        let mut total_torque = 0.0;

        if left_thruster != 0.0 {
            total_torque += Self::calculate_torque(left_thruster_offset, thrust_vector);
        }

        if right_thruster != 0.0 {
            total_torque += Self::calculate_torque(right_thruster_offset, thrust_vector);
        }

        let total_thrust = left_thruster + right_thruster;
        if total_thrust != 0.0 {
            let net_force = vector![
                thrust_vector.y * total_thrust * body_angle.sin(),
                thrust_vector.y * total_thrust * body_angle.cos()
            ];
            rocket_body.add_force(net_force, true);
        }

        if total_torque != 0.0 {
            rocket_body.add_torque(-total_torque, true);
        }
    }

    fn calculate_torque(offset: Vector<f32>, force: Vector<f32>) -> f32 {
        offset.x * force.y - offset.y * force.x
    }

    pub fn get_rocket_state(&self) -> (f32, f32, f32) {
        let rocket_body = &self.rigid_body_set[self.rocket_body_handle];
        (
            rocket_body.translation().x,
            rocket_body.translation().y,
            rocket_body.rotation().angle(),
        )
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

pub fn transform(x: f32, y: f32) -> (f32, f32) {
    let ground_y = screen_height() * 0.8;
    let screen_x = x * PIXELS_PER_METER;
    let screen_y = ground_y - y * PIXELS_PER_METER;
    (screen_x, screen_y)
}