// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The caller's camera, so the tracer and the rasteriser share one
//! `clip_from_world = projection * view`.

use kiss3d::camera::Camera3d;
use kiss3d::event::WindowEvent;
use kiss3d::glamx::{Mat4, Pose3, Vec3};
use kiss3d::window::Canvas;

/// A view and a projection, column-major as wgpu and glam store them.
///
/// The projection maps depth to `0..1` with standard z (near 0, far 1),
/// matching a depth cleared to 1.0 and a `Less` test; `near` and `far` are
/// its planes, which shadow cascades and light clustering read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    /// World to view; rigid (rotation and translation only).
    pub view: [[f32; 4]; 4],
    pub projection: [[f32; 4]; 4],
    pub near: f32,
    pub far: f32,
}

impl Camera {
    /// `projection * view`, the matrix a tracer's depth must use.
    pub fn clip_from_world(&self) -> [[f32; 4]; 4] {
        (Mat4::from_cols_array_2d(&self.projection) * Mat4::from_cols_array_2d(&self.view))
            .to_cols_array_2d()
    }

    /// Whether the view is rigid, the matrices finite and the planes ordered.
    pub fn is_valid(&self) -> bool {
        let view = Mat4::from_cols_array_2d(&self.view);
        let projection = Mat4::from_cols_array_2d(&self.projection);
        let basis = [
            view.x_axis.truncate(),
            view.y_axis.truncate(),
            view.z_axis.truncate(),
        ];
        let orthonormal = basis.iter().all(|a| (a.length() - 1.0).abs() < 1e-4)
            && basis[0].dot(basis[1]).abs() < 1e-4
            && basis[1].dot(basis[2]).abs() < 1e-4
            && basis[0].dot(basis[2]).abs() < 1e-4;
        view.is_finite()
            && projection.is_finite()
            && orthonormal
            && view.row(3).abs_diff_eq(kiss3d::glamx::Vec4::W, 1e-6)
            && self.near.is_finite()
            && self.far.is_finite()
            && self.near < self.far
    }
}

/// [`Camera`] in kiss3d's shape.
pub(crate) struct FixedCamera {
    view: Pose3,
    projection: Mat4,
    clip_from_world: Mat4,
    world_from_clip: Mat4,
    eye: Vec3,
    planes: (f32, f32),
}

impl FixedCamera {
    pub(crate) fn new(camera: &Camera) -> Self {
        let view = Mat4::from_cols_array_2d(&camera.view);
        let projection = Mat4::from_cols_array_2d(&camera.projection);
        let clip_from_world = projection * view;
        Self {
            view: Pose3::from_mat4(view),
            projection,
            clip_from_world,
            world_from_clip: clip_from_world.inverse(),
            eye: view.inverse().w_axis.truncate(),
            planes: (camera.near, camera.far),
        }
    }
}

impl Camera3d for FixedCamera {
    fn handle_event(&mut self, _: &Canvas, _: &WindowEvent) {}

    fn eye(&self) -> Vec3 {
        self.eye
    }

    fn view_transform(&self) -> Pose3 {
        self.view
    }

    fn transformation(&self) -> Mat4 {
        self.clip_from_world
    }

    fn inverse_transformation(&self) -> Mat4 {
        self.world_from_clip
    }

    fn clip_planes(&self) -> (f32, f32) {
        self.planes
    }

    fn update(&mut self, _: &Canvas) {}

    fn view_transform_pair(&self, _pass: usize) -> (Pose3, Mat4) {
        (self.view, self.projection)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiss3d::glamx::glam::camera::rh::{proj::directx, view::look_at_mat4};

    fn camera() -> Camera {
        let view = look_at_mat4(Vec3::new(4.0, 5.0, 6.0), Vec3::ZERO, Vec3::Y);
        let projection = directx::orthographic(-8.0, 8.0, -4.5, 4.5, 0.1, 100.0);
        Camera {
            view: view.to_cols_array_2d(),
            projection: projection.to_cols_array_2d(),
            near: 0.1,
            far: 100.0,
        }
    }

    #[test]
    fn fixed_camera_agrees_with_the_caller() {
        let camera = camera();
        assert!(camera.is_valid());
        let fixed = FixedCamera::new(&camera);
        let clip = Mat4::from_cols_array_2d(&camera.clip_from_world());
        assert!(fixed.transformation().abs_diff_eq(clip, 1e-5));
        assert!(fixed.eye().abs_diff_eq(Vec3::new(4.0, 5.0, 6.0), 1e-4));
        assert!(
            fixed
                .view
                .to_mat4()
                .abs_diff_eq(Mat4::from_cols_array_2d(&camera.view), 1e-5)
        );
    }

    #[test]
    fn a_scaled_view_is_refused() {
        let mut camera = camera();
        camera.view[0][0] *= 2.0;
        assert!(!camera.is_valid());
    }
}
