// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The frame entry: bodies, lights and camera in; a frame recorded into the
//! caller's encoder and target out, with the shadow atlas, light buffer and
//! joined depth handed back.

use std::collections::BTreeMap;

use kiss3d::color::Color;
use kiss3d::context::Context;
use kiss3d::scene::{AnimationPlayer, SceneNode3d};
use kiss3d::window::{CanvasSetup, NumSamples, Window};

use crate::body::{BodyError, BodyId, Palette, PaletteMesh, Pose};
use crate::camera::{Camera, FixedCamera};
use crate::device::HostDevice;
use crate::light::{self, LightBlock};

/// The tonemap from the HDR film to the target.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tonemap {
    /// Clamp only.
    None,
    /// Khronos PBR Neutral.
    #[default]
    Neutral,
    AgX,
    Aces,
    Reinhard,
}

/// What a frame did beyond recording into the caller's encoder.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameReport {
    /// Queue submissions the tenant made itself. Zero for bodies, lights
    /// and the pre-pass; the caller submits.
    pub internal_submissions: u64,
}

/// Why a frame was not recorded.
#[derive(Clone, Debug, PartialEq)]
pub enum FrameError {
    NoCamera,
    /// The target's format is neither the tenant's nor its sRGB form.
    TargetFormat(wgpu::TextureFormat),
    TargetSamples(u32),
    /// The pre-pass depth and the target differ in size.
    PrepassSize {
        prepass: [u32; 2],
        target: [u32; 2],
    },
}

/// Handles to the last frame's lighting and depth, for the tracer.
#[derive(Clone, Debug)]
pub struct Exports {
    /// `Depth32Float` 2D array, one layer per shadow view.
    pub shadow_atlas: wgpu::TextureView,
    /// Comparison sampler for the atlas.
    pub shadow_sampler: wgpu::Sampler,
    /// `ShadowUniforms`: each view's light-space matrix and per-light slots.
    pub shadow_uniforms: wgpu::Buffer,
    /// The clustered lights, `array<LightData>`, when the device clusters.
    pub lights: Option<wgpu::Buffer>,
    pub light_count: u32,
    /// The frame's depth: the pre-pass joined with the bodies.
    pub depth: wgpu::TextureView,
    /// WGSL declarations of `LightData`, `LightShadow` and `ShadowUniforms`.
    pub layouts_wgsl: &'static str,
}

struct Body {
    node: SceneNode3d,
    player: Option<AnimationPlayer>,
}

/// The lit body tenant over one host device.
pub struct Tenant {
    bodies: BTreeMap<BodyId, Body>,
    next: u64,
    scene: SceneNode3d,
    lights: SceneNode3d,
    camera: Option<FixedCamera>,
    prepass_size: Option<[u32; 2]>,
    window: Window,
    context: Context,
}

impl Tenant {
    /// A tenant drawing into targets of `target_format` (a linear format
    /// or its sRGB form; an sRGB target lists the linear format in its
    /// `view_formats`).
    pub fn new(host: &HostDevice, target_format: wgpu::TextureFormat, size: [u32; 2]) -> Self {
        let context = Context::from_host(
            host.instance.clone(),
            host.device.clone(),
            host.queue.clone(),
            host.adapter.clone(),
            target_format.remove_srgb_suffix(),
        );
        let _scope = context.enter();
        let setup = CanvasSetup {
            samples: NumSamples::One,
            ..Default::default()
        };
        let mut window = Window::new_on_context(size[0], size[1], setup);
        window.set_background_color(Color::new(0.0, 0.0, 0.0, 0.0));
        let mut scene = SceneNode3d::empty();
        let lights = scene.add_group();
        Self {
            bodies: BTreeMap::new(),
            next: 0,
            scene,
            lights,
            camera: None,
            prepass_size: None,
            window,
            context,
        }
    }

    /// Adds a rigid body coloured from `palette`.
    pub fn add_mesh(
        &mut self,
        mesh: &PaletteMesh,
        palette: &Palette,
        pose: Pose,
    ) -> Result<BodyId, BodyError> {
        mesh.validate(palette)?;
        let _scope = self.context.enter();
        let node = self.mesh_node(mesh, palette);
        self.insert(node, None, pose)
    }

    /// Replaces a body's geometry and palette, keeping its id and pose.
    pub fn set_mesh(
        &mut self,
        id: BodyId,
        mesh: &PaletteMesh,
        palette: &Palette,
    ) -> Result<(), BodyError> {
        mesh.validate(palette)?;
        let old = self.bodies.get(&id).ok_or(BodyError::Unknown(id))?;
        let (pose, scale) = (old.node.local_transformation(), old.node.local_scale());
        let _scope = self.context.enter();
        let mut node = self.mesh_node(mesh, palette);
        node.set_pose(pose);
        node.set_local_scale(scale.x, scale.y, scale.z);
        let body = self.bodies.get_mut(&id).expect("checked above");
        body.node.remove();
        body.node = node;
        Ok(())
    }

    /// Adds a glTF or GLB body, its animations stopped.
    pub fn add_gltf(&mut self, bytes: &[u8], pose: Pose) -> Result<BodyId, BodyError> {
        let _scope = self.context.enter();
        let model = kiss3d::loader::gltf::load_from_slice(bytes)
            .map_err(|e| BodyError::Gltf(e.to_string()))?;
        let mut root = self.scene.add_group();
        root.add_child(model.root);
        self.insert(root, Some(model.player), pose)
    }

    /// Plays a glTF body's animation clip; `false` when it has none so named.
    pub fn play(&mut self, id: BodyId, clip: &str) -> Result<bool, BodyError> {
        let body = self.bodies.get_mut(&id).ok_or(BodyError::Unknown(id))?;
        Ok(body.player.as_mut().is_some_and(|p| p.play(clip)))
    }

    /// Advances every playing animation by `seconds`.
    pub fn advance(&mut self, seconds: f32) {
        let _scope = self.context.enter();
        for player in self.bodies.values_mut().filter_map(|b| b.player.as_mut()) {
            player.update(seconds);
        }
    }

    pub fn set_pose(&mut self, id: BodyId, pose: Pose) -> Result<(), BodyError> {
        if !pose.is_finite() {
            return Err(BodyError::NotFinite);
        }
        let body = self.bodies.get_mut(&id).ok_or(BodyError::Unknown(id))?;
        let (pose, scale) = pose.parts();
        body.node.set_pose(pose);
        body.node.set_local_scale(scale.x, scale.y, scale.z);
        Ok(())
    }

    pub fn remove(&mut self, id: BodyId) -> Result<(), BodyError> {
        let mut body = self.bodies.remove(&id).ok_or(BodyError::Unknown(id))?;
        body.node.remove();
        Ok(())
    }

    /// Lights the next frames with `block`; `false`, changing nothing, when
    /// the block is not valid.
    pub fn set_lights(&mut self, block: &LightBlock) -> bool {
        if !block.is_valid() {
            return false;
        }
        let _scope = self.context.enter();
        light::place(block, &mut self.lights);
        let (intensity, color) = light::ambient(block);
        self.window.set_ambient(intensity);
        self.window.set_ambient_color(color);
        true
    }

    /// The colour behind the bodies; transparent by default, so the frame
    /// composites over whatever the pre-pass's tracer drew.
    pub fn set_background(&mut self, rgba: [f32; 4]) {
        self.window
            .set_background_color(Color::new(rgba[0], rgba[1], rgba[2], rgba[3]));
    }

    pub fn set_tonemap(&mut self, tonemap: Tonemap) {
        use kiss3d::post_processing::Tonemap as K;
        self.window.set_tonemap(match tonemap {
            Tonemap::None => K::None,
            Tonemap::Neutral => K::Neutral,
            Tonemap::AgX => K::AgX,
            Tonemap::Aces => K::Aces,
            Tonemap::Reinhard => K::Reinhard,
        });
    }

    pub fn set_shadows(&mut self, enabled: bool) {
        self.window.set_shadows_enabled(enabled);
    }

    /// `false`, changing nothing, when the camera is not valid.
    pub fn set_camera(&mut self, camera: &Camera) -> bool {
        if !camera.is_valid() {
            return false;
        }
        self.camera = Some(FixedCamera::new(camera));
        true
    }

    /// Starts each frame from `depth`, a tracer's surfaces in the camera's
    /// clip space: `Depth32Float`, single-sampled, `TEXTURE_BINDING`, the
    /// target's size. `None` clears to the far plane again.
    pub fn set_depth_prepass(&mut self, depth: Option<&wgpu::Texture>) {
        let _scope = self.context.enter();
        self.prepass_size = depth.map(|d| [d.width(), d.height()]);
        self.window.set_depth_prepass(depth.cloned());
    }

    /// Records one frame into `encoder`, drawing into `target`. The caller
    /// submits the encoder before recording another tenant frame.
    pub fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Texture,
    ) -> Result<FrameReport, FrameError> {
        let size = [target.width(), target.height()];
        if target.format().remove_srgb_suffix() != self.context.surface_format {
            return Err(FrameError::TargetFormat(target.format()));
        }
        if target.sample_count() != 1 {
            return Err(FrameError::TargetSamples(target.sample_count()));
        }
        if let Some(prepass) = self.prepass_size.filter(|p| *p != size) {
            return Err(FrameError::PrepassSize {
                prepass,
                target: size,
            });
        }
        let camera = self.camera.as_mut().ok_or(FrameError::NoCamera)?;
        let _scope = self.context.enter();
        let frame =
            self.window
                .render_into(encoder, target, Some(&mut self.scene), camera, &mut []);
        Ok(FrameReport {
            internal_submissions: frame.internal_submissions,
        })
    }

    /// The last frame's shadow atlas, light buffer and depth.
    pub fn exports(&self) -> Exports {
        let lights = self.window.light_exports();
        Exports {
            shadow_atlas: lights.shadow.atlas,
            shadow_sampler: lights.shadow.compare_sampler,
            shadow_uniforms: lights.shadow.uniform,
            lights: lights.clustered_lights,
            light_count: lights.clustered_light_count,
            depth: self.window.depth_view().clone(),
            layouts_wgsl: kiss3d::window::LIGHT_EXPORTS_WGSL,
        }
    }

    fn mesh_node(&mut self, mesh: &PaletteMesh, palette: &Palette) -> SceneNode3d {
        let mut node = self
            .scene
            .add_mesh(mesh.gpu(palette), kiss3d::glamx::Vec3::ONE);
        node.set_texture(palette.texture());
        node
    }

    fn insert(
        &mut self,
        mut node: SceneNode3d,
        player: Option<AnimationPlayer>,
        pose: Pose,
    ) -> Result<BodyId, BodyError> {
        if !pose.is_finite() {
            node.remove();
            return Err(BodyError::NotFinite);
        }
        let (transform, scale) = pose.parts();
        node.set_pose(transform);
        node.set_local_scale(scale.x, scale.y, scale.z);
        let id = BodyId(self.next);
        self.next += 1;
        self.bodies.insert(id, Body { node, player });
        Ok(id)
    }
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for FrameError {}

#[cfg(test)]
#[path = "frame_tests.rs"]
mod tests;
