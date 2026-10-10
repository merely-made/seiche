// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! One tenant frame on a real adapter; skips when none is present.

use kiss3d::glamx::Vec3;
use kiss3d::glamx::glam::camera::rh::{proj::directx, view::look_at_mat4};

use crate::*;

const SIZE: u32 = 64;

fn host() -> Option<HostDevice> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
    let needs = DeviceNeeds::tenant();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: needs.required_features,
        required_limits: needs.limits.unwrap_or_default(),
        ..Default::default()
    }))
    .ok()?;
    Some(HostDevice {
        instance,
        adapter,
        device,
        queue,
    })
}

fn texture(
    host: &HostDevice,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
    view_formats: &[wgpu::TextureFormat],
) -> wgpu::Texture {
    host.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats,
    })
}

// A cube of side 2 centred at the origin, one palette entry.
fn cube() -> (PaletteMesh, Palette) {
    let corner = |i: usize| {
        [
            -1.0 + 2.0 * (i & 1) as f32,
            -1.0 + 2.0 * (i >> 1 & 1) as f32,
            -1.0 + 2.0 * (i >> 2) as f32,
        ]
    };
    let quads = [
        [0, 2, 3, 1],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 6, 7, 3],
        [0, 4, 6, 2],
        [1, 3, 7, 5],
    ];
    let positions = quads
        .iter()
        .flat_map(|q| [q[0], q[1], q[2], q[0], q[2], q[3]].map(corner))
        .collect::<Vec<_>>();
    let colors = vec![[0.8, 0.1, 0.1, 1.0]; positions.len()];
    PaletteMesh::from_colored(positions, &colors)
}

fn camera() -> Camera {
    let view = look_at_mat4(Vec3::new(6.0, 5.0, 4.0), Vec3::ZERO, Vec3::Y);
    Camera {
        view: view.to_cols_array_2d(),
        projection: directx::orthographic(-3.0, 3.0, -3.0, 3.0, 0.1, 50.0).to_cols_array_2d(),
        near: 0.1,
        far: 50.0,
    }
}

// Left half at the near plane, right half far: a depth-only draw under a scissor.
fn split_depth(host: &HostDevice) -> wgpu::Texture {
    let depth = texture(
        host,
        wgpu::TextureFormat::Depth32Float,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        &[],
    );
    let shader = host
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
                let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
                return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
            }"
                .into(),
            ),
        });
    let pipeline = host
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
    let view = depth.create_view(&Default::default());
    let mut encoder = host.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_scissor_rect(0, 0, SIZE / 2, SIZE);
        pass.draw(0..3, 0..1);
    }
    host.queue.submit([encoder.finish()]);
    depth
}

fn read(host: &HostDevice, target: &wgpu::Texture) -> Vec<u8> {
    let buffer = host.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (SIZE * SIZE * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = host.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SIZE * 4),
                rows_per_image: None,
            },
        },
        target.size(),
    );
    host.queue.submit([encoder.finish()]);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    host.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    buffer.slice(..).get_mapped_range().unwrap().to_vec()
}

fn frame(tenant: &mut Tenant, host: &HostDevice, target: &wgpu::Texture) -> Vec<u8> {
    let mut encoder = host.device.create_command_encoder(&Default::default());
    let report = tenant.encode(&mut encoder, target).expect("frame");
    assert_eq!(report.internal_submissions, 0, "the caller submits");
    host.queue.submit([encoder.finish()]);
    read(host, target)
}

#[test]
fn a_lit_body_joins_a_traced_depth() {
    let Some(host) = host() else {
        eprintln!("no adapter; skipped");
        return;
    };
    let srgb = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = texture(
        &host,
        srgb,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        &[srgb.remove_srgb_suffix()],
    );
    let mut tenant = Tenant::new(&host, srgb, [SIZE, SIZE]);
    let (mesh, palette) = cube();
    let id = tenant
        .add_mesh(&mesh, &palette, Pose::default())
        .expect("cube");
    assert!(tenant.set_camera(&camera()));
    assert!(tenant.set_lights(&LightBlock {
        sun: Some(Sun {
            direction: [-0.3, -1.0, -0.5],
            color: [1.0; 3],
            intensity: 2.0,
            casts_shadows: true
        }),
        points: vec![PointLight {
            position: [3.0, 3.0, 3.0],
            color: [1.0, 0.9, 0.7],
            intensity: 5.0,
            radius: 10.0,
            casts_shadows: false,
        }],
        ..Default::default()
    }));

    let px = |image: &[u8], x: u32| {
        let at = ((SIZE / 2 * SIZE + x) * 4) as usize;
        [image[at], image[at + 1], image[at + 2], image[at + 3]]
    };
    let whole = frame(&mut tenant, &host, &target);
    let (left, right) = (px(&whole, SIZE / 2 - 3), px(&whole, SIZE / 2 + 3));
    assert!(
        left[3] > 0 && right[3] > 0,
        "cube on both sides: {left:?} {right:?}"
    );
    assert!(
        right[0] > right[1] && right[0] > right[2],
        "palette red: {right:?}"
    );

    let depth = split_depth(&host);
    tenant.set_depth_prepass(Some(&depth));
    let joined = frame(&mut tenant, &host, &target);
    assert_eq!(px(&joined, SIZE / 2 - 3)[3], 0, "behind the traced surface");
    assert!(px(&joined, SIZE / 2 + 3)[3] > 0, "in front of it");

    let exports = tenant.exports();
    assert_eq!(
        exports.depth.texture().format(),
        wgpu::TextureFormat::Depth32Float
    );
    assert_eq!(
        exports.shadow_atlas.texture().format(),
        wgpu::TextureFormat::Depth32Float
    );
    assert!(exports.layouts_wgsl.contains("struct ShadowUniforms"));

    tenant.remove(id).expect("present");
    assert_eq!(tenant.remove(id), Err(BodyError::Unknown(id)));
    drop(tenant);
    // The host's device outlives the tenant.
    let _ = read(&host, &target);
}
