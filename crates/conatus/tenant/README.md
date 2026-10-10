# tenant

The lit body tenant: the renderer for bodies in the games wing. It draws
rigid palette meshes and glTF bodies on the host's wgpu device, into the
host's colour target and the host's command encoder, lit by the stack's
light block (wing ruling 472) and depth-joined with a tracer through a
depth pre-pass (L3 interleave).

`tenant` is a plain working name; the crate's name awaits a naming round.

Beneath it is kiss3d, reshaped in a mark-ik fork (branch `mark-ik/tenant`
over upstream v0.47.0, pinned by rev): an explicit context handle over a
host-owned device, a render-into-caller-targets entry with no window and
no submission of its own, a depth pre-pass, and exports of the shadow
atlas and light buffer (wing rulings 471, 606, 735). Games depend on this
crate, never on kiss3d; no kiss3d or glam type crosses its API, which
speaks plain arrays and wgpu.

| Module | Contents |
|---|---|
| `device` | `DeviceNeeds`, what the tenant asks of a shared device, and `HostDevice`, the handles it borrows. |
| `light` | `LightBlock`: sun, ambient and point lights, owned by the stack and read by tracer and rasteriser alike. |
| `body` | `Palette`, `PaletteMesh`, `Pose` and `BodyId`: rigid bodies whose colours are a palette texture addressed by UV, as glTF's base-colour texture is. |
| `camera` | `Camera`: the caller's view and projection, the same `clip_from_world` the tracer is given. |
| `frame` | `Tenant`, `FrameReport` and `Exports`: the frame entry and what it hands back. |

The plan and its progress live in mere's conatus engine plan
(`design_docs/mere_docs/implementation_strategy/2026-08-22_conatus_engine_plan.md`),
under the engine-component comparison of 2026-10-02.

MPL-2.0.
