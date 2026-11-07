# Seiche

Seiche is the family's spatial runtime. It holds bodies and their collisions,
voxel chunks and the sparse-brick presentation ABI, fields and their algebra,
the force integration that turns fields and laws into motion, and the lit
body tenant that draws rigid and glTF bodies on a host's device. A host feeds
it node keys, positions and a device; Seiche integrates. Nothing here depends
on a graph kernel, a canvas or a renderer of its own.

## Crates

| Path | Package | What it is |
|---|---|---|
| `crates/conatus/numen` | `numen` | Field truth and evaluation: scalar and vector fields, couplings, edge paths, the registry and projection; opt-in Rhai authoring and Burn lowering |
| `crates/conatus/seiche` | `seiche` | Force integration on Rapier 2D: built-in layout forces, tensorized repulsion, field couplings, scenes, fluid, the dynamics spec; optional actor thread and lagged GPU evaluator |
| `crates/conatus/seiche-model` | `seiche-model` | The rapier-free model of seiche (node keys, colliders, scene specs, roles, the field clock, the layout view); `seiche` re-exports it under its old paths |
| `crates/conatus/conatus` | `conatus` | Host-neutral 3D body, collision, query and fixed-step runtime on Rapier 3D, and the opt-in device-resident lane on CubeCL |
| `crates/conatus/nisus` | `nisus` | Generic revisioned voxel chunk and edit mechanics |
| `crates/conatus/modulus` | `modulus` | The sparse-brick presentation ABI and WGSL voxel traversal |
| `crates/conatus/tenant` | `tenant` | The lit body tenant over a reshaped kiss3d fork (by rev), drawn into the host's targets |

`crates/conatus/README.md` gives the dependency direction. Every package keeps
the name it had in Mere. `numen` and `seiche` keep their declared dynamics
rules and their determinism bits: Rapier's `enhanced-determinism` in `seiche`
and `conatus`, and the state witness the binding replay records, which
`seiche` takes as a dev dependency from Eidetic.

## Where it came from

The seven crates and `crates/conatus/README.md` were split out of
[Mere](https://github.com/merely-made/mere) at `4fa1686da` with their history
(`git filter-repo` on their paths, which stay as they were in Mere), under
Mere's engine split plan
(`design_docs/mere_docs/implementation_strategy/2026-11-07_engine_split_plan.md`,
phase P2): D5 moved the spatial runtime and only the runtime (the projection
crates that consume the workspace kernel stay with the host), D6 named the
repository after the published crate, and D12 moved `state-witness` and
`mere-framing` to Eidetic so Seiche has no edge back into Mere. History from
before the crates reached Mere (2026-07-23) is in their former repositories.

The spatial plans (the conatus engine plan, the dynamics grammar plan, the
physics catalog plan) stay in Mere's `design_docs/`; see
`design_docs/DOC_README.md`.

## Consumers

- **Isocosm.** Eponym and Mesocosm take `conatus`, `modulus`, `nisus` and
  `tenant`, Isometer's lens takes `conatus` and `modulus`, and the
  parry-ground probe takes `conatus`. Each names
  `https://github.com/merely-made/seiche.git` with `branch = "main"` in place
  of `mere.git`, and its lock records the revision.
- **Mere's canvas.** `pictograph` takes `numen` and `seiche` (and `seiche/gpu`
  under its `gpu` feature), and `graph-kernel` takes `numen`. Mere's root rows
  for these crates name `seiche.git` with `branch = "main"`, and the
  physics-in-dev opt-level table stays in Mere's root, since a profile applies
  only from the building workspace.
- **mer3ly.** `repo-graph` takes `seiche` and moves its row to `seiche.git` in
  the same relock that moves its `mere.git` revision past Mere's repin, so the
  graph holds one `seiche`.
- Cleromancy, Knot and Murm's signalman-desktop take `seiche` the same way.

## Building

```sh
cargo check --workspace --all-targets
cargo test --workspace
```

The toolchain is pinned in `rust-toolchain.toml`. Licensing is MPL-2.0; see
`LICENSE` and `LICENSES.md`.
