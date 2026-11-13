# Conatus Engine Plan

> Moved from Mere (`design_docs/mere_docs/implementation_strategy/2026-10-28_conatus_engine_plan.md`) on 2026-11-07 at revision `4fa1686da` under engine-split D25. Paths under `archive_docs/` and the other plans it names are Mere's `design_docs/`.

**Status:** 2026-10-28: cut from `2026-08-22_conatus_engine_plan.md`, archived in `archive_docs/2026-10-28/`. Done: the spatial runtime foundation (bodies, colliders, voxel colliders, character movement, fixed-step clock, ordered phases, command buffer, frame changes), the private Rapier boundary, `BodyWorld` queries and `refresh_queries` (ruled to merge and push 2026-10-09), `nisus` CPU voxel mechanics, `conatus::resident` (Quint folded in) with its first position-plane slice, the `modulus` brick ABI with `AtlasLimits` and exact traversal, and the kiss3d tenant crate on main. Open: the T2 voxel world store, body bindings, the rest of the runtime systems, resident views, the spatial frame, field adapters, scripting, the consumers' repins, the tenant's tails, the Nexus watch and the fork upkeep.

**For:** the shared spatial runtime that Mesocosm, Paredros, Isometry and Mere projections consume through product-owned runtime profiles. Mark's ruling of 2026-08-23 set its scope: product profiles conduct, Conatus advances spatial state, reusable mechanics may move early under settled ownership, and cross-product contracts stay provisional until a second game proves them. The consolidation map of 2026-08-28 placed it among realms and the two engines.
Not here: product profiles, the composition acceptance ledger and gates C3 and C4 (projection grammar plan); the recipe's dynamics slot and the F199 pause that holds T2 (dynamics grammar plan); seiche's physics laws and GPU tier (physics catalog plan); Burn and CubeCL upkeep (distillery models plan); the games-wing decisions, whose canonical home is the Isometry and Mesocosm wing design record, and the Isocosm lane.

Receipts from this plan's runs go to `archive_docs/<date>/`, never into this
plan (DOC_POLICY §8). Evidence that matters is a test in the repository or a
line in Findings.

## Constraints

Sources are lines of `archive_docs/2026-10-28/2026-08-22_conatus_engine_plan.md`.

- Product profiles conduct; Conatus advances spatial state and owns no product clock, rules, input, durable identity, asset, scene, renderer policy, audio policy, checkpoint lifecycle or packaging. Mechanics may move early after one forcing consumer when authority is settled; contracts for the conductor, source bindings, spatial frame, trigger vocabulary and resident leases stay provisional until a second game proves them. Lines 34-93, 753-770.
- Command trust boundary: scripts and peers submit product intents; only authorized product code, the profile and its registered systems, lowers accepted consequences into local `BodyCommand`s; raw remote commands are not part of the architecture. Line 210.
- Rapier stays private. No public backend trait or selector until a named workload forces a second implementation; a backend must prove which game-facing behaviours it supports and may not silently approximate the rest. Lines 146-168.
- Nexus is decomposed, never adopted: its useful parts arrive as a Conatus backend behind the `BodyWorld` vocabulary when the first vessel gate needs dynamic bodies, never as a peer engine; licence diligence on the upstream source comes before any kernel is taken. Amended 2026-11-12 (Mark, Isocosm `sim-host` D8, "to not have gpu in the wgpu stack is insane"; he chose "C plus E where natural" and "Fine: the host is trusted"): a backend need not replay bit for bit across machines. A backend, the GPU included, proposes physical outcomes; the consuming sim admits an outcome only if it is possible and records it, and replay reads that record; voxel and crowd effects use whole-number kernels where that is easy, which are exact on every GPU. Wing ruling 604's bit-for-bit replay now binds only the CPU fallback, kept deterministic where that is cheap. Before this the line read: any backend must keep G8's bit-for-bit replay across machines (wing ruling 604). Lines 170-177, 1047-1062.
- Windowed demos, screenshots, receipt applications and benchmark gates are not the work queue; performance work starts from a named engine workload. Lines 89-92.
- Body bindings (wing rulings 346 to 352): build only when Mark says; calls borrow the caller's `BodyWorld` and the module owns no terrain, clock or durable state and never serializes a `BodyId`; a product key maps to one `BodyId` and the caller's shape revision; reconcile on an empty table is the cold rebuild and must agree with incremental set and remove; a new pose preserves identity and a changed shape revision respawns; duplicate keys and lost bodies refuse before mutation. Source meaning, admission, collider lowering, scene instances, resident slots and audio voices stay product-owned. Lines 235-288.
- T2 (wing rulings 330 to 335 and 363): `nisus` becomes the voxel world store; each consumer reads a revision log at 8-cubed brick grain and rebuilds if older than retained history; voxel edits take an optional source stamp that queries can report and the product holds dependent simulation until colliders reach it; navigation is a stamped search per query; verify inside Mere, then adopt in Mesocosm after a product repin; existing saves need an explicit migration or preserved wire format. Lines 548-612.
- Allocation ownership follows advanced state: Conatus owns the allocations whose state it advances, the profile orders passes, Netrender's tenancy stays the device seam, and Conatus is not the global allocator for ESP, Renderling or other subsystems. Lines 641-651.
- The spatial frame carries no cameras, lights, sprites or presentation policy; its shared form stays provisional until a second game consumes it. Lines 657-690.
- Renderer: kiss3d is the straightforward tenant (wing rulings 471, 472, 606, 732 to 735, 745 to 748); the scoped context slot is kept rather than a handle threaded through kiss3d; `FrameReport` counts internal submissions and does not reroute them; the lighting block is stack-owned; raster shaders are WGSL and WESL, compute is CubeCL, and rust-gpu stays for Nexus-derived compute; the water field is defined together with its reader; the ECS is hecs, kept as storage for the near-tier runtime, rebuilt from the sim's record and never saved; the record itself holds no ECS crate (wing rulings 481 to 484 archived, isocosm sim-execution D42, 2026-10-31). Lines 930-950, 962-977, 1459-1478.
- Realms own truth; the projection engine shows realms and the inference engine reads or proposes into them, never rendering; physics is a stratified capacity, not an engine. Lines 1014-1046.

## Phases

### RT. Runtime and systems, remaining work
The foundation is implemented: ordered phases (Ingest, field, before-physics, physics, after-physics, materialize, Publish), typed resources, a structural command buffer applied between phases, and frame changes on zero-step host frames.
- Done when: system access can be declared for parallel execution, the intent-lowering command boundary is enforced at the first product profile, and a lean spatial-frame resource exists (the sources state the remaining work, not a further condition).
- Needs: a first product profile; the spatial frame phase below.
- Rulings: command trust boundary (2026-08-23).
- From: `archive_docs/2026-10-28/2026-08-22_conatus_engine_plan.md:185-233,513-517`

### BB. Body bindings (document-only)
A Conatus module generic over the product's key, planned and documented, not built. Mesocosm keeps its `TactileWorld` table; the VTT has none.
- Done when: the module names no product type and adds no Conatus dependency and meets the body-binding plan's size and failure conditions; seeded accepted sets and edits give equivalent cold and incremental keys, shapes and transforms; removal and recreation invalidate the old body and reverse lookup; moved bodies retain their ids; duplicate keys and lost bodies refuse safely; product adoption preserves caller lowering and uses `refresh_queries` after its own verification and repin; Mesocosm's critter half then adopts the table (its terrain half belongs to T2); the VTT adopts only when accepted tokens need Conatus bodies, using the scene board's coordinate convention and not the retired runtime.
- Needs: Mark's go-ahead (ruling 346); Eponym's per-move query world is an adoption target through a cold reconcile.
- Rulings: 346 to 352.
- From: `archive_docs/2026-10-28/2026-08-22_conatus_engine_plan.md:235-288`

### T2. One edit, all spatial consumers (voxel world)
The world store and consumer revision path are not built; `nisus` supplies chunk value mechanics only. `Ground` becomes a thin product layer over the store or retires.
- Done when: accepted patches advance the shared source revision and refused stale patches mutate nothing; independent consumers neither consume each other's changes nor silently miss edits and an out-of-retention consumer rebuilds correctly; the Mere integration verifies removal and addition across storage, collider queries and `modulus`, source stamps included; Mesocosm's adoption verifies render, route and collision agree and holds dependent simulation while a required projection is behind, then removes or thins its terrain mirror and dirty protocol with save compatibility recorded. The voxel world is feature complete when terrain edits, body-volume edits, collision, queries and mesh or SDF preparation use one chunk and revision path and game crates carry no second voxel cache protocol.
- Needs: its own lane and Isometry's pin; Mesocosm's product repin; the F199 pause in the dynamics grammar plan; Burn baseline is stable 0.22.0 and the pre.4 condition is met (ruled 2026-10-06). T2 does not open the broader material transaction (T1) or work-consequence (T3) lanes. Body binding may share this `BodyWorld` but is a separate module and gate.
- Rulings: 330 to 335, 363.
- From: `archive_docs/2026-10-28/2026-08-22_conatus_engine_plan.md:519-612`

### RV. Resident spatial world
The first position-plane slice is done in `conatus::resident` (stable generational slots, atomic sparse patches under one stamp). Open: rotation, velocity, force and flag planes, voxel and chunk joins, direct-GPU advancement, and allocation ownership for state advanced directly on the GPU. The next resident join is incremental `ResidentChunk` publication into the brick ABI; raymarch depth and Renderling occlusion are the next presentation boundary. The lease's promotion trigger, a shipped producer and consumer, is held here as well.
- Done when: CPU tactile bodies and GPU field bodies join through one versioned profile-local spatial view and coexist in one spatial world; the cross-product vocabulary becomes stable only after a second game consumes it.
- Needs: a second game; the lease trigger above.
- Rulings: -
- From: `archive_docs/2026-10-28/2026-08-22_conatus_engine_plan.md:614-655,732-739,1074-1077`; `archive_docs/2026-10-28/2026-07-03_archived_plan_tails_plan.md:1303`

### SF. Spatial frame
A lean frame: transforms addressable by runtime identity, removals, contacts, activity, sleeping and residency changes, voxel and geometry revisions, resident products. A Renderling or other 3D adapter is added only when a product lens demands it.
- Done when: a product selects render tenants and presentation settings in its profile while body transforms, activity and geometry revisions continue to originate in Conatus.
- Needs: a second game consuming the frame before its shared form is stable; today it is the Isometry profile's neutral-shaped seam, whose first tenant was retired with `isometry-runtime`.
- Rulings: -
- From: `archive_docs/2026-10-28/2026-08-22_conatus_engine_plan.md:657-690,741-744`

### FS. Procedural and field systems
Numen and the field evaluators stay independently callable; Conatus adds adapters only for field effects on the state it advances: field sampling over bodies and volumes, force and material couplings, particle and fluid systems, SDF composition and extraction, spatial trees, neighbourhood queries and large-body broad phase.
- Done when: a profile can select a field adapter for Conatus while Seiche and another consumer keep using the same Numen definitions and evaluation without depending on Conatus.
- Needs: -
- Rulings: -
- From: `archive_docs/2026-10-28/2026-08-22_conatus_engine_plan.md:692-708`

### SC. Scripting and content
Spatial queries, events, system parameters, body and voxel descriptors and recipes are exposed to the script host; scripts get no raw backend or GPU handles and no raw command buffer; content catalogs select and combine registered features.
- Done when: a data or script-defined rule can spawn and alter bodies, edit voxel regions, query space, react to interactions and configure fields through product intents that lower into the same command vocabulary native product systems use.
- Needs: -
- Rulings: command trust boundary.
- From: `archive_docs/2026-10-28/2026-08-22_conatus_engine_plan.md:710-722`

### TN. The kiss3d tenant's tails
`crates/conatus/tenant` is a working name that awaits a naming round. Probe capture, planar reflectors and egui still submit inside kiss3d (counted, not rerouted). The water field's format has no reader yet and is to be defined with its first reader. Upstream rebases of the fork stay cheap only while the context slot stays scoped.
- Done when: the crate is named; the water field has a format and a reader; the sources state no wider condition.
- Needs: a consumer that reads the water field.
- Rulings: 471, 472, 606, 732 to 735, 745 to 748.
- From: `archive_docs/2026-10-28/2026-08-22_conatus_engine_plan.md:1366-1478`

### CR. Consumers' repins and adoptions
Work that lands in the consumers' own repositories when they repin Mere: Mesocosm's `tactile.rs` still settles queries with `step(1e-6)` and replaces it with `refresh_queries`; isometer's `bricks.rs` and Eponym's `StableResidency` adopt `AtlasLimits` (`from_keys` has no limits-aware twin yet) and Eponym drops its copied row arithmetic; consumers re-record picture receipts and isometer drops its own walk for `BrickMap::trace`, keeping only its wrapping (rulings 292, 336, 361). The `Engine` source comment still needs reconciling with the 2026-08-23 correction and the body-binding and T2 rulings.
- Done when: each consumer has repinned and adopted the shared call; the comment is reconciled.
- Needs: product repins in Isometry, Mesocosm and Eponym.
- Rulings: 292, 336, 361.
- From: `archive_docs/2026-10-28/2026-08-22_conatus_engine_plan.md:304-305,1111-1127,1265-1271,1314-1322`

### NX. Nexus watch and fork upkeep
Nexus is reconsidered as a backend seat when a product's tactile body count outgrows CPU Rapier. The renderling and crabslab forks are archived after eponym-client's Tenant and lighting move off renderling and its two probes are archived with their patch rows dropped ("L7 first, then archive"); that is Isometry wing work not tracked here. The local rust-gpu fork is retired by re-checking plain upstream.
- Done when: a Nexus seat, if taken, passes an isolated lifecycle and query receipt and the exact host-device receipt, with licence diligence, and its outcomes that become facts are admitted and recorded by the consuming sim (amended 2026-11-12, Isocosm `sim-host` D8; G8 replay is no longer a condition of the seat); the renderling and crabslab forks sit under `archive/`; the rust-gpu fork is retired or kept with a recorded reason.
- Needs: a tactile body count that outgrows CPU Rapier, or the first vessel gate needing dynamic bodies (Paredros F5 and F7); eponym-client moving off renderling.
- Rulings: 604 as narrowed by Isocosm `sim-host` D8 (2026-11-12); the 2026-10-02 archive-timing ruling.
- From: `archive_docs/2026-10-28/2026-08-22_conatus_engine_plan.md:170-177,958-961,1047-1062`; `archive_docs/2026-10-28/2026-07-03_archived_plan_tails_plan.md:1300-1302,1304`

## Open questions

- Are the `BodyWorld` queries and `refresh_queries` on main? The plan's progress heading still reads "Not on main" while the ruling of the same day said to merge and push now. From: `archive_docs/2026-10-28/2026-08-22_conatus_engine_plan.md:511,1324-1327`
- Where is the renderling and crabslab fork archive tracked? The plan calls it Isometry wing work "not tracked here" and eponym still path-depends on the live checkouts. From: `archive_docs/2026-10-28/2026-08-22_conatus_engine_plan.md:955-961`
- Does the composition acceptance ledger take body binding and T2 as gates? This plan's side of the reconciliation is the `Engine` source comment; the ledger side is the projection grammar plan's. From: `archive_docs/2026-10-28/2026-08-22_conatus_engine_plan.md:1318-1322`
- What name does `crates/conatus/tenant` take? From: `archive_docs/2026-10-28/2026-08-22_conatus_engine_plan.md:1406-1407`

## Findings

- 2026-10-09: `BodyWorld` gained `refresh_queries`, `colliders_at_point`, `voxel_filled`, `voxel_cells` and `contacts` returning `ShapeContact`; parry is asked for twice the prediction plus 1/64 and points beyond the prediction are dropped; parry's contact quirk is item 12 of the upstream-candidates ledger and is not raised upstream. No crate in mere uses `BodyWorld`. From: `archive_docs/2026-10-28/2026-08-22_conatus_engine_plan.md:1324-1364`
- 2026-09-28: the T2 starting points are `crates/conatus/nisus/src/lib.rs` (`VoxelPatch`, `VoxelChunk`, revision-gated `apply_edits`) and `crates/conatus/conatus/src/engine.rs` (`bodies_mut`) with `world.rs` (`edit_voxels`, `raycast`, `overlaps`, `step`); the document read them at an earlier checkout. From: `archive_docs/2026-10-28/2026-08-22_conatus_engine_plan.md:1300-1322`
- 2026-09-26: `modulus` sizes its atlas by `AtlasLimits` (`crates/conatus/modulus/src/limits.rs`); 8 MiB holds 16,383 bricks at wgpu's default 2,048-texel edge and 8,191 at the 256-texel web tier; the atlas stays under 4 GiB because texel offsets are 32-bit in `modulus` and in isometer's slot uploads, so widening the 16 by 16 slot plane needs that arithmetic widened first. From: `archive_docs/2026-10-28/2026-08-22_conatus_engine_plan.md:1079-1127`
- 2026-10-09: kiss3d's tonemap gamma-encodes in the shader, so an sRGB target must list `Rgba8Unorm` in its `view_formats`; uniforms reach the GPU through `Queue::write_buffer`, so a caller submits each tenant frame before recording another; entering another device while surfaces built on the current one live panics. From: `archive_docs/2026-10-28/2026-08-22_conatus_engine_plan.md:1366-1453`
- 2026-10-01: Nexus kernels execute on the development machine when built with the rust-gpu fork's cargo-gpu 0.10.0-alpha.1, which locks its install (the earlier Windows build failure was a race on the shared codegen install under cargo-gpu 0.1.0); the installed default is now cargo-gpu 0.10.0 from crates.io; shared buffer ownership with the host's device is still unproven. The rust-gpu version-gate bug has no upstream fix but is latent. From: `archive_docs/2026-10-28/2026-08-22_conatus_engine_plan.md:793-850`
- 2026-08-31: `quint` was folded into `conatus::resident`, `numen` and `seiche`; `conatus-brick` became `modulus` (2026-08-28); `conatus-voxel` became `nisus` (2026-08-28). Read "Quint" in the old plan as those owners. From: `archive_docs/2026-10-28/2026-08-22_conatus_engine_plan.md:94-118`

## Progress

- 2026-11-12: the backend constraint and NX's done-condition amended to Isocosm `sim-host` D8 (`isocosm/mesocosm/design_docs/2026-10-28_sim_host_plan.md`): a backend proposes, the sim admits and records, and 604 binds the CPU fallback only. Nothing built. Mere's dynamics grammar plan (G8) still states the cross-OS digest as a condition in conatus and is not yet amended.
- 2026-10-28: cut from `2026-08-22_conatus_engine_plan.md`; nothing new landed.
