# Seiche Documentation Index

Seiche is the family's spatial runtime: bodies, voxels, fields, force
integration and the lit tenant. Its consumers host the scene; Seiche
integrates it.

## Required reading order

1. [DOC_POLICY.md](DOC_POLICY.md) — the canonical core and the Seiche addendum.
2. [`crates/conatus/README.md`](../crates/conatus/README.md) — the crates and how
   they depend on one another.
3. `seiche_docs/` — the spatial runtime (index below), the conatus engine plan
   first.

Entries below are copied from Mere's DOC_README at `4fa1686da`. The two
documents moved here on 2026-11-07 under Mere's engine split plan, D25.

## seiche_docs/ — the spatial runtime

It describes `crates/conatus/`: `numen`, `seiche`, `seiche-model`, `conatus`,
`nisus`, `modulus`, `tenant`.

### seiche_docs/implementation_strategy/

- [conatus_engine_plan](seiche_docs/implementation_strategy/2026-10-28_conatus_engine_plan.md) — **cut 2026-10-28**: the conatus spatial runtime: voxel store, body bindings, resident views, the kiss3d tenant.

### seiche_docs/research/

- [conatus_nexus_alignment_brief](seiche_docs/research/2026-08-10_conatus_nexus_alignment_brief.md) — **dated research and rulings, 2026-08-10 through 2026-10-08**. Preserves the field-evaluation/integrator/game-physics distinction and Mark's fact-boundary authority model, which does not require identical local trajectories. Later annotations supersede the initial claim that only Seiche uses Rapier: Conatus has the tactile-body seam and Seiche the planar canvas. Both current manifests select Rapier 0.36. Nexus remains a consumer-gated backend comparison; current work and the separate wing cleanup gates are in the [Conatus engine plan](seiche_docs/implementation_strategy/2026-10-28_conatus_engine_plan.md).

## In Mere

These stay in [merely-made/mere](https://github.com/merely-made/mere), because
their subjects span Mere's canvas as well as this runtime (path citations,
core §5):

- `mere_docs/implementation_strategy/2026-10-28_dynamics_grammar_plan.md`
- `mere_docs/implementation_strategy/2026-10-28_physics_catalog_plan.md`
- `mere_docs/technical_architecture/2026-05-30_field_system_extraction.md`
- `mere_docs/implementation_strategy/2026-11-07_engine_split_plan.md` (P2; D5, D6, D12, D25)
