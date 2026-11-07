# Licenses in this repository

**This repository: MPL-2.0.** Every file Mark wrote carries Exhibit A and the
SPDX tag `MPL-2.0`, per the license posture brief of 2026-08-22 (in Mere's
`design_docs/2026-08-22_license_posture_brief.md`). The full text is in
[`LICENSE`](LICENSE).

This file is the provenance ledger, carried from Mere's `LICENSES.md` at
`4fa1686da` when the spatial runtime crates moved here with their history
(engine split plan, P2). Only the rows for paths in this repository came with
it; Mere's ledger keeps the rest. Provenance comes before license: a file gets
Exhibit A only if Mark wrote it.

## Retained licenses

Third-party code keeps its own license and its own notices. Nothing here is
relicensed, and nothing here receives a Merely copyright line.

| Path | License | Upstream | Notice files |
|---|---|---|---|
| `crates/conatus/conatus/src/resident/binning` | Apache-2.0 | [dimforge/nexus](https://github.com/dimforge/nexus) at `1cfbd76`, by Sébastien Crozet / Dimforge | `LICENSE-APACHE` in-tree; port and change notice in `mod.rs` |

`crates/conatus/conatus/src/resident/binning` is a close hand port, to CubeCL,
of Nexus's exclusive scan (`src_rbd_shaders/utils/prefix_sum.rs` and its
staging in `src_rbd/utils/prefix_sum.rs`) and of the count / cursor / finalize
pattern of its MPM particle sort (`src_mpm_shaders/grid/sort.rs`), made on
2026-10-02 for the physics catalog plan's P5a cell list. Nexus declares `MIT
OR Apache-2.0` but ships only the Apache-2.0 text, so the port is treated as
Apache-2.0 per the 2026-10-01 licensing ruling; the module header names the
upstream, the author and the changes, and the license text sits beside it. The
cell-walking force kernel that reads the bins (`kernels::exclude_cells`) and
the host code that calls them are Mark's and stay MPL-2.0.

## Published grants that arrived with a crate

None. Mere's ledger records no published-grant row for these seven crates.

## Exceptions under the fork/vendor criterion

**None.** The brief's test (a crate stays MIT OR Apache-2.0 only when a third
party would need to *modify or vendor* it rather than merely link it) admits
nothing here.

## How to add a file from elsewhere

1. Do not delete or rewrite the upstream copyright or license notice, ever.
2. Add its path to **Retained licenses** above with its license, upstream URL,
   and where its notice text lives.
3. If it is a substantial derivative rather than a verbatim import, the brief's
   rule is MPL-2.0 on the derivative *with the upstream notice retained*;
   record it here with that disposition.
4. Never add `license-file` to an owned manifest; the field is for retained
   third-party crates only.
