// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! numen — field truth, evaluation, and authoring for the Seiche substrate.
//!
//! A numen is a pervading influence, the presence felt in a place. This crate is
//! that influence as data: the portable definitions of the **fields** an app lays
//! over its graph, and the **couplings** that say how elements respond to them.
//! numen owns the analytic evaluator, registries and projection composition, plus
//! opt-in Rhai authoring and Burn lowering. Seiche owns force laws and integrates
//! them over Rapier bodies.
//!
//! Fields are the third graph primitive, beside nodes and edges. Since engine-split
//! D49 their **definitions** (`field`, `coupling`, `field_ast`, `edge_path`) live in
//! [`chartulary`](https://github.com/merely-made/eidetic), beside the generic nodes
//! and relations; this crate re-exports them for one release so existing paths keep
//! working, and evaluates them. The truth is the graph realm's, persisted beside
//! nodes and edges as content.
//!
//! - [`ScalarField`] / [`VectorField`] — the field algebra, `f: R^2 -> R` and
//!   `f: R^2 -> R^2`, as recursive data with `Sample` references to other fields by
//!   [`FieldId`].
//! - [`Field`] — a field as truth: identity + a [`FieldDefinition`] + a
//!   [`FieldExtent`] (global / region / attached-to-node) + a [`FieldLifecycle`].
//! - [`Coupling`] — `field -> `[`NodeSelector`]` x `[`CouplingResponse`]` x strength`:
//!   how elements respond to a field at their position. The response vocabulary is a
//!   recognized force core (the six `seiche` integrates) plus an open IRI tail
//!   ([`COUPLING_VOCAB`]) for visual / navigational / selection / semantic / trigger
//!   families.
//! - [`EdgePath`] / [`EdgePathRule`] — how an edge's curve is drawn, including a
//!   field-traced [`EdgePath::FieldLine`].
//!
//! The default path is plain, serde-serializable data plus analytic evaluation with
//! no GPU or host dependencies. Rhai and Burn stay behind named opt-in features.

pub mod eval;
#[cfg(feature = "field-burn")]
pub mod lower_burn;
pub mod projection;
pub mod registry;
pub mod responsive_path;
#[cfg(feature = "field-rhai")]
pub mod rhai_bindings;

// The definition half moved to chartulary (engine-split D49); the modules and
// root names stay here for one release so outside users do not break.
pub use fieldbook::{coupling, edge_path, field, field_ast};
pub use fieldbook::coupling::{COUPLING_VOCAB, Coupling, CouplingResponse, NodeSelector};
pub use fieldbook::edge_path::{EdgePath, EdgePathRule};
pub use eval::{eval_scalar, eval_vector, grad_scalar};
pub use fieldbook::field::{CouplingId, Field, FieldDefinition, FieldExtent, FieldId, FieldLifecycle};
pub use fieldbook::field_ast as ast;
pub use fieldbook::field_ast::{Falloff, ScalarField, VectorField};
pub use projection::{FieldProjection, FieldProjectionBuilder};
pub use registry::{FieldDef, FieldRegistry};
pub use responsive_path::{
    MAX_COMPATIBILITY_KEY_BYTES, MAX_CONTROL_POINTS, MAX_FIELD_AST_DEPTH, MAX_FIELD_AST_NODES,
    MAX_FIELDS, MAX_ITERATIONS, MAX_LINKS, PathField, ResponsiveLink, ResponsivePath,
    ResponsivePathConfig, ResponsivePathError, evaluate_responsive_paths,
};
