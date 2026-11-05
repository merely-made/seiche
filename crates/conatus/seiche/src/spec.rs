// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! The dynamics spec: a live layout's drive as portable data (dynamics
//! grammar plan, G4; F4 "Portable shape now, in seiche").
//!
//! A [`DynamicsSpec`] is a tree (F95). A leaf is a catalog preset, a law or
//! overlay id the host's catalog builds, or a raw seiche term with its
//! parameters (F94, [`raw`]). A mix, a grouping or a schedule may hold any
//! node, and any node may carry overlays (F109). At the root sit the seed
//! (F101), the realization (F103), an optional target (F105), the channel
//! each preset slot reads, and bars, carried and not evaluated (F102, F111).
//!
//! What the spec never holds is what its terms are: currency, class and
//! metric are read from the built forces' declarations ([`Declared`]) by
//! [`DynamicsSpec::derive`], never authored. Derive also refuses, by path:
//! a tree deeper than [`MAX_DEPTH`] (F110); an unknown preset or term name; a
//! required parameter or input left out; a reserved rung (F103); a law that
//! writes state inside a mix or a grouping (F108, by currency); and any
//! shape the canvas cannot run yet (F108, the R1–R4 of the G4a checkpoint).
//! A scene whose spec derive refuses does not open (F97).
//!
//! Weights (F96): a node's `weight` is a multiplier, 1 as calibrated; a
//! term override is the term's force at its reference where the term has
//! one ([`crate::scale`]), and a multiplier where it has none. Which reading
//! applies is derived ([`WeightReading`]), never written.
//!
//! The wire (feature `serde`, F104): one serde type for every format,
//! TOML-safe (no nulls written, `f64` numbers, `BTreeMap` maps). The spec has
//! its own version, read in `sceno::Score`'s pattern: a newer spec is refused
//! with [`SpecVersionError`], an older one read and stamped, and unknown
//! fields are refused throughout (F98).
//!
//! Plan: `design_docs/archive_docs/2026-10-28/2026-10-02_dynamics_grammar_plan.md`, G4a.

use std::collections::{BTreeMap, HashMap};

use crate::compose::{self, Admission};
use crate::scale::{self, Family};
use crate::{Currency, Force, Observable, Role, Term};

pub mod raw;
pub use raw::{InputSlot, RawTerm};

/// The spec's wire version.
pub const DYNAMICS_SPEC_VERSION: u16 = 1;

/// The deepest tree derive takes (F110): nodes on the longest path from the
/// root to a leaf, overlays and stages' nodes counted as children.
pub const MAX_DEPTH: usize = 32;

/// The seed a spec without one runs on: the catalog's `LAW_SEED`, so such a
/// spec reproduces today (F101).
pub const DEFAULT_SEED: u64 = 0x5EED_CA7A_1064;

/// See the module docs.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "SpecWire")
)]
pub struct DynamicsSpec {
    /// Stamped with the reader's version when read.
    pub version: u16,
    /// The seed every seeded term draws from (F101).
    pub seed: u64,
    pub root: Node,
    /// Which channel each preset slot reads (`"mass"` to `"mass.pagerank"`):
    /// opaque ids, resolved host-side in G4b.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "BTreeMap::is_empty"))]
    pub channels: BTreeMap<String, String>,
    pub realization: Realization,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub target: Option<Target>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Vec::is_empty"))]
    pub bars: Vec<Bar>,
}

/// A spec as written, before it is read in the reader's meaning.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SpecWire {
    version: u16,
    #[serde(default = "default_seed")]
    seed: u64,
    root: Node,
    #[serde(default)]
    channels: BTreeMap<String, String>,
    #[serde(default)]
    realization: Realization,
    #[serde(default)]
    target: Option<Target>,
    #[serde(default)]
    bars: Vec<Bar>,
}

#[cfg(feature = "serde")]
fn default_seed() -> u64 {
    DEFAULT_SEED
}

#[cfg(feature = "serde")]
impl TryFrom<SpecWire> for DynamicsSpec {
    type Error = SpecVersionError;

    fn try_from(wire: SpecWire) -> Result<Self, Self::Error> {
        DynamicsSpec::read(wire, DYNAMICS_SPEC_VERSION)
    }
}

/// A spec newer than its reader, refused rather than half-read (F98).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpecVersionError {
    /// The spec's version.
    pub found: u16,
    /// The newest version the reader knows.
    pub reader: u16,
}

impl std::fmt::Display for SpecVersionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "dynamics spec version {} is newer than this reader's {}",
            self.found, self.reader
        )
    }
}

impl std::error::Error for SpecVersionError {}

/// A node of the tree (F95). Written tagged by `node`; read through
/// [`NodeWire`], one plain struct, so a reader that tracks paths
/// ([`read`]) sees inside every node (F113).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(tag = "node", rename_all = "kebab-case", try_from = "NodeWire")
)]
pub enum Node {
    /// A catalog law or overlay by its id, as a saved scene stores it.
    Preset {
        id: String,
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "is_one"))]
        weight: f64,
        /// Per-term overrides by term name (F96).
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "BTreeMap::is_empty"))]
        terms: BTreeMap<String, f64>,
        /// Reserved for G5; refused if set (F103).
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
        rung: Option<String>,
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Vec::is_empty"))]
        overlays: Vec<Node>,
    },
    /// A raw seiche term (F106).
    Raw {
        term: RawTerm,
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "is_one"))]
        weight: f64,
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "BTreeMap::is_empty"))]
        terms: BTreeMap<String, f64>,
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
        rung: Option<String>,
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Vec::is_empty"))]
        overlays: Vec<Node>,
    },
    /// Parts summed, each at its weight.
    Mix {
        parts: Vec<Node>,
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "is_one"))]
        weight: f64,
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Vec::is_empty"))]
        overlays: Vec<Node>,
    },
    /// An outer node between the groups of a partition, an inner node within
    /// each.
    Grouped {
        /// The groups channel the partition reads (opaque until G4b).
        partition: String,
        outer: Box<Node>,
        inner: Box<Node>,
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "is_one"))]
        weight: f64,
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Vec::is_empty"))]
        overlays: Vec<Node>,
    },
    /// Stages run in order, each to its stop.
    Schedule {
        stages: Vec<Stage>,
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "is_one"))]
        weight: f64,
        #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Vec::is_empty"))]
        overlays: Vec<Node>,
    },
}

#[cfg(feature = "serde")]
fn is_one(x: &f64) -> bool {
    *x == 1.0
}

/// A node's tag, as written.
#[cfg(feature = "serde")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
enum NodeTag {
    Preset,
    Raw,
    Mix,
    Grouped,
    Schedule,
}

/// A node as written: every field any node takes, read as one struct, then
/// checked against its tag. An internally tagged enum buffers a node before
/// reading it, which hides everything inside from a path-tracking reader.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeWire {
    node: NodeTag,
    id: Option<String>,
    term: Option<RawTerm>,
    weight: Option<f64>,
    terms: Option<BTreeMap<String, f64>>,
    rung: Option<String>,
    overlays: Option<Vec<Node>>,
    parts: Option<Vec<Node>>,
    partition: Option<String>,
    outer: Option<Box<Node>>,
    inner: Option<Box<Node>>,
    stages: Option<Vec<Stage>>,
}

#[cfg(feature = "serde")]
impl TryFrom<NodeWire> for Node {
    type Error = String;

    fn try_from(wire: NodeWire) -> Result<Self, Self::Error> {
        let kind = match wire.node {
            NodeTag::Preset => "preset",
            NodeTag::Raw => "raw",
            NodeTag::Mix => "mix",
            NodeTag::Grouped => "grouped",
            NodeTag::Schedule => "schedule",
        };
        let present = [
            ("id", wire.id.is_some()),
            ("term", wire.term.is_some()),
            ("terms", wire.terms.is_some()),
            ("rung", wire.rung.is_some()),
            ("parts", wire.parts.is_some()),
            ("partition", wire.partition.is_some()),
            ("outer", wire.outer.is_some()),
            ("inner", wire.inner.is_some()),
            ("stages", wire.stages.is_some()),
        ];
        let takes: &[&str] = match wire.node {
            NodeTag::Preset => &["id", "terms", "rung"],
            NodeTag::Raw => &["term", "terms", "rung"],
            NodeTag::Mix => &["parts"],
            NodeTag::Grouped => &["partition", "outer", "inner"],
            NodeTag::Schedule => &["stages"],
        };
        if let Some((field, _)) = present
            .iter()
            .find(|(field, set)| *set && !takes.contains(field))
        {
            return Err(format!("unknown field `{field}` on a {kind} node"));
        }
        let missing = |field: &str| format!("missing field `{field}` on a {kind} node");
        let weight = wire.weight.unwrap_or(1.0);
        let overlays = wire.overlays.unwrap_or_default();
        Ok(match wire.node {
            NodeTag::Preset => Node::Preset {
                id: wire.id.ok_or_else(|| missing("id"))?,
                weight,
                terms: wire.terms.unwrap_or_default(),
                rung: wire.rung,
                overlays,
            },
            NodeTag::Raw => Node::Raw {
                term: wire.term.ok_or_else(|| missing("term"))?,
                weight,
                terms: wire.terms.unwrap_or_default(),
                rung: wire.rung,
                overlays,
            },
            NodeTag::Mix => Node::Mix {
                parts: wire.parts.ok_or_else(|| missing("parts"))?,
                weight,
                overlays,
            },
            NodeTag::Grouped => Node::Grouped {
                partition: wire.partition.ok_or_else(|| missing("partition"))?,
                outer: wire.outer.ok_or_else(|| missing("outer"))?,
                inner: wire.inner.ok_or_else(|| missing("inner"))?,
                weight,
                overlays,
            },
            NodeTag::Schedule => Node::Schedule {
                stages: wire.stages.ok_or_else(|| missing("stages"))?,
                weight,
                overlays,
            },
        })
    }
}

/// A spec refused while it is read, with where (F113): the path in the
/// spec's own terms (`root.stages[2].node.term`), as [`SpecError`]'s are.
#[cfg(feature = "serde")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecReadError {
    pub path: String,
    pub message: String,
}

#[cfg(feature = "serde")]
impl std::fmt::Display for SpecReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "at {}: {}", self.path, self.message)
    }
}

#[cfg(feature = "serde")]
impl std::error::Error for SpecReadError {}

/// Read a spec from `input`, naming the path of any refusal (F113). The
/// version is read first, so a newer spec is refused for its version even
/// where it holds names this reader does not know (F98); `input` is read
/// twice, so it is a cheap handle (`&serde_json::Value`, a TOML table).
#[cfg(feature = "serde")]
pub fn read<'de, D>(input: D) -> Result<DynamicsSpec, SpecReadError>
where
    D: serde::Deserializer<'de> + Clone,
{
    #[derive(serde::Deserialize)]
    struct Version {
        version: u16,
    }
    let refused = |error: serde_path_to_error::Error<D::Error>| SpecReadError {
        path: match error.path().to_string() {
            root if root == "." => "the spec".to_string(),
            path => path,
        },
        message: error.inner().to_string(),
    };
    let Version { version } = serde_path_to_error::deserialize(input.clone()).map_err(refused)?;
    if version > DYNAMICS_SPEC_VERSION {
        return Err(SpecReadError {
            path: "version".into(),
            message: SpecVersionError {
                found: version,
                reader: DYNAMICS_SPEC_VERSION,
            }
            .to_string(),
        });
    }
    serde_path_to_error::deserialize(input).map_err(refused)
}

/// The host's catalog of presets. seiche names none; the canvas's catalog
/// (`pictograph::canvas::dynamics_spec`) builds its laws and overlays.
pub trait SpecCatalog {
    /// The forces preset `id` builds at `at`, over inputs enough to read
    /// their declarations; `None` for an id the catalog does not know there.
    fn preset(&self, id: &str, at: PresetAt) -> Option<Vec<Box<dyn Force>>>;

    /// What law preset `law` does with overlay preset `overlay`, where the
    /// catalog rules beyond currency (Density admits three overlays, F73);
    /// `None` defers to currency ([`compose::admit`]).
    fn admits(&self, _law: &str, _overlay: &str) -> Option<Admission> {
        None
    }
}

/// A catalog with no presets: a spec of raw terms derives against it.
pub struct NoPresets;

impl SpecCatalog for NoPresets {
    fn preset(&self, _id: &str, _at: PresetAt) -> Option<Vec<Box<dyn Force>>> {
        None
    }
}

/// What derive reads off a spec.
#[derive(Clone, Debug, PartialEq)]
pub struct Derived {
    pub depth: usize,
    /// Every leaf, overlays included, in pre-order.
    pub leaves: Vec<DerivedLeaf>,
}

/// One leaf's derived fields.
#[derive(Clone, Debug, PartialEq)]
pub struct DerivedLeaf {
    /// Where the leaf sits (`root.stages[1].node.parts[0]`).
    pub path: String,
    pub leaf: LeafKind,
    /// How the leaf's motion enters the step ([`compose::law_currency`]).
    pub currency: Currency,
    pub terms: Vec<DerivedTerm>,
}

/// A leaf's kind: a preset id, or a raw kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LeafKind {
    Preset(String),
    Raw(&'static str),
}

/// One term as declared, with how a weight override on it reads.
#[derive(Clone, Debug, PartialEq)]
pub struct DerivedTerm {
    /// The declaration: currency, class, metric and the rest.
    pub term: Term,
    /// Whether the term has a resident kernel or the lagged upload.
    pub resident: bool,
    pub reading: WeightReading,
    /// The spec's override for this term, if it gives one.
    pub weight: Option<f64>,
}

/// How a term override reads (F96), derived from [`scale::family`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightReading {
    /// The term's force at its reference.
    Absolute(Family),
    /// A multiplier, 1 as calibrated: the term has no reference.
    Multiplier,
}

/// Why derive refused a spec. Each names where in the tree.
#[derive(Clone, Debug, PartialEq)]
pub enum SpecError {
    /// Deeper than [`MAX_DEPTH`] (F110).
    Depth { depth: usize, max: usize },
    /// A preset id the catalog does not know at that place.
    UnknownPreset {
        path: String,
        id: String,
        at: PresetAt,
    },
    /// A term override naming no term of its leaf.
    UnknownTerm {
        path: String,
        kind: String,
        name: String,
    },
    /// A required parameter or input left out.
    Missing { path: String, field: &'static str },
    /// A value out of its range.
    Invalid { path: String, reason: String },
    /// A reserved field set (F103's rung).
    Reserved { path: String, field: &'static str },
    /// Refused by currency (F108): a law that writes state where weights
    /// scale forces, or an overlay its node refuses.
    Currency { path: String, reason: &'static str },
    /// A shape the canvas cannot run yet (F108).
    Unrunnable { path: String, reason: &'static str },
}

impl std::fmt::Display for SpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpecError::Depth { depth, max } => {
                write!(f, "dynamics spec is {depth} deep, deeper than {max}")
            },
            SpecError::UnknownPreset { path, id, at } => {
                write!(f, "at {path}: unknown {} preset {id}", at.word())
            },
            SpecError::UnknownTerm { path, kind, name } => {
                write!(f, "at {path}: {kind} has no term named {name}")
            },
            SpecError::Missing { path, field } => write!(f, "at {path}: {field} is required"),
            SpecError::Invalid { path, reason } => write!(f, "at {path}: {reason}"),
            SpecError::Reserved { path, field } => {
                write!(f, "at {path}: {field} is reserved and must not be set")
            },
            SpecError::Currency { path, reason } => write!(f, "at {path}: {reason}"),
            SpecError::Unrunnable { path, reason } => write!(f, "at {path}: {reason}"),
        }
    }
}

impl std::error::Error for SpecError {}

/// Why a raw term is refused at run time until G4b binds it.
pub const RAW_UNBOUND: &str = "a raw term has no runner until G4b binds it";
/// Why a per-term override is refused until G4b.
pub const OVERRIDE_UNBOUND: &str = "a per-term weight has no runner until G4b";
/// Why a weight outside a mix or a grouping's outer is refused.
pub const WEIGHT_UNBOUND: &str =
    "a weight outside a mix or a grouping's outer has no runner until G4b";
/// Why overlays inside a node are refused (F109).
pub const OVERLAYS_UNBOUND: &str = "overlays run only at the root and on a stage until G4b";
/// Why a schedule's own overlays are refused.
pub const SCHEDULE_OVERLAYS: &str = "a schedule's overlays ride its stages";
/// Why a nested mix or grouping is refused.
pub const NESTED_UNBOUND: &str =
    "a mix or a grouping runs only at the root or as a stage until G4b";
/// Why an overlay that is not a leaf is refused.
pub const OVERLAY_UNBOUND: &str = "an overlay runs only as a preset or a term until G4b";
/// Why a schedule anywhere but the root is refused.
pub const SCHEDULE_NESTED: &str = "a schedule runs only at the root: a schedule inside a force \
                                   needs a time-varying runner";

/// Where a node sits, for the runnability check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Context {
    Root,
    Stage,
    MixPart,
    GroupOuter,
    GroupInner,
    Overlay,
}

#[cfg(test)]
mod tests;

#[path = "spec/derivation.rs"]
mod derivation;

#[path = "spec/leaves.rs"]
mod leaves;

use leaves::leaf;

use leaves::reserved;

use leaves::check_number;

use leaves::invalid;

use leaves::by_currency;

#[path = "spec/schedule.rs"]
mod schedule;

pub use schedule::{Bar, GroupRoles, PresetAt, Realization, Stage, Stop, Target};

#[path = "spec/document.rs"]
mod document;
