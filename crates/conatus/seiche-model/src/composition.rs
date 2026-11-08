// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Compositions in the catalog (dynamics grammar plan, G3) as data: a weighted
//! mix of force laws, and groups (an outer law between groups, an inner law
//! within each). `seiche::catalog` builds the forces; this names the choice.
//!
//! A composition runs in the law slot instead of the picked law, overlays
//! still after it. A grouping reads its partition from a `groups.*` channel
//! through the law inputs (`groups.meaning` among them), or takes one the host
//! hands in, as the separation receipt hands in its topics (F70), and between
//! groups an outer law acts by its repulsion alone (F71). A weight scales
//! forces and groups share the outer law's force among members, so both take
//! force laws only ([`PhysicsComposition::refusal`]).

use crate::{Currency, NodeKey, PhysicsKindSource, PhysicsLaw};

/// Why a weighted mix refuses a law that does not add force.
pub const UNWEIGHTED: &str = "a weight scales a law's forces, and this law writes the bodies' \
                              state instead, so a mix takes force laws only";

/// The outer weight of the first instance, Charge between groups (ruled
/// 2026-10-06, F71, "Repulsion only, weight 16").
pub const CHARGE_BETWEEN_WEIGHT: f32 = 16.0;

#[derive(Clone, Debug, PartialEq)]
pub enum PhysicsComposition {
    /// Force laws at weights; 1 is the law as calibrated, 0 leaves it out.
    Mix(Vec<(PhysicsLaw, f32)>),
    /// An outer law between groups and an inner law within each.
    Grouped(PhysicsGrouping),
}

impl PhysicsComposition {
    /// Every law the composition runs.
    pub fn laws(&self) -> Vec<PhysicsLaw> {
        match self {
            PhysicsComposition::Mix(laws) => laws.iter().map(|(law, _)| *law).collect(),
            PhysicsComposition::Grouped(g) => vec![g.outer, g.inner],
        }
    }

    /// The first law the composition refuses: a weight scales forces, and
    /// groups share the outer law's force among members, so both take force
    /// laws only.
    pub fn refusal(&self) -> Option<CompositionRefusal> {
        self.laws()
            .into_iter()
            .find(|law| law.currency() != Currency::Force)
            .map(|law| CompositionRefusal {
                law,
                reason: UNWEIGHTED,
            })
    }
}

/// Where a grouping's partition comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum GroupSource {
    /// A `groups.*` channel, read through the law inputs at each rebuild, so
    /// the partition follows the channel (F70).
    Channel(PhysicsKindSource),
    /// A partition the host hands in, a group per node: the separation
    /// receipt's topics, its ground truth.
    Given(Vec<(NodeKey, u32)>),
}

/// Groups: Charge between them and Springs within is the first instance
/// ([`Self::charge_between`]).
#[derive(Clone, Debug, PartialEq)]
pub struct PhysicsGrouping {
    /// Each node's group: a channel's, or the host's.
    pub groups: GroupSource,
    /// The law whose repulsion acts between groups, over their centroids:
    /// its repulsion alone, the rest of the law left out (F71).
    pub outer: PhysicsLaw,
    /// The law within each group, under its membership mask.
    pub inner: PhysicsLaw,
    /// The outer repulsion's weight; 1 is the law as calibrated.
    pub outer_weight: f32,
}

impl PhysicsGrouping {
    /// Charge's repulsion between groups at weight 16, Springs within (F71).
    pub fn charge_between(groups: GroupSource) -> Self {
        Self {
            groups,
            outer: PhysicsLaw::Charge,
            inner: PhysicsLaw::Springs,
            outer_weight: CHARGE_BETWEEN_WEIGHT,
        }
    }
}

/// Why a composition was refused, and which law refused it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompositionRefusal {
    pub law: PhysicsLaw,
    pub reason: &'static str,
}

impl std::fmt::Display for CompositionRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.law.label(), self.reason)
    }
}
