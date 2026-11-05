// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;

impl NodeExclusionParams {
    pub fn of(f: &NodeExclusion) -> Self {
        Self {
            strength: wide(f.strength),
            cutoff: wide(f.cutoff),
            min_distance: wide(f.min_distance),
        }
    }

    pub(super) fn build(&self) -> NodeExclusion {
        NodeExclusion {
            strength: narrow(self.strength),
            cutoff: narrow(self.cutoff),
            min_distance: narrow(self.min_distance),
        }
    }
}

impl Default for NodeExclusionParams {
    fn default() -> Self {
        Self::of(&NodeExclusion::default())
    }
}

impl EdgeSpringParams {
    pub fn of(f: &EdgeSpring) -> Self {
        Self {
            stiffness: wide(f.stiffness),
            rest_length: wide(f.rest_length),
        }
    }

    pub(super) fn build(&self) -> EdgeSpring {
        EdgeSpring {
            stiffness: narrow(self.stiffness),
            rest_length: narrow(self.rest_length),
        }
    }
}

impl Default for EdgeSpringParams {
    fn default() -> Self {
        Self::of(&EdgeSpring::default())
    }
}

impl BarnesHutRepulsionParams {
    pub fn of(f: &BarnesHutRepulsion) -> Self {
        Self {
            strength: wide(f.strength),
            min_distance: wide(f.min_distance),
            theta: wide(f.config.theta),
            min_cell_size: wide(f.config.min_cell_size),
        }
    }

    pub(super) fn build(&self) -> BarnesHutRepulsion {
        BarnesHutRepulsion {
            strength: narrow(self.strength),
            min_distance: narrow(self.min_distance),
            config: BarnesHutConfig {
                theta: narrow(self.theta),
                min_cell_size: narrow(self.min_cell_size),
            },
        }
    }
}

impl Default for BarnesHutRepulsionParams {
    fn default() -> Self {
        Self::of(&BarnesHutRepulsion::default())
    }
}

impl StressSpringParams {
    pub fn of(f: &StressSpring) -> Self {
        Self {
            pairs: None,
            unit_length: Some(wide(f.unit_length)),
            stiffness: wide(f.stiffness),
        }
    }

    pub(super) fn build(&self, path: &str) -> Result<StressSpring, SpecError> {
        let unit_length = need(self.unit_length, path, "unit_length")?;
        need(self.pairs.as_ref(), path, "pairs")?;
        Ok(
            StressSpring::from_weighted_distances(std::iter::empty(), narrow(unit_length))
                .with_stiffness(narrow(self.stiffness)),
        )
    }
}

impl Default for StressSpringParams {
    fn default() -> Self {
        let mut params = Self::of(&StressSpring::from_weighted_distances(
            std::iter::empty(),
            0.0,
        ));
        params.unit_length = None;
        params
    }
}

impl LinlogParams {
    pub fn of(f: &LinLogForce) -> Self {
        Self {
            attraction: wide(f.attraction),
            attraction_exponent: wide(f.attraction_exponent),
            repulsion: wide(f.repulsion),
            degree_weighted: f.degree_weighted,
            min_distance: wide(f.min_distance),
            gravity: wide(f.gravity),
        }
    }

    pub(super) fn build(&self) -> LinLogForce {
        LinLogForce {
            attraction: narrow(self.attraction),
            attraction_exponent: narrow(self.attraction_exponent),
            repulsion: narrow(self.repulsion),
            degree_weighted: self.degree_weighted,
            min_distance: narrow(self.min_distance),
            gravity: narrow(self.gravity),
        }
    }
}

impl Default for LinlogParams {
    fn default() -> Self {
        Self::of(&LinLogForce::default())
    }
}
