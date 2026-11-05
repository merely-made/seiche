// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;

impl AnnealParams {
    pub fn of(f: &Anneal) -> Self {
        Self {
            initial_temperature: wide(f.initial_temperature),
            cooling: wide(f.cooling),
            floor: wide(f.floor),
            step: wide(f.step),
            ideal_edge: wide(f.ideal_edge),
            repulsion: wide(f.repulsion),
            gravity: wide(f.gravity),
        }
    }

    pub(super) fn build(&self, seed: u64) -> Anneal {
        let mut f = Anneal::seeded(seed);
        f.initial_temperature = narrow(self.initial_temperature);
        f.cooling = narrow(self.cooling);
        f.floor = narrow(self.floor);
        f.step = narrow(self.step);
        f.ideal_edge = narrow(self.ideal_edge);
        f.repulsion = narrow(self.repulsion);
        f.gravity = narrow(self.gravity);
        f
    }
}

impl Default for AnnealParams {
    fn default() -> Self {
        Self::of(&Anneal::seeded(0))
    }
}

impl DensityParams {
    /// The parameters `f` was built with, its resolution aside (the grid
    /// holds it, and a spec states it).
    pub fn of(f: &Density) -> Self {
        Self {
            masses: None,
            resolution: None,
            area_per_mass: wide(f.area_per_mass),
            centre: wide2(f.centre),
            initial_blur: wide(f.initial_blur),
            seconds: wide(f.seconds),
            cfl: wide(f.cfl),
            max_substeps: f.max_substeps,
            background: wide(f.background),
            stop: DensityStopParam::of(f.stop),
            patience: f.patience,
            max_passes: f.max_passes,
            min_passes: f.min_passes,
            renew: f.renew.map(wide),
            converts: f.converts,
        }
    }

    pub(super) fn build(&self, path: &str) -> Result<Density, SpecError> {
        need(self.masses.as_ref(), path, "masses")?;
        let resolution = need(self.resolution, path, "resolution")?;
        let mut f = Density::new(std::iter::empty(), resolution as usize);
        f.area_per_mass = narrow(self.area_per_mass);
        f.centre = narrow2(self.centre);
        f.initial_blur = narrow(self.initial_blur);
        f.seconds = narrow(self.seconds);
        f.cfl = narrow(self.cfl);
        f.max_substeps = self.max_substeps;
        f.background = narrow(self.background);
        f.stop = self.stop.build();
        f.patience = self.patience;
        f.max_passes = self.max_passes;
        f.min_passes = self.min_passes;
        f.renew = self.renew.map(narrow);
        f.converts = self.converts;
        Ok(f)
    }
}

impl Default for DensityParams {
    fn default() -> Self {
        Self::of(&Density::new(std::iter::empty(), 1))
    }
}

impl DegreeRepulsionParams {
    pub fn of(f: &DegreeRepulsion) -> Self {
        Self {
            masses: None,
            strength: wide(f.strength),
            radius: wide(f.radius),
            min_distance: wide(f.min_distance),
        }
    }

    pub(super) fn build(&self) -> DegreeRepulsion {
        let mut f = DegreeRepulsion::default();
        f.strength = narrow(self.strength);
        f.radius = narrow(self.radius);
        f.min_distance = narrow(self.min_distance);
        f
    }
}

impl Default for DegreeRepulsionParams {
    fn default() -> Self {
        Self::of(&DegreeRepulsion::default())
    }
}

impl DomainClusterParams {
    pub fn of(f: &DomainCluster) -> Self {
        Self {
            groups: None,
            strength: wide(f.strength),
        }
    }

    pub(super) fn build(&self, path: &str) -> Result<DomainCluster, SpecError> {
        need(self.groups.as_ref(), path, "groups")?;
        let mut f = DomainCluster::new(std::iter::empty());
        f.strength = narrow(self.strength);
        Ok(f)
    }
}

impl Default for DomainClusterParams {
    fn default() -> Self {
        Self::of(&DomainCluster::new(std::iter::empty()))
    }
}

impl HubGravityParams {
    pub fn of(f: &HubGravity) -> Self {
        Self {
            masses: None,
            strength: wide(f.strength),
            min_distance: wide(f.min_distance),
        }
    }

    pub(super) fn build(&self) -> HubGravity {
        let mut f = HubGravity::default();
        f.strength = narrow(self.strength);
        f.min_distance = narrow(self.min_distance);
        f
    }
}

impl Default for HubGravityParams {
    fn default() -> Self {
        Self::of(&HubGravity::default())
    }
}

impl DepthGravityParams {
    pub fn of(f: &DepthGravity) -> Self {
        Self {
            depths: None,
            direction: wide2(f.direction),
            spacing: wide(f.spacing),
            strength: wide(f.strength),
        }
    }

    pub(super) fn build(&self, path: &str) -> Result<DepthGravity, SpecError> {
        need(self.depths.as_ref(), path, "depths")?;
        let mut f = DepthGravity::new(std::iter::empty());
        f.direction = narrow2(self.direction);
        f.spacing = narrow(self.spacing);
        f.strength = narrow(self.strength);
        Ok(f)
    }
}

impl Default for DepthGravityParams {
    fn default() -> Self {
        Self::of(&DepthGravity::new(std::iter::empty()))
    }
}

impl GridSnapParams {
    pub fn of(f: &GridSnap) -> Self {
        Self {
            cell: wide(f.cell),
            strength: wide(f.strength),
        }
    }

    pub(super) fn build(&self) -> GridSnap {
        GridSnap {
            cell: narrow(self.cell),
            strength: narrow(self.strength),
        }
    }
}

impl Default for GridSnapParams {
    fn default() -> Self {
        Self::of(&GridSnap::default())
    }
}

impl GravityLocusParams {
    pub fn of(f: &GravityLocus) -> Self {
        Self {
            target: Some(wide2(f.target)),
            strength: wide(f.strength),
            oscillation: f.oscillation.map(wide2),
        }
    }

    pub(super) fn build(&self, path: &str) -> Result<GravityLocus, SpecError> {
        let target = narrow2(need(self.target, path, "target")?);
        let mut f = match self.oscillation {
            None => GravityLocus::at(target),
            Some((amplitude, period)) => {
                GravityLocus::tidal(target, narrow(amplitude), narrow(period))
            },
        };
        f.strength = narrow(self.strength);
        Ok(f)
    }
}

impl Default for GravityLocusParams {
    fn default() -> Self {
        let mut params = Self::of(&GravityLocus::at((0.0, 0.0)));
        params.target = None;
        params
    }
}
