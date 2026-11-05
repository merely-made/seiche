// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;

impl Recipe {
    /// The catalog's Orbit as built today.
    pub fn catalog() -> Self {
        let g = Gravity::new([], CounterDamping::Full);
        Self {
            exclusion: Some(NodeExclusion::default()),
            strength: g.strength,
            softening: g.softening,
            kick: true,
            gravitation: true,
            counter: Counter::Full,
            radial_floor: g.radial_floor,
            centring: None,
        }
    }

    /// Exclusion whose reach ends at `diameters` node diameters.
    pub fn reach(self, diameters: f32) -> Self {
        Self {
            exclusion: Some(NodeExclusion {
                cutoff: diameters * TOUCH,
                ..NodeExclusion::default()
            }),
            ..self
        }
    }

    pub fn parts(&self, fixture: &Fixture) -> Vec<Part> {
        let gravity = |strength: f32, kick: f32, counter: CounterDamping| Gravity {
            strength,
            softening: self.softening,
            orbital_kick: kick,
            counter_damping: counter,
            radial_floor: self.radial_floor,
            ..Gravity::new(fixture.mass_map(), counter)
        };
        let mut parts = Vec::new();
        if let Some(exclusion) = self.exclusion {
            parts.push(Part::new("exclusion", exclusion));
        }
        if self.kick {
            parts.push(Part {
                label: "kick",
                force: Box::new(gravity(self.strength, 1.0, CounterDamping::Off)),
                writes_only: true,
            });
        }
        if self.gravitation {
            parts.push(Part::new(
                "gravitation",
                gravity(self.strength, 0.0, CounterDamping::Off),
            ));
        }
        match self.counter {
            Counter::Full => parts.push(Part::new(
                "counter-damping",
                gravity(0.0, 0.0, CounterDamping::Full),
            )),
            Counter::Law => parts.push(Part::new(
                "counter-damping",
                gravity(0.0, 0.0, CounterDamping::Tangential),
            )),
            Counter::Brake => parts.push(Part::new(
                "counter-damping",
                Brake {
                    masses: fixture.mass_map().into_iter().collect(),
                    strength: self.strength,
                    softening: self.softening,
                },
            )),
            Counter::Cap => parts.push(Part::new("counter-damping", Cap::default())),
            Counter::Tangential => parts.push(Part::new(
                "counter-damping",
                Tangential {
                    masses: fixture.mass_map().into_iter().collect(),
                    own: None,
                    prograde: false,
                },
            )),
            Counter::Prograde => parts.push(Part::new(
                "counter-damping",
                Tangential {
                    masses: fixture.mass_map().into_iter().collect(),
                    own: None,
                    prograde: true,
                },
            )),
            Counter::Own(rate) => {
                parts.push(Part::new(
                    "counter-damping",
                    gravity(0.0, 0.0, CounterDamping::Full),
                ));
                parts.push(Part::new(
                    "radial damping",
                    Tangential {
                        masses: fixture.mass_map().into_iter().collect(),
                        own: Some(rate),
                        prograde: false,
                    },
                ));
            },
        }
        if let Some(strength) = self.centring {
            parts.push(Part::new("centring", Boundary { strength }));
        }
        parts
    }
}
