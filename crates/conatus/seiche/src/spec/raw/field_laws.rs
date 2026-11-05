// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;

impl GravityParams {
    pub fn of(f: &Gravity) -> Self {
        Self {
            masses: None,
            counter_damping: Some(CounterDampingParam::of(f.counter_damping)),
            strength: wide(f.strength),
            softening: wide(f.softening),
            orbital_kick: wide(f.orbital_kick),
            radial_floor: wide(f.radial_floor),
        }
    }

    pub(super) fn build(&self, path: &str) -> Result<Gravity, SpecError> {
        need(self.masses.as_ref(), path, "masses")?;
        let counter = need(self.counter_damping, path, "counter_damping")?;
        let mut f = Gravity::new(std::iter::empty(), counter.build());
        f.strength = narrow(self.strength);
        f.softening = narrow(self.softening);
        f.orbital_kick = narrow(self.orbital_kick);
        f.radial_floor = narrow(self.radial_floor);
        Ok(f)
    }
}

impl Default for GravityParams {
    fn default() -> Self {
        let mut params = Self::of(&Gravity::new(std::iter::empty(), CounterDamping::Off));
        params.counter_damping = None;
        params
    }
}

impl ParticleLifeParams {
    pub fn of(f: &ParticleLife) -> Self {
        Self {
            kinds: None,
            matrix: None,
            radius: wide(f.radius),
            core: wide(f.core),
            strength: wide(f.strength),
            gravity: wide(f.gravity),
        }
    }

    pub(super) fn build(&self, path: &str, seed: u64) -> Result<ParticleLife, SpecError> {
        need(self.kinds.as_ref(), path, "kinds")?;
        let mut f = match &self.matrix {
            None => ParticleLife::seeded(std::iter::empty(), SEEDED_KINDS, seed),
            Some(matrix) => {
                let k = (matrix.len() as f64).sqrt().round() as usize;
                if k == 0 || k * k != matrix.len() || k > usize::from(u8::MAX) + 1 {
                    return Err(SpecError::Invalid {
                        path: path.to_string(),
                        reason: format!(
                            "a kind matrix is square, k × k for 1 to 256 kinds; this one has {} entries",
                            matrix.len()
                        ),
                    });
                }
                ParticleLife::new(
                    std::iter::empty(),
                    k,
                    matrix.iter().map(|x| narrow(*x)).collect(),
                )
            },
        };
        f.radius = narrow(self.radius);
        f.core = narrow(self.core);
        f.strength = narrow(self.strength);
        f.gravity = narrow(self.gravity);
        Ok(f)
    }
}

impl Default for ParticleLifeParams {
    fn default() -> Self {
        Self::of(&ParticleLife::new(std::iter::empty(), 1, Vec::new()))
    }
}

impl BoidsParams {
    pub fn of(f: &Boids) -> Self {
        Self {
            separation: wide(f.separation),
            separation_radius: wide(f.separation_radius),
            alignment: wide(f.alignment),
            cohesion: wide(f.cohesion),
            cruise_speed: wide(f.cruise_speed),
            cruise: wide(f.cruise),
            gravity: wide(f.gravity),
        }
    }

    pub(super) fn build(&self) -> Boids {
        Boids {
            separation: narrow(self.separation),
            separation_radius: narrow(self.separation_radius),
            alignment: narrow(self.alignment),
            cohesion: narrow(self.cohesion),
            cruise_speed: narrow(self.cruise_speed),
            cruise: narrow(self.cruise),
            gravity: narrow(self.gravity),
        }
    }
}

impl Default for BoidsParams {
    fn default() -> Self {
        Self::of(&Boids::default())
    }
}

impl KuramotoParams {
    pub fn of(f: &Kuramoto) -> Self {
        Self {
            radii: None,
            natural_frequency: wide(f.natural_frequency),
            coupling: wide(f.coupling),
            default_radius: wide(f.default_radius),
            centre: wide2(f.centre),
            stiffness: wide(f.stiffness),
        }
    }

    pub(super) fn build(&self) -> Kuramoto {
        let mut f = Kuramoto::new(std::iter::empty());
        f.natural_frequency = narrow(self.natural_frequency);
        f.coupling = narrow(self.coupling);
        f.default_radius = narrow(self.default_radius);
        f.centre = narrow2(self.centre);
        f.stiffness = narrow(self.stiffness);
        f
    }
}

impl Default for KuramotoParams {
    fn default() -> Self {
        Self::of(&Kuramoto::new(std::iter::empty()))
    }
}

impl MagneticSpringParams {
    pub fn of(f: &MagneticSpring) -> Self {
        Self {
            field: wide2(f.field),
            stiffness: wide(f.stiffness),
            rest_length: wide(f.rest_length),
            torque: wide(f.torque),
            repulsion: wide(f.repulsion),
            min_distance: wide(f.min_distance),
        }
    }

    pub(super) fn build(&self) -> MagneticSpring {
        MagneticSpring {
            field: narrow2(self.field),
            stiffness: narrow(self.stiffness),
            rest_length: narrow(self.rest_length),
            torque: narrow(self.torque),
            repulsion: narrow(self.repulsion),
            min_distance: narrow(self.min_distance),
        }
    }
}

impl Default for MagneticSpringParams {
    fn default() -> Self {
        Self::of(&MagneticSpring::default())
    }
}
