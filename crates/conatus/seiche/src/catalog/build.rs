use seiche_model::{PhysicsLaw, PhysicsMassSource, PhysicsOverlay};

use super::LawInputs;
use super::{
    CHARGE_STRENGTH, DENSITY_MAX_PASSES, DENSITY_MIN_PASSES, DENSITY_PATIENCE, DENSITY_RESOLUTION,
    DENSITY_STOP, LawSources, ORBIT_CENTRING, ORBIT_EXCLUSION_REACH, SKELETON_STIFFNESS,
};
use crate::{
    Anneal, BarnesHutRepulsion, Boids, Boundary, CounterDamping, DegreeRepulsion, Density,
    DepthGravity, DomainCluster, EdgeSpring, Force, Gravity, GravityLocus, GridSnap, Hold,
    HubGravity, Kuramoto, LinLogForce, MagneticSpring, NodeExclusion, NodeKey, ParticleLife,
    StressSpring,
};

/// Density at the catalog's defaults (ruled 2026-10-02 and 2026-10-03): 64²,
/// one-second passes, blur 0.25 spacings, at least 60 passes, then passes
/// repeated until the shift test stops them, the pass cap as the fallback.
pub fn density_law(masses: Vec<(NodeKey, f32)>) -> Density {
    let mut law = Density::new(masses, DENSITY_RESOLUTION);
    law.seconds = 1.0;
    law.initial_blur = 0.25;
    law.stop = DENSITY_STOP;
    law.patience = DENSITY_PATIENCE;
    law.max_passes = DENSITY_MAX_PASSES;
    law.min_passes = DENSITY_MIN_PASSES;
    law
}

impl LawInputs<'_> {
    /// The law's forces when `overlays` join it: Density's conversion is on
    /// once it takes an overlay, and off alone (F73).
    pub fn law_forces_taking(
        &self,
        law: PhysicsLaw,
        sources: LawSources,
        overlays: &[PhysicsOverlay],
    ) -> Vec<Box<dyn Force>> {
        if law == PhysicsLaw::Density && !overlays.is_empty() {
            let mut density = density_law(self.masses(sources.mass));
            density.converts = true;
            return vec![Box::new(density)];
        }
        self.law_forces(law, sources)
    }

    /// Build the law's own forces against these inputs.
    pub fn law_forces(&self, law: PhysicsLaw, sources: LawSources) -> Vec<Box<dyn Force>> {
        match law {
            PhysicsLaw::Springs => vec![
                Box::new(NodeExclusion::default()),
                Box::new(EdgeSpring::default()),
                Box::new(Boundary::default()),
            ],
            PhysicsLaw::Charge => vec![
                Box::new(BarnesHutRepulsion {
                    strength: CHARGE_STRENGTH,
                    ..BarnesHutRepulsion::default()
                }),
                Box::new(EdgeSpring::default()),
                Box::new(Boundary::default()),
            ],
            PhysicsLaw::Stress => vec![
                Box::new(NodeExclusion::default()),
                Box::new(StressSpring::from_weighted_distances(
                    self.weighted_distances(),
                    EdgeSpring::default().rest_length,
                )),
                Box::new(Boundary::default()),
            ],
            PhysicsLaw::Energy => vec![
                Box::new(NodeExclusion::default()),
                Box::new(LinLogForce::default()),
            ],
            PhysicsLaw::Orbit => vec![
                Box::new(NodeExclusion {
                    cutoff: ORBIT_EXCLUSION_REACH,
                    ..NodeExclusion::default()
                }),
                Box::new(Gravity::new(
                    self.masses(sources.mass),
                    CounterDamping::Tangential,
                )),
                Box::new(Boundary {
                    strength: ORBIT_CENTRING,
                }),
            ],
            PhysicsLaw::Kinds => {
                let (kinds, kind_count) = self.kinds(sources.kind);
                vec![
                    Box::new(NodeExclusion::default()),
                    Box::new(ParticleLife::seeded(kinds, kind_count, sources.seed)),
                ]
            },
            PhysicsLaw::Flock => vec![
                Box::new(NodeExclusion::default()),
                Box::new(Boids::default()),
            ],
            PhysicsLaw::Sync => {
                // Every node on one ring; communities become arcs of it.
                let radii = self.nodes.iter().map(|&key| (key, 240.0));
                vec![
                    Box::new(NodeExclusion::default()),
                    Box::new(Kuramoto::new(radii)),
                ]
            },
            PhysicsLaw::Flow => vec![
                Box::new(NodeExclusion::default()),
                Box::new(MagneticSpring::default()),
                Box::new(Boundary::default()),
            ],
            PhysicsLaw::Anneal => vec![Box::new(Anneal::seeded(sources.seed))],
            // Held, not empty: with no force at all rapier's contact solver
            // blasts an overlapping seed apart (the Still receipt found it).
            PhysicsLaw::Still => vec![Box::new(Hold)],
            PhysicsLaw::Density => vec![Box::new(density_law(self.masses(sources.mass)))],
        }
    }

    /// Build one overlay's force against these inputs.
    pub fn overlay_force(&self, overlay: PhysicsOverlay, sources: LawSources) -> Box<dyn Force> {
        match overlay {
            PhysicsOverlay::DegreeRepulsion => match sources.mass {
                PhysicsMassSource::Degree => Box::new(DegreeRepulsion::default()),
                PhysicsMassSource::PageRank => {
                    Box::new(DegreeRepulsion::default().with_weights(self.page_rank_weights()))
                },
            },
            PhysicsOverlay::DomainCluster => {
                Box::new(DomainCluster::new(self.groups(sources.groups)))
            },
            PhysicsOverlay::HubGravity => match sources.mass {
                PhysicsMassSource::Degree => Box::new(HubGravity::default()),
                PhysicsMassSource::PageRank => {
                    Box::new(HubGravity::default().with_weights(self.page_rank_weights()))
                },
            },
            PhysicsOverlay::DepthGravity => {
                Box::new(DepthGravity::new(self.depths(sources.depth, sources.focus)))
            },
            PhysicsOverlay::GridSnap => Box::new(GridSnap::default()),
            PhysicsOverlay::GravityLocus => Box::new(GravityLocus::at((0.0, 0.0))),
            PhysicsOverlay::Tide => Box::new(GravityLocus::tidal((0.0, 0.0), 240.0, 24.0)),
            PhysicsOverlay::Skeleton => Box::new(
                StressSpring::from_distances(
                    self.skeleton_edges().into_iter().map(|(a, b)| (a, b, 1)),
                    EdgeSpring::default().rest_length,
                )
                .with_stiffness(SKELETON_STIFFNESS),
            ),
        }
    }
}
