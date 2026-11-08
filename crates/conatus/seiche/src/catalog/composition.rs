use std::collections::HashMap;

use seiche_model::{
    ForceSetDescriptor, GroupSource, PhysicsComposition, PhysicsGrouping, PhysicsLaw,
    PhysicsOverlay,
};

use super::{LawInputs, LawSources};
use crate::{Force, Grouped, Partition, Weighted};

/// A law's repulsion terms alone, each its own force: a single-term force
/// as it is, a term of a bundle by its isolation. Every force law in the
/// catalog carries one (`NodeExclusion` at least).
fn repulsion_terms(forces: Vec<Box<dyn Force>>) -> Vec<Box<dyn Force>> {
    let is_repulsion = |t: &crate::Term| matches!(t.kernel, crate::Kernel::Repulsion { .. });
    let mut out = Vec::new();
    for force in forces {
        let terms = force.terms();
        if terms.len() == 1 {
            if is_repulsion(&terms[0]) {
                out.push(force);
            }
            continue;
        }
        for (t, term) in terms.iter().enumerate() {
            if is_repulsion(term)
                && let Some(alone) = force.isolate(t)
            {
                out.push(alone);
            }
        }
    }
    out
}

impl LawInputs<'_> {
    /// `law`'s forces, each at `weight`.
    fn weighted_law(
        &self,
        law: PhysicsLaw,
        weight: f32,
        sources: LawSources,
    ) -> Vec<Box<dyn Force>> {
        self.law_forces(law, sources)
            .into_iter()
            .map(|force| {
                Box::new(Weighted::new(force, weight).expect("a force law, checked on set"))
                    as Box<dyn Force>
            })
            .collect()
    }

    /// The grouped composition over these inputs: the outer law's repulsion
    /// built over the groups' graph at the outer weight (F71), an inner law
    /// built over each group's members and the edges inside it.
    pub fn grouped_force(&self, grouping: &PhysicsGrouping, sources: LawSources) -> Box<dyn Force> {
        let groups = match &grouping.groups {
            GroupSource::Channel(source) => self.groups(*source),
            GroupSource::Given(groups) => groups.clone(),
        };
        let partition = Partition::new(groups, self.edges());
        let (keys, edges) = partition.outer_graph();
        let outer = repulsion_terms(
            LawInputs::from_parts(keys, edges, HashMap::new()).law_forces(grouping.outer, sources),
        )
        .into_iter()
        .map(|force| {
            Box::new(
                Weighted::new(force, grouping.outer_weight).expect("a force law, checked on set"),
            ) as Box<dyn Force>
        })
        .collect();
        let inner = (0..partition.len())
            .map(|g| {
                let (members, edges) = partition.group(g);
                let sites = members
                    .iter()
                    .filter_map(|k| Some((*k, self.sites().get(k)?.clone())))
                    .collect();
                LawInputs::from_parts(members.to_vec(), edges.to_vec(), sites)
                    .law_forces(grouping.inner, sources)
            })
            .collect();
        Box::new(Grouped::new(partition, outer, inner))
    }
}

impl LawInputs<'_> {
    /// The law slot's forces, `law` or `composition` when one is set, then
    /// the overlays: what a canvas runs, and a board mirroring its stage.
    pub fn composed(
        &self,
        law: PhysicsLaw,
        composition: Option<&PhysicsComposition>,
        overlays: &[PhysicsOverlay],
        sources: LawSources,
    ) -> Vec<Box<dyn Force>> {
        let mut forces = match composition {
            None => self.law_forces_taking(law, sources, overlays),
            Some(PhysicsComposition::Mix(laws)) => laws
                .iter()
                .flat_map(|(law, weight)| self.weighted_law(*law, *weight, sources))
                .collect(),
            Some(PhysicsComposition::Grouped(grouping)) => {
                vec![self.grouped_force(grouping, sources)]
            },
        };
        forces.extend(
            overlays
                .iter()
                .map(|overlay| self.overlay_force(*overlay, sources)),
        );
        forces
    }
}

/// The forces a descriptor names: the law slot (the law, or the composition),
/// then the overlays, built over its resolved inputs. The same builders as
/// [`LawInputs::composed`], so the forces are those a host would build in
/// process from the same inputs.
pub fn build_force_set(descriptor: &ForceSetDescriptor) -> Vec<Box<dyn Force>> {
    LawInputs::from_data(&descriptor.inputs).composed(
        descriptor.law,
        descriptor.composition.as_ref(),
        &descriptor.overlays,
        descriptor.sources,
    )
}
