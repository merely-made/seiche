// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;

impl Node {
    /// A law or overlay preset at weight 1, nothing else set.
    pub fn preset(id: impl Into<String>) -> Node {
        Node::Preset {
            id: id.into(),
            weight: 1.0,
            terms: BTreeMap::new(),
            rung: None,
            overlays: Vec::new(),
        }
    }

    /// A raw term at weight 1, nothing else set.
    pub fn raw(term: RawTerm) -> Node {
        Node::Raw {
            term,
            weight: 1.0,
            terms: BTreeMap::new(),
            rung: None,
            overlays: Vec::new(),
        }
    }

    /// The node's weight, a multiplier (F96).
    pub fn weight(&self) -> f64 {
        match self {
            Node::Preset { weight, .. }
            | Node::Raw { weight, .. }
            | Node::Mix { weight, .. }
            | Node::Grouped { weight, .. }
            | Node::Schedule { weight, .. } => *weight,
        }
    }

    /// The overlays the node carries (F109).
    pub fn overlays(&self) -> &[Node] {
        match self {
            Node::Preset { overlays, .. }
            | Node::Raw { overlays, .. }
            | Node::Mix { overlays, .. }
            | Node::Grouped { overlays, .. }
            | Node::Schedule { overlays, .. } => overlays,
        }
    }

    /// The overlays, to add to.
    pub fn overlays_mut(&mut self) -> &mut Vec<Node> {
        match self {
            Node::Preset { overlays, .. }
            | Node::Raw { overlays, .. }
            | Node::Mix { overlays, .. }
            | Node::Grouped { overlays, .. }
            | Node::Schedule { overlays, .. } => overlays,
        }
    }

    /// This node's children, with the path step to each.
    pub(super) fn children(&self) -> Vec<(String, &Node)> {
        let mut out: Vec<(String, &Node)> = match self {
            Node::Preset { .. } | Node::Raw { .. } => Vec::new(),
            Node::Mix { parts, .. } => parts
                .iter()
                .enumerate()
                .map(|(i, part)| (format!("parts[{i}]"), part))
                .collect(),
            Node::Grouped { outer, inner, .. } => {
                vec![("outer".into(), &**outer), ("inner".into(), &**inner)]
            },
            Node::Schedule { stages, .. } => stages
                .iter()
                .enumerate()
                .map(|(i, stage)| (format!("stages[{i}].node"), &stage.node))
                .collect(),
        };
        out.extend(
            self.overlays()
                .iter()
                .enumerate()
                .map(|(i, overlay)| (format!("overlays[{i}]"), overlay)),
        );
        out
    }

    /// Nodes on the longest path from here to a leaf.
    pub fn depth(&self) -> usize {
        // Iterative, so a hostile tree cannot exhaust the stack here.
        let mut deepest = 0;
        let mut stack = vec![(self, 1usize)];
        while let Some((node, depth)) = stack.pop() {
            deepest = deepest.max(depth);
            stack.extend(node.children().into_iter().map(|(_, c)| (c, depth + 1)));
        }
        deepest
    }

    /// Pass one: build each leaf, refuse what cannot be built or named, and
    /// record the derived fields.
    pub(super) fn declare(
        &self,
        path: &str,
        at: PresetAt,
        seed: u64,
        catalog: &dyn SpecCatalog,
        leaves: &mut Vec<DerivedLeaf>,
    ) -> Result<(), SpecError> {
        check_number(Some(self.weight()), path, "weight")?;
        match self {
            Node::Preset {
                id, terms, rung, ..
            } => {
                reserved(rung, path)?;
                let forces = catalog
                    .preset(id, at)
                    .ok_or_else(|| SpecError::UnknownPreset {
                        path: path.into(),
                        id: id.clone(),
                        at,
                    })?;
                leaves.push(leaf(path, LeafKind::Preset(id.clone()), &forces, terms)?);
            },
            Node::Raw {
                term, terms, rung, ..
            } => {
                reserved(rung, path)?;
                let forces = vec![term.build(path, seed)?];
                leaves.push(leaf(path, LeafKind::Raw(term.kind()), &forces, terms)?);
            },
            Node::Mix { parts, .. } if parts.is_empty() => {
                return Err(invalid(path, "a mix holds at least one part"));
            },
            Node::Grouped { partition, .. } if partition.is_empty() => {
                return Err(SpecError::Missing {
                    path: path.into(),
                    field: "partition",
                });
            },
            Node::Schedule { stages, .. } if stages.is_empty() => {
                return Err(invalid(path, "a schedule holds at least one stage"));
            },
            Node::Mix { .. } | Node::Grouped { .. } | Node::Schedule { .. } => {},
        }
        for (step, child) in self.children() {
            let child_at = if step.starts_with("overlays") {
                PresetAt::Overlay
            } else {
                PresetAt::Law
            };
            child.declare(&format!("{path}.{step}"), child_at, seed, catalog, leaves)?;
        }
        Ok(())
    }

    /// Pass two, in pre-order: currency first (F108), then whether the canvas
    /// runs the shape today (R1–R4).
    pub(super) fn admit(
        &self,
        path: &str,
        context: Context,
        catalog: &dyn SpecCatalog,
        leaves: &HashMap<&str, &DerivedLeaf>,
    ) -> Result<(), SpecError> {
        let weighted = matches!(
            context,
            Context::MixPart | Context::GroupOuter | Context::GroupInner
        );
        let currency = match self {
            Node::Preset { .. } | Node::Raw { .. } => {
                let leaf = leaves[path];
                if weighted && leaf.currency != Currency::Force {
                    return Err(by_currency(path, compose::UNWEIGHTED));
                }
                if leaf.terms.iter().any(|t| {
                    t.weight.is_some()
                        && !matches!(t.term.currency, Currency::Force | Currency::Integrator)
                }) {
                    return Err(by_currency(path, compose::UNWEIGHTED));
                }
                leaf.currency
            },
            Node::Mix { .. } | Node::Grouped { .. } => Currency::Force,
            Node::Schedule { .. } => Currency::Force,
        };
        if !matches!(self, Node::Schedule { .. }) {
            for (i, overlay) in self.overlays().iter().enumerate() {
                let overlay_path = format!("{path}.overlays[{i}]");
                if let Some(refusal) =
                    self.admission(currency, overlay, &overlay_path, catalog, leaves)
                {
                    return Err(SpecError::Currency {
                        path: overlay_path,
                        reason: refusal,
                    });
                }
            }
        }
        self.runnable(path, context)?;
        for (step, child) in self.children() {
            let child_context = match (self, step.as_str()) {
                (_, s) if s.starts_with("overlays") => Context::Overlay,
                (Node::Mix { .. }, _) => Context::MixPart,
                (Node::Grouped { .. }, "outer") => Context::GroupOuter,
                (Node::Grouped { .. }, _) => Context::GroupInner,
                (Node::Schedule { .. }, _) => Context::Stage,
                _ => unreachable!("leaves have no children but overlays"),
            };
            child.admit(&format!("{path}.{step}"), child_context, catalog, leaves)?;
        }
        Ok(())
    }

    /// Why this node refuses `overlay`, if it does: the catalog's word for a
    /// pair of presets, else currency, term by term.
    pub(super) fn admission(
        &self,
        currency: Currency,
        overlay: &Node,
        overlay_path: &str,
        catalog: &dyn SpecCatalog,
        leaves: &HashMap<&str, &DerivedLeaf>,
    ) -> Option<&'static str> {
        if let (Node::Preset { id: law, .. }, Node::Preset { id, .. }) = (self, overlay)
            && let Some(admission) = catalog.admits(law, id)
        {
            return admission.refusal();
        }
        let leaf = leaves.get(overlay_path)?;
        leaf.terms
            .iter()
            .map(|t| compose::admit(currency, &t.term, t.resident))
            .fold(Admission::Compose, Admission::and)
            .refusal()
    }

    /// Whether the canvas runs this node where it sits (the G4a checkpoint's
    /// R1–R4): a law preset with overlays; a mix of law presets; a grouping
    /// of law presets; a schedule of those at the root.
    pub(super) fn runnable(&self, path: &str, context: Context) -> Result<(), SpecError> {
        let refuse = |reason| {
            Err(SpecError::Unrunnable {
                path: path.into(),
                reason,
            })
        };
        let unit = self.weight() == 1.0;
        let has_overlays = !self.overlays().is_empty();
        match self {
            Node::Raw { .. } => refuse(RAW_UNBOUND),
            Node::Preset { terms, .. } => {
                if !terms.is_empty() {
                    return refuse(OVERRIDE_UNBOUND);
                }
                match context {
                    Context::Root | Context::Stage if !unit => refuse(WEIGHT_UNBOUND),
                    Context::Root | Context::Stage => Ok(()),
                    _ if has_overlays => refuse(OVERLAYS_UNBOUND),
                    Context::GroupInner | Context::Overlay if !unit => refuse(WEIGHT_UNBOUND),
                    _ => Ok(()),
                }
            },
            Node::Mix { .. } | Node::Grouped { .. } => match context {
                Context::Root | Context::Stage if !unit => refuse(WEIGHT_UNBOUND),
                Context::Root | Context::Stage => Ok(()),
                Context::MixPart | Context::GroupOuter | Context::GroupInner => {
                    refuse(NESTED_UNBOUND)
                },
                Context::Overlay => refuse(OVERLAY_UNBOUND),
            },
            Node::Schedule { .. } => match context {
                Context::Root if has_overlays => refuse(SCHEDULE_OVERLAYS),
                Context::Root if !unit => refuse(WEIGHT_UNBOUND),
                Context::Root => Ok(()),
                _ => refuse(SCHEDULE_NESTED),
            },
        }
    }
}
