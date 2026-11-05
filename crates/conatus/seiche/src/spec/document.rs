// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;

impl DynamicsSpec {
    /// A spec of `root` at this version, the default seed, integration as
    /// the host damps it, no channels, target or bars.
    pub fn new(root: Node) -> Self {
        Self {
            version: DYNAMICS_SPEC_VERSION,
            seed: DEFAULT_SEED,
            root,
            channels: BTreeMap::new(),
            realization: Realization::default(),
            target: None,
            bars: Vec::new(),
        }
    }

    /// How a reader at version `reader` reads `wire`: a newer spec is
    /// refused; an older one is stamped with the reader's version.
    #[cfg(feature = "serde")]
    pub(super) fn read(wire: SpecWire, reader: u16) -> Result<Self, SpecVersionError> {
        if wire.version > reader {
            return Err(SpecVersionError {
                found: wire.version,
                reader,
            });
        }
        Ok(Self {
            version: reader,
            seed: wire.seed,
            root: wire.root,
            channels: wire.channels,
            realization: wire.realization,
            target: wire.target,
            bars: wire.bars,
        })
    }

    /// The tree's depth (see [`MAX_DEPTH`]).
    pub fn depth(&self) -> usize {
        self.root.depth()
    }

    /// The derived fields of every leaf, after the refusals the module docs
    /// list; `catalog` builds the presets.
    pub fn derive(&self, catalog: &dyn SpecCatalog) -> Result<Derived, SpecError> {
        let depth = self.depth();
        if depth > MAX_DEPTH {
            return Err(SpecError::Depth {
                depth,
                max: MAX_DEPTH,
            });
        }
        check_number(self.realization.damping(), "realization", "damping")?;
        if let Some(target) = &self.target {
            target.check()?;
        }
        let mut leaves = Vec::new();
        self.root
            .declare("root", PresetAt::Law, self.seed, catalog, &mut leaves)?;
        let by_path: HashMap<&str, &DerivedLeaf> = leaves
            .iter()
            .map(|leaf| (leaf.path.as_str(), leaf))
            .collect();
        self.root.admit("root", Context::Root, catalog, &by_path)?;
        Ok(Derived { depth, leaves })
    }
}
