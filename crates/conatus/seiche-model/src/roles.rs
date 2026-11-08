// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Arrangement roles: what an arrangement's positions do once physics runs.
//!
//! An arrangement is positions, not motion; physics acts on it (dynamics
//! grammar plan, F11). Each item's position plays one of three roles, chosen
//! by use (F18, F19, F26):
//!
//! - **Seeded**: the position only starts the motion. The default (F23).
//! - **Anchored**: the item returns to its position, by `AnchorSpring` while
//!   physics runs. A host with physics off puts it back when a drag ends.
//! - **Pinned**: the item stays at its position. It is a kinematic body, so
//!   laws skip it and the rest of the graph accommodates it.
//!
//! A recipe sets a default role, a group may override it, and an item may
//! override that (F22, [`RoleTable`]). An axis bound to a data field is held
//! on that axis alone ([`Axes`], F28), which rapier realizes exactly by
//! locking the body's translation on it.
//!
//! `AnchorSpring` is `seiche::AnchorSpring`.

use std::collections::{BTreeMap, HashMap};

use crate::NodeKey;

/// The role an item's arrangement position plays. Serialized by its id
/// (`seeded`, `anchored`, `pinned`), as [`Role::id`] names it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "kebab-case")
)]
pub enum Role {
    /// The position seeds the motion and is not referred to again.
    #[default]
    Seeded,
    /// The item returns to its position.
    Anchored,
    /// The item stays at its position.
    Pinned,
}

impl Role {
    pub const ALL: [Role; 3] = [Role::Seeded, Role::Anchored, Role::Pinned];

    /// The stable id hosts save.
    pub fn id(self) -> &'static str {
        match self {
            Role::Seeded => "seeded",
            Role::Anchored => "anchored",
            Role::Pinned => "pinned",
        }
    }

    /// The plain label a picker shows.
    pub fn label(self) -> &'static str {
        match self {
            Role::Seeded => "Seeded",
            Role::Anchored => "Anchored",
            Role::Pinned => "Pinned",
        }
    }

    pub fn parse(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|role| role.id() == id)
    }
}

/// Which translation axes are held. An encoded axis is pinned on that axis
/// and physics moves only the others (F28).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Axes {
    pub x: bool,
    pub y: bool,
}

impl Axes {
    pub const NONE: Axes = Axes { x: false, y: false };
    pub const BOTH: Axes = Axes { x: true, y: true };

    pub fn any(self) -> bool {
        self.x || self.y
    }
}

/// Roles at three scopes: the recipe's default, a group's override, and an
/// item's override, the most specific winning (F22).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RoleTable {
    /// The recipe's role.
    pub default: Role,
    /// Overrides by group label.
    pub groups: BTreeMap<String, Role>,
    /// Overrides by item.
    pub items: HashMap<NodeKey, Role>,
}

impl RoleTable {
    /// A table with one role for everything.
    pub fn uniform(role: Role) -> Self {
        Self {
            default: role,
            ..Self::default()
        }
    }

    /// The role `key` plays, given the group it belongs to.
    pub fn role(&self, key: NodeKey, group: Option<&str>) -> Role {
        if let Some(role) = self.items.get(&key) {
            return *role;
        }
        group
            .and_then(|group| self.groups.get(group))
            .copied()
            .unwrap_or(self.default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_most_specific_scope_wins() {
        let (a, b, c) = (NodeKey::new(0), NodeKey::new(1), NodeKey::new(2));
        let mut table = RoleTable::uniform(Role::Anchored);
        table.groups.insert("site".into(), Role::Pinned);
        table.items.insert(c, Role::Seeded);
        assert_eq!(table.role(a, None), Role::Anchored, "the recipe default");
        assert_eq!(
            table.role(b, Some("site")),
            Role::Pinned,
            "a group override"
        );
        assert_eq!(
            table.role(c, Some("site")),
            Role::Seeded,
            "an item overrides its group"
        );
        assert_eq!(RoleTable::default().role(a, None), Role::Seeded, "F23");
        for role in Role::ALL {
            assert_eq!(Role::parse(role.id()), Some(role));
        }
    }
}
