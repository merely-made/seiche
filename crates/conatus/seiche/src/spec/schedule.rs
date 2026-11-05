// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;

/// One stage of a schedule.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(deny_unknown_fields)
)]
pub struct Stage {
    pub node: Node,
    pub stop: Stop,
    /// Capture the layout as the stage ends, taken with this role (F27).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub capture: Option<Role>,
}

/// How a stage ends, as the canvas's schedules end one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "kebab-case")
)]
pub enum Stop {
    /// The bodies rest (F46's speed floor).
    Rest,
    /// After this many frames.
    Frames(u32),
    /// The stage's law stops by its own test, or rests.
    LawDone,
}

/// How the spec is realized (F103: integration only until G5).
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "kebab-case", deny_unknown_fields)
)]
pub enum Realization {
    /// Rapier integrates; `damping` is the bodies' linear damping, the
    /// host's when omitted.
    Integrate {
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "Option::is_none")
        )]
        damping: Option<f64>,
    },
}

impl Default for Realization {
    fn default() -> Self {
        Realization::Integrate { damping: None }
    }
}

impl Realization {
    pub(super) fn damping(&self) -> Option<f64> {
        match self {
            Realization::Integrate { damping } => *damping,
        }
    }
}

/// What the spec's motion acts around (F105): an arrangement, the anchored
/// role's pull, and each item's role, resolved item, then group, then the
/// default. With no target an item takes F23's initial position (seeded).
/// *Reading, not ruled:* an item id absent from the graph falls to its group
/// or the default and is counted, not refused, since a spec that travels
/// meets other graphs by design. G4a carries the target; G4b resolves it.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(deny_unknown_fields)
)]
pub struct Target {
    /// The arrangement id (a cartography adapter, or the canvas's own).
    pub arrangement: String,
    /// The anchored role's return stiffness.
    #[cfg_attr(feature = "serde", serde(default = "default_pull"))]
    pub anchored_pull: f64,
    #[cfg_attr(feature = "serde", serde(default))]
    pub default_role: Role,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub groups: Option<GroupRoles>,
    /// Roles by item id: a graph node's UUID, which resolves only on the
    /// graph that minted it.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "BTreeMap::is_empty")
    )]
    pub items: BTreeMap<String, Role>,
}

#[cfg(feature = "serde")]
fn default_pull() -> f64 {
    f64::from(crate::DEFAULT_ANCHOR_STIFFNESS)
}

/// Group roles, keyed by the groups of one `groups.*` channel (F49, F105).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(deny_unknown_fields)
)]
pub struct GroupRoles {
    pub channel: String,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "BTreeMap::is_empty")
    )]
    pub roles: BTreeMap<String, Role>,
}

impl Target {
    pub(super) fn check(&self) -> Result<(), SpecError> {
        check_number(Some(self.anchored_pull), "target", "anchored_pull")?;
        if let Some(groups) = &self.groups
            && !groups.channel.starts_with("groups.")
        {
            return Err(SpecError::Invalid {
                path: "target.groups".into(),
                reason: format!(
                    "group roles are keyed by a groups.* channel, not {}",
                    groups.channel
                ),
            });
        }
        Ok(())
    }
}

/// A signature a spec preserves, with its bounds: carried, evaluated by G6
/// and the receipts, not here (F102), and only at the root (F111).
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(deny_unknown_fields)
)]
pub struct Bar {
    pub observable: Observable,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub at_least: Option<f64>,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub at_most: Option<f64>,
}

/// Where a preset id is read: the law slot, or an overlay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresetAt {
    Law,
    Overlay,
}

impl PresetAt {
    pub(super) fn word(self) -> &'static str {
        match self {
            PresetAt::Law => "law",
            PresetAt::Overlay => "overlay",
        }
    }
}
