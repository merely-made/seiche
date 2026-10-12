// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Latest diagnostic membership frame; callbacks already ran on the owner thread.
use crate::CouplingMembershipFrame;
use std::sync::{Arc, Mutex};
type Slot = Arc<Mutex<Option<(u64, CouplingMembershipFrame)>>>;
#[derive(Default)]
pub(super) struct MembershipMailbox(Slot);
pub struct CouplingMembershipDelivery {
    generation: u64,
    slot: Slot,
}
impl MembershipMailbox {
    pub(super) fn publish(
        &self,
        generation: u64,
        frame: Option<&CouplingMembershipFrame>,
    ) -> Option<CouplingMembershipDelivery> {
        *self.0.lock().expect("membership mailbox poisoned") =
            frame.cloned().map(|frame| (generation, frame));
        frame.map(|_| CouplingMembershipDelivery {
            generation,
            slot: Arc::clone(&self.0),
        })
    }
}
impl CouplingMembershipDelivery {
    pub fn read(&self) -> Option<CouplingMembershipFrame> {
        self.slot
            .lock()
            .expect("membership mailbox poisoned")
            .as_ref()
            .filter(|(generation, _)| *generation == self.generation)
            .map(|(_, frame)| frame.clone())
    }
}
