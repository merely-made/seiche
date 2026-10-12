// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! One shared latest-result slot: queued layout updates cannot retain a trace
//! history. A packet superseded before it is read yields no evidence.
use crate::CouplingTrace;
use std::sync::{Arc, Mutex};

type Slot = Arc<Mutex<Option<(u64, CouplingTrace)>>>;

#[derive(Default)]
pub(super) struct TraceMailbox(Slot);

/// Opaque latest-result delivery accompanying an actor layout update. Its
/// contents may be superseded before folding; this must be treated as missing
/// evidence, not as an empty contribution or as another tick's trace.
pub struct CouplingTraceDelivery {
    generation: u64,
    slot: Slot,
}

impl TraceMailbox {
    pub(super) fn publish(
        &self,
        generation: u64,
        trace: Option<&CouplingTrace>,
    ) -> Option<CouplingTraceDelivery> {
        *self.0.lock().expect("trace mailbox poisoned") =
            trace.cloned().map(|trace| (generation, trace));
        trace.map(|_| CouplingTraceDelivery {
            generation,
            slot: Arc::clone(&self.0),
        })
    }
}

impl CouplingTraceDelivery {
    /// Read only this update's evidence, if it is still the latest publication.
    pub fn read(&self) -> Option<CouplingTrace> {
        self.slot
            .lock()
            .expect("trace mailbox poisoned")
            .as_ref()
            .filter(|(generation, _)| *generation == self.generation)
            .map(|(_, trace)| trace.clone())
    }

    #[cfg(test)]
    pub(super) fn fixture(generation: u64, trace: Option<&CouplingTrace>) -> Option<Self> {
        TraceMailbox::default().publish(generation, trace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;
    #[test]
    fn queued_packets_share_one_trace_and_superseded_evidence_is_unavailable() {
        let mailbox = TraceMailbox::default();
        let mut sim = crate::Simulation::new();
        sim.set_coupling_trace_limit(NonZeroUsize::new(1));
        sim.tick(0.25);
        let old = mailbox.publish(1, sim.coupling_trace()).unwrap();
        sim.tick(0.25);
        let new = mailbox.publish(2, sim.coupling_trace()).unwrap();
        assert!(Arc::ptr_eq(&old.slot, &new.slot));
        assert!(old.read().is_none());
        assert_eq!(new.read().unwrap().step, 1);
        assert!(mailbox.publish(3, None).is_none());
        assert!(new.read().is_none());
    }
}
