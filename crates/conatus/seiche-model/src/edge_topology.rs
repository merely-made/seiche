// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

//! Counted, ordered topology without relation taxonomy or a new force law.
//!
//! Only adjacent equal ordered pairs share a run. Merging distant occurrences
//! or multiplying a force by its count would change floating-point addition
//! order; reversing endpoints would also change directed laws. Expanded reads
//! therefore reproduce every original row, in order.

use std::iter::FusedIterator;
use std::num::NonZeroUsize;

use crate::NodeKey;

/// Consecutive occurrences of one ordered displayed endpoint pair.
/// This is a storage run, not a canonical relationship or a force attribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdgeRun {
    endpoints: (NodeKey, NodeKey),
    count: NonZeroUsize,
}

impl EdgeRun {
    pub fn new(endpoints: (NodeKey, NodeKey), count: NonZeroUsize) -> Self {
        Self { endpoints, count }
    }

    pub fn endpoints(&self) -> (NodeKey, NodeKey) {
        self.endpoints
    }

    pub fn count(&self) -> usize {
        self.count.get()
    }
}

/// A compact topology input shared by simulation, actor and layout view.
///
/// Counts mean repeated source rows, each with the existing unit contribution.
/// They are not arbitrary weights. No pair sorting or global deduplication is
/// performed. Hosts retain ownership of relation identity and visibility.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EdgeTopology {
    runs: Vec<EdgeRun>,
    len: usize,
}

impl EdgeTopology {
    /// Construct directly from counted runs, coalescing adjacent equal pairs.
    /// Panics if the logical row count cannot be represented by `usize`.
    pub fn from_runs(runs: impl IntoIterator<Item = EdgeRun>) -> Self {
        let mut topology = Self::default();
        for run in runs {
            topology.len = topology
                .len
                .checked_add(run.count())
                .expect("topology row overflow");
            if let Some(last) = topology.runs.last_mut()
                && last.endpoints == run.endpoints
            {
                last.count = NonZeroUsize::new(last.count() + run.count()).unwrap();
            } else {
                topology.runs.push(run);
            }
        }
        topology
    }

    /// Effective source-row count, including all multiplicity and self rows.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Stored ordered-run count; this is not the number of unique links.
    pub fn run_count(&self) -> usize {
        self.runs.len()
    }

    pub fn runs(&self) -> &[EdgeRun] {
        &self.runs
    }

    /// Visit every logical row in its original order, without allocating.
    pub fn iter(&self) -> EdgeIter<'_> {
        EdgeIter {
            runs: self.runs.iter(),
            current: None,
            remaining: self.len,
        }
    }
}

impl FromIterator<(NodeKey, NodeKey)> for EdgeTopology {
    fn from_iter<T: IntoIterator<Item = (NodeKey, NodeKey)>>(iter: T) -> Self {
        Self::from_runs(
            iter.into_iter()
                .map(|pair| EdgeRun::new(pair, NonZeroUsize::MIN)),
        )
    }
}

impl<'a> IntoIterator for &'a EdgeTopology {
    type Item = &'a (NodeKey, NodeKey);
    type IntoIter = EdgeIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Allocation-free expanded reading of an [`EdgeTopology`].
#[derive(Clone, Debug)]
pub struct EdgeIter<'a> {
    runs: std::slice::Iter<'a, EdgeRun>,
    current: Option<(&'a (NodeKey, NodeKey), usize)>,
    remaining: usize,
}

impl<'a> Iterator for EdgeIter<'a> {
    type Item = &'a (NodeKey, NodeKey);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_none() {
            let run = self.runs.next()?;
            self.current = Some((&run.endpoints, run.count()));
        }
        let (pair, count) = self.current.as_mut().unwrap();
        let pair = *pair;
        *count -= 1;
        if *count == 0 {
            self.current = None;
        }
        self.remaining -= 1;
        Some(pair)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for EdgeIter<'_> {}
impl FusedIterator for EdgeIter<'_> {}

#[cfg(test)]
mod tests;
