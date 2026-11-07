// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

use super::*;

fn rows() -> Vec<(NodeKey, NodeKey)> {
    let [a, b, c, missing] = [0, 1, 2, 99].map(NodeKey::new);
    vec![
        (a, b),
        (a, b),
        (a, b),
        (b, a),
        (b, c),
        (b, c),
        (a, b),
        (a, a),
        (a, missing),
    ]
}

#[test]
fn runs_preserve_direction_interleaving_self_rows_and_multiplicity() {
    let original = rows();
    let topology: EdgeTopology = original.iter().copied().collect();
    assert_eq!(topology.len(), 9);
    assert_eq!(topology.run_count(), 6);
    assert_eq!(
        topology
            .runs()
            .iter()
            .map(EdgeRun::count)
            .collect::<Vec<_>>(),
        [3, 1, 2, 1, 1, 1]
    );
    assert_eq!(topology.iter().copied().collect::<Vec<_>>(), original);
    let mut iter = topology.iter();
    for (i, row) in original.iter().enumerate() {
        assert_eq!(iter.len(), original.len() - i);
        assert_eq!(iter.next(), Some(row));
    }
    assert_eq!(iter.len(), 0);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);
    assert!(EdgeTopology::default().is_empty());
}
