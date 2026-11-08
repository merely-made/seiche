use super::*;

fn keys(n: usize) -> Vec<NodeKey> {
    (0..n).map(NodeKey::new).collect()
}

/// 0-1-2-3 path, 4 isolated, a double edge 2-3.
fn sample() -> (Vec<NodeKey>, Vec<(NodeKey, NodeKey)>) {
    let k = keys(5);
    let edges = vec![(k[0], k[1]), (k[1], k[2]), (k[2], k[3]), (k[2], k[3])];
    (k, edges)
}

#[test]
fn degrees_count_a_double_relation_twice() {
    let (k, e) = sample();
    let mass = mass_degree(&k, &e);
    assert_eq!(mass[2].1, 3.0);
    assert_eq!(mass[3].1, 2.0);
    assert_eq!(mass[4].1, 0.0);
    let bands = degree_bands(&k, &e);
    assert_eq!(
        bands.iter().map(|(_, b)| *b).collect::<Vec<_>>(),
        [1, 2, 2, 2, 0]
    );
}

#[test]
fn components_are_dense_in_first_node_order() {
    let (k, e) = sample();
    let c = component_groups(&k, &e);
    assert_eq!(
        c.iter().map(|(_, g)| *g).collect::<Vec<_>>(),
        [0, 0, 0, 0, 1]
    );
}

#[test]
fn the_colouring_is_proper() {
    let (k, e) = sample();
    let colour: HashMap<NodeKey, u32> = coloring_groups(&k, &e).into_iter().collect();
    for (a, b) in &e {
        assert_ne!(colour[a], colour[b]);
    }
}

#[test]
fn page_rank_has_mean_one() {
    let (k, e) = sample();
    let ranks = page_rank_weights(&k, &e);
    let mean = ranks.iter().map(|(_, r)| r).sum::<f32>() / ranks.len() as f32;
    assert!((mean - 1.0).abs() < 0.05, "{mean}");
    assert!(page_rank_weights(&[], &[]).is_empty());
}

#[test]
fn depths_follow_the_edges() {
    let (k, e) = sample();
    let roots: Vec<u32> = root_depths(&k, &e).into_iter().map(|(_, d)| d).collect();
    assert_eq!(roots, [0, 1, 2, 3, 0]);
    let layers: Vec<u32> = layer_depths(&k, &e).into_iter().map(|(_, d)| d).collect();
    assert_eq!(layers, [0, 1, 2, 3, 0]);
    let focus: Vec<u32> = focus_depths(&k, &e, k[1])
        .into_iter()
        .map(|(_, d)| d)
        .collect();
    assert_eq!(focus[1], 0);
    assert_eq!(focus[2], 1);
    // A focus outside the view reads the roots' depths.
    assert_eq!(focus_depths(&k, &e, NodeKey::new(99)), root_depths(&k, &e));
}

#[test]
fn the_spanning_tree_and_hop_distances_follow_the_undirected_view() {
    let (k, e) = sample();
    assert_eq!(spanning_edges(&k, &e).len(), 3);
    let hops = hop_distances(&k, &e);
    // The doubled pair is half a hop apart.
    let d23 = hops
        .iter()
        .find(|(a, b, _)| (*a, *b) == (k[2], k[3]))
        .unwrap()
        .2;
    assert!((d23 - 0.5).abs() < 1e-6);
    // The isolated node reaches nothing.
    assert!(hops.iter().all(|(a, b, _)| *a != k[4] && *b != k[4]));
}
