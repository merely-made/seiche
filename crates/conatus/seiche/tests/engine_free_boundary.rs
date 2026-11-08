//! The engine-free boundary: with `engine` off, the force structs, their
//! declarations, the dynamics spec and its derivation, and the catalog's
//! builders are all here; only applying a force to bodies is not. This test
//! builds with either setting of `engine`.
#![cfg(feature = "serde")]

use seiche::spec::{DynamicsSpec, Node, NoPresets, RawTerm};
use seiche::{Declared, Force};

#[test]
fn every_raw_kind_builds_and_declares_its_terms() {
    let mut built = 0;
    for term in RawTerm::defaults() {
        let kind = term.kind();
        // A kind that needs a graph-derived input refuses without one; the rest build.
        let Ok(force) = term.build(kind, seiche::spec::DEFAULT_SEED) else {
            continue;
        };
        built += 1;
        let force: Box<dyn Force> = force;
        assert!(!force.terms().is_empty(), "{kind} declares no terms");
    }
    assert!(built >= 10, "only {built} raw kinds built");
}

#[test]
fn a_raw_spec_derives_without_the_engine() {
    let spec = DynamicsSpec::new(Node::raw(RawTerm::defaults()[0].clone()));
    // A raw term has no runner yet (G4b): derive reaches that refusal, which it
    // can only do after building the force and reading its declarations.
    let refusal = spec.derive(&NoPresets).expect_err("a raw term is unrunnable");
    assert!(matches!(refusal, seiche::spec::SpecError::Unrunnable { .. }));
}

#[test]
fn the_catalog_builders_are_present() {
    let _ = seiche::catalog::build_force_set;
    let _ = seiche::catalog::build_force_set_cached;
}
