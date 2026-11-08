#!/bin/sh
# The engine-free build of seiche (the host side of a physics Worker): it must
# compile with `engine` off, and rapier, parry and nalgebra must not be in its
# normal dependency tree. Run from the repository root.
set -eu

# The remote set: the model, the wire, `Physics::Remote` and the entry relay.
FEATURES="${SEICHE_REMOTE_FEATURES:-serde}"

cargo check -p seiche --no-default-features --features "$FEATURES"

# The boundary: the force structs, their declarations, the dynamics spec and its
# derivation, and the catalog builders compile and run without the engine.
cargo test -p seiche --no-default-features --features "$FEATURES" --test engine_free_boundary
cargo test -p seiche --no-default-features --features "$FEATURES" --lib

tree=$(cargo tree -p seiche --no-default-features --features "$FEATURES" -e normal --prefix none)
if printf '%s\n' "$tree" | grep -E '^(rapier2d|rapier|parry2d|parry|nalgebra|simba|glamx) '; then
    echo "engine-free seiche still links the engine" >&2
    exit 1
fi
echo "engine-free seiche: no rapier, parry or nalgebra in the normal dependency tree"
