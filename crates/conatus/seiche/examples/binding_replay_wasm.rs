// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#[cfg(target_arch = "wasm32")]
#[path = "support/binding_replay.rs"]
mod shared;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// Run the exact JSON fixture/transcript accepted by the native example.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn run_replay(input: &str) -> Result<String, JsError> {
    shared::run(input).map_err(|error| JsError::new(&error))
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!("binding_replay_wasm is a wasm32 wrapper; use binding_replay for native runs");
}
