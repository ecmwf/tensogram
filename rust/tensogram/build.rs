// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

fn main() {
    // Capture pure-Rust dependency versions from Cargo.lock into
    // OUT_DIR/built.rs so that doctor.rs can read them via env!().
    // This covers tokio, memmap2, object_store, and other runtime deps.
    built::write_built_file().expect("failed to write built.rs");

    println!("cargo:rerun-if-changed=build.rs");
}
