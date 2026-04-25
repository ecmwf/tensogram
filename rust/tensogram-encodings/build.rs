// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

use std::env;
use std::path::PathBuf;

fn main() {
    // Capture pure-Rust dependency versions from Cargo.lock into
    // OUT_DIR/built.rs so that version.rs can read them via env!().
    built::write_built_file().expect("failed to write built.rs");

    // When the szip FFI feature is on, compile the libaec version shim.
    // The shim reads AEC_VERSION_STR from the libaec header that libaec-sys
    // built and exposes it as a callable C symbol.
    if env::var("CARGO_FEATURE_SZIP").is_ok() {
        compile_libaec_version_shim();
    }

    println!("cargo:rerun-if-changed=build_shim/libaec_version.c");
    println!("cargo:rerun-if-changed=build.rs");
}

fn compile_libaec_version_shim() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shim = manifest_dir.join("build_shim").join("libaec_version.c");

    // libaec-sys builds libaec from source and places the header in its
    // OUT_DIR/include/.  We locate that directory by searching the build
    // output tree for libaec.h.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // OUT_DIR is something like .../tensogram-encodings-<hash>/out
    // libaec-sys's OUT_DIR is a sibling: .../libaec-sys-<hash>/out
    // Walk up to the common build directory and search for the header.
    let libaec_include = match find_libaec_include(&out_dir) {
        Some(p) => p,
        None => panic!(
            "tensogram-encodings build script: could not find `libaec.h` in any \
             sibling `libaec-sys-*/out/include/` directory under `{}`. \
             The `szip` feature requires `libaec-sys` to be built first; this \
             usually means cargo's build order didn't propagate the dependency \
             — try `cargo clean` and rebuild, or report a bug if it persists. \
             Source: rust/tensogram-encodings/build.rs::find_libaec_include().",
            out_dir.display()
        ),
    };

    cc::Build::new()
        .file(&shim)
        .include(libaec_include)
        .compile("tensogram_libaec_version_shim");
}

/// Search the Cargo build output tree for the `libaec.h` header produced by
/// libaec-sys.  Returns the directory containing the header, or `None` if
/// the search either escapes the cargo build tree or finds no matching
/// sibling crate.
///
/// Callers are expected to surface a clear error on `None` rather than
/// continue to compile the shim — the C compiler's "header not found"
/// message names the wrong path and is unactionable for a Rust user.
fn find_libaec_include(out_dir: &std::path::Path) -> Option<std::path::PathBuf> {
    // OUT_DIR layout: <target>/<profile>/build/<crate>-<hash>/out
    // Go up 2 levels to reach <target>/<profile>/build/
    let build_dir = out_dir.parent()?.parent()?;

    // Walk sibling crate output directories looking for libaec.h
    for entry in std::fs::read_dir(build_dir).ok()?.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with("libaec-sys") {
            let candidate = entry.path().join("out").join("include");
            if candidate.join("libaec.h").exists() {
                return Some(candidate);
            }
        }
    }
    None
}
