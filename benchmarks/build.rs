// (C) Copyright 2024- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

// build.rs — benchmarks crate
//
// When the `eccodes` feature is active, tell the linker to link against
// libeccodes (the ecCodes C library). The location is discovered via
// pkg-config, falling back to well-known Homebrew paths.
//
// This is necessary because our grib_comparison module calls eccodes C
// functions via `extern "C"` declarations, which require an explicit link.

fn main() {
    #[cfg(feature = "eccodes")]
    link_eccodes();
}

#[cfg(feature = "eccodes")]
fn link_eccodes() {
    // Try pkg-config first — the recommended way when available.
    if try_pkg_config() {
        return;
    }

    // Fallback: known Homebrew install paths on macOS.
    let homebrew_paths = [
        "/opt/homebrew/lib", // Apple Silicon
        "/usr/local/lib",    // Intel Mac
    ];
    for path in &homebrew_paths {
        let lib = std::path::Path::new(path).join("libeccodes.dylib");
        if lib.exists() {
            println!("cargo:rustc-link-search=native={path}");
            println!("cargo:rustc-link-lib=dylib=eccodes");
            return;
        }
    }

    // System-wide (Linux with apt-installed libeccodes-dev).
    println!("cargo:rustc-link-lib=dylib=eccodes");
}

#[cfg(feature = "eccodes")]
fn try_pkg_config() -> bool {
    let output = std::process::Command::new("pkg-config")
        .args(["--libs", "eccodes"])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let flags = String::from_utf8_lossy(&out.stdout);
            let mut linked_eccodes = false;

            // Parse -L paths and -l libs from pkg-config output.
            for flag in flags.split_whitespace() {
                if let Some(path) = flag.strip_prefix("-L") {
                    println!("cargo:rustc-link-search=native={path}");
                } else if let Some(lib) = flag.strip_prefix("-l") {
                    if lib == "eccodes" {
                        linked_eccodes = true;
                    }
                    println!("cargo:rustc-link-lib=dylib={lib}");
                }
            }

            if !linked_eccodes {
                println!("cargo:rustc-link-lib=dylib=eccodes");
            }
            true
        }
        _ => false,
    }
}
