// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

// Strategy:
// 1. SZ3 is header-only. We generate `version.hpp` from `version.hpp.in` by
//    substituting the cmake variables ourselves (avoids a full cmake configure
//    step, which would try to build targets we don't need).
// 2. Use `cc` crate to compile our thin C++ shim (`cpp/sz3_ffi.cpp`) with C++17.
// 3. Link zstd — its include path comes from the `zstd-sys` crate via the
//    `DEP_ZSTD_INCLUDE` env var.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR not set"));

    // -----------------------------------------------------------------------
    // 1. Generate version.hpp from version.hpp.in
    // -----------------------------------------------------------------------
    let sz3_source = manifest_dir.join("SZ3");
    if !sz3_source.join("CMakeLists.txt").exists() {
        panic!(
            "SZ3 source not found at {}. \
             Please clone https://github.com/szcompressor/SZ3.git (tag v3.3.2) there.",
            sz3_source.display()
        );
    }

    // Parse version from CMakeLists.txt:  project(SZ3 VERSION x.y.z)
    let cmake_txt = fs::read_to_string(sz3_source.join("CMakeLists.txt"))
        .expect("failed to read SZ3/CMakeLists.txt");
    let (major, minor, patch) = parse_project_version(&cmake_txt);
    let project_version = format!("{major}.{minor}.{patch}");

    // Parse data version: set(SZ3_DATA_VERSION x.y.z)
    let data_version =
        parse_cmake_set(&cmake_txt, "SZ3_DATA_VERSION").unwrap_or_else(|| project_version.clone());

    // Read the template
    let version_in = fs::read_to_string(sz3_source.join("include/SZ3/version.hpp.in"))
        .expect("failed to read version.hpp.in");

    // Substitute cmake variables
    let version_hpp = version_in
        .replace("@PROJECT_NAME@", "SZ3")
        .replace("@PROJECT_VERSION@", &project_version)
        .replace("@PROJECT_VERSION_MAJOR@", &major.to_string())
        .replace("@PROJECT_VERSION_MINOR@", &minor.to_string())
        .replace("@PROJECT_VERSION_PATCH@", &patch.to_string())
        .replace("@PROJECT_VERSION_TWEAK@", "0")
        .replace("@SZ3_DATA_VERSION@", &data_version);

    // Write the generated header into OUT_DIR/include/SZ3/version.hpp
    let gen_include_dir = out_dir.join("include");
    let gen_sz3_dir = gen_include_dir.join("SZ3");
    fs::create_dir_all(&gen_sz3_dir).expect("failed to create include/SZ3 dir");
    fs::write(gen_sz3_dir.join("version.hpp"), version_hpp).expect("failed to write version.hpp");

    // SZ3 source include directory
    let sz3_include_dir = sz3_source.join("include");

    // -----------------------------------------------------------------------
    // 2. zstd include path from zstd-sys dependency
    // -----------------------------------------------------------------------
    let zstd_includes = env::var("DEP_ZSTD_INCLUDE").unwrap_or_default();

    // -----------------------------------------------------------------------
    // 3. Compile the C++ FFI shim
    // -----------------------------------------------------------------------
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file(manifest_dir.join("cpp/sz3_ffi.cpp"))
        .include(&sz3_include_dir) // SZ3 source headers
        .include(&gen_include_dir) // generated version.hpp
        .warnings(false); // suppress SZ3 header warnings

    // Add zstd include paths (semicolon-separated list from zstd-sys)
    for inc in zstd_includes.split(';') {
        let inc = inc.trim();
        if !inc.is_empty() {
            build.include(inc);
        }
    }

    // OpenMP support (optional)
    if cfg!(feature = "openmp") {
        build.flag("-fopenmp");
    }

    build.compile("sz3_ffi");

    // -----------------------------------------------------------------------
    // 4. Cargo metadata
    // -----------------------------------------------------------------------
    println!("cargo:rerun-if-changed=cpp/sz3_ffi.cpp");
    println!("cargo:rerun-if-changed=build.rs");

    // The zstd-sys crate already links libzstd into the final binary through
    // its own build script.  We depend on it in Cargo.toml, so Cargo ensures
    // it is linked.  No additional `rustc-link-lib=zstd` is needed here.

    // Link C++ standard library
    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("apple") || target.contains("freebsd") {
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("linux") || target.contains("android") {
        println!("cargo:rustc-link-lib=stdc++");
    }
    // On Windows with MSVC the C++ stdlib is linked automatically.
}

// ---------------------------------------------------------------------------
// CMakeLists.txt parsing helpers
// ---------------------------------------------------------------------------

/// Parse `project(SZ3 VERSION x.y.z)` from CMakeLists.txt.
fn parse_project_version(cmake_txt: &str) -> (u32, u32, u32) {
    for line in cmake_txt.lines() {
        let trimmed = line.trim();
        if let Some(idx) = trimmed
            .starts_with("project(")
            .then(|| trimmed.find("VERSION"))
            .flatten()
        {
            let rest = trimmed[idx + "VERSION".len()..]
                .trim()
                .trim_end_matches(')');
            let parts: Vec<&str> = rest.split('.').collect();
            if parts.len() >= 3 {
                let major = parts[0].trim().parse().unwrap_or(0);
                let minor = parts[1].trim().parse().unwrap_or(0);
                let patch = parts[2].trim().parse().unwrap_or(0);
                return (major, minor, patch);
            }
        }
    }
    panic!("Could not parse project VERSION from CMakeLists.txt");
}

/// Parse `set(VAR_NAME value)` from CMakeLists.txt.
fn parse_cmake_set(cmake_txt: &str, var_name: &str) -> Option<String> {
    let prefix = format!("set({var_name}");
    for line in cmake_txt.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(&prefix) {
            let rest = &trimmed[prefix.len()..];
            let value = rest.trim().trim_end_matches(')').trim();
            return Some(value.to_string());
        }
    }
    None
}
