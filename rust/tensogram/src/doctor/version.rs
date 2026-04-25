// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

//! Backend version lookups for tensogram-core features.
//!
//! Covers `remote` (object_store), `mmap` (memmap2), and `async` (tokio).
//! All versions are sourced from the compile-time `Cargo.lock` snapshot
//! via the `built` crate.
//!
//! Converter backend versions (`grib`, `netcdf`) are handled by the CLI
//! layer which has access to those crates' transitive C libraries.
//!
//! The canonical public path for [`BackendVersion`] / [`Linkage`] is
//! [`crate::doctor`]; this module imports them only as helpers for its
//! own functions.

#[cfg(any(feature = "remote", feature = "mmap", feature = "async"))]
use tensogram_encodings::version::BackendVersion;

#[cfg(any(feature = "remote", feature = "mmap", feature = "async"))]
mod built_info {
    include!(concat!(env!("OUT_DIR"), "/built.rs"));
}

#[cfg(any(feature = "remote", feature = "mmap", feature = "async"))]
fn dep_version(crate_name: &str) -> Option<String> {
    built_info::DEPENDENCIES
        .iter()
        .find(|(name, _)| *name == crate_name)
        .map(|(_, ver)| ver.to_string())
}

// ── remote (object_store) ────────────────────────────────────────────────────

/// Version of the object_store crate used for remote I/O.
#[cfg(feature = "remote")]
pub fn remote_version() -> BackendVersion {
    BackendVersion::pure_rust("object_store", dep_version("object_store"))
}

// ── mmap (memmap2) ───────────────────────────────────────────────────────────

/// Version of the memmap2 crate used for memory-mapped I/O.
#[cfg(feature = "mmap")]
pub fn mmap_version() -> BackendVersion {
    BackendVersion::pure_rust("memmap2", dep_version("memmap2"))
}

// ── async (tokio) ────────────────────────────────────────────────────────────

/// Version of the tokio crate used for async I/O.
#[cfg(feature = "async")]
pub fn async_version() -> BackendVersion {
    BackendVersion::pure_rust("tokio", dep_version("tokio"))
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #[cfg(any(feature = "remote", feature = "mmap", feature = "async"))]
    use super::*;

    #[cfg(any(feature = "remote", feature = "mmap", feature = "async"))]
    fn has_digit(s: &str) -> bool {
        s.chars().any(|c| c.is_ascii_digit())
    }

    #[test]
    #[cfg(feature = "remote")]
    fn remote_version_non_empty() {
        let v = remote_version();
        let ver = v.version.expect("object_store version should be present");
        assert!(!ver.is_empty());
        assert!(has_digit(&ver), "object_store version has no digit: {ver}");
    }

    #[test]
    #[cfg(feature = "mmap")]
    fn mmap_version_non_empty() {
        let v = mmap_version();
        let ver = v.version.expect("memmap2 version should be present");
        assert!(!ver.is_empty());
        assert!(has_digit(&ver), "memmap2 version has no digit: {ver}");
    }

    #[test]
    #[cfg(feature = "async")]
    fn async_version_non_empty() {
        let v = async_version();
        let ver = v.version.expect("tokio version should be present");
        assert!(!ver.is_empty());
        assert!(has_digit(&ver), "tokio version has no digit: {ver}");
    }
}
