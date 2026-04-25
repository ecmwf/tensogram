// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Reader-side scan-walker configuration shared by every backend.
//!
//! The fields configure how a [`crate::TensogramFile`] discovers
//! per-message layouts on open.  Local-file backends ignore them
//! (a single forward sweep over the file is the only sensible
//! strategy); the remote backend honours `bidirectional`.
//!
//! This module is cfg-independent so the public open methods on
//! [`crate::TensogramFile`] can accept `Option<&RemoteScanOptions>`
//! whether or not the `remote` Cargo feature is enabled.  Without
//! `remote`, the type is still public but `open_source` falls
//! through to the local-file path and the value is ignored.

/// Reader-side options for the scan walker.
///
/// Defaults to a forward-only walk, which is byte-identical across
/// every backend to behaviour before this option existed.
///
/// ```
/// use tensogram::RemoteScanOptions;
/// let opts = RemoteScanOptions::default();
/// assert!(!opts.bidirectional);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RemoteScanOptions {
    /// Enable the bidirectional walker on the remote backend.
    ///
    /// When `true`, the remote backend alternates forward and
    /// backward hops across the file, using the v3 postamble's
    /// mirrored `total_length` field to walk inward from EOF in
    /// parallel with the forward sweep.  This roughly halves the
    /// number of HTTP `GET` requests needed for tail / full-scan
    /// access on header-indexed files.
    ///
    /// `false` keeps the forward-only walker — every existing call
    /// site lands on this path unchanged.
    ///
    /// Local-file backends ignore this flag entirely; the field is
    /// only meaningful for [`crate::TensogramFile::open_remote`] and
    /// its async sibling.
    pub bidirectional: bool,
}

#[cfg(test)]
mod tests {
    use super::RemoteScanOptions;

    #[test]
    fn default_is_forward_only() {
        let opts = RemoteScanOptions::default();
        assert!(!opts.bidirectional);
    }

    #[test]
    fn equality_is_structural() {
        assert_eq!(
            RemoteScanOptions {
                bidirectional: true
            },
            RemoteScanOptions {
                bidirectional: true
            },
        );
        assert_ne!(
            RemoteScanOptions {
                bidirectional: true
            },
            RemoteScanOptions::default(),
        );
    }

    #[test]
    fn struct_is_copy() {
        let a = RemoteScanOptions {
            bidirectional: true,
        };
        let b = a;
        assert_eq!(a, b);
    }
}
