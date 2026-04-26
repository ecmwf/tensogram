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
//! [`crate::TensogramFile`] can accept `Option<RemoteScanOptions>`
//! whether or not the `remote` Cargo feature is enabled.  Without
//! `remote`, the type is still public but `open_source` falls
//! through to the local-file path and the value is ignored.

/// Reader-side options for the scan walker.
///
/// Defaults to a bidirectional walk: the remote backend pairs
/// forward preamble fetches with backward postamble fetches, and a
/// pipelined full-discovery scanner overlaps each round's backward
/// validation with the next round's primary fetches.  On real-network
/// workloads this roughly halves wall-clock for full layout discovery
/// and tail / middle access.
///
/// ```
/// use tensogram::RemoteScanOptions;
/// let opts = RemoteScanOptions::default();
/// assert!(opts.bidirectional);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteScanOptions {
    /// Enable the bidirectional walker on the remote backend.
    ///
    /// When `true` (the default), the remote backend pairs forward
    /// preamble fetches with backward postamble fetches across the
    /// file, using the v3 postamble's mirrored `total_length` field
    /// to walk inward from EOF in parallel with the forward sweep.
    /// The pipelined scanner overlaps each round's candidate-preamble
    /// validation with the next round's primary fetches, collapsing
    /// the per-round critical path from 2 RTTs to 1 RTT.
    ///
    /// Set `false` to force a forward-only walk — useful when an
    /// adversarial server might serve disagreeing forward and backward
    /// reads, or when the remote source returns paired ranges over a
    /// transport whose connection-pool serialises them anyway.
    ///
    /// `RemoteScanOptions` only configures the remote backend.  The
    /// local-file backend uses a separate in-memory walker
    /// configured by [`crate::framing::ScanOptions`]; this flag is
    /// silently ignored when the source resolves to a local path.
    pub bidirectional: bool,
}

impl Default for RemoteScanOptions {
    fn default() -> Self {
        Self {
            bidirectional: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RemoteScanOptions;

    #[test]
    fn default_is_bidirectional() {
        let opts = RemoteScanOptions::default();
        assert!(opts.bidirectional);
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
                bidirectional: false
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
