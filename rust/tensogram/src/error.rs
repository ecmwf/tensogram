// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use thiserror::Error;

/// Top-level error type for the `tensogram` crate.
///
/// **Variant guarantee.** The enum is `#[non_exhaustive]` so future
/// versions can add new error categories (e.g. for new wire-format
/// features or external system integrations) without breaking
/// downstream `match` arms.  Callers should always include a `_ =>`
/// fallback when matching on `TensogramError`.
///
/// Inner-crate errors (`tensogram_encodings::PipelineError`,
/// `CompressionError`, `PackingError`, `ShuffleError`,
/// `bitmask::MaskError`) are converted into `Encoding` /
/// `Compression` variants via `e.to_string()` at the boundary —
/// the source error chain is collapsed.  This keeps the error
/// strings stable across language bindings (Python `ValueError`,
/// C++ typed exceptions, TypeScript `EncodingError`) at the cost
/// of losing programmatic access to the source.  Tightening this
/// to `#[from]` source chains is a deliberate non-goal until a
/// concrete consumer needs structured introspection.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TensogramError {
    /// Wire-format structural error: bad magic, unknown frame type,
    /// length mismatches, missing ENDF marker, etc.
    #[error("framing error: {0}")]
    Framing(String),
    /// CBOR metadata error: bad CBOR, missing required keys,
    /// type mismatches on per-codec params, mask method validation,
    /// etc.
    #[error("metadata error: {0}")]
    Metadata(String),
    /// Encoding / pipeline error: simple_packing rejection, shuffle
    /// failure, allocation failure on a descriptor-derived size,
    /// codec param invalid for the dtype, etc.
    #[error("encoding error: {0}")]
    Encoding(String),
    /// Compression error: codec library failure, decompression
    /// short-fetch, codec not compiled in, etc.
    #[error("compression error: {0}")]
    Compression(String),
    /// Object error: out-of-range index into a data-object array.
    #[error("object error: {0}")]
    Object(String),
    /// I/O error from the underlying file or socket.  Source-chained
    /// via `#[from]` so `e.source()` returns the original
    /// [`std::io::Error`].
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Frame-body hash digest disagreement between the stored slot
    /// and the recomputed digest.  Both digests are 16-character
    /// lowercase hex strings of the xxh3-64 value.
    #[error("hash mismatch: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },
    /// Remote object-store error: HTTP transport failure, malformed
    /// URL, missing credentials, etc.
    #[error("remote error: {0}")]
    Remote(String),
}

pub type Result<T> = std::result::Result<T, TensogramError>;
