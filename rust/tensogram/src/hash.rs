// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use crate::error::{Result, TensogramError};
use crate::types::HashDescriptor;
use crate::wire::{FRAME_HEADER_SIZE, FrameType, footer_size_for};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlgorithm {
    Xxh3,
}

impl HashAlgorithm {
    pub fn as_str(&self) -> &'static str {
        match self {
            HashAlgorithm::Xxh3 => "xxh3",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "xxh3" => Ok(HashAlgorithm::Xxh3),
            _ => Err(TensogramError::Metadata(format!("unknown hash type: {s}"))),
        }
    }

    /// Length in characters of the hex-encoded digest string produced by
    /// [`compute_hash`] for this algorithm.
    ///
    /// Used by the streaming encoder to size the CBOR descriptor before
    /// the payload has been hashed — the digest must always serialise to
    /// the same number of bytes regardless of value, otherwise the frame
    /// header's `total_length` would be wrong.  Adding a new variant
    /// forces an explicit answer here: the match is exhaustive at
    /// compile time.
    pub fn hex_digest_len(&self) -> usize {
        match self {
            HashAlgorithm::Xxh3 => 16, // 64 bits → 16 hex chars
        }
    }
}

/// Compute a hash of the given data, returning the hex-encoded digest.
pub fn compute_hash(data: &[u8], algorithm: HashAlgorithm) -> String {
    match algorithm {
        HashAlgorithm::Xxh3 => format_xxh3_digest(xxhash_rust::xxh3::xxh3_64(data)),
    }
}

// ── Inline per-frame hashing (v3) ────────────────────────────────────────────

/// Compute the xxh3-64 digest of a frame's *body* — the bytes
/// between the 16-byte header and the type-specific footer.
///
/// This is the canonical definition of the frame's hash scope per
/// `plans/WIRE_FORMAT.md` §2.4.  The scope covers:
///
/// - The CBOR payload (for metadata / index / hash / preceder frames)
/// - The encoded tensor payload + mask blobs + CBOR descriptor
///   (for NTensorFrame — `cbor_offset` is part of the footer, so
///   it is **not** covered)
///
/// Excluded in every case: the 16-byte frame header, the full
/// type-specific footer (hash slot, ENDF, plus any type-specific
/// fields like `cbor_offset`), and any 8-byte alignment padding
/// after `ENDF`.
///
/// Returns an error when the frame is smaller than
/// `FRAME_HEADER_SIZE + footer_size_for(frame_type)` — i.e. the
/// frame-body slice would be invalid.
pub fn hash_frame_body(frame_bytes: &[u8], frame_type: FrameType) -> Result<u64> {
    let footer = footer_size_for(frame_type);
    let min_size = FRAME_HEADER_SIZE + footer;
    if frame_bytes.len() < min_size {
        return Err(TensogramError::Framing(format!(
            "frame too small to hash: {} < header+footer ({})",
            frame_bytes.len(),
            min_size
        )));
    }
    let body = &frame_bytes[FRAME_HEADER_SIZE..frame_bytes.len() - footer];
    Ok(xxhash_rust::xxh3::xxh3_64(body))
}

/// Verify the inline hash slot of a frame against its body.
///
/// Reads the uint64 BE value at `frame_end - FRAME_COMMON_FOOTER_SIZE
/// .. frame_end - 4` (the 8 bytes immediately before `ENDF`),
/// recomputes the xxh3-64 of the body region via
/// [`hash_frame_body`], and errors with [`TensogramError::
/// HashMismatch`] on disagreement.
///
/// Pass-through: a stored hash of `0x0000000000000000` is a
/// signal that the frame was written without hashing (message
/// flag `HASHES_PRESENT = 0`), and this function treats that case
/// as "no hash to verify" and returns `Ok(())`.  Callers that want
/// to *require* hashing should check the preamble flag themselves.
pub fn verify_frame_hash(frame_bytes: &[u8], frame_type: FrameType) -> Result<()> {
    use crate::wire::{FRAME_COMMON_FOOTER_SIZE, FRAME_END, read_u64_be};
    let total = frame_bytes.len();
    // Buffer must be large enough to contain the common 12-byte
    // tail.  The body hasher does its own min-size check too.
    if total < FRAME_COMMON_FOOTER_SIZE {
        return Err(TensogramError::Framing(format!(
            "frame too small to read hash slot: {total} < {FRAME_COMMON_FOOTER_SIZE}"
        )));
    }
    // ENDF at the very end — sanity check so we never hash a
    // misaligned frame.
    let endf_start = total - FRAME_END.len();
    if &frame_bytes[endf_start..total] != FRAME_END {
        return Err(TensogramError::Framing(
            "frame missing ENDF marker while verifying inline hash".to_string(),
        ));
    }
    let slot_start = total - FRAME_COMMON_FOOTER_SIZE;
    let stored = read_u64_be(frame_bytes, slot_start);
    if stored == 0 {
        // HASHES_PRESENT = 0 for this message.  No hash to verify.
        return Ok(());
    }
    let computed = hash_frame_body(frame_bytes, frame_type)?;
    if computed != stored {
        return Err(TensogramError::HashMismatch {
            expected: format_xxh3_digest(stored),
            actual: format_xxh3_digest(computed),
        });
    }
    Ok(())
}

/// Format a raw xxh3-64 digest into the canonical 16-char hex string used
/// in [`HashDescriptor::value`].
///
/// Exposed `pub(crate)` so that the buffered encoder can reuse this
/// formatting when it consumes the digest produced by the
/// `tensogram-encodings` pipeline (see
/// [`pipeline::PipelineResult::hash`]).
#[inline]
pub(crate) fn format_xxh3_digest(digest: u64) -> String {
    format!("{digest:016x}")
}

/// Verify a hash descriptor against data.
///
/// If the hash algorithm is not recognized, a warning is logged and
/// verification is skipped (returns Ok). This ensures forward compatibility
/// when new hash algorithms are added.
pub fn verify_hash(data: &[u8], descriptor: &HashDescriptor) -> Result<()> {
    let algorithm = match HashAlgorithm::parse(&descriptor.hash_type) {
        Ok(algo) => algo,
        Err(_) => {
            tracing::warn!(
                hash_type = %descriptor.hash_type,
                "unknown hash algorithm, skipping verification"
            );
            return Ok(());
        }
    };
    let actual = compute_hash(data, algorithm);
    if actual != descriptor.value {
        return Err(TensogramError::HashMismatch {
            expected: descriptor.value.clone(),
            actual,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xxh3() {
        let data = b"hello world";
        let hash = compute_hash(data, HashAlgorithm::Xxh3);
        assert_eq!(hash.len(), 16); // 64-bit = 16 hex chars
        // Verify deterministic
        assert_eq!(hash, compute_hash(data, HashAlgorithm::Xxh3));
    }

    #[test]
    fn test_verify_hash() {
        let data = b"test data";
        let hash = compute_hash(data, HashAlgorithm::Xxh3);
        let descriptor = HashDescriptor {
            hash_type: "xxh3".to_string(),
            value: hash,
        };
        assert!(verify_hash(data, &descriptor).is_ok());
    }

    #[test]
    fn test_verify_hash_mismatch() {
        let data = b"test data";
        let descriptor = HashDescriptor {
            hash_type: "xxh3".to_string(),
            value: "0000000000000000".to_string(),
        };
        assert!(verify_hash(data, &descriptor).is_err());
    }

    #[test]
    fn test_unknown_hash_type_skips_verification() {
        let data = b"test data";
        let descriptor = HashDescriptor {
            hash_type: "sha256".to_string(),
            value: "abc123".to_string(),
        };
        // Unknown hash algorithms skip verification with a warning (forward compatibility)
        assert!(verify_hash(data, &descriptor).is_ok());
    }
}
