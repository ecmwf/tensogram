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

/// Cryptographic / checksum algorithm identifiers understood by
/// this crate's hashing surface.
///
/// v3 recognises a single algorithm — xxh3-64 — named `"xxh3"`
/// on the wire (see the `algorithm` field of the
/// [`crate::types::HashFrame`] CBOR schema).  Any other algorithm
/// name parsed from a wire message surfaces as an
/// `UnknownHashAlgorithm` validation warning; the inline per-frame
/// hash slot still serves as the authoritative integrity source.
///
/// Adding a new variant here is deliberate: the `as_str` /
/// `parse` match arms are exhaustive at compile time, so every
/// call site that names an algorithm forces an explicit answer
/// for the new variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlgorithm {
    /// xxh3-64 — the canonical v3 algorithm.  64-bit digest,
    /// rendered on the wire as a lowercase 16-character hex
    /// string (see [`format_xxh3_digest`]).
    Xxh3,
}

impl HashAlgorithm {
    /// Returns the wire-format string identifier for this
    /// algorithm (e.g. `"xxh3"`).  Used by the HashFrame CBOR
    /// encoder and the FFI `tgm_object_hash_type` accessor.
    pub fn as_str(&self) -> &'static str {
        match self {
            HashAlgorithm::Xxh3 => "xxh3",
        }
    }

    /// Parse a wire-format algorithm string into a
    /// [`HashAlgorithm`].
    ///
    /// # Errors
    ///
    /// Returns `TensogramError::Metadata` when `s` is not a
    /// recognised algorithm name.  v3 readers treat the error
    /// loosely at the HashFrame layer (the inline slot remains
    /// authoritative and an unknown algorithm becomes a warning
    /// rather than a hard failure).
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "xxh3" => Ok(HashAlgorithm::Xxh3),
            _ => Err(TensogramError::Metadata(format!("unknown hash type: {s}"))),
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
            "frame too small to hash: frame_bytes.len() = {} < header({}) + footer({}) = {}; \
             for frame_type = {:?}.  Likely truncated; re-read from source.",
            frame_bytes.len(),
            FRAME_HEADER_SIZE,
            footer,
            min_size,
            frame_type,
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
    let frame_len = frame_bytes.len();
    // Buffer must be large enough to contain the 12-byte common tail
    // (hash + ENDF).  The body hasher below does its own header+footer
    // size check against `frame_type`.
    if frame_len < FRAME_COMMON_FOOTER_SIZE {
        return Err(TensogramError::Framing(format!(
            "frame too small to read hash slot: frame_bytes.len() = {frame_len} \
             < FRAME_COMMON_FOOTER_SIZE ({FRAME_COMMON_FOOTER_SIZE}); \
             truncated or not a v3 frame"
        )));
    }
    // ENDF at the very end — sanity check so we never hash a
    // misaligned frame.
    let endf_start = frame_len - FRAME_END.len();
    if &frame_bytes[endf_start..frame_len] != FRAME_END {
        return Err(TensogramError::Framing(
            "frame missing ENDF marker while verifying inline hash — \
             likely truncated or not a v3 frame; re-read from source"
                .to_string(),
        ));
    }
    let slot_start = frame_len - FRAME_COMMON_FOOTER_SIZE;
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
    let algorithm = match HashAlgorithm::parse(&descriptor.algorithm) {
        Ok(algo) => algo,
        Err(_) => {
            tracing::warn!(
                algorithm = %descriptor.algorithm,
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
            algorithm: "xxh3".to_string(),
            value: hash,
        };
        assert!(verify_hash(data, &descriptor).is_ok());
    }

    #[test]
    fn test_verify_hash_mismatch() {
        let data = b"test data";
        let descriptor = HashDescriptor {
            algorithm: "xxh3".to_string(),
            value: "0000000000000000".to_string(),
        };
        assert!(verify_hash(data, &descriptor).is_err());
    }

    #[test]
    fn test_unknown_hash_type_skips_verification() {
        let data = b"test data";
        let descriptor = HashDescriptor {
            algorithm: "sha256".to_string(),
            value: "abc123".to_string(),
        };
        // Unknown hash algorithms skip verification with a warning (forward compatibility)
        assert!(verify_hash(data, &descriptor).is_ok());
    }

    // ── Inline-slot error paths ─────────────────────────────────────

    #[test]
    fn hash_frame_body_rejects_below_minimum_size() {
        // Minimum size for an NTensorFrame is header(16)+footer(20)=36.
        let buf = vec![0u8; 30];
        let err = hash_frame_body(&buf, FrameType::NTensorFrame).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("frame too small to hash"));
        assert!(msg.contains("frame_bytes.len() = 30"));
        assert!(msg.contains("NTensorFrame"));
    }

    #[test]
    fn verify_frame_hash_rejects_below_common_footer_size() {
        // Below 12 B we can't even read the hash slot.
        let buf = vec![0u8; 10];
        let err = verify_frame_hash(&buf, FrameType::HeaderMetadata).unwrap_err();
        assert!(
            err.to_string()
                .contains("frame too small to read hash slot")
        );
    }

    #[test]
    fn verify_frame_hash_rejects_missing_endf() {
        // 12-byte buffer: 8 B of zeros (hash slot = 0) + 4 B non-ENDF.
        // Must fail at the ENDF check before the stored==0 fast path.
        let mut buf = vec![0u8; 12];
        buf[8..12].copy_from_slice(b"XXXX");
        let err = verify_frame_hash(&buf, FrameType::HeaderMetadata).unwrap_err();
        assert!(err.to_string().contains("ENDF"));
    }

    #[test]
    fn verify_frame_hash_accepts_zero_slot_as_no_hash_to_verify() {
        // Minimal frame: 16 B header + empty body + 0-hash slot + ENDF.
        use crate::wire::{FRAME_END, FRAME_MAGIC};
        let mut buf = Vec::new();
        buf.extend_from_slice(FRAME_MAGIC);
        buf.extend_from_slice(&1u16.to_be_bytes()); // HeaderMetadata
        buf.extend_from_slice(&1u16.to_be_bytes()); // version
        buf.extend_from_slice(&0u16.to_be_bytes()); // flags
        buf.extend_from_slice(&28u64.to_be_bytes()); // total_length
        buf.extend_from_slice(&0u64.to_be_bytes()); // hash slot = 0
        buf.extend_from_slice(FRAME_END);
        assert!(verify_frame_hash(&buf, FrameType::HeaderMetadata).is_ok());
    }

    #[test]
    fn verify_frame_hash_reports_mismatch_on_tampered_slot() {
        // Build a frame with a valid body but a wrong hash slot.
        use crate::wire::{FRAME_END, FRAME_MAGIC};
        let body = b"hello";
        let mut buf = Vec::new();
        buf.extend_from_slice(FRAME_MAGIC);
        buf.extend_from_slice(&1u16.to_be_bytes());
        buf.extend_from_slice(&1u16.to_be_bytes());
        buf.extend_from_slice(&0u16.to_be_bytes());
        let total_length = 16 + body.len() + 12;
        buf.extend_from_slice(&(total_length as u64).to_be_bytes());
        buf.extend_from_slice(body);
        buf.extend_from_slice(&0xDEADBEEFCAFEBABEu64.to_be_bytes()); // wrong
        buf.extend_from_slice(FRAME_END);

        let err = verify_frame_hash(&buf, FrameType::HeaderMetadata).unwrap_err();
        assert!(matches!(err, TensogramError::HashMismatch { .. }));
        let msg = err.to_string();
        assert!(msg.contains("deadbeefcafebabe"));
    }
}
