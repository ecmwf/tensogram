// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use crate::error::{Result, TensogramError};
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
/// **Strict-compare contract.** This function unconditionally
/// compares the stored slot against the recomputed digest.  A
/// stored value of `0x0000000000000000` on a frame whose body
/// hashes to something non-zero is a mismatch — it does **not**
/// mean "no hash to verify".  Callers that need to handle the
/// message-wide `HASHES_PRESENT = 0` case (every slot is zero by
/// design) must check the preamble flag themselves and skip
/// calling this function; the validator does this in
/// `validate_integrity`.
///
/// Rationale: treating zero as a pass-through would be an
/// integrity bypass — an attacker zeroing a single frame's slot
/// (while leaving `HASHES_PRESENT = 1`) would otherwise silently
/// pass verification.
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
/// by [`crate::types::HashFrame`] entries.
///
/// Exposed `pub(crate)` so that the buffered encoder can reuse this
/// formatting when it consumes the digest produced by the
/// `tensogram-encodings` pipeline (see
/// [`pipeline::PipelineResult::hash`]).
#[inline]
pub(crate) fn format_xxh3_digest(digest: u64) -> String {
    format!("{digest:016x}")
}

// (Wave 2.2 removed the standalone
// `verify_hash(&[u8], &HashDescriptor)` helper — `HashDescriptor`
// itself is gone in v3.  Frame-level integrity goes through
// [`hash_frame_body`] / [`verify_frame_hash`] over raw frame bytes.)

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

    // (Wave 2.2 removed `HashDescriptor` and its `verify_hash`
    // helper.  Frame-level integrity is exercised by the
    // `verify_frame_hash_*` tests below; the strict-input contract
    // for unknown algorithm names now lives in the validator —
    // see `validate::integrity::validate_integrity` and
    // `IssueCode::UnknownHashAlgorithm`.)

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
    fn verify_frame_hash_rejects_zero_slot_when_body_is_nonempty() {
        // Post-Copilot-review: `verify_frame_hash` now strict-compares
        // the slot against the recomputed digest.  A zero slot on a
        // frame with a non-zero body hash is a mismatch — NOT a
        // pass-through.  Callers that know HASHES_PRESENT = 0 at the
        // message level must check that flag themselves and skip
        // calling this helper; the validator does this via
        // `validate_integrity`.
        use crate::wire::{FRAME_END, FRAME_MAGIC};
        let body = b"hello";
        let mut buf = Vec::new();
        buf.extend_from_slice(FRAME_MAGIC);
        buf.extend_from_slice(&1u16.to_be_bytes()); // HeaderMetadata
        buf.extend_from_slice(&1u16.to_be_bytes()); // version
        buf.extend_from_slice(&0u16.to_be_bytes()); // flags
        let total_length = 16 + body.len() + 12;
        buf.extend_from_slice(&(total_length as u64).to_be_bytes());
        buf.extend_from_slice(body);
        buf.extend_from_slice(&0u64.to_be_bytes()); // hash slot = 0 (attacker-zeroed)
        buf.extend_from_slice(FRAME_END);
        let err = verify_frame_hash(&buf, FrameType::HeaderMetadata).unwrap_err();
        assert!(matches!(err, TensogramError::HashMismatch { .. }));
    }

    #[test]
    fn verify_frame_hash_accepts_zero_slot_only_when_body_hashes_to_zero() {
        // A frame whose body happens to hash to zero (not xxh3 in
        // practice but a boundary case for the strict-compare
        // contract): stored slot = 0 matches computed = 0 → Ok.
        // xxh3 of an empty body is non-zero in reality, so we
        // can't actually trigger the all-zero case with real bytes
        // — this test documents the contract via negation.
        use crate::wire::{FRAME_END, FRAME_HEADER_SIZE, FRAME_MAGIC};
        let body: &[u8] = b"";
        let mut buf = Vec::new();
        buf.extend_from_slice(FRAME_MAGIC);
        buf.extend_from_slice(&1u16.to_be_bytes());
        buf.extend_from_slice(&1u16.to_be_bytes());
        buf.extend_from_slice(&0u16.to_be_bytes());
        let total_length = FRAME_HEADER_SIZE + body.len() + 12;
        buf.extend_from_slice(&(total_length as u64).to_be_bytes());
        buf.extend_from_slice(body);
        buf.extend_from_slice(&0u64.to_be_bytes()); // stored = 0
        buf.extend_from_slice(FRAME_END);
        // Empty body xxh3 is 0x2d06800538d394c2 (non-zero), so
        // stored=0 != computed → must mismatch.
        let result = verify_frame_hash(&buf, FrameType::HeaderMetadata);
        assert!(matches!(result, Err(TensogramError::HashMismatch { .. })));
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
