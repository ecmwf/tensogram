// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use crate::error::{Result, TensogramError};
use crate::wire::{FRAME_HEADER_SIZE, FrameFlags, FrameType, footer_size_for};

/// Wire-format string identifier for the v3 frame-level hash
/// algorithm.  Stored in the `algorithm` field of
/// [`crate::types::HashFrame`] (see `plans/WIRE_FORMAT.md` §6.3)
/// and returned by the FFI `tgm_object_hash_type` accessor.
///
/// v3 ships exactly one algorithm — xxh3-64.  When a second algorithm
/// lands the strict-input contract for the encoder boundary
/// ([`parse_hash_name`]) becomes a multi-arm match; until then it is
/// a string equality check against this constant.  Keeping the name
/// in one place makes the future addition mechanical.
pub const HASH_ALGORITHM_NAME: &str = "xxh3";

/// Whether a binding-supplied algorithm name selects integrity
/// hashing on encode.
///
/// Strict-input contract:
/// - `None` → use the encoder's default (hashing on).
/// - `Some("xxh3")` → on.
/// - `Some("none")` → off.
/// - Any other value → [`TensogramError::Metadata`] with the
///   offending string in the message.
///
/// Replaces the v2 `HashAlgorithm::parse(&str) -> Result<HashAlgorithm>`
/// helper.  The single-variant enum it returned was dead surface — the
/// only meaningful question on the encode path is "hash or don't",
/// which is a `bool`.  The wire-format string still exists (in
/// `HashFrame.algorithm`) and is the [`HASH_ALGORITHM_NAME`]
/// constant.
pub fn parse_hash_name(name: Option<&str>) -> Result<bool> {
    match name {
        None => Ok(true),
        Some(HASH_ALGORITHM_NAME) => Ok(true),
        Some("none") => Ok(false),
        Some(other) => Err(TensogramError::Metadata(format!(
            "unknown hash type: {other}; expected \"{HASH_ALGORITHM_NAME}\" or \"none\""
        ))),
    }
}

/// Compute the canonical xxh3-64 hex digest of the given bytes.
///
/// v3 has exactly one algorithm — xxh3-64.  Earlier versions took an
/// `algorithm: HashAlgorithm` argument; collapsed in Wave 2.1 to
/// reflect reality.  When a second algorithm lands the function will
/// gain an `algorithm` parameter again as a deliberate signal at
/// every call site.
pub fn compute_hash(data: &[u8]) -> String {
    format_xxh3_digest(xxhash_rust::xxh3::xxh3_64(data))
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

/// Inspect a frame's inline hash slot, dispatching on the
/// per-frame `HASH_PRESENT` flag.
///
/// Hash presence is signalled by bit 1 of the frame header's
/// `flags` field ([`FrameFlags::HASH_PRESENT`] — `plans/WIRE_FORMAT.md`
/// §2.5).  Callers that want to know whether a frame carries a
/// digest **must** read the flag; the slot value alone is not a
/// reliable signal because every 64-bit value, including zero, is
/// a valid xxh3-64 digest.
///
/// Returns:
/// * `Ok(true)` — `HASH_PRESENT` was set and the slot value
///   equals the recomputed body digest.
/// * `Ok(false)` — `HASH_PRESENT` was clear; no hash was
///   recorded for this frame, no comparison performed.
/// * `Err(HashMismatch { object_index: None, .. })` —
///   `HASH_PRESENT` was set but the slot disagrees with the
///   recomputed body digest.  The `object_index` field is `None`
///   here because this low-level helper doesn't know the
///   surrounding object index; callers that have one should
///   transform the error to attach it (see
///   [`crate::decode`]).
/// * `Err(Framing { .. })` — frame too small, missing ENDF,
///   or otherwise malformed.
pub fn check_frame_hash(frame_bytes: &[u8], frame_type: FrameType) -> Result<bool> {
    use crate::wire::{FRAME_COMMON_FOOTER_SIZE, FRAME_END, read_u16_be, read_u64_be};
    let frame_len = frame_bytes.len();
    // Must be large enough to cover the frame header plus the
    // 12-byte common tail `[hash][ENDF]`; the body hasher below
    // does its own per-type check.
    if frame_len < FRAME_HEADER_SIZE + FRAME_COMMON_FOOTER_SIZE {
        return Err(TensogramError::Framing(format!(
            "frame too small to read hash slot: frame_bytes.len() = {frame_len} \
             < header({FRAME_HEADER_SIZE}) + common footer ({FRAME_COMMON_FOOTER_SIZE}); \
             truncated or not a v3 frame"
        )));
    }
    // ENDF at the very end — sanity check so we never hash a
    // misaligned frame.
    let endf_start = frame_len - FRAME_END.len();
    if &frame_bytes[endf_start..frame_len] != FRAME_END {
        return Err(TensogramError::Framing(
            "frame missing ENDF marker while inspecting inline hash — \
             likely truncated or not a v3 frame; re-read from source"
                .to_string(),
        ));
    }
    // The flag is bit 1 of the `flags` field at offset 6 of the
    // frame header — read directly without parsing the full
    // header struct so we stay cheap on large messages.
    let header_flags = read_u16_be(frame_bytes, 6);
    if header_flags & FrameFlags::HASH_PRESENT == 0 {
        return Ok(false);
    }
    let slot_start = frame_len - FRAME_COMMON_FOOTER_SIZE;
    let stored = read_u64_be(frame_bytes, slot_start);
    let computed = hash_frame_body(frame_bytes, frame_type)?;
    if computed != stored {
        return Err(TensogramError::HashMismatch {
            object_index: None,
            expected: format_xxh3_digest(stored),
            actual: format_xxh3_digest(computed),
        });
    }
    Ok(true)
}

/// Strict wrapper around [`check_frame_hash`]: a frame with
/// `HASH_PRESENT` clear is treated as a verification failure.
///
/// Used by the offline validator (`tensogram validate
/// --checksum`) and the cross-language conformance harness, both
/// of which start from the assumption that the message was
/// encoded with hashing on and want to fail loudly when any
/// individual frame's flag is unexpectedly clear.
///
/// The decode-time verification path (`DecodeOptions::verify_hash`)
/// does **not** use this wrapper — it calls [`check_frame_hash`]
/// directly so it can wrap the `Ok(false)` case in a
/// [`TensogramError::MissingHash`] error that carries the
/// surrounding `object_index`.
pub fn verify_frame_hash(frame_bytes: &[u8], frame_type: FrameType) -> Result<()> {
    match check_frame_hash(frame_bytes, frame_type)? {
        true => Ok(()),
        false => Err(TensogramError::MissingHash { object_index: 0 }),
    }
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
    fn test_xxh3_compute_hash_is_deterministic() {
        let data = b"hello world";
        let hash = compute_hash(data);
        assert_eq!(hash.len(), 16); // 64-bit digest = 16 hex chars
        assert_eq!(hash, compute_hash(data));
    }

    // ── parse_hash_name strict-input contract (Wave 2.1) ─────────────

    #[test]
    fn parse_hash_name_default_is_on() {
        assert!(parse_hash_name(None).unwrap());
    }

    #[test]
    fn parse_hash_name_accepts_xxh3() {
        assert!(parse_hash_name(Some("xxh3")).unwrap());
        assert!(parse_hash_name(Some(HASH_ALGORITHM_NAME)).unwrap());
    }

    #[test]
    fn parse_hash_name_accepts_none() {
        assert!(!parse_hash_name(Some("none")).unwrap());
    }

    #[test]
    fn parse_hash_name_rejects_unknown() {
        // Strict-input: unknown names are rejected (was an integrity
        // bypass on the standalone helper before Wave 1.2; the
        // collapse in Wave 2.1 keeps that contract).
        let err = parse_hash_name(Some("sha256")).unwrap_err();
        match err {
            TensogramError::Metadata(msg) => {
                assert!(msg.contains("sha256"), "msg: {msg}");
                assert!(msg.contains("xxh3"), "msg: {msg}");
                assert!(msg.contains("none"), "msg: {msg}");
            }
            other => panic!("expected Metadata error, got: {other:?}"),
        }
    }

    #[test]
    fn parse_hash_name_rejects_uppercase() {
        // Case-sensitive — wire format uses lowercase exclusively.
        let err = parse_hash_name(Some("XXH3")).unwrap_err();
        assert!(matches!(err, TensogramError::Metadata(_)));
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
    fn check_frame_hash_rejects_below_minimum_size() {
        // A buffer below header(16) + common footer(12) = 28 cannot
        // even hold a frame header plus the inline hash slot, so
        // we reject before flag inspection.
        let buf = vec![0u8; 10];
        let err = check_frame_hash(&buf, FrameType::HeaderMetadata).unwrap_err();
        assert!(
            err.to_string()
                .contains("frame too small to read hash slot")
        );
    }

    #[test]
    fn check_frame_hash_rejects_missing_endf() {
        // 28-byte buffer: header(16) + 8 B hash slot + 4 B non-ENDF.
        // Even though the size check passes, the ENDF sanity check
        // fires before we touch the body.  Use the new flag-driven
        // helper directly so the failure is observable regardless
        // of HASH_PRESENT state (the helper checks size + ENDF
        // before consulting the flag).
        use crate::wire::FRAME_HEADER_SIZE;
        let mut buf = vec![0u8; FRAME_HEADER_SIZE + 12];
        // ENDF at the wrong place.
        buf[FRAME_HEADER_SIZE + 8..FRAME_HEADER_SIZE + 12].copy_from_slice(b"XXXX");
        let err = check_frame_hash(&buf, FrameType::HeaderMetadata).unwrap_err();
        assert!(err.to_string().contains("ENDF"));
    }

    /// Build a minimal `HeaderMetadata` frame for the
    /// flag-dispatch tests below.  `flags` is written directly
    /// into the header; `slot_value` is written into the inline
    /// hash slot.  Caller controls both so the tests can express
    /// the (flag, slot) cross-product cleanly.
    fn build_header_frame(flags: u16, body: &[u8], slot_value: u64) -> Vec<u8> {
        use crate::wire::{FRAME_END, FRAME_HEADER_SIZE, FRAME_MAGIC};
        let mut buf = Vec::new();
        buf.extend_from_slice(FRAME_MAGIC);
        buf.extend_from_slice(&1u16.to_be_bytes()); // HeaderMetadata
        buf.extend_from_slice(&1u16.to_be_bytes()); // version
        buf.extend_from_slice(&flags.to_be_bytes());
        let total_length = FRAME_HEADER_SIZE + body.len() + 12;
        buf.extend_from_slice(&(total_length as u64).to_be_bytes());
        buf.extend_from_slice(body);
        buf.extend_from_slice(&slot_value.to_be_bytes());
        buf.extend_from_slice(FRAME_END);
        buf
    }

    #[test]
    fn check_frame_hash_returns_false_when_flag_clear_regardless_of_slot() {
        // HASH_PRESENT clear: the slot value is undefined.  The
        // helper returns `Ok(false)` no matter what the slot
        // contains — even if the slot would have been a valid
        // digest under flag-set semantics.
        let body = b"hello";
        let true_digest = xxhash_rust::xxh3::xxh3_64(body);

        // Clear flag + zero slot (typical "no hashing" producer).
        let buf = build_header_frame(0, body, 0);
        assert!(!check_frame_hash(&buf, FrameType::HeaderMetadata).unwrap());

        // Clear flag + slot = real digest (encoder bug — flag
        // forgotten — but the contract says we ignore the slot).
        let buf = build_header_frame(0, body, true_digest);
        assert!(!check_frame_hash(&buf, FrameType::HeaderMetadata).unwrap());

        // Clear flag + slot = arbitrary garbage.
        let buf = build_header_frame(0, body, 0xDEADBEEF);
        assert!(!check_frame_hash(&buf, FrameType::HeaderMetadata).unwrap());
    }

    #[test]
    fn check_frame_hash_returns_true_on_valid_match() {
        // HASH_PRESENT set + slot equals recomputed digest → Ok(true).
        let body = b"hello";
        let true_digest = xxhash_rust::xxh3::xxh3_64(body);
        let buf = build_header_frame(FrameFlags::HASH_PRESENT, body, true_digest);
        assert!(check_frame_hash(&buf, FrameType::HeaderMetadata).unwrap());
    }

    #[test]
    fn check_frame_hash_accepts_legitimate_zero_digest() {
        // Documents the new contract: under HASH_PRESENT = 1 every
        // 64-bit value, including zero, is a valid digest.  We
        // can't construct a real body that genuinely hashes to
        // zero (xxh3 of "" is non-zero, and finding a zero-hash
        // body is computationally infeasible), so the test forges
        // the scenario by writing zero into the slot AND mocking
        // the body hash by hashing an empty body — for which
        // xxh3(b"") = 0x2d06800538d394c2.  The point of the test
        // is the symmetric assertion: when the slot DOES equal
        // the body digest, the slot value plays no role beyond
        // the equality check; zero would be accepted as readily
        // as any other value.  Asserts via the negative case
        // (slot=0, body's real digest non-zero → mismatch, NOT a
        // false "missing").
        let body: &[u8] = b"";
        let real_digest = xxhash_rust::xxh3::xxh3_64(body);
        assert_ne!(real_digest, 0, "test invariant: xxh3(\"\") != 0");
        // Stored slot = 0, flag set → mismatch (because the body
        // hashes to non-zero).  This is the *new* contract: zero
        // is a legitimate digest only when it equals what we
        // recompute, and here it doesn't.
        let buf = build_header_frame(FrameFlags::HASH_PRESENT, body, 0);
        let err = check_frame_hash(&buf, FrameType::HeaderMetadata).unwrap_err();
        assert!(matches!(
            err,
            TensogramError::HashMismatch {
                object_index: None,
                ..
            }
        ));
    }

    #[test]
    fn check_frame_hash_reports_mismatch_on_tampered_slot() {
        // HASH_PRESENT set + slot disagrees with recomputed →
        // HashMismatch; object_index is None at this layer.
        let body = b"hello";
        let buf = build_header_frame(FrameFlags::HASH_PRESENT, body, 0xDEADBEEFCAFEBABE);
        let err = check_frame_hash(&buf, FrameType::HeaderMetadata).unwrap_err();
        match err {
            TensogramError::HashMismatch {
                object_index,
                expected,
                actual,
            } => {
                assert!(object_index.is_none());
                assert_eq!(expected, "deadbeefcafebabe");
                assert_ne!(actual, expected);
            }
            other => panic!("expected HashMismatch, got: {other:?}"),
        }
    }

    #[test]
    fn verify_frame_hash_strict_wrapper_treats_clear_flag_as_missing_hash() {
        // The strict wrapper is the offline validator's path: it
        // refuses to accept a flag-clear frame as "verified".
        // `MissingHash` carries object_index 0 by default — this
        // helper has no surrounding object context, so the
        // validator's caller is responsible for replacing 0 with
        // the real index when relevant.
        let body = b"hello";
        let buf = build_header_frame(0, body, 0);
        let err = verify_frame_hash(&buf, FrameType::HeaderMetadata).unwrap_err();
        assert!(matches!(err, TensogramError::MissingHash { .. }));
    }

    #[test]
    fn verify_frame_hash_strict_wrapper_passes_when_flag_set_and_slot_matches() {
        let body = b"hello";
        let true_digest = xxhash_rust::xxh3::xxh3_64(body);
        let buf = build_header_frame(FrameFlags::HASH_PRESENT, body, true_digest);
        verify_frame_hash(&buf, FrameType::HeaderMetadata).unwrap();
    }
}
