// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Run-length encoding for bit-level bitmasks.
//!
//! Best on highly-clustered masks (land / sea masks, swath gaps,
//! contiguous missing-data regions).  Worst on random alternating
//! bits — the encoder-integration layer handles that case by falling
//! back to [`crate::bitmask::MaskMethod::None`] (raw packed bytes) for
//! masks below the `small_mask_threshold_bytes`.
//!
//! # On-wire format
//!
//! ```text
//! [u8 start_bit][varint run_1][varint run_2] ... [varint run_k]
//! ```
//!
//! - `start_bit`: `0x00` or `0x01` — the value of the first run.  Any
//!   other byte is a malformed-payload error.
//! - Each `run_i` is unsigned LEB128 (ULEB128) — count of consecutive
//!   bits of the alternating value.  Minimum run length 1.
//! - Run counts sum to exactly the declared element count; decode
//!   returns [`crate::bitmask::MaskError::Rle`] otherwise.
//! - The element count itself is NOT stored in the run-length blob —
//!   it comes from the enclosing CBOR descriptor's `shape`.  The
//!   caller tells us how many elements to decode.
//!
//! # Edge cases
//!
//! - Empty mask (`n_elements == 0`): encodes to a zero-byte blob.
//! - All-zero mask: `[0x00, varint(n)]`.
//! - All-one mask:  `[0x01, varint(n)]`.
//! - Alternating bits (worst case): `[start, 1, 1, 1, ...]` — at ≥1
//!   byte per run, this inflates the byte count vs bit-packed.  The
//!   small-mask fallback should prevent this being an issue in
//!   practice.

use super::{Bitmask, MaskError};

/// Encode a `&[bool]` as the RLE wire format described in the module
/// doc.
pub fn encode(bits: &[bool]) -> Vec<u8> {
    if bits.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(1 + bits.len() / 4);
    out.push(if bits[0] { 1 } else { 0 });

    let mut current = bits[0];
    let mut run: u64 = 1;
    for &b in &bits[1..] {
        if b == current {
            run += 1;
        } else {
            write_uleb128(&mut out, run);
            current = b;
            run = 1;
        }
    }
    write_uleb128(&mut out, run);
    out
}

/// Decode `n_elements` bits from the RLE wire format.
///
/// Returns [`MaskError::Rle`] on:
/// - empty blob with `n_elements > 0`,
/// - invalid `start_bit` byte (not 0 or 1),
/// - truncated varint,
/// - run-count sum ≠ `n_elements`.
pub fn decode(bytes: &[u8], n_elements: usize) -> Result<Bitmask, MaskError> {
    if n_elements == 0 {
        // Empty blob is valid only for zero-element masks.  Any payload
        // bytes on a zero-element request is a malformed input.
        return if bytes.is_empty() {
            Ok(Vec::new())
        } else {
            Err(MaskError::Rle(format!(
                "zero-element mask must have empty payload; got {} bytes",
                bytes.len()
            )))
        };
    }
    if bytes.is_empty() {
        return Err(MaskError::Rle(format!(
            "empty RLE payload but {n_elements} elements declared"
        )));
    }

    let start_bit = match bytes[0] {
        0 => false,
        1 => true,
        other => {
            return Err(MaskError::Rle(format!(
                "invalid start_bit byte {other:#04x} (must be 0x00 or 0x01)"
            )));
        }
    };

    let mut out = Vec::with_capacity(n_elements);
    let mut current = start_bit;
    let mut cursor = 1;
    while cursor < bytes.len() {
        let (run, consumed) = read_uleb128(&bytes[cursor..])?;
        cursor += consumed;
        if run == 0 {
            return Err(MaskError::Rle(
                "zero-length run is not permitted by the RLE format".to_string(),
            ));
        }
        // Check overflow of the decoded count
        let run_usize = usize::try_from(run).map_err(|_| {
            MaskError::Rle(format!(
                "run count {run} exceeds usize — malformed or truncated"
            ))
        })?;
        if out.len() + run_usize > n_elements {
            return Err(MaskError::Rle(format!(
                "run overruns element count: decoded {}..{} but only {n_elements} declared",
                out.len(),
                out.len() + run_usize
            )));
        }
        for _ in 0..run_usize {
            out.push(current);
        }
        current = !current;
    }

    if out.len() != n_elements {
        return Err(MaskError::Rle(format!(
            "RLE decoded {} elements but {n_elements} declared",
            out.len()
        )));
    }
    Ok(out)
}

// ── ULEB128 helpers ─────────────────────────────────────────────────────────

fn write_uleb128(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn read_uleb128(bytes: &[u8]) -> Result<(u64, usize), MaskError> {
    let mut value: u64 = 0;
    let mut shift = 0;
    for (i, &b) in bytes.iter().enumerate() {
        // 10 bytes is the maximum ULEB128 encoding for u64.
        if i >= 10 {
            return Err(MaskError::Rle(
                "ULEB128 integer overflows u64 — malformed payload".to_string(),
            ));
        }
        let chunk = (b & 0x7F) as u64;
        value |= chunk
            .checked_shl(shift as u32)
            .ok_or_else(|| MaskError::Rle(format!("ULEB128 shift overflow at byte {i}")))?;
        if (b & 0x80) == 0 {
            return Ok((value, i + 1));
        }
        shift += 7;
    }
    Err(MaskError::Rle(
        "truncated ULEB128: missing terminator byte".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_empty() {
        assert_eq!(encode(&[]), Vec::<u8>::new());
    }

    #[test]
    fn decode_empty() {
        assert_eq!(decode(&[], 0).unwrap(), Vec::<bool>::new());
    }

    #[test]
    fn decode_empty_mismatched() {
        let err = decode(&[], 10).unwrap_err();
        assert!(matches!(err, MaskError::Rle(_)));
    }

    #[test]
    fn single_bit_zero() {
        let enc = encode(&[false]);
        assert_eq!(enc, vec![0x00, 0x01]);
        assert_eq!(decode(&enc, 1).unwrap(), vec![false]);
    }

    #[test]
    fn single_bit_one() {
        let enc = encode(&[true]);
        assert_eq!(enc, vec![0x01, 0x01]);
        assert_eq!(decode(&enc, 1).unwrap(), vec![true]);
    }

    #[test]
    fn all_zeros_encodes_as_single_run() {
        let bits = vec![false; 1000];
        let enc = encode(&bits);
        // start_bit=0, varint(1000) = 0xE8 0x07
        assert_eq!(enc, vec![0x00, 0xE8, 0x07]);
        assert_eq!(decode(&enc, 1000).unwrap(), bits);
    }

    #[test]
    fn all_ones_encodes_as_single_run() {
        let bits = vec![true; 64];
        let enc = encode(&bits);
        assert_eq!(enc, vec![0x01, 0x40]);
        assert_eq!(decode(&enc, 64).unwrap(), bits);
    }

    #[test]
    fn alternating_bits_is_worst_case() {
        // N=8 alternating → 9 bytes of output (1 start + 8 × 1-byte varint),
        // whereas bit-packing would be 1 byte.  Documented inflation; the
        // small-mask threshold avoids emitting this in practice.
        let bits: Vec<bool> = (0..8).map(|i| i % 2 == 0).collect();
        let enc = encode(&bits);
        assert_eq!(enc.len(), 1 + 8);
        assert_eq!(decode(&enc, 8).unwrap(), bits);
    }

    #[test]
    fn clustered_mask_round_trips() {
        // 100 zeros, 50 ones, 200 zeros, 50 ones.
        let mut bits = Vec::new();
        bits.extend(std::iter::repeat_n(false, 100));
        bits.extend(std::iter::repeat_n(true, 50));
        bits.extend(std::iter::repeat_n(false, 200));
        bits.extend(std::iter::repeat_n(true, 50));
        let enc = encode(&bits);
        // start_bit + 4 varints
        assert!(
            enc.len() < 10,
            "clustered mask should be tiny: got {} bytes",
            enc.len()
        );
        assert_eq!(decode(&enc, bits.len()).unwrap(), bits);
    }

    #[test]
    fn large_mask_roundtrip() {
        // 100k elements, pseudo-random clusters.
        let mut bits = Vec::new();
        let mut on = false;
        for run_len in [50, 100, 5, 1000, 200, 7, 42, 98765] {
            bits.extend(std::iter::repeat_n(on, run_len));
            on = !on;
        }
        let n = bits.len();
        let enc = encode(&bits);
        assert_eq!(decode(&enc, n).unwrap(), bits);
    }

    #[test]
    fn decode_invalid_start_bit() {
        let err = decode(&[0x42, 0x01], 1).unwrap_err();
        assert!(matches!(err, MaskError::Rle(_)));
    }

    #[test]
    fn decode_truncated_varint() {
        // start_bit + varint with continuation bit set but no following byte.
        let err = decode(&[0x00, 0x80], 1).unwrap_err();
        assert!(matches!(err, MaskError::Rle(_)));
    }

    #[test]
    fn decode_run_overruns_declared_count() {
        let err = decode(&[0x00, 0x0A], 3).unwrap_err();
        // 0x0A = 10, but n=3 → overruns.
        assert!(matches!(err, MaskError::Rle(_)));
    }

    #[test]
    fn decode_run_undershoots_declared_count() {
        // 2 zeros, 1 one = 3 bits, but declare 5.
        let enc = vec![0x00, 0x02, 0x01];
        let err = decode(&enc, 5).unwrap_err();
        assert!(matches!(err, MaskError::Rle(_)));
    }

    #[test]
    fn uleb128_multi_byte_boundary() {
        // Values > 0x7F exercise multi-byte varints.
        for val in [
            0x80_u64,
            0x3FFF,
            0x4000,
            0xFFFF,
            0x100_000,
            u32::MAX as u64,
            u64::MAX,
        ] {
            let mut buf = Vec::new();
            write_uleb128(&mut buf, val);
            let (got, consumed) = read_uleb128(&buf).unwrap();
            assert_eq!(got, val);
            assert_eq!(consumed, buf.len());
        }
    }

    // ── Property-style sweep: many random-ish masks round-trip ─────────

    #[test]
    fn sweep_random_masks_round_trip() {
        // Pseudo-random masks of various sizes; deterministic seed so
        // the test is reproducible.
        let mut rng = rand_state(0xDEADBEEF_CAFEBABE);
        for size in [0, 1, 2, 7, 8, 9, 63, 64, 100, 1_000, 10_000] {
            let bits: Vec<bool> = (0..size).map(|_| rng_next(&mut rng) & 1 == 1).collect();
            let enc = encode(&bits);
            let dec = decode(&enc, size).unwrap();
            assert_eq!(dec, bits, "size={size}");
        }
    }

    // ── tiny xorshift64* for the sweep test ──────────────────────────────

    type State = u64;
    fn rand_state(seed: u64) -> State {
        seed.max(1)
    }
    fn rng_next(s: &mut State) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }
}
