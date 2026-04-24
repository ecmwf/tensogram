// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! MSB-first bit packing / unpacking for the bitmask companion frame.
//!
//! Layout (matches `plans/WIRE_FORMAT.md` §6.5.2):
//!
//! ```text
//! element index    0  1  2  3  4  5  6  7 | 8  9 10 11 12 13 14 15 | ...
//! bit position     b7 b6 b5 b4 b3 b2 b1 b0 | b7 b6 b5 b4 b3 b2 b1 b0 | ...
//! ```
//!
//! Element `i` lives in byte `i / 8` at bit position `7 - (i % 8)`.
//! Trailing bits in the last byte (when `N % 8 != 0`) are
//! **zero-filled** so the packed representation is deterministic and
//! bit-exact across encoders — required for hashing and golden-file
//! parity.

use super::{Bitmask, MaskError};

/// Pack a `&[bool]` into MSB-first bit-packed bytes.
///
/// Returns `ceil(bits.len() / 8)` bytes.  Trailing bits in the final
/// byte are zero-filled.
pub fn pack(bits: &[bool]) -> Vec<u8> {
    let n_bytes = bits.len().div_ceil(8);
    let mut out = vec![0u8; n_bytes];
    for (i, &b) in bits.iter().enumerate() {
        if b {
            out[i / 8] |= 1 << (7 - (i % 8));
        }
    }
    out
}

/// Unpack MSB-first bit-packed bytes into `n_elements` bools.
///
/// Returns [`MaskError::LengthMismatch`] if the input byte count does
/// not match `ceil(n_elements / 8)`.  Trailing bits in the final byte
/// are read but ignored — per spec they are zero-filled on encode, but
/// we do not fail on non-zero trailing padding (lenient on decode,
/// strict on encode).
pub fn unpack(bytes: &[u8], n_elements: usize) -> Result<Bitmask, MaskError> {
    let expected = n_elements.div_ceil(8);
    if bytes.len() != expected {
        return Err(MaskError::LengthMismatch {
            expected,
            actual: bytes.len(),
        });
    }
    let mut out: Vec<bool> = Vec::new();
    super::try_reserve_mask(&mut out, n_elements)?;
    for i in 0..n_elements {
        let bit = (bytes[i / 8] >> (7 - (i % 8))) & 1;
        out.push(bit == 1);
    }
    Ok(out)
}

/// Count the `true` bits in a bitmask.  Equivalent to `bits.iter().filter(|b| **b).count()`
/// but kept as a named helper for readability at call sites that are
/// computing mask density.
pub fn popcount(bits: &[bool]) -> usize {
    bits.iter().filter(|b| **b).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_empty() {
        assert!(pack(&[]).is_empty());
    }

    #[test]
    fn pack_single_bit_set() {
        assert_eq!(pack(&[true]), vec![0b1000_0000]);
        assert_eq!(pack(&[false]), vec![0b0000_0000]);
    }

    #[test]
    fn pack_full_byte_msb_first() {
        let bits = [true, true, true, true, false, false, false, false];
        assert_eq!(pack(&bits), vec![0b1111_0000]);
    }

    #[test]
    fn pack_partial_final_byte_zero_filled() {
        // 10 bits: 1010_1010_10_______ → 0xAA, 0b1000_0000
        let bits = [
            true, false, true, false, true, false, true, false, true, false,
        ];
        assert_eq!(pack(&bits), vec![0b1010_1010, 0b1000_0000]);
    }

    #[test]
    fn pack_multi_byte() {
        // 16 bits, alternating — yields two 0xAA bytes.
        let bits: Vec<bool> = (0..16).map(|i| i % 2 == 0).collect();
        assert_eq!(pack(&bits), vec![0xAA, 0xAA]);
    }

    #[test]
    fn unpack_roundtrip() {
        for n in [1usize, 7, 8, 9, 15, 16, 17, 64, 65, 256, 1000] {
            let bits: Vec<bool> = (0..n).map(|i| i % 3 == 0 || i % 7 == 1).collect();
            let packed = pack(&bits);
            assert_eq!(packed.len(), n.div_ceil(8));
            let unpacked = unpack(&packed, n).unwrap();
            assert_eq!(unpacked, bits, "roundtrip failed for n={n}");
        }
    }

    #[test]
    fn unpack_length_mismatch() {
        let packed = vec![0xFF; 3]; // 3 bytes = up to 24 bits
        // Declaring 32 elements needs 4 bytes, so 3-byte input is wrong.
        let err = unpack(&packed, 32).unwrap_err();
        assert!(matches!(
            err,
            MaskError::LengthMismatch {
                expected: 4,
                actual: 3
            }
        ));
    }

    #[test]
    fn unpack_lenient_on_trailing_bits() {
        // N=10 packs to 2 bytes but has 6 trailing padding bits.
        // Unpack should NOT care about the padding's value.
        // Packed: 0b1111_1111, 0b1111_1100 — first 10 bits all set, rest set.
        let packed = vec![0xFF, 0xFC];
        let unpacked = unpack(&packed, 10).unwrap();
        assert_eq!(unpacked, vec![true; 10]);
    }

    #[test]
    fn popcount_various() {
        assert_eq!(popcount(&[]), 0);
        assert_eq!(popcount(&[false; 10]), 0);
        assert_eq!(popcount(&[true; 10]), 10);
        assert_eq!(popcount(&[true, false, true, true, false, true]), 4);
    }
}
