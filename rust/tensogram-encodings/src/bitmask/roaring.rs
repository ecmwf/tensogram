// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Roaring-bitmap codec for the bitmask companion frame.
//!
//! Uses the standard
//! [Roaring Portable Serialization Format](https://github.com/RoaringBitmap/RoaringFormatSpec)
//! — the `roaring = "0.11"` crate's `serialize_into` produces it by
//! default.  We store the raw serialised bytes as-is; no
//! Tensogram-specific framing.
//!
//! Pure-Rust and verified to build on `wasm32-unknown-unknown`, so
//! available on every platform without feature-gating — this is the
//! default [`crate::bitmask::MaskMethod`] per
//! `plans/WIRE_FORMAT.md` §6.5.
//!
//! # Index range
//!
//! Roaring operates on `u32` keys.  Tensor element counts up to
//! `u32::MAX ≈ 4.29 × 10⁹` are representable; larger tensors would
//! overflow the key space.  At `u32::MAX` elements at `f64 = 8`
//! bytes/element, that's ~32 GB of payload — well beyond what a
//! single in-memory tensor is in practice.  We surface
//! [`crate::bitmask::MaskError::Malformed`] on overflow rather than
//! silently truncating.

use roaring::RoaringBitmap;

use super::{Bitmask, MaskError};

/// Encode a `&[bool]` as a Roaring-bitmap serialisation.
///
/// Returns [`MaskError::Malformed`] if `bits.len() > u32::MAX` —
/// Roaring's u32 key space can't address positions beyond that.
pub fn encode(bits: &[bool]) -> Result<Vec<u8>, MaskError> {
    if bits.len() > u32::MAX as usize {
        return Err(MaskError::Malformed(format!(
            "roaring mask length {} exceeds u32::MAX addressable positions",
            bits.len()
        )));
    }
    let mut bm = RoaringBitmap::new();
    for (i, &b) in bits.iter().enumerate() {
        if b {
            // Safe: bounds-checked above.
            bm.insert(i as u32);
        }
    }
    // Apply run-container optimisation — dramatically shrinks
    // clustered masks (land / sea, swath gaps) by replacing long
    // array / bitmap containers with per-run descriptors.
    bm.optimize();
    let mut out = Vec::with_capacity(bm.serialized_size());
    bm.serialize_into(&mut out)
        .map_err(|e| MaskError::Roaring(format!("serialize: {e}")))?;
    Ok(out)
}

/// Decode a Roaring-serialised blob into `n_elements` bits.
///
/// The Roaring blob only carries the set bits; the caller tells us
/// the total element count so we know how many `false` bits to emit
/// between and after the set positions.
///
/// Returns [`MaskError::Roaring`] on parse errors, and
/// [`MaskError::Malformed`] if any key in the blob is `≥ n_elements`.
pub fn decode(bytes: &[u8], n_elements: usize) -> Result<Bitmask, MaskError> {
    if n_elements > u32::MAX as usize {
        return Err(MaskError::Malformed(format!(
            "roaring mask length {n_elements} exceeds u32::MAX addressable positions"
        )));
    }
    let bm = RoaringBitmap::deserialize_from(bytes)
        .map_err(|e| MaskError::Roaring(format!("deserialize: {e}")))?;
    let mut out = vec![false; n_elements];
    for key in bm.iter() {
        let k = key as usize;
        if k >= n_elements {
            return Err(MaskError::Malformed(format!(
                "roaring mask has key {k} but only {n_elements} elements declared"
            )));
        }
        out[k] = true;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_empty() {
        let enc = encode(&[]).unwrap();
        // Roaring's empty serialised form is a small fixed-size
        // preamble, not zero bytes — just pin that it round-trips.
        let dec = decode(&enc, 0).unwrap();
        assert!(dec.is_empty());
    }

    #[test]
    fn single_set_bit() {
        for n in [1, 2, 8, 64, 1000] {
            for idx in [0, n / 2, n - 1] {
                let mut bits = vec![false; n];
                bits[idx] = true;
                let enc = encode(&bits).unwrap();
                let dec = decode(&enc, n).unwrap();
                assert_eq!(dec, bits, "n={n} idx={idx}");
            }
        }
    }

    #[test]
    fn all_zeros() {
        let bits = vec![false; 1024];
        let enc = encode(&bits).unwrap();
        let dec = decode(&enc, 1024).unwrap();
        assert_eq!(dec, bits);
    }

    #[test]
    fn all_ones() {
        let bits = vec![true; 1024];
        let enc = encode(&bits).unwrap();
        let dec = decode(&enc, 1024).unwrap();
        assert_eq!(dec, bits);
    }

    #[test]
    fn clustered_mask_compresses_well() {
        // 1000 zeros, 500 ones, 5000 zeros, 500 ones, 500 zeros.
        let mut bits = Vec::new();
        bits.extend(std::iter::repeat_n(false, 1000));
        bits.extend(std::iter::repeat_n(true, 500));
        bits.extend(std::iter::repeat_n(false, 5000));
        bits.extend(std::iter::repeat_n(true, 500));
        bits.extend(std::iter::repeat_n(false, 500));
        let enc = encode(&bits).unwrap();
        // A naive bit-pack of 7500 elements is ~938 bytes; Roaring with
        // run-containers should be substantially smaller.
        assert!(
            enc.len() < 100,
            "roaring should compress clustered masks well, got {} bytes",
            enc.len()
        );
        let dec = decode(&enc, bits.len()).unwrap();
        assert_eq!(dec, bits);
    }

    #[test]
    fn random_sparse_mask() {
        // 10k elements, ~1% set — sparse workload roaring excels at.
        let mut rng: u64 = 0xABCDEF0123456789;
        let bits: Vec<bool> = (0..10_000)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng.is_multiple_of(100)
            })
            .collect();
        let enc = encode(&bits).unwrap();
        let dec = decode(&enc, bits.len()).unwrap();
        assert_eq!(dec, bits);
    }

    #[test]
    fn random_dense_mask() {
        // 10k elements, ~50% set — dense workload roaring uses
        // array / bitmap containers adaptively for.
        let mut rng: u64 = 0xFEDCBA9876543210;
        let bits: Vec<bool> = (0..10_000)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                (rng & 1) == 1
            })
            .collect();
        let enc = encode(&bits).unwrap();
        let dec = decode(&enc, bits.len()).unwrap();
        assert_eq!(dec, bits);
    }

    #[test]
    fn decode_key_out_of_range_rejected() {
        // Hand-craft a mask with position 10 set, then declare only 5 elements.
        let bits = (0..11).map(|i| i == 10).collect::<Vec<_>>();
        let enc = encode(&bits).unwrap();
        let err = decode(&enc, 5).unwrap_err();
        assert!(matches!(err, MaskError::Malformed(_)));
    }

    #[test]
    fn decode_garbage_input_rejected() {
        // Random bytes that don't parse as a roaring blob.
        let garbage = b"this is not a roaring bitmap";
        let err = decode(garbage, 100).unwrap_err();
        assert!(matches!(err, MaskError::Roaring(_)));
    }
}
