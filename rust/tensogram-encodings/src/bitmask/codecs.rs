// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Thin mask-codec wrappers around existing compressors:
//! Zstd, LZ4, Blosc2 (BitShuffle + sub-codec), and `None` (passthrough).
//!
//! The RLE and Roaring codecs have their own modules ([`super::rle`],
//! [`super::roaring`]) because they operate directly on bit patterns
//! rather than bit-packed bytes.  The codecs in this module compress
//! the **bit-packed** representation (see [`super::packing`]).

use super::{Bitmask, MaskError, packing};

// ── None (passthrough — raw bit-packed bytes) ───────────────────────────────

/// Pack `bits` MSB-first into bytes; no further compression.  This is
/// the `method = "none"` on-wire format, chosen automatically by the
/// encoder-integration layer for masks smaller than
/// `small_mask_threshold_bytes` (default 128).
pub fn encode_none(bits: &[bool]) -> Vec<u8> {
    packing::pack(bits)
}

/// Unpack raw MSB-first bit-packed bytes into a [`Bitmask`].
pub fn decode_none(bytes: &[u8], n_elements: usize) -> Result<Bitmask, MaskError> {
    packing::unpack(bytes, n_elements)
}

// ── LZ4 ─────────────────────────────────────────────────────────────────────

#[cfg(feature = "lz4")]
pub fn encode_lz4(bits: &[bool]) -> Result<Vec<u8>, MaskError> {
    let packed = packing::pack(bits);
    Ok(lz4_flex::compress_prepend_size(&packed))
}

#[cfg(feature = "lz4")]
pub fn decode_lz4(bytes: &[u8], n_elements: usize) -> Result<Bitmask, MaskError> {
    let packed = lz4_flex::decompress_size_prepended(bytes)
        .map_err(|e| MaskError::Codec(format!("lz4 decompress: {e}")))?;
    packing::unpack(&packed, n_elements)
}

#[cfg(not(feature = "lz4"))]
pub fn encode_lz4(_bits: &[bool]) -> Result<Vec<u8>, MaskError> {
    Err(MaskError::FeatureDisabled { method: "lz4" })
}

#[cfg(not(feature = "lz4"))]
pub fn decode_lz4(_bytes: &[u8], _n_elements: usize) -> Result<Bitmask, MaskError> {
    Err(MaskError::FeatureDisabled { method: "lz4" })
}

// ── Zstd ────────────────────────────────────────────────────────────────────

#[cfg(feature = "zstd")]
pub fn encode_zstd(bits: &[bool], level: Option<i32>) -> Result<Vec<u8>, MaskError> {
    let packed = packing::pack(bits);
    let level = level.unwrap_or(3);
    zstd::encode_all(packed.as_slice(), level)
        .map_err(|e| MaskError::Codec(format!("zstd encode: {e}")))
}

#[cfg(feature = "zstd")]
pub fn decode_zstd(bytes: &[u8], n_elements: usize) -> Result<Bitmask, MaskError> {
    let packed =
        zstd::decode_all(bytes).map_err(|e| MaskError::Codec(format!("zstd decode: {e}")))?;
    packing::unpack(&packed, n_elements)
}

#[cfg(not(feature = "zstd"))]
pub fn encode_zstd(_bits: &[bool], _level: Option<i32>) -> Result<Vec<u8>, MaskError> {
    Err(MaskError::FeatureDisabled { method: "zstd" })
}

#[cfg(not(feature = "zstd"))]
pub fn decode_zstd(_bytes: &[u8], _n_elements: usize) -> Result<Bitmask, MaskError> {
    Err(MaskError::FeatureDisabled { method: "zstd" })
}

// ── Blosc2 (BitShuffle + sub-codec) ─────────────────────────────────────────
//
// Uses blosc2's SChunk interface with a BitShuffle filter layered on
// top of the chosen sub-codec.  BitShuffle works at the bit level
// within each typesize-aligned group; we pack the mask into bytes
// first and set typesize=1 so blosc2 bit-shuffles at byte granularity
// (which produces strong compression on packed bitmasks).

#[cfg(feature = "blosc2")]
pub fn encode_blosc2(
    bits: &[bool],
    codec: crate::pipeline::Blosc2Codec,
    level: i32,
) -> Result<Vec<u8>, MaskError> {
    use blosc2::chunk::SChunk;
    use blosc2::{CParams, DParams, Filter};

    let packed = packing::pack(bits);
    if packed.is_empty() {
        return Ok(Vec::new());
    }

    let algo = crate::compression::blosc2::codec_to_algo(&codec);

    let mut cparams = CParams::default();
    cparams
        .compressor(algo)
        .clevel(level.clamp(1, 9) as u32)
        .typesize(1)
        .map_err(|e| MaskError::Codec(format!("blosc2 cparams: {e}")))?;
    cparams
        .filters(&[Filter::BitShuffle])
        .map_err(|e| MaskError::Codec(format!("blosc2 filter: {e}")))?;

    let dparams = DParams::default();
    let mut schunk = SChunk::new(cparams, dparams)
        .map_err(|e| MaskError::Codec(format!("blosc2 schunk: {e}")))?;
    schunk
        .append(&packed)
        .map_err(|e| MaskError::Codec(format!("blosc2 append: {e}")))?;
    let buf = schunk
        .to_buffer()
        .map_err(|e| MaskError::Codec(format!("blosc2 to_buffer: {e}")))?;
    Ok(buf.as_slice().to_vec())
}

#[cfg(feature = "blosc2")]
pub fn decode_blosc2(bytes: &[u8], n_elements: usize) -> Result<Bitmask, MaskError> {
    use blosc2::chunk::SChunk;

    if bytes.is_empty() {
        return packing::unpack(&[], n_elements);
    }
    let schunk = SChunk::from_buffer(bytes.into())
        .map_err(|e| MaskError::Codec(format!("blosc2 from_buffer: {e}")))?;
    let num_items = schunk.items_num();
    let packed = if num_items == 0 {
        Vec::new()
    } else {
        schunk
            .items(0..num_items)
            .map_err(|e| MaskError::Codec(format!("blosc2 items: {e}")))?
    };
    packing::unpack(&packed, n_elements)
}

#[cfg(not(feature = "blosc2"))]
pub fn encode_blosc2(_bits: &[bool], _codec: (), _level: i32) -> Result<Vec<u8>, MaskError> {
    Err(MaskError::FeatureDisabled { method: "blosc2" })
}

#[cfg(not(feature = "blosc2"))]
pub fn decode_blosc2(_bytes: &[u8], _n_elements: usize) -> Result<Bitmask, MaskError> {
    Err(MaskError::FeatureDisabled { method: "blosc2" })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_mask() -> Bitmask {
        let mut bits = vec![false; 1024];
        // A few clustered runs + a sparse region.
        bits[100..200].fill(true);
        bits[500..510].fill(true);
        for idx in [700, 750, 800] {
            bits[idx] = true;
        }
        bits
    }

    #[test]
    fn none_roundtrip() {
        let bits = sample_mask();
        let enc = encode_none(&bits);
        let dec = decode_none(&enc, bits.len()).unwrap();
        assert_eq!(dec, bits);
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn lz4_roundtrip() {
        let bits = sample_mask();
        let enc = encode_lz4(&bits).unwrap();
        let dec = decode_lz4(&enc, bits.len()).unwrap();
        assert_eq!(dec, bits);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn zstd_roundtrip_default_level() {
        let bits = sample_mask();
        let enc = encode_zstd(&bits, None).unwrap();
        let dec = decode_zstd(&enc, bits.len()).unwrap();
        assert_eq!(dec, bits);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn zstd_roundtrip_level_1_and_22() {
        let bits = sample_mask();
        for level in [1, 3, 22] {
            let enc = encode_zstd(&bits, Some(level)).unwrap();
            let dec = decode_zstd(&enc, bits.len()).unwrap();
            assert_eq!(dec, bits, "level {level}");
        }
    }

    #[cfg(feature = "blosc2")]
    #[test]
    fn blosc2_roundtrip_all_codecs() {
        use crate::pipeline::Blosc2Codec;
        let bits = sample_mask();
        for codec in [
            Blosc2Codec::Blosclz,
            Blosc2Codec::Lz4,
            Blosc2Codec::Lz4hc,
            Blosc2Codec::Zlib,
            Blosc2Codec::Zstd,
        ] {
            let enc = encode_blosc2(&bits, codec, 5).unwrap();
            let dec = decode_blosc2(&enc, bits.len()).unwrap();
            assert_eq!(dec, bits, "codec {codec:?}");
        }
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn lz4_garbage_rejected() {
        let err = decode_lz4(b"not a valid lz4 stream", 8).unwrap_err();
        assert!(matches!(err, MaskError::Codec(_)));
    }
}
