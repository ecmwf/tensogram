// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! RLE compressor for `Dtype::Bitmask` tensor payloads.
//!
//! This codec takes MSB-first packed bitmask bytes (the canonical
//! on-wire form for bitmask tensors — see `plans/WIRE_FORMAT.md`
//! §8), unpacks them to a `Vec<bool>`, applies the bit-level
//! run-length encoding from
//! [`crate::bitmask::rle`], and prepends a 4-byte big-endian
//! `n_elements` header so the decoder can unambiguously recover
//! the pre-pack element count (required for byte-exact unpack
//! when the element count is not a multiple of 8).
//!
//! Wire layout of the compressed blob:
//!
//! ```text
//! [u32 BE n_elements][ rle-encoded bytes from crate::bitmask::rle ]
//! ```
//!
//! # Dtype restriction
//!
//! RLE is a bitmask-specific codec and has no meaning on byte /
//! integer / float payloads.  The pipeline-build layer
//! (`pipeline::build_compressor`) rejects
//! `compression = "rle"` unless `dtype = "bitmask"`; this
//! compressor's `decompress_range` additionally returns
//! [`CompressionError::RangeNotSupported`] so callers get a clean
//! error rather than mysterious behaviour when they try partial
//! decode.

use super::{CompressResult, CompressionError, Compressor};
use crate::bitmask::{packing, rle};

/// RLE compressor over bitmask-packed bytes.
pub struct RleCompressor;

impl Compressor for RleCompressor {
    fn compress(&self, data: &[u8]) -> Result<CompressResult, CompressionError> {
        // The pipeline passes the packed bitmask bytes.  We treat
        // `n_elements = data.len() * 8` as the canonical upper
        // bound so `unpack` reads the full input; the pipeline is
        // responsible for trimming trailing bits when the caller
        // hands in a packed payload whose element count isn't a
        // multiple of 8.
        let n_elements = data.len() * 8;
        let bits = packing::unpack(data, n_elements)
            .map_err(|e| CompressionError::Unknown(format!("RLE compress unpack: {e}")))?;
        let rle_bytes = rle::encode(&bits);

        let mut out = Vec::with_capacity(4 + rle_bytes.len());
        out.extend_from_slice(&(n_elements as u32).to_be_bytes());
        out.extend_from_slice(&rle_bytes);
        Ok(CompressResult {
            data: out,
            block_offsets: None,
        })
    }

    fn decompress(&self, data: &[u8], expected_size: usize) -> Result<Vec<u8>, CompressionError> {
        // Split off the 4-byte n_elements prefix without
        // `try_into().unwrap()` so the function is panic-free by
        // construction on any input, even when a future caller
        // bypasses the length check.
        let Some((prefix, rle_bytes)) = data.split_first_chunk::<4>() else {
            return Err(CompressionError::Unknown(
                "RLE blob too short: missing 4-byte n_elements prefix".to_string(),
            ));
        };
        let n_elements = u32::from_be_bytes(*prefix) as usize;
        let bits = rle::decode(rle_bytes, n_elements)
            .map_err(|e| CompressionError::Unknown(format!("RLE decode: {e}")))?;
        let packed = packing::pack(&bits)
            .map_err(|e| CompressionError::Unknown(format!("RLE repack: {e}")))?;
        if packed.len() != expected_size {
            return Err(CompressionError::Unknown(format!(
                "RLE decompressed size {} != expected {}",
                packed.len(),
                expected_size
            )));
        }
        Ok(packed)
    }

    fn decompress_range(
        &self,
        _data: &[u8],
        _block_offsets: &[u64],
        _byte_pos: usize,
        _byte_size: usize,
    ) -> Result<Vec<u8>, CompressionError> {
        Err(CompressionError::RangeNotSupported)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Packed bitmask bytes round-trip through compress → decompress
    /// byte-for-byte.  Exercises the `n_elements` header write/read, the
    /// pack/unpack bridge, and the expected-size check on the happy path.
    #[test]
    fn rle_compress_decompress_round_trip() {
        // A mix of runs so the RLE body is non-trivial: 0xFF (8 set),
        // 0x00 (8 clear), 0xAA (alternating).
        let packed = vec![0xFFu8, 0x00, 0xAA, 0x0F];
        let result = RleCompressor.compress(&packed).expect("compress");
        // The blob carries the 4-byte big-endian element count prefix.
        assert!(result.data.len() >= 4);
        assert_eq!(
            &result.data[0..4],
            &((packed.len() * 8) as u32).to_be_bytes()
        );
        assert!(result.block_offsets.is_none());

        let restored = RleCompressor
            .decompress(&result.data, packed.len())
            .expect("decompress");
        assert_eq!(restored, packed);
    }

    /// An all-zero bitmask is the RLE best case (one long run); it must
    /// still round-trip byte-exactly.
    #[test]
    fn rle_round_trip_all_zero() {
        let packed = vec![0x00u8; 16];
        let blob = RleCompressor.compress(&packed).unwrap().data;
        let restored = RleCompressor.decompress(&blob, packed.len()).unwrap();
        assert_eq!(restored, packed);
    }

    /// A blob shorter than the 4-byte `n_elements` prefix is rejected
    /// with a clear error rather than panicking on the missing chunk.
    #[test]
    fn rle_decompress_rejects_short_blob() {
        let err = RleCompressor
            .decompress(&[0x00, 0x01, 0x02], 1)
            .expect_err("blob shorter than the 4-byte prefix must be rejected");
        match err {
            CompressionError::Unknown(msg) => assert!(
                msg.contains("too short"),
                "expected a 'too short' diagnostic, got: {msg}"
            ),
            other => panic!("expected Unknown, got: {other:?}"),
        }
    }

    /// When the caller's `expected_size` disagrees with the repacked
    /// length, decompress reports the mismatch instead of silently
    /// returning the wrong number of bytes.
    #[test]
    fn rle_decompress_rejects_size_mismatch() {
        let packed = vec![0xFFu8, 0x00];
        let blob = RleCompressor.compress(&packed).unwrap().data;
        // Real size is 2 bytes; claim 3.
        let err = RleCompressor
            .decompress(&blob, 3)
            .expect_err("size mismatch must be rejected");
        match err {
            CompressionError::Unknown(msg) => assert!(
                msg.contains("!= expected"),
                "expected a size-mismatch diagnostic, got: {msg}"
            ),
            other => panic!("expected Unknown, got: {other:?}"),
        }
    }

    /// Partial range decode is structurally unsupported for RLE.
    #[test]
    fn rle_decompress_range_unsupported() {
        let err = RleCompressor
            .decompress_range(&[0u8; 8], &[], 0, 1)
            .expect_err("range decode must be unsupported");
        assert!(matches!(err, CompressionError::RangeNotSupported));
    }
}
