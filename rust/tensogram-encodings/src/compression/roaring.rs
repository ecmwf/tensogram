// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Roaring-bitmap compressor for `Dtype::Bitmask` tensor payloads.
//!
//! See [`crate::compression::rle`] for the rationale; this codec
//! wraps the roaring-bitmap encoder from [`crate::bitmask::roaring`]
//! with the same `[u32 BE n_elements][blob]` framing so the
//! compressed output is self-describing.
//!
//! # Dtype restriction
//!
//! Roaring is bitmask-only.  Pipeline-build rejects
//! `compression = "roaring"` on any other dtype, and
//! `decompress_range` returns
//! [`CompressionError::RangeNotSupported`].

use super::{CompressResult, CompressionError, Compressor};
use crate::bitmask::{packing, roaring};

/// Roaring-bitmap compressor over bitmask-packed bytes.
pub struct RoaringCompressor;

impl Compressor for RoaringCompressor {
    fn compress(&self, data: &[u8]) -> Result<CompressResult, CompressionError> {
        // As in `rle`, the packed byte count gives an upper-bound
        // element count.  Callers with a non-byte-aligned element
        // count must pad their packed bytes to the next byte;
        // the pipeline layer handles this automatically.
        let n_elements = data.len() * 8;
        let bits = packing::unpack(data, n_elements)
            .map_err(|e| CompressionError::Unknown(format!("Roaring compress unpack: {e}")))?;
        let blob = roaring::encode(&bits)
            .map_err(|e| CompressionError::Unknown(format!("Roaring encode: {e}")))?;

        let mut out = Vec::with_capacity(4 + blob.len());
        out.extend_from_slice(&(n_elements as u32).to_be_bytes());
        out.extend_from_slice(&blob);
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
        let Some((prefix, blob)) = data.split_first_chunk::<4>() else {
            return Err(CompressionError::Unknown(
                "Roaring blob too short: missing 4-byte n_elements prefix".to_string(),
            ));
        };
        let n_elements = u32::from_be_bytes(*prefix) as usize;
        let bits = roaring::decode(blob, n_elements)
            .map_err(|e| CompressionError::Unknown(format!("Roaring decode: {e}")))?;
        let packed = packing::pack(&bits)
            .map_err(|e| CompressionError::Unknown(format!("Roaring repack: {e}")))?;
        if packed.len() != expected_size {
            return Err(CompressionError::Unknown(format!(
                "Roaring decompressed size {} != expected {}",
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
    /// byte-for-byte, exercising the `n_elements` header and the
    /// pack/unpack bridge around the roaring codec.
    #[test]
    fn roaring_compress_decompress_round_trip() {
        // Sparse set bits suit the roaring representation.
        let packed = vec![0x80u8, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x20];
        let result = RoaringCompressor.compress(&packed).expect("compress");
        assert!(result.data.len() >= 4);
        assert_eq!(
            &result.data[0..4],
            &((packed.len() * 8) as u32).to_be_bytes()
        );
        assert!(result.block_offsets.is_none());

        let restored = RoaringCompressor
            .decompress(&result.data, packed.len())
            .expect("decompress");
        assert_eq!(restored, packed);
    }

    /// An all-set bitmask round-trips byte-exactly.
    #[test]
    fn roaring_round_trip_all_set() {
        let packed = vec![0xFFu8; 12];
        let blob = RoaringCompressor.compress(&packed).unwrap().data;
        let restored = RoaringCompressor.decompress(&blob, packed.len()).unwrap();
        assert_eq!(restored, packed);
    }

    /// A blob shorter than the 4-byte `n_elements` prefix is rejected
    /// without panicking.
    #[test]
    fn roaring_decompress_rejects_short_blob() {
        let err = RoaringCompressor
            .decompress(&[0x00, 0x01], 1)
            .expect_err("blob shorter than the 4-byte prefix must be rejected");
        match err {
            CompressionError::Unknown(msg) => assert!(
                msg.contains("too short"),
                "expected a 'too short' diagnostic, got: {msg}"
            ),
            other => panic!("expected Unknown, got: {other:?}"),
        }
    }

    /// `expected_size` disagreement is reported, not silently accepted.
    #[test]
    fn roaring_decompress_rejects_size_mismatch() {
        let packed = vec![0x01u8, 0x80];
        let blob = RoaringCompressor.compress(&packed).unwrap().data;
        let err = RoaringCompressor
            .decompress(&blob, 99)
            .expect_err("size mismatch must be rejected");
        match err {
            CompressionError::Unknown(msg) => assert!(
                msg.contains("!= expected"),
                "expected a size-mismatch diagnostic, got: {msg}"
            ),
            other => panic!("expected Unknown, got: {other:?}"),
        }
    }

    /// Partial range decode is structurally unsupported for roaring.
    #[test]
    fn roaring_decompress_range_unsupported() {
        let err = RoaringCompressor
            .decompress_range(&[0u8; 8], &[], 0, 1)
            .expect_err("range decode must be unsupported");
        assert!(matches!(err, CompressionError::RangeNotSupported));
    }
}
