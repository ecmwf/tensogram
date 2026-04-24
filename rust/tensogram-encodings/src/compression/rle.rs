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
