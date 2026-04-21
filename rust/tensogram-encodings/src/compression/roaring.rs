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
        let packed = packing::pack(&bits);
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
