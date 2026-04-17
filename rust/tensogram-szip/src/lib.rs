// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Pure-Rust CCSDS 121.0-B-3 Adaptive Entropy Coding (AEC/SZIP).
//!
//! This crate provides encode, decode, and range-decode functions with
//! the same `AecParams` interface as the C libaec library. It can be
//! used as a drop-in replacement for `libaec-sys` in environments where
//! C FFI is unavailable (e.g. WebAssembly).
//!
//! # Example
//!
//! ```
//! use tensogram_szip::{aec_compress, aec_decompress, AecParams, AEC_DATA_PREPROCESS};
//!
//! let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
//! let params = AecParams {
//!     bits_per_sample: 8,
//!     block_size: 16,
//!     rsi: 128,
//!     flags: AEC_DATA_PREPROCESS,
//! };
//!
//! let (compressed, offsets) = aec_compress(&data, &params).unwrap();
//! let decompressed = aec_decompress(&compressed, data.len(), &params).unwrap();
//! assert_eq!(decompressed, data);
//! ```

mod bitstream;
mod decoder;
mod encoder;
mod error;
pub mod params;
mod preprocessor;

pub use error::AecError;
pub use params::{
    AEC_ALLOW_K13, AEC_DATA_3BYTE, AEC_DATA_MSB, AEC_DATA_PREPROCESS, AEC_DATA_SIGNED,
    AEC_NOT_ENFORCE, AEC_PAD_RSI, AEC_RESTRICTED, AecParams,
};

/// Compress data using CCSDS 121.0-B-3 adaptive entropy coding.
///
/// Returns `(compressed_bytes, rsi_block_bit_offsets)` where the
/// offsets track the bit position of each RSI boundary in the
/// compressed stream — needed for [`aec_decompress_range`].
pub fn aec_compress(data: &[u8], params: &AecParams) -> Result<(Vec<u8>, Vec<u64>), AecError> {
    params::validate(params)?;
    encoder::encode(data, params, true)
}

/// Compress data without tracking RSI block offsets (slightly faster).
pub fn aec_compress_no_offsets(data: &[u8], params: &AecParams) -> Result<Vec<u8>, AecError> {
    params::validate(params)?;
    let (bytes, _) = encoder::encode(data, params, false)?;
    Ok(bytes)
}

/// Decompress an entire AEC-compressed stream.
///
/// `expected_size` is the expected decompressed size in bytes.
pub fn aec_decompress(
    data: &[u8],
    expected_size: usize,
    params: &AecParams,
) -> Result<Vec<u8>, AecError> {
    params::validate(params)?;
    decoder::decode(data, expected_size, params)
}

/// Decompress a partial byte range from AEC-compressed data using
/// pre-computed RSI block bit offsets.
///
/// `block_offsets` are the bit offsets returned by [`aec_compress`].
/// `byte_pos` and `byte_size` specify the byte range within the
/// decompressed output to extract.
pub fn aec_decompress_range(
    data: &[u8],
    block_offsets: &[u64],
    byte_pos: usize,
    byte_size: usize,
    params: &AecParams,
) -> Result<Vec<u8>, AecError> {
    params::validate(params)?;
    decoder::decode_range(data, block_offsets, byte_pos, byte_size, params)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params(bits_per_sample: u32) -> AecParams {
        AecParams {
            bits_per_sample,
            block_size: 16,
            rsi: 128,
            flags: AEC_DATA_PREPROCESS,
        }
    }

    #[test]
    fn round_trip_u8() {
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let params = default_params(8);

        let (compressed, offsets) = aec_compress(&data, &params).unwrap();
        assert!(!compressed.is_empty());
        assert!(!offsets.is_empty());

        let decompressed = aec_decompress(&compressed, data.len(), &params).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn round_trip_u16() {
        let values: Vec<u16> = (0..2048).map(|i| (i * 7 % 65536) as u16).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let params = default_params(16);

        let (compressed, offsets) = aec_compress(&data, &params).unwrap();
        assert!(!compressed.is_empty());
        assert!(!offsets.is_empty());

        let decompressed = aec_decompress(&compressed, data.len(), &params).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn round_trip_u24() {
        let n = 4096;
        let data: Vec<u8> = (0..n * 3).map(|i| (i % 256) as u8).collect();
        let params = default_params(24);

        let (compressed, offsets) = aec_compress(&data, &params).unwrap();
        assert!(!compressed.is_empty());
        assert!(!offsets.is_empty());

        let decompressed = aec_decompress(&compressed, data.len(), &params).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn round_trip_u32() {
        let values: Vec<u32> = (0..4096).map(|i| i * 13).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let params = default_params(32);

        let (compressed, offsets) = aec_compress(&data, &params).unwrap();
        assert!(!compressed.is_empty());
        assert!(!offsets.is_empty());

        let decompressed = aec_decompress(&compressed, data.len(), &params).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn empty_input_returns_empty() {
        let params = default_params(8);
        let (compressed, offsets) = aec_compress(&[], &params).unwrap();
        assert!(compressed.is_empty());
        assert!(offsets.is_empty());

        let decompressed = aec_decompress(&[], 0, &params).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn misaligned_data_returns_error() {
        let data = vec![1u8, 2, 3]; // 3 bytes, not multiple of 2 (u16)
        let params = default_params(16);
        assert!(aec_compress(&data, &params).is_err());
    }

    #[test]
    fn offsets_match_rsi_count() {
        // 4096 8-bit samples / (128 RSI * 16 block_size) = 2 RSIs
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let params = default_params(8);
        let (_, offsets) = aec_compress(&data, &params).unwrap();
        let expected_rsis = 4096usize.div_ceil(128 * 16);
        assert_eq!(offsets.len(), expected_rsis);
    }

    #[test]
    fn round_trip_constant_data() {
        let data = vec![42u8; 2048];
        let params = default_params(8);

        let (compressed, _) = aec_compress(&data, &params).unwrap();
        let decompressed = aec_decompress(&compressed, data.len(), &params).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn round_trip_no_preprocess() {
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let params = AecParams {
            bits_per_sample: 8,
            block_size: 16,
            rsi: 128,
            flags: 0, // no preprocessing
        };

        let (compressed, _) = aec_compress(&data, &params).unwrap();
        let decompressed = aec_decompress(&compressed, data.len(), &params).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn range_decode_matches_full() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let params = default_params(8);

        let (compressed, offsets) = aec_compress(&data, &params).unwrap();
        let full = aec_decompress(&compressed, data.len(), &params).unwrap();

        // Decode a range from the middle
        let pos = 100;
        let size = 200;
        let partial = aec_decompress_range(&compressed, &offsets, pos, size, &params).unwrap();

        assert_eq!(partial.len(), size);
        assert_eq!(&partial[..], &full[pos..pos + size]);
    }

    #[test]
    fn range_decode_first_block() {
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let params = default_params(8);

        let (compressed, offsets) = aec_compress(&data, &params).unwrap();
        let full = aec_decompress(&compressed, data.len(), &params).unwrap();

        let partial = aec_decompress_range(&compressed, &offsets, 0, 50, &params).unwrap();
        assert_eq!(&partial[..], &full[..50]);
    }

    #[test]
    fn range_decode_zero_size() {
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let params = default_params(8);

        let (compressed, offsets) = aec_compress(&data, &params).unwrap();

        let partial = aec_decompress_range(&compressed, &offsets, 0, 0, &params).unwrap();
        assert!(partial.is_empty());
    }

    #[test]
    fn round_trip_msb_data() {
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let params = AecParams {
            bits_per_sample: 8,
            block_size: 16,
            rsi: 128,
            flags: AEC_DATA_PREPROCESS | AEC_DATA_MSB,
        };

        let (compressed, _) = aec_compress(&data, &params).unwrap();
        let decompressed = aec_decompress(&compressed, data.len(), &params).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn round_trip_small_block_size() {
        let data: Vec<u8> = (0..512).map(|i| (i % 256) as u8).collect();
        let params = AecParams {
            bits_per_sample: 8,
            block_size: 8,
            rsi: 64,
            flags: AEC_DATA_PREPROCESS,
        };

        let (compressed, _) = aec_compress(&data, &params).unwrap();
        let decompressed = aec_decompress(&compressed, data.len(), &params).unwrap();
        assert_eq!(decompressed, data);
    }

    // ── Coverage: aec_compress_no_offsets ────────────────────────────────

    #[test]
    fn compress_no_offsets_round_trip() {
        let data: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let params = default_params(8);

        let compressed = aec_compress_no_offsets(&data, &params).unwrap();
        assert!(!compressed.is_empty());

        let decompressed = aec_decompress(&compressed, data.len(), &params).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn compress_no_offsets_empty() {
        let params = default_params(8);
        let compressed = aec_compress_no_offsets(&[], &params).unwrap();
        assert!(compressed.is_empty());
    }

    // ── Coverage: signed data round-trip ─────────────────────────────────

    #[test]
    fn round_trip_signed_8bit() {
        // 8-bit signed: values encoded as unsigned representation
        let data: Vec<u8> = (-128..=127i8).map(|v| v as u8).collect();
        let params = AecParams {
            bits_per_sample: 8,
            block_size: 16,
            rsi: 128,
            flags: AEC_DATA_PREPROCESS | AEC_DATA_SIGNED,
        };

        let (compressed, _) = aec_compress(&data, &params).unwrap();
        let decompressed = aec_decompress(&compressed, data.len(), &params).unwrap();
        assert_eq!(decompressed, data);
    }

    // ── Coverage: range decode out-of-bounds ─────────────────────────────

    #[test]
    fn range_decode_oob_rsi_returns_error() {
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let params = default_params(8);
        let (compressed, offsets) = aec_compress(&data, &params).unwrap();

        // Request beyond available RSIs
        let result = aec_decompress_range(&compressed, &offsets, 999999, 100, &params);
        assert!(result.is_err());
    }

    #[test]
    fn range_decode_empty_data_returns_error() {
        let params = default_params(8);
        let result = aec_decompress_range(&[], &[0], 0, 100, &params);
        assert!(result.is_err());
    }

    // ── Coverage: validation error paths ─────────────────────────────────

    #[test]
    fn compress_bad_rsi_returns_error() {
        let params = AecParams {
            bits_per_sample: 8,
            block_size: 16,
            rsi: 0, // invalid
            flags: 0,
        };
        assert!(aec_compress(&[0u8; 256], &params).is_err());
    }

    #[test]
    fn decompress_bad_block_size_returns_error() {
        let params = AecParams {
            bits_per_sample: 8,
            block_size: 12, // invalid without NOT_ENFORCE
            rsi: 128,
            flags: 0,
        };
        assert!(aec_decompress(&[0u8; 256], 256, &params).is_err());
    }

    #[test]
    fn compress_restricted_high_bps_returns_error() {
        let params = AecParams {
            bits_per_sample: 8,
            block_size: 16,
            rsi: 128,
            flags: AEC_RESTRICTED, // requires bps ≤ 4
        };
        assert!(aec_compress(&[0u8; 256], &params).is_err());
    }

    // ── Coverage: 32-bit round trip ──────────────────────────────────────

    #[test]
    fn round_trip_u32_max_values() {
        // Extreme u32 values
        let values: Vec<u32> = vec![0, u32::MAX, u32::MAX / 2, 1, u32::MAX - 1];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let params = default_params(32);

        let (compressed, _) = aec_compress(&data, &params).unwrap();
        let decompressed = aec_decompress(&compressed, data.len(), &params).unwrap();
        assert_eq!(decompressed, data);
    }
}
