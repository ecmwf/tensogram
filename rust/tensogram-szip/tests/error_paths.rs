// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Error handling tests for the pure-Rust AEC/SZIP codec.
//!
//! Verifies that invalid inputs, corrupt data, and misaligned buffers
//! produce clean errors (not panics).

use tensogram_szip::{
    AEC_DATA_PREPROCESS, AEC_NOT_ENFORCE, AEC_RESTRICTED, AecParams, aec_compress, aec_decompress,
    aec_decompress_range,
};

fn default_params() -> AecParams {
    AecParams {
        bits_per_sample: 8,
        block_size: 16,
        rsi: 64,
        flags: AEC_DATA_PREPROCESS,
    }
}

fn compress_ramp() -> (Vec<u8>, Vec<u64>, usize) {
    let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
    let len = data.len();
    let (compressed, offsets) = aec_compress(&data, &default_params()).unwrap();
    (compressed, offsets, len)
}

// ── Truncated compressed data ───────────────────────────────────────────────

mod truncated {
    use super::*;

    #[test]
    fn truncated_by_one_byte() {
        let (mut compressed, _, len) = compress_ramp();
        compressed.pop();
        assert!(aec_decompress(&compressed, len, &default_params()).is_err());
    }

    #[test]
    fn truncated_to_half() {
        let (compressed, _, len) = compress_ramp();
        let half = &compressed[..compressed.len() / 2];
        assert!(aec_decompress(half, len, &default_params()).is_err());
    }

    #[test]
    fn truncated_to_zero() {
        // Empty compressed data with nonzero expected size — should error or
        // return incomplete output (not panic).
        let result = aec_decompress(&[], 1024, &default_params());
        // Codec may return Ok(empty) or Err — both are acceptable, just no panic.
        if let Ok(out) = result {
            assert!(
                out.len() < 1024,
                "should not produce full output from empty input"
            );
        }
    }
}

// ── Corrupt bitstream ───────────────────────────────────────────────────────

mod corrupt {
    use super::*;

    #[test]
    fn flip_first_byte() {
        let (mut compressed, _, len) = compress_ramp();
        compressed[0] ^= 0xFF;
        // Should error or produce wrong output — not panic
        let _ = aec_decompress(&compressed, len, &default_params());
    }

    #[test]
    fn flip_middle_byte() {
        let (mut compressed, _, len) = compress_ramp();
        let mid = compressed.len() / 2;
        compressed[mid] ^= 0xFF;
        let _ = aec_decompress(&compressed, len, &default_params());
    }

    #[test]
    fn flip_last_byte() {
        let (mut compressed, _, len) = compress_ramp();
        let last = compressed.len() - 1;
        compressed[last] ^= 0xFF;
        let _ = aec_decompress(&compressed, len, &default_params());
    }

    #[test]
    fn all_zeros() {
        let compressed = vec![0u8; 100];
        // Should not panic
        let _ = aec_decompress(&compressed, 1024, &default_params());
    }

    #[test]
    fn all_ones() {
        let compressed = vec![0xFFu8; 100];
        let _ = aec_decompress(&compressed, 1024, &default_params());
    }
}

// ── Invalid parameters ──────────────────────────────────────────────────────

mod invalid_params {
    use super::*;

    #[test]
    fn bps_zero() {
        let params = AecParams {
            bits_per_sample: 0,
            block_size: 16,
            rsi: 64,
            flags: 0,
        };
        assert!(aec_compress(&[0u8; 16], &params).is_err());
    }

    #[test]
    fn bps_33() {
        let params = AecParams {
            bits_per_sample: 33,
            block_size: 16,
            rsi: 64,
            flags: 0,
        };
        assert!(aec_compress(&[0u8; 16], &params).is_err());
    }

    #[test]
    fn block_size_zero() {
        let params = AecParams {
            bits_per_sample: 8,
            block_size: 0,
            rsi: 64,
            flags: 0,
        };
        assert!(aec_compress(&[0u8; 16], &params).is_err());
    }

    #[test]
    fn block_size_odd_without_not_enforce() {
        let params = AecParams {
            bits_per_sample: 8,
            block_size: 3,
            rsi: 64,
            flags: 0,
        };
        assert!(aec_compress(&[0u8; 16], &params).is_err());
    }

    #[test]
    fn block_size_odd_with_not_enforce() {
        let params = AecParams {
            bits_per_sample: 8,
            block_size: 7,
            rsi: 64,
            flags: AEC_NOT_ENFORCE,
        };
        assert!(aec_compress(&[0u8; 16], &params).is_err());
    }

    #[test]
    fn rsi_zero() {
        let params = AecParams {
            bits_per_sample: 8,
            block_size: 16,
            rsi: 0,
            flags: 0,
        };
        assert!(aec_compress(&[0u8; 16], &params).is_err());
    }

    #[test]
    fn rsi_too_large() {
        let params = AecParams {
            bits_per_sample: 8,
            block_size: 16,
            rsi: 4097,
            flags: 0,
        };
        assert!(aec_compress(&[0u8; 16], &params).is_err());
    }

    #[test]
    fn restricted_bps_5() {
        let params = AecParams {
            bits_per_sample: 5,
            block_size: 16,
            rsi: 64,
            flags: AEC_RESTRICTED,
        };
        assert!(aec_compress(&[0u8; 16], &params).is_err());
    }
}

// ── Misaligned data ─────────────────────────────────────────────────────────

mod misaligned_data {
    use super::*;

    #[test]
    fn bps16_odd_length() {
        let params = AecParams {
            bits_per_sample: 16,
            block_size: 16,
            rsi: 64,
            flags: 0,
        };
        assert!(aec_compress(&[0u8; 3], &params).is_err()); // 3 bytes not multiple of 2
    }

    #[test]
    fn bps32_non_multiple_of_4() {
        let params = AecParams {
            bits_per_sample: 32,
            block_size: 16,
            rsi: 64,
            flags: 0,
        };
        assert!(aec_compress(&[0u8; 5], &params).is_err()); // 5 bytes not multiple of 4
    }
}

// ── Range decode errors ─────────────────────────────────────────────────────

mod range_errors {
    use super::*;

    #[test]
    fn byte_pos_beyond_data() {
        let (compressed, offsets, _) = compress_ramp();
        let result = aec_decompress_range(&compressed, &offsets, 9999, 32, &default_params());
        assert!(result.is_err());
    }

    #[test]
    fn empty_offsets_with_nonzero_pos() {
        let (compressed, _, _) = compress_ramp();
        let result = aec_decompress_range(&compressed, &[], 64, 32, &default_params());
        assert!(result.is_err());
    }

    #[test]
    fn oversized_byte_range() {
        let (compressed, offsets, _) = compress_ramp();
        // Request a range larger than one RSI (32 bytes per RSI at bps=8, bs=16, rsi=4→64 samples)
        // but starting at pos 0. The range decoder decodes one RSI, so requesting more should error.
        let rsi_bytes = 64 * 16; // rsi=64 blocks × 16 samples = 1024 samples per RSI
        let result =
            aec_decompress_range(&compressed, &offsets, 0, rsi_bytes + 1, &default_params());
        assert!(result.is_err());
    }
}
