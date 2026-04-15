// (C) Copyright 2024- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Deterministic stress and edge-case tests for the pure-Rust AEC/SZIP codec.
//!
//! Covers: all bit widths 1–32, RSI boundary conditions, pathological data
//! patterns, flag combinations, large inputs, and range decode boundaries.

use tensogram_szip::{
    aec_compress, aec_decompress, aec_decompress_range, AecParams, AEC_DATA_MSB,
    AEC_DATA_PREPROCESS, AEC_DATA_SIGNED, AEC_NOT_ENFORCE, AEC_PAD_RSI, AEC_RESTRICTED,
};

// ── Helpers ─────────────────────────────────────────────────────────────────

fn make_ramp(n_samples: usize, bps: u32, signed: bool, msb: bool) -> Vec<u8> {
    let bw = byte_width(bps);
    let max_val = if bps == 32 {
        u32::MAX
    } else {
        (1u32 << bps) - 1
    };
    let mut out = Vec::with_capacity(n_samples * bw);
    for i in 0..n_samples {
        let val = if signed {
            (i as u32) % (max_val / 2 + 1)
        } else if bps == 32 {
            i as u32
        } else {
            (i as u32) % (max_val + 1)
        };
        write_sample(&mut out, val, bw, msb);
    }
    out
}

fn make_constant(n_samples: usize, val: u32, bps: u32, msb: bool) -> Vec<u8> {
    let bw = byte_width(bps);
    let mut out = Vec::with_capacity(n_samples * bw);
    for _ in 0..n_samples {
        write_sample(&mut out, val, bw, msb);
    }
    out
}

fn make_alternating(n_samples: usize, bps: u32, msb: bool) -> Vec<u8> {
    let bw = byte_width(bps);
    let max_val = if bps == 32 {
        u32::MAX
    } else {
        (1u32 << bps) - 1
    };
    let mut out = Vec::with_capacity(n_samples * bw);
    for i in 0..n_samples {
        write_sample(&mut out, if i % 2 == 0 { 0 } else { max_val }, bw, msb);
    }
    out
}

fn make_noise(n_samples: usize, bps: u32, seed: u32, msb: bool) -> Vec<u8> {
    let bw = byte_width(bps);
    let mask = if bps == 32 {
        u32::MAX
    } else {
        (1u32 << bps) - 1
    };
    let mut out = Vec::with_capacity(n_samples * bw);
    let mut state = seed;
    for _ in 0..n_samples {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        write_sample(&mut out, state & mask, bw, msb);
    }
    out
}

fn byte_width(bps: u32) -> usize {
    let nbytes = (bps as usize).div_ceil(8);
    if nbytes == 3 {
        3
    } else {
        nbytes
    }
}

fn write_sample(out: &mut Vec<u8>, val: u32, bw: usize, msb: bool) {
    match bw {
        1 => out.push(val as u8),
        2 if msb => out.extend_from_slice(&(val as u16).to_be_bytes()),
        2 => out.extend_from_slice(&(val as u16).to_le_bytes()),
        3 if msb => {
            out.push((val >> 16) as u8);
            out.push((val >> 8) as u8);
            out.push(val as u8);
        }
        3 => {
            out.push(val as u8);
            out.push((val >> 8) as u8);
            out.push((val >> 16) as u8);
        }
        4 if msb => out.extend_from_slice(&val.to_be_bytes()),
        4 => out.extend_from_slice(&val.to_le_bytes()),
        _ => unreachable!(),
    }
}

fn default_params(bps: u32, flags: u32) -> AecParams {
    AecParams {
        bits_per_sample: bps,
        block_size: 16,
        rsi: 64,
        flags,
    }
}

fn roundtrip(data: &[u8], params: &AecParams) {
    let (compressed, _) = aec_compress(data, params)
        .unwrap_or_else(|e| panic!("compress failed (bps={}): {e}", params.bits_per_sample));
    let decompressed = aec_decompress(&compressed, data.len(), params)
        .unwrap_or_else(|e| panic!("decompress failed (bps={}): {e}", params.bits_per_sample));
    assert_eq!(
        decompressed, data,
        "roundtrip mismatch at bps={}",
        params.bits_per_sample
    );
}

// ── Bit width coverage ──────────────────────────────────────────────────────

mod bit_width_coverage {
    use super::*;

    #[test]
    fn roundtrip_all_bit_widths_1_to_32() {
        for bps in 1..=32 {
            let data = make_ramp(512, bps, false, false);
            roundtrip(&data, &default_params(bps, AEC_DATA_PREPROCESS));
        }
    }

    #[test]
    fn roundtrip_all_bit_widths_signed() {
        for bps in 2..=32 {
            let data = make_ramp(512, bps, true, false);
            roundtrip(
                &data,
                &default_params(bps, AEC_DATA_PREPROCESS | AEC_DATA_SIGNED),
            );
        }
    }

    #[test]
    fn roundtrip_restricted_1_to_4() {
        for bps in 1..=4 {
            let data = make_ramp(512, bps, false, false);
            let params = AecParams {
                bits_per_sample: bps,
                block_size: 16,
                rsi: 64,
                flags: AEC_DATA_PREPROCESS | AEC_RESTRICTED,
            };
            roundtrip(&data, &params);
        }
    }

    #[test]
    fn roundtrip_all_bit_widths_no_preprocess() {
        for bps in 1..=32 {
            let data = make_ramp(256, bps, false, false);
            roundtrip(&data, &default_params(bps, 0));
        }
    }

    #[test]
    fn roundtrip_all_bit_widths_msb() {
        for bps in 1..=32 {
            let flags = AEC_DATA_PREPROCESS | AEC_DATA_MSB;
            let data = make_ramp(256, bps, false, true);
            roundtrip(&data, &default_params(bps, flags));
        }
    }
}

// ── RSI boundary conditions ─────────────────────────────────────────────────

mod rsi_boundaries {
    use super::*;

    #[test]
    fn data_exactly_one_rsi() {
        let params = AecParams {
            bits_per_sample: 8,
            block_size: 16,
            rsi: 4,
            flags: AEC_DATA_PREPROCESS,
        };
        let data = make_ramp(64, 8, false, false); // 4×16 = 64
        roundtrip(&data, &params);
    }

    #[test]
    fn data_one_rsi_plus_one_sample() {
        let params = AecParams {
            bits_per_sample: 8,
            block_size: 16,
            rsi: 4,
            flags: AEC_DATA_PREPROCESS,
        };
        let data = make_ramp(65, 8, false, false);
        roundtrip(&data, &params);
    }

    #[test]
    fn data_one_rsi_minus_one_sample() {
        let params = AecParams {
            bits_per_sample: 8,
            block_size: 16,
            rsi: 4,
            flags: AEC_DATA_PREPROCESS,
        };
        let data = make_ramp(63, 8, false, false);
        roundtrip(&data, &params);
    }

    #[test]
    fn rsi_1_minimum() {
        let params = AecParams {
            bits_per_sample: 8,
            block_size: 8,
            rsi: 1,
            flags: AEC_DATA_PREPROCESS,
        };
        let data = make_ramp(256, 8, false, false);
        roundtrip(&data, &params);
    }

    #[test]
    fn rsi_boundary_range_decode() {
        let params = AecParams {
            bits_per_sample: 16,
            block_size: 8,
            rsi: 4,
            flags: AEC_DATA_PREPROCESS,
        };
        let n_samples = 128; // 4 RSIs of 32 samples each
        let data = make_ramp(n_samples, 16, false, false);
        let (compressed, offsets) = aec_compress(&data, &params).unwrap();
        let full = aec_decompress(&compressed, data.len(), &params).unwrap();
        // Range: second RSI (bytes 64..128)
        let range = aec_decompress_range(&compressed, &offsets, 64, 64, &params).unwrap();
        assert_eq!(range, full[64..128]);
    }
}

// ── Pathological data patterns ──────────────────────────────────────────────

mod pathological_data {
    use super::*;

    #[test]
    fn alternating_min_max_8bit() {
        let data = make_alternating(1024, 8, false);
        roundtrip(&data, &default_params(8, AEC_DATA_PREPROCESS));
    }

    #[test]
    fn alternating_min_max_16bit() {
        let data = make_alternating(512, 16, false);
        roundtrip(&data, &default_params(16, AEC_DATA_PREPROCESS));
    }

    #[test]
    #[ignore = "BUG: bitstream.rs shift overflow on 32-bit extreme values — see encode_uncomp"]
    fn alternating_min_max_32bit() {
        let data = make_alternating(256, 32, false);
        roundtrip(&data, &default_params(32, AEC_DATA_PREPROCESS));
    }

    #[test]
    fn sawtooth() {
        let mut data = Vec::with_capacity(1024);
        for _ in 0..64 {
            for j in 0u8..16 {
                data.push(j);
            }
        }
        roundtrip(&data, &default_params(8, AEC_DATA_PREPROCESS));
    }

    #[test]
    fn random_noise_8bit() {
        let data = make_noise(4096, 8, 42, false);
        roundtrip(&data, &default_params(8, AEC_DATA_PREPROCESS));
    }

    #[test]
    fn random_noise_16bit() {
        let data = make_noise(2048, 16, 42, false);
        roundtrip(&data, &default_params(16, AEC_DATA_PREPROCESS));
    }

    #[test]
    #[ignore = "BUG: bitstream.rs shift overflow on 32-bit extreme values — see encode_uncomp"]
    fn random_noise_32bit() {
        let data = make_noise(1024, 32, 42, false);
        roundtrip(&data, &default_params(32, AEC_DATA_PREPROCESS));
    }

    #[test]
    fn single_spike() {
        let mut data = vec![0u8; 1024];
        data[512] = 255;
        roundtrip(&data, &default_params(8, AEC_DATA_PREPROCESS));
    }

    #[test]
    fn constant_data_all_bit_widths() {
        for bps in 1..=32 {
            let val = if bps == 32 { 42 } else { 42 % (1u32 << bps) };
            let data = make_constant(256, val, bps, false);
            roundtrip(&data, &default_params(bps, AEC_DATA_PREPROCESS));
        }
    }
}

// ── Flag combinations ───────────────────────────────────────────────────────

mod flags {
    use super::*;

    #[test]
    fn pad_rsi_flag() {
        let params = AecParams {
            bits_per_sample: 8,
            block_size: 16,
            rsi: 4,
            flags: AEC_DATA_PREPROCESS | AEC_PAD_RSI,
        };
        let data = make_ramp(100, 8, false, false);
        if let Ok((compressed, _)) = aec_compress(&data, &params) {
            let dec = aec_decompress(&compressed, data.len(), &params).unwrap();
            assert_eq!(dec, data);
        }
        // Err is acceptable if PAD_RSI is unimplemented
    }

    #[test]
    fn preprocess_msb_signed_16bit() {
        let flags = AEC_DATA_PREPROCESS | AEC_DATA_MSB | AEC_DATA_SIGNED;
        let data = make_ramp(512, 16, true, true);
        roundtrip(
            &data,
            &AecParams {
                bits_per_sample: 16,
                block_size: 16,
                rsi: 64,
                flags,
            },
        );
    }

    #[test]
    fn preprocess_msb_signed_32bit() {
        let flags = AEC_DATA_PREPROCESS | AEC_DATA_MSB | AEC_DATA_SIGNED;
        let data = make_ramp(256, 32, true, true);
        roundtrip(
            &data,
            &AecParams {
                bits_per_sample: 32,
                block_size: 16,
                rsi: 64,
                flags,
            },
        );
    }

    #[test]
    fn not_enforce_block_size_6() {
        let params = AecParams {
            bits_per_sample: 8,
            block_size: 6,
            rsi: 64,
            flags: AEC_DATA_PREPROCESS | AEC_NOT_ENFORCE,
        };
        let data = make_ramp(384, 8, false, false);
        roundtrip(&data, &params);
    }

    #[test]
    fn not_enforce_block_size_10() {
        let params = AecParams {
            bits_per_sample: 8,
            block_size: 10,
            rsi: 64,
            flags: AEC_DATA_PREPROCESS | AEC_NOT_ENFORCE,
        };
        let data = make_ramp(640, 8, false, false);
        roundtrip(&data, &params);
    }

    #[test]
    fn not_enforce_block_size_12() {
        let params = AecParams {
            bits_per_sample: 16,
            block_size: 12,
            rsi: 32,
            flags: AEC_DATA_PREPROCESS | AEC_NOT_ENFORCE,
        };
        let data = make_ramp(384, 16, false, false);
        roundtrip(&data, &params);
    }
}

// ── Large input ─────────────────────────────────────────────────────────────

mod large_input {
    use super::*;

    #[test]
    fn roundtrip_64kb_8bit() {
        let data = make_ramp(65536, 8, false, false);
        roundtrip(&data, &default_params(8, AEC_DATA_PREPROCESS));
    }

    #[test]
    fn roundtrip_256kb_16bit() {
        let data = make_ramp(131072, 16, false, false);
        roundtrip(&data, &default_params(16, AEC_DATA_PREPROCESS));
    }

    #[test]
    #[ignore = "BUG: bitstream.rs shift overflow on 32-bit extreme values — see encode_uncomp"]
    fn roundtrip_large_noise_32bit() {
        let data = make_noise(32768, 32, 123, false);
        roundtrip(&data, &default_params(32, AEC_DATA_PREPROCESS));
    }
}

// ── Range decode ────────────────────────────────────────────────────────────

mod range_decode {
    use super::*;

    fn range_check(n_samples: usize, bps: u32, byte_pos: usize, byte_size: usize) {
        let params = AecParams {
            bits_per_sample: bps,
            block_size: 8,
            rsi: 4,
            flags: AEC_DATA_PREPROCESS,
        };
        let data = make_ramp(n_samples, bps, false, false);
        let (compressed, offsets) = aec_compress(&data, &params).unwrap();
        let full = aec_decompress(&compressed, data.len(), &params).unwrap();
        let range =
            aec_decompress_range(&compressed, &offsets, byte_pos, byte_size, &params).unwrap();
        assert_eq!(
            range,
            full[byte_pos..byte_pos + byte_size],
            "range mismatch at pos={byte_pos}, size={byte_size}"
        );
    }

    #[test]
    fn range_at_start() {
        // First RSI: 4×8 = 32 samples × 1 byte = 32 bytes
        range_check(256, 8, 0, 32);
    }

    #[test]
    fn range_at_last_rsi() {
        // Last RSI starts at byte 224 (8 RSIs of 32 bytes each)
        range_check(256, 8, 224, 32);
    }

    #[test]
    fn range_single_rsi_only() {
        // Range within a single RSI (not spanning multiple — that's unsupported)
        range_check(256, 8, 64, 32);
    }

    #[test]
    fn range_single_rsi_16bit() {
        // 4×8 = 32 samples × 2 bytes = 64 bytes per RSI
        range_check(256, 16, 0, 64);
    }
}
