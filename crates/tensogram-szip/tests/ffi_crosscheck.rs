//! Cross-validation tests: compress with libaec FFI, decompress with tensogram-szip
//! (and vice versa). Verifies byte-level compatibility.
//!
//! Run with: cargo test -p tensogram-szip --test ffi_crosscheck

use tensogram_szip::{AecParams, AEC_DATA_MSB, AEC_DATA_PREPROCESS};

// ── Helpers: libaec FFI wrappers ─────────────────────────────────────────────

fn ffi_compress(data: &[u8], params: &AecParams) -> (Vec<u8>, Vec<u64>) {
    use libaec_sys::*;
    let flags = effective_flags_ffi(params);
    let out_capacity = data.len() + data.len() / 4 + 256;
    let mut out = vec![0u8; out_capacity];
    let mut offsets_raw: Vec<usize>;
    #[allow(unused_assignments)]
    let mut offsets = Vec::new();

    unsafe {
        let mut strm: aec_stream = std::mem::zeroed();
        strm.next_in = data.as_ptr();
        strm.avail_in = data.len();
        strm.next_out = out.as_mut_ptr();
        strm.avail_out = out.len();
        strm.bits_per_sample = params.bits_per_sample;
        strm.block_size = params.block_size;
        strm.rsi = params.rsi;
        strm.flags = flags;

        assert_eq!(aec_encode_init(&mut strm), AEC_OK as i32);
        assert_eq!(aec_encode_enable_offsets(&mut strm), AEC_OK as i32);
        assert_eq!(aec_encode(&mut strm, AEC_FLUSH as i32), AEC_OK as i32);

        let compressed_len = out.len() - strm.avail_out;
        out.truncate(compressed_len);

        let mut offset_count: usize = 0;
        assert_eq!(
            aec_encode_count_offsets(&mut strm, &mut offset_count),
            AEC_OK as i32
        );

        offsets_raw = vec![0usize; offset_count];
        if offset_count > 0 {
            assert_eq!(
                aec_encode_get_offsets(&mut strm, offsets_raw.as_mut_ptr(), offset_count),
                AEC_OK as i32
            );
        }
        offsets = offsets_raw.iter().map(|&o| o as u64).collect();

        aec_encode_end(&mut strm);
    }

    (out, offsets)
}

fn ffi_decompress(data: &[u8], expected_size: usize, params: &AecParams) -> Vec<u8> {
    use libaec_sys::*;
    let flags = effective_flags_ffi(params);
    let mut out = vec![0u8; expected_size];

    unsafe {
        let mut strm: aec_stream = std::mem::zeroed();
        strm.next_in = data.as_ptr();
        strm.avail_in = data.len();
        strm.next_out = out.as_mut_ptr();
        strm.avail_out = out.len();
        strm.bits_per_sample = params.bits_per_sample;
        strm.block_size = params.block_size;
        strm.rsi = params.rsi;
        strm.flags = flags;

        assert_eq!(aec_decode_init(&mut strm), AEC_OK as i32);
        assert_eq!(aec_decode(&mut strm, AEC_FLUSH as i32), AEC_OK as i32);

        let decoded_len = out.len() - strm.avail_out;
        out.truncate(decoded_len);
        aec_decode_end(&mut strm);
    }

    out
}

fn effective_flags_ffi(params: &AecParams) -> u32 {
    let mut flags = params.flags;
    if params.bits_per_sample > 16 && params.bits_per_sample <= 24 {
        flags |= libaec_sys::AEC_DATA_3BYTE;
    }
    flags
}

// ── Cross-validation tests ───────────────────────────────────────────────────

/// Compress with tensogram-szip, decompress with libaec FFI → verify identical
fn roundtrip_pure_to_ffi(data: &[u8], params: &AecParams) {
    let (compressed, _offsets) = tensogram_szip::aec_compress(data, params).unwrap();
    let decompressed = ffi_decompress(&compressed, data.len(), params);
    assert_eq!(
        decompressed, data,
        "pure→FFI mismatch for bps={} block={} rsi={} flags={}",
        params.bits_per_sample, params.block_size, params.rsi, params.flags
    );
}

/// Compress with libaec FFI, decompress with tensogram-szip → verify identical
fn roundtrip_ffi_to_pure(data: &[u8], params: &AecParams) {
    let (compressed, _offsets) = ffi_compress(data, params);
    let decompressed = tensogram_szip::aec_decompress(&compressed, data.len(), params).unwrap();
    assert_eq!(
        decompressed, data,
        "FFI→pure mismatch for bps={} block={} rsi={} flags={}",
        params.bits_per_sample, params.block_size, params.rsi, params.flags
    );
}

// ── 8-bit tests ──────────────────────────────────────────────────────────────

#[test]
fn crosscheck_8bit_ramp_pure_to_ffi() {
    let data: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
    let params = AecParams {
        bits_per_sample: 8,
        block_size: 16,
        rsi: 128,
        flags: AEC_DATA_PREPROCESS,
    };
    roundtrip_pure_to_ffi(&data, &params);
}

#[test]
fn crosscheck_8bit_ramp_ffi_to_pure() {
    let data: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
    let params = AecParams {
        bits_per_sample: 8,
        block_size: 16,
        rsi: 128,
        flags: AEC_DATA_PREPROCESS,
    };
    roundtrip_ffi_to_pure(&data, &params);
}

#[test]
fn crosscheck_8bit_constant_pure_to_ffi() {
    let data = vec![42u8; 2048];
    let params = AecParams {
        bits_per_sample: 8,
        block_size: 16,
        rsi: 128,
        flags: AEC_DATA_PREPROCESS,
    };
    roundtrip_pure_to_ffi(&data, &params);
}

#[test]
fn crosscheck_8bit_constant_ffi_to_pure() {
    let data = vec![42u8; 2048];
    let params = AecParams {
        bits_per_sample: 8,
        block_size: 16,
        rsi: 128,
        flags: AEC_DATA_PREPROCESS,
    };
    roundtrip_ffi_to_pure(&data, &params);
}

#[test]
fn crosscheck_8bit_no_preprocess_pure_to_ffi() {
    let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
    let params = AecParams {
        bits_per_sample: 8,
        block_size: 16,
        rsi: 128,
        flags: 0,
    };
    roundtrip_pure_to_ffi(&data, &params);
}

#[test]
fn crosscheck_8bit_no_preprocess_ffi_to_pure() {
    let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
    let params = AecParams {
        bits_per_sample: 8,
        block_size: 16,
        rsi: 128,
        flags: 0,
    };
    roundtrip_ffi_to_pure(&data, &params);
}

// ── 16-bit tests ─────────────────────────────────────────────────────────────

#[test]
fn crosscheck_16bit_ramp_pure_to_ffi() {
    let values: Vec<u16> = (0..2048).map(|i| (i * 7 % 65536) as u16).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let params = AecParams {
        bits_per_sample: 16,
        block_size: 16,
        rsi: 128,
        flags: AEC_DATA_PREPROCESS,
    };
    roundtrip_pure_to_ffi(&data, &params);
}

#[test]
fn crosscheck_16bit_ramp_ffi_to_pure() {
    let values: Vec<u16> = (0..2048).map(|i| (i * 7 % 65536) as u16).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let params = AecParams {
        bits_per_sample: 16,
        block_size: 16,
        rsi: 128,
        flags: AEC_DATA_PREPROCESS,
    };
    roundtrip_ffi_to_pure(&data, &params);
}

// ── 32-bit tests ─────────────────────────────────────────────────────────────

#[test]
fn crosscheck_32bit_ramp_pure_to_ffi() {
    let values: Vec<u32> = (0..4096).map(|i| i * 13).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let params = AecParams {
        bits_per_sample: 32,
        block_size: 16,
        rsi: 128,
        flags: AEC_DATA_PREPROCESS,
    };
    roundtrip_pure_to_ffi(&data, &params);
}

#[test]
fn crosscheck_32bit_ramp_ffi_to_pure() {
    let values: Vec<u32> = (0..4096).map(|i| i * 13).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let params = AecParams {
        bits_per_sample: 32,
        block_size: 16,
        rsi: 128,
        flags: AEC_DATA_PREPROCESS,
    };
    roundtrip_ffi_to_pure(&data, &params);
}

// ── MSB tests ────────────────────────────────────────────────────────────────

#[test]
fn crosscheck_8bit_msb_pure_to_ffi() {
    let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
    let params = AecParams {
        bits_per_sample: 8,
        block_size: 16,
        rsi: 128,
        flags: AEC_DATA_PREPROCESS | AEC_DATA_MSB,
    };
    roundtrip_pure_to_ffi(&data, &params);
}

#[test]
fn crosscheck_8bit_msb_ffi_to_pure() {
    let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
    let params = AecParams {
        bits_per_sample: 8,
        block_size: 16,
        rsi: 128,
        flags: AEC_DATA_PREPROCESS | AEC_DATA_MSB,
    };
    roundtrip_ffi_to_pure(&data, &params);
}

// ── Block size variations ────────────────────────────────────────────────────

#[test]
fn crosscheck_block8_pure_to_ffi() {
    let data: Vec<u8> = (0..512).map(|i| (i % 256) as u8).collect();
    let params = AecParams {
        bits_per_sample: 8,
        block_size: 8,
        rsi: 64,
        flags: AEC_DATA_PREPROCESS,
    };
    roundtrip_pure_to_ffi(&data, &params);
}

#[test]
fn crosscheck_block8_ffi_to_pure() {
    let data: Vec<u8> = (0..512).map(|i| (i % 256) as u8).collect();
    let params = AecParams {
        bits_per_sample: 8,
        block_size: 8,
        rsi: 64,
        flags: AEC_DATA_PREPROCESS,
    };
    roundtrip_ffi_to_pure(&data, &params);
}

// ── Range decode cross-check ─────────────────────────────────────────────────

#[test]
fn crosscheck_range_decode_ffi_compressed() {
    let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
    let params = AecParams {
        bits_per_sample: 8,
        block_size: 16,
        rsi: 128,
        flags: AEC_DATA_PREPROCESS,
    };

    // Compress with FFI
    let (compressed, offsets) = ffi_compress(&data, &params);

    // Full decode with pure
    let full = tensogram_szip::aec_decompress(&compressed, data.len(), &params).unwrap();
    assert_eq!(full, data);

    // Range decode with pure using FFI offsets
    let pos = 200;
    let size = 500;
    let partial =
        tensogram_szip::aec_decompress_range(&compressed, &offsets, pos, size, &params).unwrap();
    assert_eq!(partial.len(), size);
    assert_eq!(&partial[..], &data[pos..pos + size]);
}
