// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Integration tests for `EncodeOptions::reject_nan` and
//! `EncodeOptions::reject_inf`.
//!
//! The strict-finite check runs **before** any pipeline stage and is
//! pipeline-independent — a caller who turns it on gets the same
//! contract regardless of `encoding`, `filter`, or `compression`.
//!
//! See `plans/RESEARCH_NAN_HANDLING.md` §4.1.1 for the design
//! rationale; these tests pin §6 "Test matrix for future work".

use std::collections::BTreeMap;
use std::io::Cursor;
use tensogram::*;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn make_global_meta() -> GlobalMetadata {
    GlobalMetadata {
        version: 2,
        ..Default::default()
    }
}

fn make_descriptor(
    shape: Vec<u64>,
    dtype: Dtype,
    byte_order: ByteOrder,
    encoding: &str,
    compression: &str,
) -> DataObjectDescriptor {
    let strides = if shape.is_empty() {
        vec![]
    } else {
        let mut s = vec![1u64; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            s[i] = s[i + 1] * shape[i + 1];
        }
        s
    };
    DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: shape.len() as u64,
        shape,
        strides,
        dtype,
        byte_order,
        encoding: encoding.to_string(),
        filter: "none".to_string(),
        compression: compression.to_string(),
        params: BTreeMap::new(),
        hash: None,
    }
}

fn strict_opts(reject_nan: bool, reject_inf: bool) -> EncodeOptions {
    EncodeOptions {
        reject_nan,
        reject_inf,
        ..Default::default()
    }
}

fn f32_bytes(values: &[f32], order: ByteOrder) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| match order {
            ByteOrder::Big => v.to_be_bytes(),
            ByteOrder::Little => v.to_le_bytes(),
        })
        .collect()
}

fn f64_bytes(values: &[f64], order: ByteOrder) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| match order {
            ByteOrder::Big => v.to_be_bytes(),
            ByteOrder::Little => v.to_le_bytes(),
        })
        .collect()
}

fn u16_bytes(values: &[u16], order: ByteOrder) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| match order {
            ByteOrder::Big => v.to_be_bytes(),
            ByteOrder::Little => v.to_le_bytes(),
        })
        .collect()
}

fn assert_encoding_error_mentions(err: &TensogramError, needles: &[&str]) {
    let msg = err.to_string();
    assert!(
        matches!(err, TensogramError::Encoding(_)),
        "expected TensogramError::Encoding, got {err:?} ({msg})"
    );
    for needle in needles {
        assert!(
            msg.contains(needle),
            "error message {msg:?} did not contain {needle:?}"
        );
    }
}

// ── 1. Default options preserve current behaviour ───────────────────────────

#[test]
fn default_options_do_not_reject_nan_in_float32_passthrough() {
    // Regression guard: with default options, NaN-bearing float32 data
    // must still round-trip bit-exactly through encoding="none". This
    // pins §2.1 of the research memo.
    let data = f32_bytes(&[1.0, f32::NAN, 3.0], ByteOrder::native());
    let desc = make_descriptor(vec![3], Dtype::Float32, ByteOrder::native(), "none", "none");
    let meta = make_global_meta();
    let bytes = encode(&meta, &[(&desc, &data)], &EncodeOptions::default())
        .expect("default options must not reject NaN");
    let (_, decoded) = decode(&bytes, &DecodeOptions::default()).unwrap();
    assert_eq!(decoded[0].1, data, "NaN bits must round-trip by default");
}

#[test]
fn default_options_do_not_reject_inf_in_float64_passthrough() {
    let data = f64_bytes(
        &[1.0, f64::INFINITY, f64::NEG_INFINITY, 4.0],
        ByteOrder::native(),
    );
    let desc = make_descriptor(vec![4], Dtype::Float64, ByteOrder::native(), "none", "none");
    let meta = make_global_meta();
    encode(&meta, &[(&desc, &data)], &EncodeOptions::default())
        .expect("default options must not reject Inf");
}

// ── 2. reject_nan rejects NaN across float dtypes ───────────────────────────

#[test]
fn reject_nan_rejects_float32() {
    let data = f32_bytes(&[1.0, 2.0, f32::NAN, 4.0], ByteOrder::native());
    let desc = make_descriptor(vec![4], Dtype::Float32, ByteOrder::native(), "none", "none");
    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(true, false),
    )
    .unwrap_err();
    assert_encoding_error_mentions(&err, &["NaN", "element 2", "float32"]);
}

#[test]
fn reject_nan_rejects_float64() {
    let data = f64_bytes(&[1.0, 2.0, 3.0, f64::NAN], ByteOrder::native());
    let desc = make_descriptor(vec![4], Dtype::Float64, ByteOrder::native(), "none", "none");
    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(true, false),
    )
    .unwrap_err();
    assert_encoding_error_mentions(&err, &["NaN", "element 3", "float64"]);
}

#[test]
fn reject_nan_rejects_float16_bit_level() {
    // IEEE half: exp=0x1F (all 1s), mantissa != 0 => NaN. Pick 0x7E00.
    // Pattern layout: [finite, NaN, finite, finite]
    let nan_bits: u16 = 0x7E00;
    let data = u16_bytes(&[0x3C00, nan_bits, 0x4000, 0x4200], ByteOrder::native());
    let desc = make_descriptor(vec![4], Dtype::Float16, ByteOrder::native(), "none", "none");
    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(true, false),
    )
    .unwrap_err();
    assert_encoding_error_mentions(&err, &["NaN", "element 1", "float16"]);
}

#[test]
fn reject_nan_rejects_bfloat16_bit_level() {
    // BF16: exp=0xFF (all 1s), mantissa != 0 => NaN. Pick 0x7FC0.
    let nan_bits: u16 = 0x7FC0;
    let data = u16_bytes(&[0x3F80, 0x4000, nan_bits], ByteOrder::native());
    let desc = make_descriptor(
        vec![3],
        Dtype::Bfloat16,
        ByteOrder::native(),
        "none",
        "none",
    );
    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(true, false),
    )
    .unwrap_err();
    assert_encoding_error_mentions(&err, &["NaN", "element 2", "bfloat16"]);
}

#[test]
fn reject_nan_rejects_complex64_real_component() {
    // complex64 = (real f32, imag f32) interleaved
    let data: Vec<u8> = [1.0f32, 2.0, f32::NAN, 3.0, 4.0, 5.0]
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    let desc = make_descriptor(
        vec![3],
        Dtype::Complex64,
        ByteOrder::native(),
        "none",
        "none",
    );
    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(true, false),
    )
    .unwrap_err();
    assert_encoding_error_mentions(&err, &["NaN", "element 1", "complex64", "real"]);
}

#[test]
fn reject_nan_rejects_complex64_imag_component() {
    // real finite at every slot, imag NaN at slot 2
    let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, f32::NAN]
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    let desc = make_descriptor(
        vec![3],
        Dtype::Complex64,
        ByteOrder::native(),
        "none",
        "none",
    );
    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(true, false),
    )
    .unwrap_err();
    assert_encoding_error_mentions(&err, &["NaN", "element 2", "complex64", "imag"]);
}

#[test]
fn reject_nan_rejects_complex128_real_component() {
    let data: Vec<u8> = [f64::NAN, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    let desc = make_descriptor(
        vec![2],
        Dtype::Complex128,
        ByteOrder::native(),
        "none",
        "none",
    );
    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(true, false),
    )
    .unwrap_err();
    assert_encoding_error_mentions(&err, &["NaN", "element 0", "complex128", "real"]);
}

// ── 3. reject_inf rejects +Inf and -Inf across float dtypes ─────────────────

#[test]
fn reject_inf_rejects_positive_inf_in_float32() {
    let data = f32_bytes(&[1.0, f32::INFINITY, 3.0], ByteOrder::native());
    let desc = make_descriptor(vec![3], Dtype::Float32, ByteOrder::native(), "none", "none");
    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(false, true),
    )
    .unwrap_err();
    assert_encoding_error_mentions(&err, &["Inf", "element 1", "float32"]);
}

#[test]
fn reject_inf_rejects_negative_inf_in_float64() {
    let data = f64_bytes(&[1.0, 2.0, f64::NEG_INFINITY], ByteOrder::native());
    let desc = make_descriptor(vec![3], Dtype::Float64, ByteOrder::native(), "none", "none");
    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(false, true),
    )
    .unwrap_err();
    assert_encoding_error_mentions(&err, &["Inf", "element 2", "float64"]);
}

#[test]
fn reject_inf_rejects_float16_bit_level() {
    // IEEE half +Inf: exp=0x1F, mantissa=0 → 0x7C00
    let inf_bits: u16 = 0x7C00;
    let data = u16_bytes(&[0x3C00, inf_bits, 0x4000], ByteOrder::native());
    let desc = make_descriptor(vec![3], Dtype::Float16, ByteOrder::native(), "none", "none");
    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(false, true),
    )
    .unwrap_err();
    assert_encoding_error_mentions(&err, &["Inf", "element 1", "float16"]);
}

#[test]
fn reject_inf_rejects_bfloat16_bit_level() {
    // BF16 -Inf: exp=0xFF, mantissa=0, sign=1 → 0xFF80
    let inf_bits: u16 = 0xFF80;
    let data = u16_bytes(&[0x3F80, inf_bits], ByteOrder::native());
    let desc = make_descriptor(
        vec![2],
        Dtype::Bfloat16,
        ByteOrder::native(),
        "none",
        "none",
    );
    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(false, true),
    )
    .unwrap_err();
    assert_encoding_error_mentions(&err, &["Inf", "element 1", "bfloat16"]);
}

// ── 4. Byte-order awareness ─────────────────────────────────────────────────

#[test]
fn reject_nan_is_byte_order_aware_for_float32() {
    // Big-endian NaN bit pattern in a message with byte_order = Big
    let data = f32_bytes(&[1.0, f32::NAN, 3.0], ByteOrder::Big);
    let desc = make_descriptor(vec![3], Dtype::Float32, ByteOrder::Big, "none", "none");
    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(true, false),
    )
    .unwrap_err();
    assert_encoding_error_mentions(&err, &["NaN", "element 1"]);
}

// ── 5. Integer dtypes skip the scan (zero-cost guarantee) ───────────────────

#[test]
fn reject_nan_skips_integer_dtypes() {
    // 0xFFFFFFFF is NaN if interpreted as f32 but uint32 is not scanned.
    let data: Vec<u8> = vec![0xFFu8; 16]; // 4 u32 elements
    let desc = make_descriptor(vec![4], Dtype::Uint32, ByteOrder::native(), "none", "none");
    encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(true, true),
    )
    .expect("uint32 must not be scanned for NaN/Inf");
}

#[test]
fn reject_nan_skips_bitmask_dtype() {
    let data: Vec<u8> = vec![0xFFu8; 2]; // 16 bits of all-set
    let desc = make_descriptor(
        vec![16],
        Dtype::Bitmask,
        ByteOrder::native(),
        "none",
        "none",
    );
    encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(true, true),
    )
    .expect("bitmask must not be scanned for NaN/Inf");
}

// ── 6. Orthogonality: flag-off = no rejection, flag-on = rejection ──────────

#[test]
fn reject_inf_does_not_reject_nan() {
    // NaN is present but only `reject_inf` is on — NaN must pass through.
    let data = f32_bytes(&[1.0, f32::NAN, 3.0], ByteOrder::native());
    let desc = make_descriptor(vec![3], Dtype::Float32, ByteOrder::native(), "none", "none");
    encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(false, true),
    )
    .expect("reject_inf must not catch NaN");
}

#[test]
fn reject_nan_does_not_reject_inf() {
    let data = f32_bytes(&[1.0, f32::INFINITY, 3.0], ByteOrder::native());
    let desc = make_descriptor(vec![3], Dtype::Float32, ByteOrder::native(), "none", "none");
    encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(true, false),
    )
    .expect("reject_nan must not catch Inf");
}

#[test]
fn reject_both_rejects_either() {
    let data_nan = f32_bytes(&[1.0, f32::NAN], ByteOrder::native());
    let data_inf = f32_bytes(&[1.0, f32::INFINITY], ByteOrder::native());
    let desc = make_descriptor(vec![2], Dtype::Float32, ByteOrder::native(), "none", "none");
    let err_nan = encode(
        &make_global_meta(),
        &[(&desc, &data_nan)],
        &strict_opts(true, true),
    )
    .unwrap_err();
    assert!(err_nan.to_string().contains("NaN"));
    let err_inf = encode(
        &make_global_meta(),
        &[(&desc, &data_inf)],
        &strict_opts(true, true),
    )
    .unwrap_err();
    assert!(err_inf.to_string().contains("Inf"));
}

// ── 7. The §3.1 gotcha: Inf in simple_packing silent-corruption mitigation ──

#[test]
fn reject_inf_fires_before_simple_packing_silent_corruption() {
    // Without the flag, `[1.0, +Inf, 3.0]` through simple_packing is the
    // canonical silent-corruption path documented in RESEARCH §3.1:
    // compute_params yields binary_scale_factor = i32::MAX, encode
    // produces all-zero packed ints, decode recovers NaN everywhere.
    //
    // The strict-Inf flag fires upstream of this so the user sees a
    // clean EncodingError instead of silently-corrupt output.
    let data = f64_bytes(&[1.0, f64::INFINITY, 3.0], ByteOrder::native());
    let mut desc = make_descriptor(
        vec![3],
        Dtype::Float64,
        ByteOrder::native(),
        "simple_packing",
        "none",
    );
    // Simple packing needs these params or compute-from-data; encode()
    // requires the user to have filled them in first. For this test we
    // provide them so the pipeline would run; the strict check fires
    // upstream and we never get that far.
    desc.params.insert(
        "bits_per_value".to_string(),
        ciborium::Value::Integer(16.into()),
    );
    desc.params
        .insert("reference_value".to_string(), ciborium::Value::Float(1.0));
    desc.params.insert(
        "binary_scale_factor".to_string(),
        ciborium::Value::Integer(0.into()),
    );
    desc.params.insert(
        "decimal_scale_factor".to_string(),
        ciborium::Value::Integer(0.into()),
    );

    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(false, true),
    )
    .unwrap_err();
    assert_encoding_error_mentions(&err, &["Inf", "element 1"]);
}

// ── 8. Interaction with compression ─────────────────────────────────────────

#[test]
fn reject_nan_fires_before_lz4_compression() {
    let data = f64_bytes(&[1.0, f64::NAN], ByteOrder::native());
    let desc = make_descriptor(vec![2], Dtype::Float64, ByteOrder::native(), "none", "lz4");
    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(true, false),
    )
    .unwrap_err();
    assert_encoding_error_mentions(&err, &["NaN"]);
}

#[test]
fn nan_passes_lz4_compression_when_flag_off() {
    // lz4 is byte-level lossless so NaN bits round-trip; we just pin
    // that the strict check isn't gate-keeping this case.
    let data = f64_bytes(&[1.0, f64::NAN, 3.0], ByteOrder::native());
    let desc = make_descriptor(vec![3], Dtype::Float64, ByteOrder::native(), "none", "lz4");
    let bytes = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .expect("lz4 passthrough must accept NaN with flag off");
    let (_, decoded) = decode(&bytes, &DecodeOptions::default()).unwrap();
    assert_eq!(decoded[0].1, data);
}

// ── 9. encode_pre_encoded: strict flags are explicitly rejected ─────────────

#[test]
fn encode_pre_encoded_errors_when_reject_nan_is_set() {
    // Pre-encoded bytes are opaque to the library; the strict flags
    // cannot be meaningfully applied.  Rather than silently discarding
    // the caller's intent, the API returns a clear error — mirroring
    // the Python binding (which raises TypeError on the kwarg).
    let data = f32_bytes(&[1.0, 2.0], ByteOrder::native());
    let desc = make_descriptor(vec![2], Dtype::Float32, ByteOrder::native(), "none", "none");
    let err = encode_pre_encoded(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(true, false),
    )
    .unwrap_err();
    assert_encoding_error_mentions(&err, &["reject_nan", "encode_pre_encoded"]);
}

#[test]
fn encode_pre_encoded_errors_when_reject_inf_is_set() {
    let data = f32_bytes(&[1.0, 2.0], ByteOrder::native());
    let desc = make_descriptor(vec![2], Dtype::Float32, ByteOrder::native(), "none", "none");
    let err = encode_pre_encoded(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(false, true),
    )
    .unwrap_err();
    assert_encoding_error_mentions(&err, &["reject_inf", "encode_pre_encoded"]);
}

#[test]
fn encode_pre_encoded_accepts_default_options() {
    // Regression: the check must not fire when flags are off (default).
    let data = f32_bytes(&[1.0, f32::NAN], ByteOrder::native());
    let desc = make_descriptor(vec![2], Dtype::Float32, ByteOrder::native(), "none", "none");
    encode_pre_encoded(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .expect("encode_pre_encoded with default options must succeed");
}

// ── 10. StreamingEncoder honours strict flags ──────────────────────────────

#[test]
fn streaming_encoder_rejects_nan_with_flag() {
    let data = f32_bytes(&[1.0, f32::NAN], ByteOrder::native());
    let desc = make_descriptor(vec![2], Dtype::Float32, ByteOrder::native(), "none", "none");
    let mut buf = Vec::new();
    let mut enc = StreamingEncoder::new(
        Cursor::new(&mut buf),
        &make_global_meta(),
        &strict_opts(true, false),
    )
    .unwrap();
    let err = enc.write_object(&desc, &data).unwrap_err();
    assert_encoding_error_mentions(&err, &["NaN", "float32"]);
}

#[test]
fn streaming_encoder_rejects_inf_with_flag() {
    let data = f64_bytes(&[1.0, f64::INFINITY], ByteOrder::native());
    let desc = make_descriptor(vec![2], Dtype::Float64, ByteOrder::native(), "none", "none");
    let mut buf = Vec::new();
    let mut enc = StreamingEncoder::new(
        Cursor::new(&mut buf),
        &make_global_meta(),
        &strict_opts(false, true),
    )
    .unwrap();
    let err = enc.write_object(&desc, &data).unwrap_err();
    assert_encoding_error_mentions(&err, &["Inf", "float64"]);
}

#[test]
fn streaming_encoder_default_passes_nan() {
    let data = f64_bytes(&[1.0, f64::NAN], ByteOrder::native());
    let desc = make_descriptor(vec![2], Dtype::Float64, ByteOrder::native(), "none", "none");
    let mut buf = Vec::new();
    {
        let mut enc = StreamingEncoder::new(
            Cursor::new(&mut buf),
            &make_global_meta(),
            &EncodeOptions::default(),
        )
        .unwrap();
        enc.write_object(&desc, &data)
            .expect("default streaming must accept NaN");
        enc.finish().unwrap();
    }
    assert!(!buf.is_empty());
}

#[test]
fn streaming_write_object_pre_encoded_errors_when_reject_nan_is_set() {
    // When the StreamingEncoder was configured with reject_nan=true,
    // write_object_pre_encoded must fail loudly — pre-encoded bytes
    // are opaque and the flag cannot be meaningfully enforced.  This
    // mirrors the buffered encode_pre_encoded contract.
    let data = f32_bytes(&[1.0, 2.0], ByteOrder::native());
    let desc = make_descriptor(vec![2], Dtype::Float32, ByteOrder::native(), "none", "none");
    let mut buf = Vec::new();
    let mut enc = StreamingEncoder::new(
        Cursor::new(&mut buf),
        &make_global_meta(),
        &strict_opts(true, false),
    )
    .unwrap();
    let err = enc.write_object_pre_encoded(&desc, &data).unwrap_err();
    assert_encoding_error_mentions(&err, &["reject_nan", "write_object_pre_encoded"]);
}

#[test]
fn streaming_write_object_pre_encoded_errors_when_reject_inf_is_set() {
    let data = f64_bytes(&[1.0, 2.0], ByteOrder::native());
    let desc = make_descriptor(vec![2], Dtype::Float64, ByteOrder::native(), "none", "none");
    let mut buf = Vec::new();
    let mut enc = StreamingEncoder::new(
        Cursor::new(&mut buf),
        &make_global_meta(),
        &strict_opts(false, true),
    )
    .unwrap();
    let err = enc.write_object_pre_encoded(&desc, &data).unwrap_err();
    assert_encoding_error_mentions(&err, &["reject_inf", "write_object_pre_encoded"]);
}

#[test]
fn streaming_write_object_pre_encoded_succeeds_with_default_options() {
    // Regression: the check must not fire when flags are off.
    let data = f32_bytes(&[1.0, 2.0], ByteOrder::native());
    let desc = make_descriptor(vec![2], Dtype::Float32, ByteOrder::native(), "none", "none");
    let mut buf = Vec::new();
    let mut enc = StreamingEncoder::new(
        Cursor::new(&mut buf),
        &make_global_meta(),
        &EncodeOptions::default(),
    )
    .unwrap();
    enc.write_object_pre_encoded(&desc, &data)
        .expect("default options must accept pre-encoded writes");
    enc.finish().unwrap();
}

// ── 11. Multi-object messages ───────────────────────────────────────────────

#[test]
fn reject_nan_fails_whole_message_on_any_bad_object() {
    let clean = f64_bytes(&[1.0, 2.0, 3.0], ByteOrder::native());
    let dirty = f64_bytes(&[1.0, f64::NAN, 3.0], ByteOrder::native());
    let desc = make_descriptor(vec![3], Dtype::Float64, ByteOrder::native(), "none", "none");
    let err = encode(
        &make_global_meta(),
        &[(&desc, &clean), (&desc, &dirty), (&desc, &clean)],
        &strict_opts(true, false),
    )
    .unwrap_err();
    assert!(err.to_string().contains("NaN"));
}

// ── 12. Parallel path still rejects ─────────────────────────────────────────

#[cfg(feature = "threads")]
#[test]
fn reject_nan_parallel_large_input() {
    // Force the parallel path: threads=4, input well above 64 KiB threshold
    let mut values = vec![1.0_f64; 16_384]; // 128 KiB
    values[10_000] = f64::NAN;
    let data = f64_bytes(&values, ByteOrder::native());
    let desc = make_descriptor(
        vec![16_384],
        Dtype::Float64,
        ByteOrder::native(),
        "none",
        "none",
    );
    let opts = EncodeOptions {
        reject_nan: true,
        threads: 4,
        ..Default::default()
    };
    let err = encode(&make_global_meta(), &[(&desc, &data)], &opts).unwrap_err();
    // Parallel path reports per-worker-first-seen (§3.4 of research
    // memo); just assert an NaN error was raised.
    assert!(err.to_string().contains("NaN"));
}

#[cfg(feature = "threads")]
#[test]
fn reject_inf_parallel_large_input() {
    let mut values = vec![1.0_f64; 16_384];
    values[5_000] = f64::INFINITY;
    let data = f64_bytes(&values, ByteOrder::native());
    let desc = make_descriptor(
        vec![16_384],
        Dtype::Float64,
        ByteOrder::native(),
        "none",
        "none",
    );
    let opts = EncodeOptions {
        reject_inf: true,
        threads: 4,
        ..Default::default()
    };
    let err = encode(&make_global_meta(), &[(&desc, &data)], &opts).unwrap_err();
    assert!(err.to_string().contains("Inf"));
}

// ── 13. Edge cases ──────────────────────────────────────────────────────────

#[test]
fn reject_nan_negative_zero_is_not_nan() {
    let data = f64_bytes(&[1.0, -0.0, 3.0], ByteOrder::native());
    let desc = make_descriptor(vec![3], Dtype::Float64, ByteOrder::native(), "none", "none");
    encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(true, true),
    )
    .expect("-0.0 is finite, must pass strict-finite checks");
}

#[test]
fn reject_nan_subnormal_is_not_nan() {
    let subnormal = f64::from_bits(0x0000_0000_0000_0001);
    let data = f64_bytes(&[subnormal, 1.0], ByteOrder::native());
    let desc = make_descriptor(vec![2], Dtype::Float64, ByteOrder::native(), "none", "none");
    encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(true, true),
    )
    .expect("subnormal is finite, must pass strict-finite checks");
}

#[test]
fn reject_nan_empty_array_passes() {
    let data: Vec<u8> = Vec::new();
    let desc = make_descriptor(vec![0], Dtype::Float64, ByteOrder::native(), "none", "none");
    encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(true, true),
    )
    .expect("empty array has no elements to scan");
}

#[test]
fn reject_nan_signalling_nan_is_also_rejected() {
    // Signalling NaN for f32: exp=0xFF (all ones), mantissa has top bit
    // clear but is non-zero. Pattern: 0x7F800001.
    let snan = f32::from_bits(0x7F80_0001);
    let data = f32_bytes(&[1.0, snan], ByteOrder::native());
    let desc = make_descriptor(vec![2], Dtype::Float32, ByteOrder::native(), "none", "none");
    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(true, false),
    )
    .unwrap_err();
    assert!(err.to_string().contains("NaN"));
}

// ── 14. Multi-dimensional tensors ───────────────────────────────────────────

#[test]
fn reject_nan_reports_flat_index_for_3d_tensor() {
    // 3-D tensor: shape=[2, 3, 4], row-major, NaN at (1, 1, 2).
    // Flat index = 1*(3*4) + 1*4 + 2 = 12 + 4 + 2 = 18.
    let mut values = vec![1.0_f64; 24];
    values[18] = f64::NAN;
    let data = f64_bytes(&values, ByteOrder::native());
    let desc = make_descriptor(
        vec![2, 3, 4],
        Dtype::Float64,
        ByteOrder::native(),
        "none",
        "none",
    );
    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(true, false),
    )
    .unwrap_err();
    assert_encoding_error_mentions(&err, &["NaN", "element 18", "float64"]);
}

#[test]
fn reject_inf_reports_flat_index_for_2d_float32() {
    // 2-D tensor: shape=[10, 5], row-major, +Inf at (3, 2).
    // Flat index = 3*5 + 2 = 17.
    let mut values = vec![1.0_f32; 50];
    values[17] = f32::INFINITY;
    let data = f32_bytes(&values, ByteOrder::native());
    let desc = make_descriptor(
        vec![10, 5],
        Dtype::Float32,
        ByteOrder::native(),
        "none",
        "none",
    );
    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &strict_opts(false, true),
    )
    .unwrap_err();
    assert_encoding_error_mentions(&err, &["+Inf", "element 17"]);
}

// ── 15. Multi-object encoding with axis-A parallelism ──────────────────────

#[cfg(feature = "threads")]
#[test]
fn reject_nan_axis_a_parallel_catches_bad_object() {
    // Axis A = par_iter across objects.  With threads=4 and multiple
    // objects, object encodings run in parallel.  One of them has NaN
    // and should trigger the strict-finite rejection regardless of
    // which worker handled it.
    let clean = f64_bytes(&[1.0, 2.0, 3.0, 4.0], ByteOrder::native());
    let mut dirty_vals = vec![1.0_f64; 4];
    dirty_vals[2] = f64::NAN;
    let dirty = f64_bytes(&dirty_vals, ByteOrder::native());
    let desc = make_descriptor(vec![4], Dtype::Float64, ByteOrder::native(), "none", "none");
    let opts = EncodeOptions {
        reject_nan: true,
        threads: 4,
        ..Default::default()
    };
    // Mix several clean objects with one dirty one; axis A should pick
    // whichever ordering the scheduler chooses but MUST still report
    // the NaN.
    let objects: Vec<(&DataObjectDescriptor, &[u8])> = vec![
        (&desc, &clean),
        (&desc, &clean),
        (&desc, &dirty),
        (&desc, &clean),
    ];
    let err = encode(&make_global_meta(), &objects, &opts).unwrap_err();
    assert!(err.to_string().contains("NaN"));
}
