// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! BDD scenarios for the `simple_packing` auto-compute encoder path.
//!
//! User story:
//!
//! > I want to write a descriptor with just ``encoding: "simple_packing"``
//! > and ``sp_bits_per_value: 16`` (the knob), and have the encoder
//! > derive ``sp_reference_value`` / ``sp_binary_scale_factor`` from my
//! > data automatically.
//!
//! The tests cover the seven design-plan scenarios (S1–S7) plus a
//! parity check that auto-compute and explicit-params paths produce
//! identical decoded output, each written as a `#[test]` that reads
//! like a Given/When/Then.

use std::collections::BTreeMap;

use ciborium::Value as CborValue;
use tensogram::{
    ByteOrder, DataObjectDescriptor, DecodeOptions, Dtype, EncodeOptions, GlobalMetadata,
    HashAlgorithm, decode, encode,
};

fn make_global_meta() -> GlobalMetadata {
    GlobalMetadata::default()
}

/// Build a minimal simple_packing descriptor carrying only the user
/// knobs.  The encoder should auto-compute the rest.
fn make_auto_desc(shape: Vec<u64>, bits_per_value: i64) -> DataObjectDescriptor {
    let mut params = BTreeMap::new();
    params.insert(
        "sp_bits_per_value".to_string(),
        CborValue::Integer(bits_per_value.into()),
    );
    DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: shape.len() as u64,
        shape: shape.clone(),
        strides: c_strides(&shape),
        dtype: Dtype::Float64,
        byte_order: ByteOrder::native(),
        encoding: "simple_packing".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params,
        masks: None,
    }
}

fn c_strides(shape: &[u64]) -> Vec<u64> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1u64; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn get_i64(params: &BTreeMap<String, CborValue>, key: &str) -> i64 {
    match params.get(key) {
        Some(CborValue::Integer(i)) => {
            let n: i128 = (*i).into();
            i64::try_from(n).unwrap_or_else(|_| panic!("{key} not in i64 range"))
        }
        other => panic!("{key} missing or wrong type: {other:?}"),
    }
}

fn get_f64(params: &BTreeMap<String, CborValue>, key: &str) -> f64 {
    match params.get(key) {
        Some(CborValue::Float(f)) => *f,
        Some(CborValue::Integer(i)) => {
            let n: i128 = (*i).into();
            n as f64
        }
        other => panic!("{key} missing or wrong type: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// S1 — happy path: user supplies only sp_bits_per_value; auto-compute
//      fills in sp_reference_value + sp_binary_scale_factor; decode
//      round-trips within quantization tolerance.
// ---------------------------------------------------------------------------

#[test]
fn s1_auto_compute_roundtrips_within_tolerance() {
    let desc = make_auto_desc(vec![4], 16);
    let values: Vec<f64> = vec![270.0, 275.0, 280.0, 285.0];
    let data = f64_bytes(&values);

    let bytes = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .expect("auto-compute encode should succeed");

    let (_meta, objects) = decode(&bytes, &DecodeOptions::default()).expect("decode");
    assert_eq!(objects.len(), 1);
    let (returned_desc, payload) = &objects[0];

    // Every 4 sp_* keys must be present after encode.
    assert!(returned_desc.params.contains_key("sp_reference_value"));
    assert!(returned_desc.params.contains_key("sp_binary_scale_factor"));
    assert!(returned_desc.params.contains_key("sp_bits_per_value"));
    assert!(returned_desc.params.contains_key("sp_decimal_scale_factor"));
    assert_eq!(get_i64(&returned_desc.params, "sp_bits_per_value"), 16);
    assert_eq!(get_i64(&returned_desc.params, "sp_decimal_scale_factor"), 0);

    // Values round-trip within 16-bit simple_packing tolerance.
    assert_eq!(payload.len(), values.len() * 8);
    let decoded: Vec<f64> = payload
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    let tolerance = (values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - values.iter().cloned().fold(f64::INFINITY, f64::min))
        / f64::from(1u32 << 16);
    for (i, (v, d)) in values.iter().zip(decoded.iter()).enumerate() {
        assert!(
            (v - d).abs() <= tolerance,
            "index {i}: original {v}, decoded {d}, tolerance {tolerance}"
        );
    }
}

// ---------------------------------------------------------------------------
// S2 — explicit computed params win (Q2 option b).
// ---------------------------------------------------------------------------

#[test]
fn s2_explicit_computed_params_are_used_verbatim() {
    let mut desc = make_auto_desc(vec![4], 16);
    // Explicit sp_reference_value + sp_binary_scale_factor — the auto
    // computer should see these and short-circuit, using them as-is.
    desc.params
        .insert("sp_reference_value".to_string(), CborValue::Float(200.0));
    desc.params.insert(
        "sp_binary_scale_factor".to_string(),
        CborValue::Integer(5i64.into()),
    );
    let values: Vec<f64> = vec![270.0, 275.0, 280.0, 285.0];
    let data = f64_bytes(&values);

    let bytes = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .expect("explicit params encode should succeed");

    let (_meta, objects) = decode(&bytes, &DecodeOptions::default()).expect("decode");
    let (returned_desc, _payload) = &objects[0];
    // Values must be preserved verbatim — not replaced by the derived
    // ones that auto-compute would have produced.
    assert_eq!(get_f64(&returned_desc.params, "sp_reference_value"), 200.0);
    assert_eq!(get_i64(&returned_desc.params, "sp_binary_scale_factor"), 5);
}

// ---------------------------------------------------------------------------
// S3 — missing sp_bits_per_value → clear error.
// ---------------------------------------------------------------------------

#[test]
fn s3_missing_bits_per_value_is_a_clear_error() {
    let mut desc = make_auto_desc(vec![4], 16);
    desc.params.clear(); // drop sp_bits_per_value
    let values: Vec<f64> = vec![270.0, 275.0, 280.0, 285.0];
    let data = f64_bytes(&values);

    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("sp_bits_per_value"),
        "error must name the missing key: {err}"
    );
    assert!(
        err.contains("auto-compute"),
        "error must mention the auto-compute facility: {err}"
    );
}

// ---------------------------------------------------------------------------
// S4 — NaN in data: auto-compute must surface the underlying
//      PackingError::NanValue, not a cryptic failure.
// ---------------------------------------------------------------------------

#[test]
fn s4_nan_in_data_is_rejected_cleanly() {
    let desc = make_auto_desc(vec![4], 16);
    let values: Vec<f64> = vec![270.0, f64::NAN, 280.0, 285.0];
    let data = f64_bytes(&values);

    // Default options have allow_nan=false — the substitute_and_mask
    // stage errors out BEFORE auto-compute runs.  Either way the
    // caller sees "NaN".
    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.to_lowercase().contains("nan"),
        "error must mention NaN: {err}"
    );
}

// ---------------------------------------------------------------------------
// S5 — auto-compute works in the streaming encoder too.
// ---------------------------------------------------------------------------

#[test]
fn s5_streaming_encoder_auto_computes() {
    use std::io::Cursor;
    use tensogram::StreamingEncoder;

    let desc = make_auto_desc(vec![4], 16);
    let values: Vec<f64> = vec![270.0, 275.0, 280.0, 285.0];
    let data = f64_bytes(&values);

    let buf = Cursor::new(Vec::<u8>::new());
    let meta = make_global_meta();
    let opts = EncodeOptions::default();
    let mut enc = StreamingEncoder::new(buf, &meta, &opts).expect("streaming encoder");
    enc.write_object(&desc, &data)
        .expect("streaming write_object auto-compute");
    let finished = enc.finish().expect("finish");

    let bytes = finished.into_inner();
    let (_meta, objects) = decode(&bytes, &DecodeOptions::default()).expect("decode streamed");
    let (returned_desc, _payload) = &objects[0];
    assert!(returned_desc.params.contains_key("sp_reference_value"));
    assert!(returned_desc.params.contains_key("sp_binary_scale_factor"));
}

// ---------------------------------------------------------------------------
// S6 — sp_decimal_scale_factor defaults to 0 when absent.
// ---------------------------------------------------------------------------

#[test]
fn s6_decimal_scale_factor_defaults_to_zero() {
    let desc = make_auto_desc(vec![4], 16);
    let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let data = f64_bytes(&values);

    let bytes = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .expect("encode");
    let (_meta, objects) = decode(&bytes, &DecodeOptions::default()).unwrap();
    let (returned_desc, _) = &objects[0];
    assert_eq!(
        get_i64(&returned_desc.params, "sp_decimal_scale_factor"),
        0,
        "default must be 0 when the user omits the key"
    );
}

// ---------------------------------------------------------------------------
// S7 — non-float64 dtype + simple_packing is rejected with a clear
//      error at auto-compute time (same message as the explicit path).
// ---------------------------------------------------------------------------

#[test]
fn s7_non_f64_dtype_rejected_at_auto_compute() {
    let mut desc = make_auto_desc(vec![4], 16);
    desc.dtype = Dtype::Float32;
    let values: Vec<f32> = vec![270.0, 275.0, 280.0, 285.0];
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("simple_packing only supports float64"),
        "error must name the dtype constraint: {err}"
    );
}

// ---------------------------------------------------------------------------
// Parity check — auto-computed output equals explicit path on the
// same data.  This is the key invariant: the ergonomic form produces
// byte-equivalent output to the expert form.
// ---------------------------------------------------------------------------

#[test]
fn auto_compute_matches_explicit_path_byte_for_byte() {
    let values: Vec<f64> = vec![100.0, 105.5, 110.25, 115.125, 120.0625, 125.0, 130.0];
    let data = f64_bytes(&values);

    // Explicit path — caller pre-computes params.
    let explicit_params = tensogram_encodings::simple_packing::compute_params(&values, 16, 0)
        .expect("compute_params");
    let mut explicit_desc = make_auto_desc(vec![values.len() as u64], 16);
    explicit_desc.params.insert(
        "sp_reference_value".to_string(),
        CborValue::Float(explicit_params.reference_value),
    );
    explicit_desc.params.insert(
        "sp_binary_scale_factor".to_string(),
        CborValue::Integer(i64::from(explicit_params.binary_scale_factor).into()),
    );
    explicit_desc.params.insert(
        "sp_decimal_scale_factor".to_string(),
        CborValue::Integer(0i64.into()),
    );

    // Auto-compute path — caller only sets sp_bits_per_value.
    let auto_desc = make_auto_desc(vec![values.len() as u64], 16);

    // Deterministic encode options — both paths must produce the same
    // payload bytes (provenance timestamps still differ at the CBOR
    // metadata level, so we compare the decoded descriptor params +
    // decoded payload rather than raw wire bytes).
    let opts = EncodeOptions {
        hash_algorithm: Some(HashAlgorithm::Xxh3),
        ..EncodeOptions::default()
    };

    let explicit_bytes = encode(&make_global_meta(), &[(&explicit_desc, &data)], &opts).unwrap();
    let auto_bytes = encode(&make_global_meta(), &[(&auto_desc, &data)], &opts).unwrap();

    let (_e_meta, e_objects) = decode(&explicit_bytes, &DecodeOptions::default()).unwrap();
    let (_a_meta, a_objects) = decode(&auto_bytes, &DecodeOptions::default()).unwrap();

    let (e_desc, e_payload) = &e_objects[0];
    let (a_desc, a_payload) = &a_objects[0];

    for key in [
        "sp_reference_value",
        "sp_binary_scale_factor",
        "sp_decimal_scale_factor",
        "sp_bits_per_value",
    ] {
        assert_eq!(
            e_desc.params.get(key),
            a_desc.params.get(key),
            "descriptor key {key} must match"
        );
    }
    assert_eq!(
        e_payload, a_payload,
        "encoded payloads must match byte-for-byte"
    );
}

// ---------------------------------------------------------------------------
// E1 — partial-explicit params reject (Pass-3 hardening).
//
// Providing exactly one of sp_reference_value / sp_binary_scale_factor
// is ambiguous: auto-compute would silently overwrite the user-supplied
// half, which is a precision footgun.  The encoder must reject it
// up-front with a clear error pointing at the missing partner key.
// ---------------------------------------------------------------------------

#[test]
fn e1_only_sp_reference_value_rejected() {
    let mut desc = make_auto_desc(vec![4], 16);
    desc.params
        .insert("sp_reference_value".to_string(), CborValue::Float(200.0));
    let values: Vec<f64> = vec![270.0, 275.0, 280.0, 285.0];
    let data = f64_bytes(&values);

    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("sp_reference_value") && err.contains("sp_binary_scale_factor"),
        "error must name both keys: {err}"
    );
}

#[test]
fn e1_only_sp_binary_scale_factor_rejected() {
    let mut desc = make_auto_desc(vec![4], 16);
    desc.params.insert(
        "sp_binary_scale_factor".to_string(),
        CborValue::Integer(5i64.into()),
    );
    let values: Vec<f64> = vec![270.0, 275.0, 280.0, 285.0];
    let data = f64_bytes(&values);

    let err = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("sp_binary_scale_factor") && err.contains("sp_reference_value"),
        "error must name both keys: {err}"
    );
}

// ---------------------------------------------------------------------------
// E2 — auto-compute uses the ORIGINAL data, not post-substitute bytes.
//
// When `allow_nan = true` substitutes NaN with 0.0, running auto-compute
// on the substituted bytes would derive a distorted sp_reference_value
// (silent precision loss).  Auto-compute must run on the pre-substitute
// data so non-finite inputs surface as a clean PackingError, telling
// the user to either fix the data or supply explicit params.
// ---------------------------------------------------------------------------

#[test]
fn e2_nan_data_with_allow_nan_still_rejected_by_auto_compute() {
    use tensogram::MaskMethod;

    let desc = make_auto_desc(vec![4], 16);
    let values: Vec<f64> = vec![270.0, f64::NAN, 280.0, 285.0];
    let data = f64_bytes(&values);

    // Even with allow_nan=true (which would normally substitute NaN
    // with 0.0 and build a mask), the auto-compute step runs on the
    // ORIGINAL data — `compute_params` rejects the NaN before any
    // substitution-induced precision distortion can land in the
    // descriptor.
    let opts = EncodeOptions {
        allow_nan: true,
        nan_mask_method: MaskMethod::Roaring,
        ..EncodeOptions::default()
    };
    let err = encode(&make_global_meta(), &[(&desc, &data)], &opts)
        .unwrap_err()
        .to_string();
    assert!(
        err.to_lowercase().contains("nan"),
        "auto-compute should surface NaN: {err}"
    );
}

// ---------------------------------------------------------------------------
// E3 — auto-compute composes with shuffle + zstd (Pass-4 hardening).
//
// The auto-compute step happens before the pipeline-config build, so
// shuffle / zstd / blosc2 / szip should all compose cleanly with it.
// Pin it.
// ---------------------------------------------------------------------------

#[test]
fn e3_auto_compute_composes_with_shuffle_and_zstd() {
    let mut desc = make_auto_desc(vec![1024], 16);
    desc.filter = "shuffle".to_string();
    desc.compression = "zstd".to_string();
    // shuffle on simple_packing → element size = ceil(bits/8) = 2 bytes
    desc.params.insert(
        "shuffle_element_size".to_string(),
        CborValue::Integer(2i64.into()),
    );
    desc.params
        .insert("zstd_level".to_string(), CborValue::Integer(3i64.into()));

    let values: Vec<f64> = (0..1024).map(|i| 270.0 + (i as f64) * 0.05).collect();
    let data = f64_bytes(&values);

    let bytes = encode(
        &make_global_meta(),
        &[(&desc, &data)],
        &EncodeOptions::default(),
    )
    .expect("auto-compute + shuffle + zstd should work");

    let (_meta, objects) = decode(&bytes, &DecodeOptions::default()).expect("decode");
    let (returned_desc, payload) = &objects[0];

    // All four sp_* keys present after encode.
    assert!(returned_desc.params.contains_key("sp_reference_value"));
    assert!(returned_desc.params.contains_key("sp_binary_scale_factor"));
    // Round-trip within 16-bit tolerance.
    let decoded: Vec<f64> = payload
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    let tolerance = (values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - values.iter().cloned().fold(f64::INFINITY, f64::min))
        / f64::from(1u32 << 16);
    for (v, d) in values.iter().zip(decoded.iter()) {
        assert!((v - d).abs() <= tolerance);
    }
}

// ---------------------------------------------------------------------------
// E4 — auto-compute honours non-native byte_order (Pass-4 hardening).
//
// `bytes_as_f64_vec` reads the descriptor's `byte_order` to decide
// `from_be_bytes` vs `from_le_bytes`.  When the user declares the
// non-native order on the descriptor, the input bytes must be in that
// order; auto-compute interprets them correctly and produces the same
// `sp_reference_value` / `sp_binary_scale_factor` as a native-order
// encode of the same values.
// ---------------------------------------------------------------------------

#[test]
fn e4_auto_compute_handles_big_endian_descriptor() {
    let values: Vec<f64> = vec![270.0, 275.0, 280.0, 285.0];

    // Native-endian reference encode.
    let native_desc = make_auto_desc(vec![4], 16);
    let native_bytes = f64_bytes(&values);
    let native_buf = encode(
        &make_global_meta(),
        &[(&native_desc, &native_bytes)],
        &EncodeOptions::default(),
    )
    .unwrap();

    // Big-endian descriptor: caller declares the byte order AND
    // supplies bytes in that order.
    let mut be_desc = make_auto_desc(vec![4], 16);
    be_desc.byte_order = ByteOrder::Big;
    let be_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();
    let be_buf = encode(
        &make_global_meta(),
        &[(&be_desc, &be_bytes)],
        &EncodeOptions::default(),
    )
    .unwrap();

    // Decoded sp_reference_value and sp_binary_scale_factor must match
    // — auto-compute should not be byte-order-sensitive.
    let (_, n_objs) = decode(&native_buf, &DecodeOptions::default()).unwrap();
    let (_, b_objs) = decode(&be_buf, &DecodeOptions::default()).unwrap();
    let n_ref = get_f64(&n_objs[0].0.params, "sp_reference_value");
    let b_ref = get_f64(&b_objs[0].0.params, "sp_reference_value");
    assert_eq!(
        n_ref, b_ref,
        "sp_reference_value must be byte-order invariant"
    );

    let n_bsf = get_i64(&n_objs[0].0.params, "sp_binary_scale_factor");
    b_endian_assertion(b_objs[0].0.params.get("sp_binary_scale_factor"), n_bsf);

    // Also check we get nearly-identical decoded values (modulo
    // platform-native re-encoding from the decoded path).
    let n_payload: Vec<f64> = n_objs[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    let b_payload: Vec<f64> = b_objs[0]
        .1
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    for (n, b) in n_payload.iter().zip(b_payload.iter()) {
        assert!((n - b).abs() < 1e-9);
    }
}

fn b_endian_assertion(actual: Option<&CborValue>, expected_native: i64) {
    match actual {
        Some(CborValue::Integer(i)) => {
            let n: i128 = (*i).into();
            assert_eq!(
                i64::try_from(n).unwrap(),
                expected_native,
                "sp_binary_scale_factor must be byte-order invariant"
            );
        }
        other => panic!("missing sp_binary_scale_factor: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// E5 — multi-object messages can mix auto-compute and explicit-params
//      objects (Pass-4 hardening).
//
// Each object's descriptor is processed independently in
// `encode_one_object`, so a single message can hold one auto-compute
// simple_packing object next to an explicit-params one and a
// no-encoding one.  Pin that.
// ---------------------------------------------------------------------------

#[test]
fn e5_multi_object_mixed_encoding_round_trips() {
    // Object 0: auto-compute simple_packing.
    let mut auto_desc = make_auto_desc(vec![4], 16);
    auto_desc.params.insert(
        "sp_decimal_scale_factor".to_string(),
        CborValue::Integer(0i64.into()),
    );
    let auto_values: Vec<f64> = vec![270.0, 275.0, 280.0, 285.0];
    let auto_data = f64_bytes(&auto_values);

    // Object 1: explicit-params simple_packing.
    let mut explicit_desc = make_auto_desc(vec![4], 16);
    explicit_desc
        .params
        .insert("sp_reference_value".to_string(), CborValue::Float(100.0));
    explicit_desc.params.insert(
        "sp_binary_scale_factor".to_string(),
        CborValue::Integer(2i64.into()),
    );
    explicit_desc.params.insert(
        "sp_decimal_scale_factor".to_string(),
        CborValue::Integer(0i64.into()),
    );
    let explicit_values: Vec<f64> = vec![100.0, 100.5, 101.0, 101.5];
    let explicit_data = f64_bytes(&explicit_values);

    // Object 2: encoding="none".
    let mut none_desc = make_auto_desc(vec![4], 16);
    none_desc.params.clear(); // drop sp_bits_per_value
    none_desc.encoding = "none".to_string();
    let none_values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let none_data = f64_bytes(&none_values);

    let bytes = encode(
        &make_global_meta(),
        &[
            (&auto_desc, &auto_data),
            (&explicit_desc, &explicit_data),
            (&none_desc, &none_data),
        ],
        &EncodeOptions::default(),
    )
    .expect("multi-object mixed encoding");

    let (_meta, objects) = decode(&bytes, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 3);

    // Object 0 gained the auto-computed keys.
    assert!(objects[0].0.params.contains_key("sp_reference_value"));
    assert!(objects[0].0.params.contains_key("sp_binary_scale_factor"));

    // Object 1 kept its explicit values verbatim.
    assert_eq!(get_f64(&objects[1].0.params, "sp_reference_value"), 100.0);
    assert_eq!(get_i64(&objects[1].0.params, "sp_binary_scale_factor"), 2);

    // Object 2 has no sp_* keys (it's encoding=none).
    assert_eq!(objects[2].0.encoding, "none");
    assert!(!objects[2].0.params.contains_key("sp_reference_value"));
}
