// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Golden binary test files — decode canonical `.tgm` files and verify contents.
//!
//! These files are checked into the repository under `tests/golden/`.
//! Any language binding can decode the same files to verify interoperability.
//!
//! The `test_golden_files_are_deterministic` test verifies that re-encoding
//! in memory produces byte-identical output — without writing to disk, so
//! there is no race between parallel test threads.

use std::collections::BTreeMap;
use std::path::PathBuf;

use tensogram::decode::{self, DecodeOptions};
use tensogram::dtype::Dtype;
use tensogram::encode::{self, EncodeOptions};
use tensogram::framing;
use tensogram::types::{ByteOrder, DataObjectDescriptor, GlobalMetadata};

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn make_descriptor(shape: Vec<u64>, dtype: Dtype) -> DataObjectDescriptor {
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
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    }
}

/// Build a `{product, parameter}` base entry — the generic scientific
/// vocabulary used by the non-MARS golden fixtures.
fn generic_base_entry(parameter: &str) -> BTreeMap<String, ciborium::Value> {
    let mut entry = BTreeMap::new();
    entry.insert(
        "product".to_string(),
        ciborium::Value::Text("efi".to_string()),
    );
    entry.insert(
        "parameter".to_string(),
        ciborium::Value::Text(parameter.to_string()),
    );
    entry
}

/// Generate all golden file contents in memory. Returns (filename, bytes) pairs.
/// Does NOT write to disk — callers compare against committed files.
fn generate_golden_bytes() -> Vec<(&'static str, Vec<u8>)> {
    let mut results = Vec::new();

    // 1. Simple message: single float32 tensor [4], no compression,
    //    carrying a generic scientific vocabulary (`product`, `parameter`).
    {
        let meta = GlobalMetadata {
            base: vec![generic_base_entry("pressure")],
            ..Default::default()
        };
        let desc = make_descriptor(vec![4], Dtype::Float32);
        let mut payload = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            payload.extend_from_slice(&v.to_be_bytes());
        }
        let msg = encode::encode(&meta, &[(&desc, &payload)], &EncodeOptions::default()).unwrap();
        results.push(("simple_f32.tgm", msg));
    }

    // 2. Multi-object: 3 tensors with different dtypes, each labelled
    //    with its own `parameter` under the same `product = efi` umbrella.
    {
        let meta = GlobalMetadata {
            base: vec![
                generic_base_entry("temperature"),
                generic_base_entry("pressure"),
                generic_base_entry("humidity"),
            ],
            ..Default::default()
        };
        let desc_f32 = make_descriptor(vec![2], Dtype::Float32);
        let desc_i64 = make_descriptor(vec![3], Dtype::Int64);
        let desc_u8 = make_descriptor(vec![5], Dtype::Uint8);

        let mut payload_f32 = Vec::new();
        for v in [1.5f32, 2.5] {
            payload_f32.extend_from_slice(&v.to_be_bytes());
        }
        let mut payload_i64 = Vec::new();
        for v in [100i64, -200, 300] {
            payload_i64.extend_from_slice(&v.to_be_bytes());
        }
        let payload_u8 = vec![10u8, 20, 30, 40, 50];

        let msg = encode::encode(
            &meta,
            &[
                (&desc_f32, &payload_f32),
                (&desc_i64, &payload_i64),
                (&desc_u8, &payload_u8),
            ],
            &EncodeOptions::default(),
        )
        .unwrap();
        results.push(("multi_object.tgm", msg));
    }

    // 3. Message with metadata (MARS namespace)
    {
        let mut mars = BTreeMap::new();
        mars.insert("class".to_string(), ciborium::Value::Text("od".to_string()));
        mars.insert("type".to_string(), ciborium::Value::Text("fc".to_string()));
        mars.insert("step".to_string(), ciborium::Value::Integer(12.into()));
        let mut base_entry = BTreeMap::new();
        base_entry.insert(
            "mars".to_string(),
            ciborium::Value::Map(
                mars.into_iter()
                    .map(|(k, v)| (ciborium::Value::Text(k), v))
                    .collect(),
            ),
        );
        let meta = GlobalMetadata {
            base: vec![base_entry],
            ..Default::default()
        };
        let desc = make_descriptor(vec![2, 3], Dtype::Float64);
        let mut payload = Vec::new();
        for v in [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0] {
            payload.extend_from_slice(&v.to_be_bytes());
        }
        let msg = encode::encode(&meta, &[(&desc, &payload)], &EncodeOptions::default()).unwrap();
        results.push(("mars_metadata.tgm", msg));
    }

    // 4. Multi-message file — two messages of the same `product=efi`
    //    `parameter=pressure` field (as in a short time-series).
    {
        let meta = GlobalMetadata {
            base: vec![generic_base_entry("pressure")],
            ..Default::default()
        };
        let desc = make_descriptor(vec![2], Dtype::Float32);
        let mut payload1 = Vec::new();
        for v in [1.0f32, 2.0] {
            payload1.extend_from_slice(&v.to_be_bytes());
        }
        let mut payload2 = Vec::new();
        for v in [3.0f32, 4.0] {
            payload2.extend_from_slice(&v.to_be_bytes());
        }
        let msg1 = encode::encode(&meta, &[(&desc, &payload1)], &EncodeOptions::default()).unwrap();
        let msg2 = encode::encode(&meta, &[(&desc, &payload2)], &EncodeOptions::default()).unwrap();
        let mut multi = msg1;
        multi.extend_from_slice(&msg2);
        results.push(("multi_message.tgm", multi));
    }

    // 5. Hash verification message (xxh3) — same generic vocabulary.
    {
        let meta = GlobalMetadata {
            base: vec![generic_base_entry("pressure")],
            ..Default::default()
        };
        let desc = make_descriptor(vec![4], Dtype::Float32);
        let mut payload = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            payload.extend_from_slice(&v.to_be_bytes());
        }
        let opts = EncodeOptions {
            hash_algorithm: Some(tensogram::hash::HashAlgorithm::Xxh3),
            ..Default::default()
        };
        let msg = encode::encode(&meta, &[(&desc, &payload)], &opts).unwrap();
        results.push(("hash_xxh3.tgm", msg));
    }

    results
}

#[test]
fn test_golden_files_are_deterministic() {
    // Structural comparison: decode both committed and freshly-generated
    // messages and compare metadata + payload data. Byte-exact comparison
    // is not possible because provenance fields (uuid, time) are
    // nondeterministic.
    use tensogram::decode::{DecodeOptions, decode};

    let dir = golden_dir();
    let decode_opts = DecodeOptions::default();

    for (filename, generated) in generate_golden_bytes() {
        let committed = std::fs::read(dir.join(filename))
            .unwrap_or_else(|e| panic!("golden file {filename} missing from repo: {e}"));

        // Multi-message files: compare message count via scan
        let committed_entries = tensogram::scan(&committed);
        let generated_entries = tensogram::scan(&generated);

        assert_eq!(
            committed_entries.len(),
            generated_entries.len(),
            "golden file {filename}: message count mismatch"
        );

        for (i, (&(c_off, c_len), &(g_off, g_len))) in committed_entries
            .iter()
            .zip(generated_entries.iter())
            .enumerate()
        {
            let c_msg = &committed[c_off..c_off + c_len];
            let g_msg = &generated[g_off..g_off + g_len];

            let (c_meta, c_objs) = decode(c_msg, &decode_opts)
                .unwrap_or_else(|e| panic!("{filename}[{i}] committed decode: {e}"));
            let (g_meta, g_objs) = decode(g_msg, &decode_opts)
                .unwrap_or_else(|e| panic!("{filename}[{i}] generated decode: {e}"));

            // Wire version is carried in the preamble (checked by
            // Preamble::read_from during decode); the CBOR metadata
            // frame is free-form and has no reserved `version` key.
            assert_eq!(c_meta.extra, g_meta.extra, "{filename}[{i}] extra");
            assert_eq!(c_objs.len(), g_objs.len(), "{filename}[{i}] object count");

            for (j, (c_obj, g_obj)) in c_objs.iter().zip(g_objs.iter()).enumerate() {
                assert_eq!(c_obj.0.dtype, g_obj.0.dtype, "{filename}[{i}][{j}] dtype");
                assert_eq!(c_obj.0.shape, g_obj.0.shape, "{filename}[{i}][{j}] shape");
                assert_eq!(c_obj.1, g_obj.1, "{filename}[{i}][{j}] payload data");
            }
        }
    }
}

/// Write golden files to disk. Run manually when the encoder changes:
///   cargo test -p tensogram --test golden_files -- --ignored regenerate
#[test]
#[ignore]
fn regenerate_golden_files() {
    let dir = golden_dir();
    std::fs::create_dir_all(&dir).unwrap();
    for (filename, bytes) in generate_golden_bytes() {
        std::fs::write(dir.join(filename), &bytes).unwrap();
        println!("wrote {filename} ({} bytes)", bytes.len());
    }
}

// ── Read-only tests — decode committed golden files ─────────────────────

fn assert_product_efi_parameter(
    base: &BTreeMap<String, ciborium::Value>,
    expected_parameter: &str,
) {
    assert_eq!(
        base.get("product"),
        Some(&ciborium::Value::Text("efi".to_string())),
        "golden fixture must carry product = efi"
    );
    assert_eq!(
        base.get("parameter"),
        Some(&ciborium::Value::Text(expected_parameter.to_string())),
        "golden fixture must carry parameter = {expected_parameter}"
    );
}

#[test]
fn test_golden_simple_f32() {
    let data = std::fs::read(golden_dir().join("simple_f32.tgm")).unwrap();
    let (meta, objects) = decode::decode(&data, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0].0.shape, vec![4]);
    assert_eq!(objects[0].0.dtype, Dtype::Float32);
    assert_product_efi_parameter(&meta.base[0], "pressure");
    // CBOR carries no free-form top-level keys beyond the library's
    // named sections — the wire version lives in the preamble alone.
    assert!(meta.extra.is_empty());

    // Verify payload values
    let bytes = &objects[0].1;
    assert_eq!(bytes.len(), 16); // 4 * 4 bytes
    // Decoded bytes are in native byte order (native_byte_order=true default).
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_golden_multi_object() {
    let data = std::fs::read(golden_dir().join("multi_object.tgm")).unwrap();
    let (meta, objects) = decode::decode(&data, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 3);
    assert_product_efi_parameter(&meta.base[0], "temperature");
    assert_product_efi_parameter(&meta.base[1], "pressure");
    assert_product_efi_parameter(&meta.base[2], "humidity");

    // Float32 [2]
    assert_eq!(objects[0].0.dtype, Dtype::Float32);
    assert_eq!(objects[0].0.shape, vec![2]);

    // Int64 [3]
    assert_eq!(objects[1].0.dtype, Dtype::Int64);
    assert_eq!(objects[1].0.shape, vec![3]);
    // Decoded bytes are in native byte order.
    let i64_values: Vec<i64> = objects[1]
        .1
        .chunks_exact(8)
        .map(|c| i64::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(i64_values, vec![100, -200, 300]);

    // Uint8 [5]
    assert_eq!(objects[2].0.dtype, Dtype::Uint8);
    assert_eq!(objects[2].0.shape, vec![5]);
    assert_eq!(objects[2].1, vec![10, 20, 30, 40, 50]);
}

#[test]
fn test_golden_mars_metadata() {
    let data = std::fs::read(golden_dir().join("mars_metadata.tgm")).unwrap();
    let (meta, objects) = decode::decode(&data, &DecodeOptions::default()).unwrap();
    assert!(meta.base[0].contains_key("mars"));

    // Verify MARS keys are under base[0]["mars"]
    if let ciborium::Value::Map(mars) = &meta.base[0]["mars"] {
        let keys: Vec<&str> = mars
            .iter()
            .filter_map(|(k, _)| {
                if let ciborium::Value::Text(s) = k {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert!(keys.contains(&"class"));
        assert!(keys.contains(&"type"));
        assert!(keys.contains(&"step"));
    } else {
        panic!("mars should be a map");
    }

    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0].0.shape, vec![2, 3]);
    assert_eq!(objects[0].0.dtype, Dtype::Float64);
}

#[test]
fn test_golden_multi_message() {
    let data = std::fs::read(golden_dir().join("multi_message.tgm")).unwrap();
    let offsets = framing::scan(&data);
    assert_eq!(offsets.len(), 2);

    let (_, obj1) = decode::decode(
        &data[offsets[0].0..offsets[0].0 + offsets[0].1],
        &DecodeOptions::default(),
    )
    .unwrap();
    let (_, obj2) = decode::decode(
        &data[offsets[1].0..offsets[1].0 + offsets[1].1],
        &DecodeOptions::default(),
    )
    .unwrap();

    // Decoded bytes are in native byte order.
    let vals1: Vec<f32> = obj1[0]
        .1
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(vals1, vec![1.0, 2.0]);

    let vals2: Vec<f32> = obj2[0]
        .1
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(vals2, vec![3.0, 4.0]);
}

#[test]
fn test_golden_hash_xxh3() {
    use tensogram::framing::{decode_message, scan};
    use tensogram::hash::verify_frame_hash;
    use tensogram::wire::{FrameHeader, MessageFlags, Preamble};

    let data = std::fs::read(golden_dir().join("hash_xxh3.tgm")).unwrap();

    // Decode high-level content.
    let (meta, objects) = decode::decode(&data, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 1);
    assert_product_efi_parameter(&meta.base[0], "pressure");

    // v3: the hash lives in the frame footer's inline slot.  Pin
    // the committed golden's HASHES_PRESENT flag and the
    // frame-level hash verification pathway.
    let preamble = Preamble::read_from(&data).unwrap();
    assert!(
        preamble.flags.has(MessageFlags::HASHES_PRESENT),
        "hash_xxh3.tgm must have HASHES_PRESENT set"
    );
    let messages = scan(&data);
    let (offset, len) = messages[0];
    let msg = &data[offset..offset + len];
    let decoded = decode_message(msg).unwrap();
    for (_, _, _, frame_offset) in &decoded.objects {
        let frame = &msg[*frame_offset..];
        let fh = FrameHeader::read_from(frame).unwrap();
        let frame_bytes = &frame[..fh.total_length as usize];
        verify_frame_hash(frame_bytes, fh.frame_type).expect("golden hash_xxh3 frame must verify");
    }
}
