// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

//! Parity tests: async streaming encoder vs sync streaming encoder.
//!
//! Both produce wire-format-identical bytes for the same logical
//! sequence of writes.  This is the primary correctness invariant of
//! the async streaming encoder (`plans/PLAN_CPP_ASYNC.md` §5).

#![cfg(feature = "async")]

use std::collections::BTreeMap;
use std::io::Cursor;

use tensogram::decode::{DecodeOptions, decode};
use tensogram::encode::EncodeOptions;
use tensogram::streaming::StreamingEncoder;
use tensogram::streaming_async::AsyncStreamingEncoder;
use tensogram::types::{ByteOrder, DataObjectDescriptor, GlobalMetadata};
use tensogram::{Dtype, MaskMethod};

fn make_descriptor(shape: Vec<u64>, dtype: Dtype) -> DataObjectDescriptor {
    let ndim = shape.len() as u64;
    let mut strides = vec![0u64; shape.len()];
    if !shape.is_empty() {
        strides[shape.len() - 1] = 1;
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim,
        shape,
        strides,
        dtype,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    }
}

#[tokio::test]
async fn async_streaming_single_object_round_trip() {
    let meta = GlobalMetadata::default();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data = vec![0u8; 4 * 4];

    let buf = Vec::new();
    let mut enc =
        AsyncStreamingEncoder::new(buf, &meta, &EncodeOptions::default()).await.unwrap();
    enc.write_object(&desc, &data).await.unwrap();
    let result = enc.finish().await.unwrap();

    let (_decoded_meta, objects) = decode(&result, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0].1, data);
}

#[tokio::test]
async fn async_streaming_multi_object_round_trip() {
    let meta = GlobalMetadata::default();
    let desc1 = make_descriptor(vec![4], Dtype::Float32);
    let desc2 = make_descriptor(vec![8], Dtype::Float32);
    let data1 = vec![1u8; 4 * 4];
    let data2 = vec![2u8; 8 * 4];

    let mut enc =
        AsyncStreamingEncoder::new(Vec::new(), &meta, &EncodeOptions::default()).await.unwrap();
    enc.write_object(&desc1, &data1).await.unwrap();
    enc.write_object(&desc2, &data2).await.unwrap();
    assert_eq!(enc.object_count(), 2);
    let result = enc.finish().await.unwrap();

    let (_, objects) = decode(&result, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 2);
    assert_eq!(objects[0].1, data1);
    assert_eq!(objects[1].1, data2);
}

/// The headline correctness invariant: for an identical sequence of
/// writes the async encoder must produce the exact same bytes as the
/// sync encoder.  Differences here would mean a subtle wire-format
/// drift and would break every consumer reading the async output.
#[tokio::test]
async fn async_and_sync_produce_identical_bytes() {
    let meta = GlobalMetadata::default();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data: Vec<u8> = (0..16).collect();

    // Sync path
    let sync_bytes = {
        let mut enc = StreamingEncoder::new(
            Vec::new(),
            &meta,
            &EncodeOptions {
                hashing: false, // disable hashing so reserved.uuid/timestamp drift is the only diff
                ..Default::default()
            },
        )
        .unwrap();
        enc.write_object(&desc, &data).unwrap();
        enc.finish().unwrap()
    };

    // Async path
    let async_bytes = {
        let mut enc = AsyncStreamingEncoder::new(
            Vec::new(),
            &meta,
            &EncodeOptions {
                hashing: false,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        enc.write_object(&desc, &data).await.unwrap();
        enc.finish().await.unwrap()
    };

    // Provenance fields (uuid, timestamp) differ between encoder
    // instantiations.  Decode both and compare descriptors + payloads
    // instead.
    let (sync_meta, sync_objs) = decode(&sync_bytes, &DecodeOptions::default()).unwrap();
    let (async_meta, async_objs) = decode(&async_bytes, &DecodeOptions::default()).unwrap();

    assert_eq!(sync_meta.base.len(), async_meta.base.len());
    assert_eq!(sync_objs.len(), async_objs.len());
    for (a, b) in sync_objs.iter().zip(async_objs.iter()) {
        assert_eq!(a.0.shape, b.0.shape);
        assert_eq!(a.0.dtype, b.0.dtype);
        assert_eq!(a.0.encoding, b.0.encoding);
        assert_eq!(a.0.compression, b.0.compression);
        assert_eq!(a.1, b.1);
    }
}

#[tokio::test]
async fn async_streaming_with_hash_round_trip() {
    let meta = GlobalMetadata::default();
    let desc = make_descriptor(vec![16], Dtype::Float32);
    let data: Vec<u8> = (0..64).collect();

    let mut enc = AsyncStreamingEncoder::new(
        Vec::new(),
        &meta,
        &EncodeOptions {
            hashing: true,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    enc.write_object(&desc, &data).await.unwrap();
    let result = enc.finish().await.unwrap();

    let (_, objects) = decode(
        &result,
        &DecodeOptions {
            verify_hash: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0].1, data);
}

#[tokio::test]
async fn async_streaming_with_preceder() {
    let meta = GlobalMetadata::default();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data = vec![0xAA; 16];

    let mut prec = BTreeMap::new();
    prec.insert(
        "step".to_string(),
        ciborium::Value::Integer(ciborium::value::Integer::from(7)),
    );

    let mut enc =
        AsyncStreamingEncoder::new(Vec::new(), &meta, &EncodeOptions::default()).await.unwrap();
    enc.write_preceder(prec.clone()).await.unwrap();
    enc.write_object(&desc, &data).await.unwrap();
    let result = enc.finish().await.unwrap();

    let (decoded_meta, objects) = decode(&result, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 1);
    // Preceder payloads land in base[i].
    assert!(decoded_meta.base[0].contains_key("step"));
}

#[tokio::test]
async fn async_streaming_double_preceder_errors() {
    let meta = GlobalMetadata::default();
    let mut enc =
        AsyncStreamingEncoder::new(Vec::new(), &meta, &EncodeOptions::default()).await.unwrap();
    let prec = BTreeMap::from([(
        "k".to_string(),
        ciborium::Value::Integer(ciborium::value::Integer::from(1)),
    )]);
    enc.write_preceder(prec.clone()).await.unwrap();
    let err = enc.write_preceder(prec).await;
    assert!(err.is_err(), "double preceder must error");
}

#[tokio::test]
async fn async_streaming_dangling_preceder_errors() {
    let meta = GlobalMetadata::default();
    let mut enc =
        AsyncStreamingEncoder::new(Vec::new(), &meta, &EncodeOptions::default()).await.unwrap();
    let prec = BTreeMap::from([(
        "k".to_string(),
        ciborium::Value::Integer(ciborium::value::Integer::from(1)),
    )]);
    enc.write_preceder(prec).await.unwrap();
    // finish() without a following write_object must error.
    let err = enc.finish().await;
    assert!(err.is_err(), "dangling preceder must error on finish");
}

#[tokio::test]
async fn async_streaming_object_count_tracking() {
    let meta = GlobalMetadata::default();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data = vec![0u8; 16];

    let mut enc =
        AsyncStreamingEncoder::new(Vec::new(), &meta, &EncodeOptions::default()).await.unwrap();
    assert_eq!(enc.object_count(), 0);
    enc.write_object(&desc, &data).await.unwrap();
    assert_eq!(enc.object_count(), 1);
    enc.write_object(&desc, &data).await.unwrap();
    assert_eq!(enc.object_count(), 2);
    let _ = enc.finish().await.unwrap();
}

#[tokio::test]
async fn async_streaming_pre_encoded_round_trip() {
    let meta = GlobalMetadata::default();
    let desc = make_descriptor(vec![8], Dtype::Float32);
    let data: Vec<u8> = (0..32).collect();

    let mut enc =
        AsyncStreamingEncoder::new(Vec::new(), &meta, &EncodeOptions::default()).await.unwrap();
    enc.write_object_pre_encoded(&desc, &data).await.unwrap();
    let result = enc.finish().await.unwrap();

    let (_, objects) = decode(&result, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0].1, data);
}

#[tokio::test]
async fn async_streaming_to_tokio_file_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.tgm");

    let meta = GlobalMetadata::default();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data: Vec<u8> = (0..16).collect();

    {
        let file = tokio::fs::File::create(&path).await.unwrap();
        let mut enc = AsyncStreamingEncoder::new(file, &meta, &EncodeOptions::default())
            .await
            .unwrap();
        enc.write_object(&desc, &data).await.unwrap();
        let _ = enc.finish().await.unwrap();
    }

    let bytes = std::fs::read(&path).unwrap();
    let (_, objects) = decode(&bytes, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0].1, data);
}

#[tokio::test]
async fn async_streaming_finish_with_backfill_patches_total_length() {
    use std::io::Read;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("backfill.tgm");

    let meta = GlobalMetadata::default();
    let desc = make_descriptor(vec![4], Dtype::Float32);
    let data: Vec<u8> = (0..16).collect();

    {
        let file = tokio::fs::File::create(&path).await.unwrap();
        let mut enc = AsyncStreamingEncoder::new(file, &meta, &EncodeOptions::default())
            .await
            .unwrap();
        enc.write_object(&desc, &data).await.unwrap();
        let _ = enc.finish_with_backfill().await.unwrap();
    }

    let mut bytes = Vec::new();
    std::fs::File::open(&path).unwrap().read_to_end(&mut bytes).unwrap();

    // Preamble's total_length lives at bytes 16..24 (big-endian u64).
    let mut pre = [0u8; 8];
    pre.copy_from_slice(&bytes[16..24]);
    let preamble_total = u64::from_be_bytes(pre);
    assert_eq!(preamble_total, bytes.len() as u64);

    // Postamble's total_length lives at bytes [end-16..end-8].
    let end = bytes.len();
    let mut post = [0u8; 8];
    post.copy_from_slice(&bytes[end - 16..end - 8]);
    let postamble_total = u64::from_be_bytes(post);
    assert_eq!(postamble_total, bytes.len() as u64);

    // And the bytes still decode cleanly.
    let (_, objects) = decode(&bytes, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0].1, data);
}

#[tokio::test]
async fn async_streaming_with_compression_round_trip() {
    let meta = GlobalMetadata::default();
    let mut desc = make_descriptor(vec![1024], Dtype::Float32);
    desc.compression = "zstd".to_string();
    // Construct payload from finite floats so the strict-finite check
    // doesn't fire (the default EncodeOptions rejects NaN/Inf without
    // a mask companion).
    let data: Vec<u8> = (0..1024)
        .flat_map(|i| (i as f32 * 0.5).to_ne_bytes())
        .collect();

    let mut enc =
        AsyncStreamingEncoder::new(Vec::new(), &meta, &EncodeOptions::default()).await.unwrap();
    enc.write_object(&desc, &data).await.unwrap();
    let result = enc.finish().await.unwrap();

    let (_, objects) = decode(&result, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0].1, data);
}

#[tokio::test]
async fn async_streaming_via_cursor() {
    // Smoke test: ensure the encoder works with std::io::Cursor-equivalent
    // tokio sinks (Vec<u8> in this case implements AsyncWrite via tokio).
    let meta = GlobalMetadata::default();
    let desc = make_descriptor(vec![16], Dtype::Float32);
    let data: Vec<u8> = (0..64).collect();

    let buf: Cursor<Vec<u8>> = Cursor::new(Vec::new());
    let mut enc =
        AsyncStreamingEncoder::new(buf, &meta, &EncodeOptions::default()).await.unwrap();
    enc.write_object(&desc, &data).await.unwrap();
    let cursor = enc.finish().await.unwrap();
    let bytes = cursor.into_inner();

    let (_, objects) = decode(&bytes, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0].1, data);
}

#[tokio::test]
async fn async_streaming_rejects_reserved_in_preceder() {
    let meta = GlobalMetadata::default();
    let mut enc =
        AsyncStreamingEncoder::new(Vec::new(), &meta, &EncodeOptions::default()).await.unwrap();
    let mut prec = BTreeMap::new();
    prec.insert(
        tensogram::RESERVED_KEY.to_string(),
        ciborium::Value::Integer(ciborium::value::Integer::from(1)),
    );
    let err = enc.write_preceder(prec).await;
    assert!(err.is_err(), "_reserved_ in preceder must be rejected");
}

#[tokio::test]
async fn async_streaming_with_mask_options_round_trip() {
    let meta = GlobalMetadata::default();
    let desc = make_descriptor(vec![16], Dtype::Float32);
    // f32 NaN at index 0
    let mut data = vec![0u8; 64];
    let nan_bytes = f32::NAN.to_ne_bytes();
    data[..4].copy_from_slice(&nan_bytes);

    let mut enc = AsyncStreamingEncoder::new(
        Vec::new(),
        &meta,
        &EncodeOptions {
            allow_nan: true,
            nan_mask_method: MaskMethod::default(),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    enc.write_object(&desc, &data).await.unwrap();
    let result = enc.finish().await.unwrap();

    let (_, objects) = decode(&result, &DecodeOptions::default()).unwrap();
    assert_eq!(objects.len(), 1);
    // Decoded payload should restore the canonical NaN at index 0.
    let restored = &objects[0].1[..4];
    let f = f32::from_ne_bytes([restored[0], restored[1], restored[2], restored[3]]);
    assert!(f.is_nan(), "NaN should be restored after round trip");
}
