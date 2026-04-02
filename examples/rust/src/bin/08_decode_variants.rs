//! Example 08 — All decode variants
//!
//! Tensogram provides four decode functions to match different use cases:
//!
//!   decode()           — all objects, full pipeline
//!   decode_metadata()  — CBOR only, no payload bytes touched
//!   decode_object()    — single object by index (O(1) via binary header)
//!   decode_range()     — contiguous sub-slice of an uncompressed object
//!
//! This example builds a 3-object message and demonstrates each variant.

use std::collections::BTreeMap;

use tensogram_core::{
    decode, decode_metadata, decode_object, decode_range, encode, ByteOrder, DecodeOptions, Dtype,
    EncodeOptions, Metadata, ObjectDescriptor, PayloadDescriptor,
};

fn make_metadata_3obj() -> Metadata {
    let make_obj = |shape: Vec<u64>, dtype: Dtype| {
        let strides: Vec<u64> = {
            let mut s = vec![1u64; shape.len()];
            for i in (0..shape.len() - 1).rev() {
                s[i] = s[i + 1] * shape[i + 1];
            }
            s
        };
        ObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: shape.len() as u64,
            shape,
            strides,
            dtype,
            extra: BTreeMap::new(),
        }
    };

    let make_payload = || PayloadDescriptor {
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    };

    Metadata {
        version: 1,
        objects: vec![
            make_obj(vec![10, 20], Dtype::Float32), // obj 0: 10×20 f32
            make_obj(vec![5], Dtype::Float64),      // obj 1: 5 f64
            make_obj(vec![8, 8], Dtype::Uint8),     // obj 2: 8×8 u8
        ],
        payload: vec![make_payload(), make_payload(), make_payload()],
        extra: BTreeMap::new(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let metadata = make_metadata_3obj();

    // Known data: each object filled with a distinct byte value
    let data0 = vec![0xAAu8; 10 * 20 * 4]; // float32
    let data1 = vec![0xBBu8; 5 * 8]; // float64
    let data2 = vec![0xCCu8; 8 * 8]; // uint8

    let message = encode(
        &metadata,
        &[&data0, &data1, &data2],
        &EncodeOptions::default(),
    )?;

    println!("Message: {} bytes  (3 objects)\n", message.len());

    // ── decode_metadata() ─────────────────────────────────────────────────────
    //
    // Parses binary header + CBOR only. Zero payload bytes allocated.
    // Use this for filtering/listing large files efficiently.
    {
        let meta = decode_metadata(&message)?;
        println!("decode_metadata():");
        for (i, obj) in meta.objects.iter().enumerate() {
            println!("  [{i}] dtype={}, shape={:?}", obj.dtype, obj.shape);
        }
        // Verify hash descriptor was stored by encoder
        let hash = &meta.payload[0].hash;
        println!(
            "  payload[0].hash: {:?}",
            hash.as_ref().map(|h| &h.hash_type)
        );
    }

    // ── decode() — all objects ────────────────────────────────────────────────
    {
        let (_meta, objects) = decode(&message, &DecodeOptions::default())?;
        println!("\ndecode() — all objects:");
        for (i, obj) in objects.iter().enumerate() {
            println!("  [{i}] {} bytes, first_byte=0x{:02X}", obj.len(), obj[0]);
        }
        assert_eq!(objects[0], data0);
        assert_eq!(objects[1], data1);
        assert_eq!(objects[2], data2);
    }

    // ── decode_object() — single object ──────────────────────────────────────
    //
    // The binary header stores an offset for every object, so this is O(1):
    // it seeks directly to the requested object and decodes only that one.
    {
        let (_meta, obj1) = decode_object(&message, 1, &DecodeOptions::default())?;
        println!("\ndecode_object(index=1):");
        println!("  {} bytes, first_byte=0x{:02X}", obj1.len(), obj1[0]);
        assert_eq!(obj1, data1);

        // Out-of-range index → TensogramError::Object
        let result = decode_object(&message, 99, &DecodeOptions::default());
        println!("  index=99 → error: {}", result.unwrap_err());
    }

    // ── decode_object() with hash verification ─────────────────────────────────
    {
        let verify_opts = DecodeOptions { verify_hash: true };
        let (_meta, obj0) = decode_object(&message, 0, &verify_opts)?;
        println!("\ndecode_object(index=0, verify_hash=true):");
        println!("  {} bytes, hash OK", obj0.len());
    }

    // ── decode_range() — partial sub-slice ────────────────────────────────────
    //
    // Extracts a contiguous slice of elements from object 2 (uint8, 8×8).
    // Works only for encoding="none" and compression="none".
    // ranges = [(element_offset, element_count), ...]
    {
        // Extract elements 10..18 (8 elements from the second row of the 8×8 grid)
        let partial = decode_range(
            &message,
            2,          // object index
            &[(10, 8)], // offset=10, count=8
            &DecodeOptions::default(),
        )?;

        println!("\ndecode_range(obj=2, offset=10, count=8):");
        println!(
            "  {} bytes, all=0x{:02X}: {}",
            partial.len(),
            0xCC,
            partial.iter().all(|&b| b == 0xCC)
        );
        assert_eq!(partial.len(), 8);

        // Multiple ranges in one call (non-contiguous slices)
        let partial2 = decode_range(
            &message,
            2,
            &[(0, 4), (60, 4)], // first 4 bytes + last 4 bytes of the 8×8 grid
            &DecodeOptions::default(),
        )?;
        println!(
            "  Two ranges [(0,4),(60,4)]: {} bytes total",
            partial2.len()
        );
        assert_eq!(partial2.len(), 8);
    }

    println!("\nAll decode variants OK.");
    Ok(())
}
