// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Example 06 — Integrity and hash verification
//!
//! Every frame carries an inline xxh3-64 hash slot.  When the
//! preamble's `HASHES_PRESENT` flag is set (the default), those slots
//! are populated at encode time and `validate_message` at level
//! `Integrity` or above recomputes the per-frame body hashes and
//! compares them to the inline values.
//!
//! Importantly, **`decode` is not the integrity surface.**  Both the
//! buffered and streaming decoders treat the inline hash slots as
//! opaque: they round-trip corrupted messages without complaining.
//! The canonical path to detect corruption is `validate_message`
//! (or the `tensogram validate --checksum` CLI).
//!
//! Run:
//!
//! ```bash
//! cargo run --release -p tensogram-rust-examples --bin 06_hash_verification
//! ```

use std::collections::BTreeMap;

use tensogram::{
    ByteOrder, DataObjectDescriptor, DecodeOptions, Dtype, EncodeOptions, GlobalMetadata,
    HashAlgorithm, IssueCode, ValidateOptions, decode, encode, validate_message,
};

fn make_descriptor() -> DataObjectDescriptor {
    DataObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 1,
        // Large enough that a byte-flip near the message midpoint
        // reliably lands in the data-object frame body (hashed region).
        shape: vec![4096],
        strides: vec![1],
        dtype: Dtype::Float32,
        byte_order: ByteOrder::native(),
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        masks: None,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = vec![42u8; 4096 * 4];
    let desc = make_descriptor();
    let meta = GlobalMetadata::default();

    // ── 1. Default encode populates the inline hash slots ─────────────────────
    //
    // EncodeOptions::default() sets hash_algorithm = Some(Xxh3).  When
    // hash_algorithm is Some(_) the preamble's HASHES_PRESENT flag is
    // set and every frame's inline body-hash slot is populated with
    // xxh3-64 of its body — that's the surface validate_message reads
    // at level Integrity.  The separate `create_header_hashes` /
    // `create_footer_hashes` options control an optional aggregate
    // HashFrame that lists per-object hashes in a single CBOR frame
    // (a listing convenience, not the integrity surface).

    let hashed = encode(&meta, &[(&desc, &data)], &EncodeOptions::default())?;
    println!("Hashed message:   {} bytes", hashed.len());

    // ── 2. Encode with hashing turned off ─────────────────────────────────────
    //
    // `HASHES_PRESENT` is set iff `hash_algorithm.is_some()` — setting it
    // to `None` clears the preamble flag and leaves every inline slot
    // zero.  validate_message then reports `hash_verified = false`.

    let no_hash_opts = EncodeOptions {
        hash_algorithm: None,
        ..Default::default()
    };
    let unhashed = encode(&meta, &[(&desc, &data)], &no_hash_opts)?;
    println!("Unhashed message: {} bytes", unhashed.len());

    // ── 3. Decode is hash-agnostic ────────────────────────────────────────────
    //
    // Both messages round-trip through `decode`.  `verify_hash: true`
    // is accepted for API compatibility but does not perform integrity
    // checking — use `validate_message` for that.

    decode(&hashed, &DecodeOptions::default())?;
    decode(&unhashed, &DecodeOptions::default())?;
    println!("\nBoth messages decode cleanly (decode is hash-agnostic).");

    // ── 4. validate_message at Integrity level checks inline hashes ───────────

    println!("\n=== validate_message (clean, hashed) ===");
    let report = validate_message(&hashed, &ValidateOptions::default());
    println!(
        "  issues={}  hash_verified={}  algorithm={:?}",
        report.issues.len(),
        report.hash_verified,
        HashAlgorithm::Xxh3,
    );
    assert!(report.is_ok());
    assert!(report.hash_verified);
    assert_eq!(report.issues.len(), 0);

    println!("\n=== validate_message (clean, unhashed) ===");
    let report = validate_message(&unhashed, &ValidateOptions::default());
    println!(
        "  issues={}  hash_verified={} (HASHES_PRESENT=0, nothing to verify)",
        report.issues.len(),
        report.hash_verified,
    );
    assert!(report.is_ok());
    assert!(!report.hash_verified);
    // The integrity level emits a single `NoHashAvailable` warning when
    // `HASHES_PRESENT=0`, not an error.  Pinning the code here documents
    // the contract the example is teaching.
    assert!(
        report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::NoHashAvailable),
        "expected NoHashAvailable warning on unhashed clean message",
    );

    // ── 5. Corruption detection ───────────────────────────────────────────────
    //
    // Flip one byte near the middle of the message.  With a 16 KiB
    // payload the midpoint is deep inside the data-object frame body,
    // so the recomputed xxh3 will disagree with the inline slot and
    // validate_message will report a `HashMismatch` at level
    // `Integrity`.  (Had the flip landed in a header or CBOR region
    // instead, validate would still flag it — as a structural or
    // metadata issue at levels 1 or 2.)

    let mut corrupted = hashed.clone();
    let mid = corrupted.len() / 2;
    corrupted[mid] ^= 0xFF;

    println!("\n=== validate_message (corrupted at byte {mid}) ===");
    let report = validate_message(&corrupted, &ValidateOptions::default());
    for issue in &report.issues {
        println!(
            "  [{:?}/{:?}] {:?}: {}",
            issue.severity, issue.level, issue.code, issue.description,
        );
    }
    // At 16 KiB the message midpoint lands deep inside the single data
    // object frame body, so the recomputed xxh3 must disagree with the
    // inline slot.  Assert the exact contract this example teaches.
    assert!(
        report
            .issues
            .iter()
            .any(|i| i.code == IssueCode::HashMismatch),
        "expected IssueCode::HashMismatch on a payload-byte flip",
    );
    println!("  -> inline xxh3 slot disagreed with recomputed body hash.");

    Ok(())
}
