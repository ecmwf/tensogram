// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::collections::BTreeMap;

use crate::error::{Result, TensogramError};
use crate::types::{DataObjectDescriptor, GlobalMetadata, HashFrame, IndexFrame};

/// Key reserved for library-managed metadata (ndim/shape/strides/dtype/provenance).
pub const RESERVED_KEY: &str = "_reserved_";

/// Serialize global metadata to deterministic CBOR bytes (RFC 8949 Section 4.2).
pub fn global_metadata_to_cbor(metadata: &GlobalMetadata) -> Result<Vec<u8>> {
    let mut value: ciborium::Value = ciborium::Value::serialized(metadata)
        .map_err(|e| TensogramError::Metadata(format!("failed to serialize metadata: {e}")))?;
    canonicalize(&mut value)?;
    value_to_bytes(&value)
}

/// Deserialize global metadata from CBOR bytes.
///
/// The CBOR metadata frame is free-form: only `base`, `_reserved_`, and
/// `_extra_` are named sections the library interprets.  Any other
/// top-level key (including a stray legacy `"version"` key emitted by
/// pre-0.17 encoders) is routed into `_extra_` for forward-compatibility.
/// The wire-format version lives exclusively in the preamble (see
/// [`crate::wire::WIRE_VERSION`]).
pub fn cbor_to_global_metadata(cbor_bytes: &[u8]) -> Result<GlobalMetadata> {
    let value: ciborium::Value = ciborium::from_reader(cbor_bytes)
        .map_err(|e| TensogramError::Metadata(format!("failed to parse CBOR: {e}")))?;

    let map = match value {
        ciborium::Value::Map(m) => m,
        _ => {
            return Err(TensogramError::Metadata(
                "global metadata CBOR is not a map".to_string(),
            ));
        }
    };

    let mut meta = GlobalMetadata::default();

    for (k, v) in map {
        // Skip non-text map keys defensively.  Canonical CBOR uses
        // text keys for our schema, but a malformed producer could
        // emit anything; we want to neither panic nor lose data.
        let key = match k {
            ciborium::Value::Text(s) => s,
            _ => continue,
        };

        match key.as_str() {
            "base" => {
                let entries: Vec<BTreeMap<String, ciborium::Value>> =
                    v.deserialized().map_err(|e| {
                        TensogramError::Metadata(format!("failed to deserialize base: {e}"))
                    })?;
                meta.base = entries;
            }
            "_reserved_" => {
                let entries: BTreeMap<String, ciborium::Value> = v.deserialized().map_err(|e| {
                    TensogramError::Metadata(format!("failed to deserialize _reserved_: {e}"))
                })?;
                meta.reserved = entries;
            }
            "_extra_" => {
                let entries: BTreeMap<String, ciborium::Value> = v.deserialized().map_err(|e| {
                    TensogramError::Metadata(format!("failed to deserialize _extra_: {e}"))
                })?;
                for (ek, ev) in entries {
                    meta.extra.insert(ek, ev);
                }
            }
            // Anything else (including a stray legacy `"version"` key)
            // flows into `_extra_`.  This is the free-form rule: no
            // top-level key is reserved beyond `base`, `_reserved_`,
            // and `_extra_` itself.
            _ => {
                meta.extra.insert(key, v);
            }
        }
    }

    Ok(meta)
}

/// Serialize a data object descriptor to deterministic CBOR bytes.
pub fn object_descriptor_to_cbor(desc: &DataObjectDescriptor) -> Result<Vec<u8>> {
    let mut value: ciborium::Value = ciborium::Value::serialized(desc)
        .map_err(|e| TensogramError::Metadata(format!("failed to serialize descriptor: {e}")))?;
    canonicalize(&mut value)?;
    value_to_bytes(&value)
}

/// Deserialize a data object descriptor from CBOR bytes.
pub fn cbor_to_object_descriptor(cbor_bytes: &[u8]) -> Result<DataObjectDescriptor> {
    let value: ciborium::Value = ciborium::from_reader(cbor_bytes)
        .map_err(|e| TensogramError::Metadata(format!("failed to parse descriptor CBOR: {e}")))?;
    value
        .deserialized()
        .map_err(|e| TensogramError::Metadata(format!("failed to deserialize descriptor: {e}")))
}

/// Serialize an index frame to deterministic CBOR bytes.
///
/// v3 CBOR structure (see `plans/WIRE_FORMAT.md` §6.2):
/// ```cbor
/// { "offsets": [uint, ...], "lengths": [uint, ...] }
/// ```
///
/// Object count is derived from `offsets.len()` — no separate
/// `object_count` key.
pub fn index_to_cbor(index: &IndexFrame) -> Result<Vec<u8>> {
    use ciborium::Value;
    let map = Value::Map(vec![
        (
            Value::Text("offsets".to_string()),
            Value::Array(
                index
                    .offsets
                    .iter()
                    .map(|&o| Value::Integer(o.into()))
                    .collect(),
            ),
        ),
        (
            Value::Text("lengths".to_string()),
            Value::Array(
                index
                    .lengths
                    .iter()
                    .map(|&l| Value::Integer(l.into()))
                    .collect(),
            ),
        ),
    ]);
    let mut sorted = map;
    canonicalize(&mut sorted)?;
    value_to_bytes(&sorted)
}

/// Deserialize an index frame from CBOR bytes.
pub fn cbor_to_index(cbor_bytes: &[u8]) -> Result<IndexFrame> {
    let value: ciborium::Value = ciborium::from_reader(cbor_bytes)
        .map_err(|e| TensogramError::Metadata(format!("failed to parse index CBOR: {e}")))?;

    let map = match &value {
        ciborium::Value::Map(m) => m,
        _ => {
            return Err(TensogramError::Metadata(
                "index CBOR is not a map".to_string(),
            ));
        }
    };

    let mut index = IndexFrame::default();

    for (k, v) in map {
        let key = match k {
            ciborium::Value::Text(s) => s.as_str(),
            _ => continue,
        };
        match key {
            "offsets" => {
                index.offsets = cbor_to_u64_array(v, "offsets")?;
            }
            "lengths" => {
                index.lengths = cbor_to_u64_array(v, "lengths")?;
            }
            _ => {} // ignore unknown keys (forward compat)
        }
    }

    // v3: object count is derived from offsets.len(); cross-check
    // lengths has the same cardinality.
    if index.offsets.len() != index.lengths.len() {
        return Err(TensogramError::Metadata(format!(
            "index offsets.len() ({}) != lengths.len() ({})",
            index.offsets.len(),
            index.lengths.len()
        )));
    }

    Ok(index)
}

/// Serialize a hash frame to deterministic CBOR bytes.
///
/// v3 CBOR structure (see `plans/WIRE_FORMAT.md` §6.3):
/// ```cbor
/// { "algorithm": "xxh3", "hashes": ["hex", ...] }
/// ```
///
/// Object count is derived from `hashes.len()` — no separate
/// `object_count` key.
pub fn hash_frame_to_cbor(hf: &HashFrame) -> Result<Vec<u8>> {
    use ciborium::Value;
    let map = Value::Map(vec![
        (
            Value::Text("algorithm".to_string()),
            Value::Text(hf.algorithm.clone()),
        ),
        (
            Value::Text("hashes".to_string()),
            Value::Array(hf.hashes.iter().map(|h| Value::Text(h.clone())).collect()),
        ),
    ]);
    let mut sorted = map;
    canonicalize(&mut sorted)?;
    value_to_bytes(&sorted)
}

/// Deserialize a hash frame from CBOR bytes.
///
/// v3 only accepts the new schema (`algorithm`, `hashes`).  The old
/// v2 keys (`hash_type`, `object_count`) are silently ignored as
/// unknown — callers that need to detect the legacy form should
/// check for their presence explicitly.
pub fn cbor_to_hash_frame(cbor_bytes: &[u8]) -> Result<HashFrame> {
    let value: ciborium::Value = ciborium::from_reader(cbor_bytes)
        .map_err(|e| TensogramError::Metadata(format!("failed to parse hash CBOR: {e}")))?;

    let map = match &value {
        ciborium::Value::Map(m) => m,
        _ => {
            return Err(TensogramError::Metadata(
                "hash frame CBOR is not a map".to_string(),
            ));
        }
    };

    let mut algorithm = String::new();
    let mut hashes = Vec::new();

    for (k, v) in map {
        let key = match k {
            ciborium::Value::Text(s) => s.as_str(),
            _ => continue,
        };
        match key {
            "algorithm" => {
                algorithm = match v {
                    ciborium::Value::Text(s) => s.clone(),
                    _ => {
                        return Err(TensogramError::Metadata(
                            "algorithm must be text".to_string(),
                        ));
                    }
                };
            }
            "hashes" => {
                hashes = match v {
                    ciborium::Value::Array(arr) => arr
                        .iter()
                        .map(|item| match item {
                            ciborium::Value::Text(s) => Ok(s.clone()),
                            _ => Err(TensogramError::Metadata(
                                "hash entry must be text".to_string(),
                            )),
                        })
                        .collect::<Result<Vec<_>>>()?,
                    _ => {
                        return Err(TensogramError::Metadata(
                            "hashes must be an array".to_string(),
                        ));
                    }
                };
            }
            _ => {} // ignore unknown keys (forward compat)
        }
    }

    Ok(HashFrame { algorithm, hashes })
}

// ── Common-key extraction ────────────────────────────────────────────────────

/// Extract keys common to ALL base entries.
///
/// Returns `(common_keys, remaining_per_object_entries)` where:
/// - `common_keys`: keys present in ALL entries with identical CBOR values
/// - `remaining`: per-object entries with only the varying keys
///
/// The `_reserved_` key is excluded from common computation (it's library-managed
/// and varies per object due to ndim/shape/strides/dtype).
///
/// Edge cases:
/// - 0 entries: returns (empty, empty vec)
/// - 1 entry: all keys (except `_reserved_`) are common, remaining is empty
///
/// **NaN handling:** CBOR `Float(NaN)` values use IEEE 754 equality
/// (`NaN != NaN`), so a key whose value is NaN in all entries will NOT
/// be classified as common.  This is conservative — no data is lost,
/// the key simply appears per-object rather than in the common set.
pub fn compute_common(
    base: &[BTreeMap<String, ciborium::Value>],
) -> (
    BTreeMap<String, ciborium::Value>,
    Vec<BTreeMap<String, ciborium::Value>>,
) {
    if base.is_empty() {
        return (BTreeMap::new(), Vec::new());
    }

    let first = &base[0];
    let mut common = BTreeMap::new();

    // Candidate keys come from the first entry (minus _reserved_).
    // A key is common only if every other entry has the same key with the same value.
    // Note: uses cbor_values_equal() for NaN-safe float comparison.
    for (key, value) in first {
        if key == RESERVED_KEY {
            continue;
        }
        let is_common = base[1..]
            .iter()
            .all(|entry| entry.get(key).is_some_and(|v| cbor_values_equal(v, value)));
        if is_common {
            common.insert(key.clone(), value.clone());
        }
    }

    // Build per-object remaining entries: everything that isn't common or _reserved_.
    let remaining = base
        .iter()
        .map(|entry| {
            entry
                .iter()
                .filter(|(k, _)| *k != RESERVED_KEY && !common.contains_key(*k))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        })
        .collect();

    (common, remaining)
}

// ── CBOR helpers ─────────────────────────────────────────────────────────────

/// NaN-safe equality for CBOR values.
///
/// Standard `PartialEq` for `ciborium::Value` derives from `f64::eq`,
/// so `Float(NaN) != Float(NaN)`.  This function compares floats
/// bitwise (via `to_bits`) so NaN values with identical bit patterns
/// are considered equal — matching RFC 8949 deterministic encoding
/// where the canonical NaN bit pattern is fixed.
fn cbor_values_equal(a: &ciborium::Value, b: &ciborium::Value) -> bool {
    use ciborium::Value;
    match (a, b) {
        (Value::Float(fa), Value::Float(fb)) => fa.to_bits() == fb.to_bits(),
        (Value::Array(aa), Value::Array(ab)) => {
            aa.len() == ab.len()
                && aa
                    .iter()
                    .zip(ab.iter())
                    .all(|(x, y)| cbor_values_equal(x, y))
        }
        (Value::Map(ma), Value::Map(mb)) => {
            ma.len() == mb.len()
                && ma.iter().zip(mb.iter()).all(|((ka, va), (kb, vb))| {
                    cbor_values_equal(ka, kb) && cbor_values_equal(va, vb)
                })
        }
        (Value::Tag(ta, va), Value::Tag(tb, vb)) => ta == tb && cbor_values_equal(va, vb),
        // All other variants: delegate to standard PartialEq
        _ => a == b,
    }
}

fn value_to_bytes(value: &ciborium::Value) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    ciborium::into_writer(value, &mut buf)
        .map_err(|e| TensogramError::Metadata(format!("failed to write CBOR: {e}")))?;
    Ok(buf)
}

fn cbor_to_u64(v: &ciborium::Value, field: &str) -> Result<u64> {
    match v {
        ciborium::Value::Integer(i) => {
            let n: i128 = (*i).into();
            u64::try_from(n).map_err(|_| {
                TensogramError::Metadata(format!("{field} value {n} out of u64 range"))
            })
        }
        _ => Err(TensogramError::Metadata(format!(
            "{field} must be an integer"
        ))),
    }
}

fn cbor_to_u64_array(v: &ciborium::Value, field: &str) -> Result<Vec<u64>> {
    match v {
        ciborium::Value::Array(arr) => arr.iter().map(|item| cbor_to_u64(item, field)).collect(),
        _ => Err(TensogramError::Metadata(format!(
            "{field} must be an array"
        ))),
    }
}

/// Recursively sort all map keys in a ciborium::Value tree for canonical encoding.
/// Keys are sorted by their CBOR-encoded byte representation (lexicographic).
pub(crate) fn canonicalize(value: &mut ciborium::Value) -> Result<()> {
    match value {
        ciborium::Value::Map(entries) => {
            for (k, v) in entries.iter_mut() {
                canonicalize(k)?;
                canonicalize(v)?;
            }
            let mut keyed: Vec<(Vec<u8>, (ciborium::Value, ciborium::Value))> = Vec::new();
            for (k, v) in entries.drain(..) {
                let mut key_bytes = Vec::new();
                ciborium::into_writer(&k, &mut key_bytes).map_err(|e| {
                    TensogramError::Metadata(format!("CBOR key serialisation failed: {e}"))
                })?;
                keyed.push((key_bytes, (k, v)));
            }
            keyed.sort_by(|(a, _), (b, _)| a.cmp(b));
            *entries = keyed.into_iter().map(|(_, kv)| kv).collect();
        }
        ciborium::Value::Array(items) => {
            for item in items.iter_mut() {
                canonicalize(item)?;
            }
        }
        ciborium::Value::Tag(_, inner) => {
            canonicalize(inner)?;
        }
        _ => {}
    }
    Ok(())
}

/// Verify that CBOR bytes are in RFC 8949 §4.2.1 canonical form.
///
/// Checks that all map keys are sorted by their encoded byte representation
/// (length-first, then lexicographic). Returns `Ok(())` if canonical, or an
/// error describing the first violation.
pub fn verify_canonical_cbor(cbor_bytes: &[u8]) -> Result<()> {
    let value: ciborium::Value = ciborium::from_reader(cbor_bytes)
        .map_err(|e| TensogramError::Metadata(format!("failed to parse CBOR: {e}")))?;
    verify_canonical_value(&value)
}

fn verify_canonical_value(value: &ciborium::Value) -> Result<()> {
    match value {
        ciborium::Value::Map(entries) => {
            // Check keys are sorted by encoded byte representation
            let mut prev_key_bytes: Option<Vec<u8>> = None;
            for (k, v) in entries {
                let mut key_bytes = Vec::new();
                ciborium::into_writer(k, &mut key_bytes).map_err(|e| {
                    TensogramError::Metadata(format!("CBOR key serialisation failed: {e}"))
                })?;

                if let Some(ref prev) = prev_key_bytes
                    && key_bytes <= *prev
                {
                    return Err(TensogramError::Metadata(format!(
                        "CBOR map keys not in canonical order: key {:?} should come before {:?}",
                        prev, key_bytes
                    )));
                }
                prev_key_bytes = Some(key_bytes);

                // Recurse into key and value
                verify_canonical_value(k)?;
                verify_canonical_value(v)?;
            }
        }
        ciborium::Value::Array(items) => {
            for item in items {
                verify_canonical_value(item)?;
            }
        }
        ciborium::Value::Tag(_, inner) => {
            verify_canonical_value(inner)?;
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;
    use crate::types::ByteOrder;
    use std::collections::BTreeMap;

    fn make_test_global_metadata() -> GlobalMetadata {
        let mut mars = BTreeMap::new();
        mars.insert("class".to_string(), ciborium::Value::Text("od".to_string()));
        mars.insert("type".to_string(), ciborium::Value::Text("fc".to_string()));

        let mars_value = ciborium::Value::Map(
            mars.into_iter()
                .map(|(k, v)| (ciborium::Value::Text(k), v))
                .collect(),
        );

        let mut base_entry = BTreeMap::new();
        base_entry.insert("mars".to_string(), mars_value);

        GlobalMetadata {
            base: vec![base_entry],
            ..Default::default()
        }
    }

    fn make_test_descriptor() -> DataObjectDescriptor {
        DataObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 2,
            shape: vec![10, 20],
            strides: vec![20, 1],
            dtype: Dtype::Float32,
            byte_order: ByteOrder::native(),
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            masks: None,
        }
    }

    #[test]
    fn test_global_metadata_round_trip() {
        let meta = make_test_global_metadata();
        let bytes = global_metadata_to_cbor(&meta).unwrap();
        let decoded = cbor_to_global_metadata(&bytes).unwrap();
        assert_eq!(decoded.base.len(), 1);
        assert!(decoded.base[0].contains_key("mars"));
    }

    #[test]
    fn test_cbor_has_no_version_key() {
        // The wire-format version is carried in the preamble, NOT in the
        // CBOR metadata frame.  A round-trip through the encoder must
        // never stamp a top-level `"version"` key.
        let meta = make_test_global_metadata();
        let bytes = global_metadata_to_cbor(&meta).unwrap();
        let value: ciborium::Value = ciborium::from_reader(bytes.as_slice()).unwrap();
        let map = match value {
            ciborium::Value::Map(m) => m,
            other => panic!("expected CBOR map, got {other:?}"),
        };
        for (k, _) in &map {
            if let ciborium::Value::Text(key) = k {
                assert_ne!(
                    key, "version",
                    "encoder must not write a top-level `version` key to CBOR"
                );
            }
        }
    }

    #[test]
    fn test_legacy_version_routed_to_extra() {
        // A producer that still emits a stray `version` top-level key
        // (legacy encoders pre-0.17) must decode cleanly under the
        // free-form rule — the key flows into `_extra_` instead of
        // being rejected or silently dropped.
        use ciborium::Value;

        let legacy_map = Value::Map(vec![
            (Value::Text("version".to_string()), Value::Integer(3.into())),
            (
                Value::Text("_extra_".to_string()),
                Value::Map(vec![(
                    Value::Text("note".to_string()),
                    Value::Text("hi".to_string()),
                )]),
            ),
        ]);
        let mut bytes = Vec::new();
        ciborium::into_writer(&legacy_map, &mut bytes).unwrap();

        let decoded = cbor_to_global_metadata(&bytes).unwrap();
        assert!(decoded.base.is_empty());
        assert_eq!(
            decoded.extra.get("version"),
            Some(&Value::Integer(3.into())),
            "legacy `version` key must land in `_extra_`"
        );
        assert_eq!(
            decoded.extra.get("note"),
            Some(&Value::Text("hi".to_string())),
            "explicit `_extra_` entries must still be preserved"
        );
    }

    #[test]
    fn test_free_form_top_level_keys_routed_to_extra() {
        // Arbitrary top-level keys the caller puts in the CBOR (e.g.
        // `"product"`, `"foo"`) must also flow into `_extra_` on
        // decode.  This is the free-form invariant.
        use ciborium::Value;

        let free_form = Value::Map(vec![
            (
                Value::Text("foo".to_string()),
                Value::Text("bar".to_string()),
            ),
            (
                Value::Text("product".to_string()),
                Value::Text("efi".to_string()),
            ),
        ]);
        let mut bytes = Vec::new();
        ciborium::into_writer(&free_form, &mut bytes).unwrap();

        let decoded = cbor_to_global_metadata(&bytes).unwrap();
        assert_eq!(
            decoded.extra.get("foo"),
            Some(&Value::Text("bar".to_string()))
        );
        assert_eq!(
            decoded.extra.get("product"),
            Some(&Value::Text("efi".to_string()))
        );
    }

    #[test]
    fn test_global_metadata_deterministic() {
        let meta = make_test_global_metadata();
        let b1 = global_metadata_to_cbor(&meta).unwrap();
        let b2 = global_metadata_to_cbor(&meta).unwrap();
        assert_eq!(b1, b2);
    }

    #[test]
    fn test_descriptor_round_trip() {
        let desc = make_test_descriptor();
        let bytes = object_descriptor_to_cbor(&desc).unwrap();
        let decoded = cbor_to_object_descriptor(&bytes).unwrap();
        assert_eq!(decoded.obj_type, "ntensor");
        assert_eq!(decoded.shape, vec![10, 20]);
        assert_eq!(decoded.dtype, Dtype::Float32);
        assert_eq!(decoded.encoding, "none");
    }

    #[test]
    fn test_index_round_trip() {
        let index = IndexFrame {
            offsets: vec![100, 500, 1200],
            lengths: vec![400, 700, 300],
        };
        let bytes = index_to_cbor(&index).unwrap();
        let decoded = cbor_to_index(&bytes).unwrap();
        assert_eq!(decoded.offsets.len(), 3);
        assert_eq!(decoded.offsets, vec![100, 500, 1200]);
        assert_eq!(decoded.lengths, vec![400, 700, 300]);
    }

    #[test]
    fn test_hash_frame_round_trip() {
        let hf = HashFrame {
            algorithm: "xxh3".to_string(),
            hashes: vec![
                "abcdef0123456789".to_string(),
                "1234567890abcdef".to_string(),
            ],
        };
        let bytes = hash_frame_to_cbor(&hf).unwrap();
        let decoded = cbor_to_hash_frame(&bytes).unwrap();
        assert_eq!(decoded.hashes.len(), 2);
        assert_eq!(decoded.algorithm, "xxh3");
        assert_eq!(decoded.hashes, hf.hashes);
    }

    #[test]
    fn test_empty_global_metadata() {
        let meta = GlobalMetadata::default();
        let bytes = global_metadata_to_cbor(&meta).unwrap();
        let decoded = cbor_to_global_metadata(&bytes).unwrap();
        assert!(decoded.base.is_empty());
        assert!(decoded.extra.is_empty());
    }

    // ── Phase 7: Canonical CBOR verification tests ───────────────────

    #[test]
    fn test_global_metadata_cbor_is_canonical() {
        let meta = make_test_global_metadata();
        let bytes = global_metadata_to_cbor(&meta).unwrap();
        verify_canonical_cbor(&bytes).unwrap();
    }

    #[test]
    fn test_descriptor_cbor_is_canonical() {
        let desc = make_test_descriptor();
        let bytes = object_descriptor_to_cbor(&desc).unwrap();
        verify_canonical_cbor(&bytes).unwrap();
    }

    #[test]
    fn test_index_cbor_is_canonical() {
        let index = IndexFrame {
            offsets: vec![100, 500, 1200],
            lengths: vec![400, 700, 300],
        };
        let bytes = index_to_cbor(&index).unwrap();
        verify_canonical_cbor(&bytes).unwrap();
    }

    #[test]
    fn test_hash_frame_cbor_is_canonical() {
        let hf = HashFrame {
            algorithm: "xxh3".to_string(),
            hashes: vec!["abc".to_string(), "def".to_string()],
        };
        let bytes = hash_frame_to_cbor(&hf).unwrap();
        verify_canonical_cbor(&bytes).unwrap();
    }

    #[test]
    fn test_verify_rejects_non_canonical() {
        // Build a map with keys intentionally out of order (longer key first)
        use ciborium::Value;
        let non_canonical = Value::Map(vec![
            (
                Value::Text("zzz_long_key".to_string()),
                Value::Integer(1.into()),
            ),
            (Value::Text("a".to_string()), Value::Integer(2.into())),
        ]);
        let mut buf = Vec::new();
        ciborium::into_writer(&non_canonical, &mut buf).unwrap();

        let result = verify_canonical_cbor(&buf);
        assert!(result.is_err(), "non-canonical CBOR should be rejected");
    }

    #[test]
    fn test_canonicalize_sorts_nested_maps() {
        use ciborium::Value;
        // Create a map with nested maps, keys out of order
        let mut value = Value::Map(vec![
            (
                Value::Text("z".to_string()),
                Value::Map(vec![
                    (Value::Text("b".to_string()), Value::Integer(2.into())),
                    (Value::Text("a".to_string()), Value::Integer(1.into())),
                ]),
            ),
            (Value::Text("a".to_string()), Value::Integer(0.into())),
        ]);

        canonicalize(&mut value).unwrap();

        // Serialize and verify canonical
        let mut bytes = Vec::new();
        ciborium::into_writer(&value, &mut bytes).unwrap();
        verify_canonical_cbor(&bytes).unwrap();
    }

    #[test]
    fn test_encoding_determinism_across_key_insertion_orders() {
        // Insert keys in two different orders, verify same CBOR output
        let mut base1 = BTreeMap::new();
        base1.insert("zebra".to_string(), ciborium::Value::Integer(1.into()));
        base1.insert("apple".to_string(), ciborium::Value::Integer(2.into()));
        let meta1 = GlobalMetadata {
            base: vec![base1],
            ..Default::default()
        };

        let mut base2 = BTreeMap::new();
        base2.insert("apple".to_string(), ciborium::Value::Integer(2.into()));
        base2.insert("zebra".to_string(), ciborium::Value::Integer(1.into()));
        let meta2 = GlobalMetadata {
            base: vec![base2],
            ..Default::default()
        };

        let bytes1 = global_metadata_to_cbor(&meta1).unwrap();
        let bytes2 = global_metadata_to_cbor(&meta2).unwrap();
        assert_eq!(
            bytes1, bytes2,
            "CBOR output must be independent of insertion order"
        );
    }

    // ── compute_common tests ─────────────────────────────────────────

    #[test]
    fn test_compute_common_empty() {
        let (common, remaining) = compute_common(&[]);
        assert!(common.is_empty());
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_compute_common_single_entry() {
        let mut entry = BTreeMap::new();
        entry.insert("class".to_string(), ciborium::Value::Text("od".to_string()));
        entry.insert("_reserved_".to_string(), ciborium::Value::Integer(1.into()));

        let (common, remaining) = compute_common(&[entry]);
        assert_eq!(common.len(), 1); // only "class", _reserved_ excluded
        assert!(common.contains_key("class"));
        assert!(!common.contains_key("_reserved_"));
        assert_eq!(remaining.len(), 1);
        assert!(remaining[0].is_empty());
    }

    #[test]
    fn test_compute_common_all_identical() {
        let mut e1 = BTreeMap::new();
        e1.insert("class".to_string(), ciborium::Value::Text("od".to_string()));
        e1.insert(
            "date".to_string(),
            ciborium::Value::Text("20260401".to_string()),
        );
        let e2 = e1.clone();

        let (common, remaining) = compute_common(&[e1, e2]);
        assert_eq!(common.len(), 2);
        assert!(remaining[0].is_empty());
        assert!(remaining[1].is_empty());
    }

    #[test]
    fn test_compute_common_one_varying() {
        let mut e1 = BTreeMap::new();
        e1.insert("class".to_string(), ciborium::Value::Text("od".to_string()));
        e1.insert("param".to_string(), ciborium::Value::Text("2t".to_string()));

        let mut e2 = BTreeMap::new();
        e2.insert("class".to_string(), ciborium::Value::Text("od".to_string()));
        e2.insert(
            "param".to_string(),
            ciborium::Value::Text("msl".to_string()),
        );

        let (common, remaining) = compute_common(&[e1, e2]);
        assert_eq!(common.len(), 1); // only "class"
        assert!(common.contains_key("class"));
        assert_eq!(remaining[0].len(), 1); // "param"
        assert_eq!(remaining[1].len(), 1); // "param"
    }

    #[test]
    fn test_compute_common_reserved_excluded() {
        let mut e1 = BTreeMap::new();
        e1.insert("class".to_string(), ciborium::Value::Text("od".to_string()));
        e1.insert("_reserved_".to_string(), ciborium::Value::Map(vec![]));
        let mut e2 = BTreeMap::new();
        e2.insert("class".to_string(), ciborium::Value::Text("od".to_string()));
        e2.insert("_reserved_".to_string(), ciborium::Value::Map(vec![]));

        let (common, remaining) = compute_common(&[e1, e2]);
        assert!(common.contains_key("class"));
        assert!(!common.contains_key("_reserved_"));
        // _reserved_ excluded from remaining too
        assert!(remaining[0].is_empty());
        assert!(remaining[1].is_empty());
    }

    #[test]
    fn test_compute_common_key_missing_in_some_entries() {
        // Key present in first entry but absent in second => not common
        let mut e1 = BTreeMap::new();
        e1.insert("class".to_string(), ciborium::Value::Text("od".to_string()));
        e1.insert(
            "extra".to_string(),
            ciborium::Value::Text("only_here".to_string()),
        );

        let mut e2 = BTreeMap::new();
        e2.insert("class".to_string(), ciborium::Value::Text("od".to_string()));

        let (common, remaining) = compute_common(&[e1, e2]);
        assert_eq!(common.len(), 1);
        assert!(common.contains_key("class"));
        assert_eq!(remaining[0].len(), 1); // "extra" is per-object for e1
        assert!(remaining[1].is_empty()); // e2 has nothing left
    }

    #[test]
    fn test_compute_common_three_entries() {
        let mut e1 = BTreeMap::new();
        e1.insert("class".to_string(), ciborium::Value::Text("od".to_string()));
        e1.insert("step".to_string(), ciborium::Value::Integer(0.into()));

        let mut e2 = BTreeMap::new();
        e2.insert("class".to_string(), ciborium::Value::Text("od".to_string()));
        e2.insert("step".to_string(), ciborium::Value::Integer(6.into()));

        let mut e3 = BTreeMap::new();
        e3.insert("class".to_string(), ciborium::Value::Text("od".to_string()));
        e3.insert("step".to_string(), ciborium::Value::Integer(12.into()));

        let (common, remaining) = compute_common(&[e1, e2, e3]);
        assert_eq!(common.len(), 1); // "class"
        assert!(common.contains_key("class"));
        // Each remaining has "step"
        assert_eq!(remaining[0].len(), 1);
        assert_eq!(remaining[1].len(), 1);
        assert_eq!(remaining[2].len(), 1);
    }

    #[test]
    fn test_compute_common_nan_values_treated_as_equal() {
        // NaN values with identical bit patterns should be treated as common
        let mut e1 = BTreeMap::new();
        e1.insert("class".to_string(), ciborium::Value::Text("od".to_string()));
        e1.insert("fill".to_string(), ciborium::Value::Float(f64::NAN));

        let mut e2 = BTreeMap::new();
        e2.insert("class".to_string(), ciborium::Value::Text("od".to_string()));
        e2.insert("fill".to_string(), ciborium::Value::Float(f64::NAN));

        let (common, remaining) = compute_common(&[e1, e2]);
        // Both "class" and "fill" should be common (NaN-safe comparison)
        assert_eq!(common.len(), 2);
        assert!(common.contains_key("class"));
        assert!(common.contains_key("fill"));
        assert!(remaining[0].is_empty());
        assert!(remaining[1].is_empty());
    }

    #[test]
    fn test_compute_common_nested_maps_with_nan() {
        // Nested CBOR maps containing NaN should still compare correctly
        let make_entry = || {
            let nested = ciborium::Value::Map(vec![(
                ciborium::Value::Text("fill_value".to_string()),
                ciborium::Value::Float(f64::NAN),
            )]);
            let mut e = BTreeMap::new();
            e.insert("params".to_string(), nested);
            e
        };

        let (common, remaining) = compute_common(&[make_entry(), make_entry()]);
        assert_eq!(common.len(), 1);
        assert!(common.contains_key("params"));
        assert!(remaining[0].is_empty());
        assert!(remaining[1].is_empty());
    }

    #[test]
    fn test_cbor_values_equal_basic() {
        use super::cbor_values_equal;

        // NaN == NaN (bitwise)
        assert!(cbor_values_equal(
            &ciborium::Value::Float(f64::NAN),
            &ciborium::Value::Float(f64::NAN)
        ));

        // Normal floats
        assert!(cbor_values_equal(
            &ciborium::Value::Float(1.5),
            &ciborium::Value::Float(1.5)
        ));
        assert!(!cbor_values_equal(
            &ciborium::Value::Float(1.5),
            &ciborium::Value::Float(2.5)
        ));

        // Mixed types
        assert!(!cbor_values_equal(
            &ciborium::Value::Float(1.0),
            &ciborium::Value::Integer(1.into())
        ));

        // Text
        assert!(cbor_values_equal(
            &ciborium::Value::Text("a".to_string()),
            &ciborium::Value::Text("a".to_string())
        ));
    }

    // ── Edge case: compute_common with different key sets ────────────────

    #[test]
    fn test_compute_common_different_key_sets() {
        // Entry 0 has "a", "b"; entry 1 has "b", "c".
        // Only "b" can be common if values match.
        let mut e1 = BTreeMap::new();
        e1.insert("a".to_string(), ciborium::Value::Integer(1.into()));
        e1.insert("b".to_string(), ciborium::Value::Text("shared".to_string()));

        let mut e2 = BTreeMap::new();
        e2.insert("b".to_string(), ciborium::Value::Text("shared".to_string()));
        e2.insert("c".to_string(), ciborium::Value::Integer(3.into()));

        let (common, remaining) = compute_common(&[e1, e2]);
        assert_eq!(common.len(), 1);
        assert!(common.contains_key("b"));
        // Note: "c" is NOT considered as a common candidate because
        // compute_common only examines keys from the first entry.
        // "a" is in remaining[0], "c" is in remaining[1].
        assert_eq!(remaining[0].len(), 1); // "a"
        assert!(remaining[0].contains_key("a"));
        assert_eq!(remaining[1].len(), 1); // "c"
        assert!(remaining[1].contains_key("c"));
    }

    #[test]
    fn test_compute_common_key_in_later_entry_only() {
        // Key "z" present in entry 1 only, not in entry 0 — never a common candidate.
        let mut e1 = BTreeMap::new();
        e1.insert("a".to_string(), ciborium::Value::Integer(1.into()));

        let mut e2 = BTreeMap::new();
        e2.insert("a".to_string(), ciborium::Value::Integer(1.into()));
        e2.insert("z".to_string(), ciborium::Value::Integer(99.into()));

        let (common, remaining) = compute_common(&[e1, e2]);
        assert_eq!(common.len(), 1);
        assert!(common.contains_key("a"));
        // "z" only in entry 1's remaining
        assert!(remaining[0].is_empty());
        assert_eq!(remaining[1].len(), 1);
        assert!(remaining[1].contains_key("z"));
    }

    // ── cbor_values_equal edge cases ────────────────────────────────────

    #[test]
    fn test_cbor_values_equal_booleans() {
        use super::cbor_values_equal;
        assert!(cbor_values_equal(
            &ciborium::Value::Bool(true),
            &ciborium::Value::Bool(true)
        ));
        assert!(!cbor_values_equal(
            &ciborium::Value::Bool(true),
            &ciborium::Value::Bool(false)
        ));
    }

    #[test]
    fn test_cbor_values_equal_bytes() {
        use super::cbor_values_equal;
        assert!(cbor_values_equal(
            &ciborium::Value::Bytes(vec![1, 2, 3]),
            &ciborium::Value::Bytes(vec![1, 2, 3])
        ));
        assert!(!cbor_values_equal(
            &ciborium::Value::Bytes(vec![1, 2, 3]),
            &ciborium::Value::Bytes(vec![1, 2, 4])
        ));
    }

    #[test]
    fn test_cbor_values_equal_arrays_different_lengths() {
        use super::cbor_values_equal;
        let a = ciborium::Value::Array(vec![ciborium::Value::Integer(1.into())]);
        let b = ciborium::Value::Array(vec![
            ciborium::Value::Integer(1.into()),
            ciborium::Value::Integer(2.into()),
        ]);
        assert!(!cbor_values_equal(&a, &b));
    }

    #[test]
    fn test_cbor_values_equal_nested_arrays_with_nan() {
        use super::cbor_values_equal;
        let a = ciborium::Value::Array(vec![
            ciborium::Value::Float(f64::NAN),
            ciborium::Value::Integer(1.into()),
        ]);
        let b = ciborium::Value::Array(vec![
            ciborium::Value::Float(f64::NAN),
            ciborium::Value::Integer(1.into()),
        ]);
        assert!(cbor_values_equal(&a, &b));
    }

    #[test]
    fn test_cbor_values_equal_maps_different_lengths() {
        use super::cbor_values_equal;
        let a = ciborium::Value::Map(vec![(
            ciborium::Value::Text("a".to_string()),
            ciborium::Value::Integer(1.into()),
        )]);
        let b = ciborium::Value::Map(vec![
            (
                ciborium::Value::Text("a".to_string()),
                ciborium::Value::Integer(1.into()),
            ),
            (
                ciborium::Value::Text("b".to_string()),
                ciborium::Value::Integer(2.into()),
            ),
        ]);
        assert!(!cbor_values_equal(&a, &b));
    }

    #[test]
    fn test_cbor_values_equal_tags() {
        use super::cbor_values_equal;
        let a = ciborium::Value::Tag(1, Box::new(ciborium::Value::Integer(42.into())));
        let b = ciborium::Value::Tag(1, Box::new(ciborium::Value::Integer(42.into())));
        assert!(cbor_values_equal(&a, &b));

        let c = ciborium::Value::Tag(2, Box::new(ciborium::Value::Integer(42.into())));
        assert!(!cbor_values_equal(&a, &c));

        let d = ciborium::Value::Tag(1, Box::new(ciborium::Value::Integer(99.into())));
        assert!(!cbor_values_equal(&a, &d));
    }

    #[test]
    fn test_cbor_values_equal_null() {
        use super::cbor_values_equal;
        assert!(cbor_values_equal(
            &ciborium::Value::Null,
            &ciborium::Value::Null
        ));
        assert!(!cbor_values_equal(
            &ciborium::Value::Null,
            &ciborium::Value::Bool(false)
        ));
    }

    #[test]
    fn test_cbor_values_equal_tags_with_nan_inside() {
        use super::cbor_values_equal;
        let a = ciborium::Value::Tag(1, Box::new(ciborium::Value::Float(f64::NAN)));
        let b = ciborium::Value::Tag(1, Box::new(ciborium::Value::Float(f64::NAN)));
        assert!(cbor_values_equal(&a, &b));
    }

    #[test]
    fn test_compute_common_cbor_maps_with_different_key_order() {
        // CBOR Maps with same content but different key ordering should be
        // considered equal because cbor_values_equal compares element-by-element
        // after canonicalization by BTreeMap (CBOR maps are compared positionally).
        // But here we use ciborium::Value::Map which preserves insertion order.
        // Two maps with same keys/values in different order: NOT equal by the
        // current positional comparison (this is correct — CBOR canonical encoding
        // ensures all maps are sorted, so different-order maps from a canonical
        // encoder can't happen).
        let map1 = ciborium::Value::Map(vec![
            (
                ciborium::Value::Text("a".to_string()),
                ciborium::Value::Integer(1.into()),
            ),
            (
                ciborium::Value::Text("b".to_string()),
                ciborium::Value::Integer(2.into()),
            ),
        ]);
        let map2 = ciborium::Value::Map(vec![
            (
                ciborium::Value::Text("b".to_string()),
                ciborium::Value::Integer(2.into()),
            ),
            (
                ciborium::Value::Text("a".to_string()),
                ciborium::Value::Integer(1.into()),
            ),
        ]);
        // With the current implementation, positional comparison means different
        // key orders are NOT equal.
        let mut e1 = BTreeMap::new();
        e1.insert("data".to_string(), map1);
        let mut e2 = BTreeMap::new();
        e2.insert("data".to_string(), map2);

        let (common, remaining) = compute_common(&[e1, e2]);
        // "data" will NOT be common because the maps differ positionally.
        // This is the expected behavior: canonical encoding ensures maps are
        // always in sorted order, so this scenario only arises with
        // non-canonical input.
        assert!(common.is_empty());
        assert_eq!(remaining[0].len(), 1);
        assert_eq!(remaining[1].len(), 1);
    }
}
