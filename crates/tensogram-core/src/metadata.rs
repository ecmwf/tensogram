use crate::error::{Result, TensogramError};
use crate::types::{DataObjectDescriptor, GlobalMetadata, HashFrame, IndexFrame};

/// Serialize global metadata to deterministic CBOR bytes (RFC 8949 Section 4.2).
pub fn global_metadata_to_cbor(metadata: &GlobalMetadata) -> Result<Vec<u8>> {
    let mut value: ciborium::Value = ciborium::Value::serialized(metadata)
        .map_err(|e| TensogramError::Metadata(format!("failed to serialize metadata: {e}")))?;
    canonicalize(&mut value)?;
    value_to_bytes(&value)
}

/// Deserialize global metadata from CBOR bytes.
pub fn cbor_to_global_metadata(cbor_bytes: &[u8]) -> Result<GlobalMetadata> {
    let value: ciborium::Value = ciborium::from_reader(cbor_bytes)
        .map_err(|e| TensogramError::Metadata(format!("failed to parse CBOR: {e}")))?;
    value
        .deserialized()
        .map_err(|e| TensogramError::Metadata(format!("failed to deserialize metadata: {e}")))
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
/// CBOR structure: { "object_count": uint, "offsets": [uint, ...], "lengths": [uint, ...] }
pub fn index_to_cbor(index: &IndexFrame) -> Result<Vec<u8>> {
    use ciborium::Value;
    let map = Value::Map(vec![
        (
            Value::Text("object_count".to_string()),
            Value::Integer(index.object_count.into()),
        ),
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
            ))
        }
    };

    let mut index = IndexFrame::default();

    for (k, v) in map {
        let key = match k {
            ciborium::Value::Text(s) => s.as_str(),
            _ => continue,
        };
        match key {
            "object_count" => {
                index.object_count = cbor_to_u64(v, "object_count")?;
            }
            "offsets" => {
                index.offsets = cbor_to_u64_array(v, "offsets")?;
            }
            "lengths" => {
                index.lengths = cbor_to_u64_array(v, "lengths")?;
            }
            _ => {} // ignore unknown keys
        }
    }

    if index.object_count != index.offsets.len() as u64 {
        return Err(TensogramError::Metadata(format!(
            "index object_count ({}) != offsets.len() ({})",
            index.object_count,
            index.offsets.len()
        )));
    }

    Ok(index)
}

/// Serialize a hash frame to deterministic CBOR bytes.
///
/// CBOR structure: { "object_count": uint, "hash_type": text, "hashes": [text, ...] }
pub fn hash_frame_to_cbor(hf: &HashFrame) -> Result<Vec<u8>> {
    use ciborium::Value;
    let map = Value::Map(vec![
        (
            Value::Text("hash_type".to_string()),
            Value::Text(hf.hash_type.clone()),
        ),
        (
            Value::Text("hashes".to_string()),
            Value::Array(hf.hashes.iter().map(|h| Value::Text(h.clone())).collect()),
        ),
        (
            Value::Text("object_count".to_string()),
            Value::Integer(hf.object_count.into()),
        ),
    ]);
    let mut sorted = map;
    canonicalize(&mut sorted)?;
    value_to_bytes(&sorted)
}

/// Deserialize a hash frame from CBOR bytes.
pub fn cbor_to_hash_frame(cbor_bytes: &[u8]) -> Result<HashFrame> {
    let value: ciborium::Value = ciborium::from_reader(cbor_bytes)
        .map_err(|e| TensogramError::Metadata(format!("failed to parse hash CBOR: {e}")))?;

    let map = match &value {
        ciborium::Value::Map(m) => m,
        _ => {
            return Err(TensogramError::Metadata(
                "hash frame CBOR is not a map".to_string(),
            ))
        }
    };

    let mut object_count = 0u64;
    let mut hash_type = String::new();
    let mut hashes = Vec::new();

    for (k, v) in map {
        let key = match k {
            ciborium::Value::Text(s) => s.as_str(),
            _ => continue,
        };
        match key {
            "object_count" => {
                object_count = cbor_to_u64(v, "object_count")?;
            }
            "hash_type" => {
                hash_type = match v {
                    ciborium::Value::Text(s) => s.clone(),
                    _ => {
                        return Err(TensogramError::Metadata(
                            "hash_type must be text".to_string(),
                        ))
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
                        ))
                    }
                };
            }
            _ => {}
        }
    }

    if object_count != hashes.len() as u64 {
        return Err(TensogramError::Metadata(format!(
            "hash frame object_count ({object_count}) != hashes.len() ({})",
            hashes.len()
        )));
    }

    Ok(HashFrame {
        object_count,
        hash_type,
        hashes,
    })
}

// ── CBOR helpers ─────────────────────────────────────────────────────────────

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

                if let Some(ref prev) = prev_key_bytes {
                    if key_bytes <= *prev {
                        return Err(TensogramError::Metadata(format!(
                            "CBOR map keys not in canonical order: key {:?} should come before {:?}",
                            prev, key_bytes
                        )));
                    }
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

        let mut extra = BTreeMap::new();
        extra.insert(
            "mars".to_string(),
            ciborium::Value::Map(
                mars.into_iter()
                    .map(|(k, v)| (ciborium::Value::Text(k), v))
                    .collect(),
            ),
        );

        GlobalMetadata { version: 2, extra }
    }

    fn make_test_descriptor() -> DataObjectDescriptor {
        DataObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 2,
            shape: vec![10, 20],
            strides: vec![20, 1],
            dtype: Dtype::Float32,
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        }
    }

    #[test]
    fn test_global_metadata_round_trip() {
        let meta = make_test_global_metadata();
        let bytes = global_metadata_to_cbor(&meta).unwrap();
        let decoded = cbor_to_global_metadata(&bytes).unwrap();
        assert_eq!(decoded.version, 2);
        assert!(decoded.extra.contains_key("mars"));
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
            object_count: 3,
            offsets: vec![100, 500, 1200],
            lengths: vec![400, 700, 300],
        };
        let bytes = index_to_cbor(&index).unwrap();
        let decoded = cbor_to_index(&bytes).unwrap();
        assert_eq!(decoded.object_count, 3);
        assert_eq!(decoded.offsets, vec![100, 500, 1200]);
        assert_eq!(decoded.lengths, vec![400, 700, 300]);
    }

    #[test]
    fn test_hash_frame_round_trip() {
        let hf = HashFrame {
            object_count: 2,
            hash_type: "xxh3".to_string(),
            hashes: vec![
                "abcdef0123456789".to_string(),
                "1234567890abcdef".to_string(),
            ],
        };
        let bytes = hash_frame_to_cbor(&hf).unwrap();
        let decoded = cbor_to_hash_frame(&bytes).unwrap();
        assert_eq!(decoded.object_count, 2);
        assert_eq!(decoded.hash_type, "xxh3");
        assert_eq!(decoded.hashes, hf.hashes);
    }

    #[test]
    fn test_empty_global_metadata() {
        let meta = GlobalMetadata {
            version: 2,
            extra: BTreeMap::new(),
        };
        let bytes = global_metadata_to_cbor(&meta).unwrap();
        let decoded = cbor_to_global_metadata(&bytes).unwrap();
        assert_eq!(decoded.version, 2);
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
            object_count: 3,
            offsets: vec![100, 500, 1200],
            lengths: vec![400, 700, 300],
        };
        let bytes = index_to_cbor(&index).unwrap();
        verify_canonical_cbor(&bytes).unwrap();
    }

    #[test]
    fn test_hash_frame_cbor_is_canonical() {
        let hf = HashFrame {
            object_count: 2,
            hash_type: "xxh3".to_string(),
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
        let mut extra1 = BTreeMap::new();
        extra1.insert("zebra".to_string(), ciborium::Value::Integer(1.into()));
        extra1.insert("apple".to_string(), ciborium::Value::Integer(2.into()));
        let meta1 = GlobalMetadata {
            version: 2,
            extra: extra1,
        };

        let mut extra2 = BTreeMap::new();
        extra2.insert("apple".to_string(), ciborium::Value::Integer(2.into()));
        extra2.insert("zebra".to_string(), ciborium::Value::Integer(1.into()));
        let meta2 = GlobalMetadata {
            version: 2,
            extra: extra2,
        };

        let bytes1 = global_metadata_to_cbor(&meta1).unwrap();
        let bytes2 = global_metadata_to_cbor(&meta2).unwrap();
        assert_eq!(
            bytes1, bytes2,
            "CBOR output must be independent of insertion order"
        );
    }
}
