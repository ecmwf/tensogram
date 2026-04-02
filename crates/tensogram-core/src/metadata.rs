use crate::error::{Result, TensogramError};
use crate::types::Metadata;

/// Serialize metadata to deterministic CBOR bytes (RFC 8949 Section 4.2).
/// Two-step process: serialize to ciborium::Value, canonicalize key ordering, write bytes.
pub fn metadata_to_cbor(metadata: &Metadata) -> Result<Vec<u8>> {
    // Step 1: Serialize to ciborium::Value via serde
    let mut value: ciborium::Value = ciborium::Value::serialized(metadata)
        .map_err(|e| TensogramError::Metadata(format!("failed to serialize metadata: {e}")))?;

    // Step 2: Canonicalize all map keys (sort by CBOR-encoded bytes)
    canonicalize(&mut value);

    // Step 3: Write to bytes
    let mut buf = Vec::new();
    ciborium::into_writer(&value, &mut buf)
        .map_err(|e| TensogramError::Metadata(format!("failed to write CBOR: {e}")))?;

    Ok(buf)
}

/// Deserialize metadata from CBOR bytes. Unknown keys are preserved in `extra` fields.
pub fn cbor_to_metadata(cbor_bytes: &[u8]) -> Result<Metadata> {
    let value: ciborium::Value = ciborium::from_reader(cbor_bytes)
        .map_err(|e| TensogramError::Metadata(format!("failed to parse CBOR: {e}")))?;

    let metadata: Metadata = value
        .deserialized()
        .map_err(|e| TensogramError::Metadata(format!("failed to deserialize metadata: {e}")))?;

    // Validate objects.len == payload.len
    if metadata.objects.len() != metadata.payload.len() {
        return Err(TensogramError::Metadata(format!(
            "objects.len ({}) != payload.len ({})",
            metadata.objects.len(),
            metadata.payload.len()
        )));
    }

    Ok(metadata)
}

/// Recursively sort all map keys in a ciborium::Value tree for canonical encoding.
/// Keys are sorted by their CBOR-encoded byte representation (lexicographic).
fn canonicalize(value: &mut ciborium::Value) {
    match value {
        ciborium::Value::Map(entries) => {
            // Recursively canonicalize all values (and keys if they're maps)
            for (k, v) in entries.iter_mut() {
                canonicalize(k);
                canonicalize(v);
            }
            // Sort by CBOR-encoded key bytes
            entries.sort_by(|(a, _), (b, _)| {
                let mut a_bytes = Vec::new();
                let mut b_bytes = Vec::new();
                // These shouldn't fail for already-valid Values
                let _ = ciborium::into_writer(a, &mut a_bytes);
                let _ = ciborium::into_writer(b, &mut b_bytes);
                a_bytes.cmp(&b_bytes)
            });
        }
        ciborium::Value::Array(items) => {
            for item in items.iter_mut() {
                canonicalize(item);
            }
        }
        ciborium::Value::Tag(_, inner) => {
            canonicalize(inner);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;
    use crate::types::{ByteOrder, ObjectDescriptor, PayloadDescriptor};
    use std::collections::BTreeMap;

    fn make_test_metadata() -> Metadata {
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

        Metadata {
            version: 1,
            objects: vec![ObjectDescriptor {
                obj_type: "ntensor".to_string(),
                ndim: 2,
                shape: vec![10, 20],
                strides: vec![20, 1],
                dtype: Dtype::Float32,
                extra: BTreeMap::new(),
            }],
            payload: vec![PayloadDescriptor {
                byte_order: ByteOrder::Big,
                encoding: "none".to_string(),
                filter: "none".to_string(),
                compression: "none".to_string(),
                params: BTreeMap::new(),
                hash: None,
            }],
            extra,
        }
    }

    #[test]
    fn test_deterministic_encoding() {
        let metadata = make_test_metadata();
        let bytes1 = metadata_to_cbor(&metadata).unwrap();
        let bytes2 = metadata_to_cbor(&metadata).unwrap();
        assert_eq!(bytes1, bytes2, "CBOR encoding must be deterministic");
    }

    #[test]
    fn test_round_trip() {
        let metadata = make_test_metadata();
        let bytes = metadata_to_cbor(&metadata).unwrap();
        let decoded = cbor_to_metadata(&bytes).unwrap();
        assert_eq!(decoded.version, 1);
        assert_eq!(decoded.objects.len(), 1);
        assert_eq!(decoded.objects[0].shape, vec![10, 20]);
        assert_eq!(decoded.payload[0].encoding, "none");
    }

    #[test]
    fn test_objects_payload_length_mismatch() {
        let mut metadata = make_test_metadata();
        metadata.payload.clear(); // mismatch: 1 object, 0 payload
        let bytes = metadata_to_cbor(&metadata).unwrap();
        assert!(cbor_to_metadata(&bytes).is_err());
    }

    #[test]
    fn test_empty_objects() {
        let metadata = Metadata {
            version: 1,
            objects: vec![],
            payload: vec![],
            extra: BTreeMap::new(),
        };
        let bytes = metadata_to_cbor(&metadata).unwrap();
        let decoded = cbor_to_metadata(&bytes).unwrap();
        assert_eq!(decoded.objects.len(), 0);
        assert_eq!(decoded.payload.len(), 0);
    }

    #[test]
    fn test_unknown_keys_preserved() {
        let mut metadata = make_test_metadata();
        metadata.extra.insert(
            "custom_namespace".to_string(),
            ciborium::Value::Text("test_value".to_string()),
        );
        let bytes = metadata_to_cbor(&metadata).unwrap();
        let decoded = cbor_to_metadata(&bytes).unwrap();
        assert!(decoded.extra.contains_key("custom_namespace"));
    }
}
