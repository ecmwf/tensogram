use tensogram_core::{DataObjectDescriptor, GlobalMetadata};

use crate::filter::lookup_key;

/// Format metadata values as a table row.
pub fn format_table_row<K: AsRef<str>>(metadata: &GlobalMetadata, keys: &[K]) -> String {
    keys.iter()
        .map(|key| lookup_key(metadata, key.as_ref()).unwrap_or_else(|| "N/A".to_string()))
        .collect::<Vec<_>>()
        .join("\t")
}

/// Format global metadata as JSON. If keys is Some, only include those keys.
///
/// `objects` optionally includes per-object descriptor info in the output.
pub fn format_json<K: AsRef<str>>(
    metadata: &GlobalMetadata,
    keys: Option<&[K]>,
    objects: Option<&[(DataObjectDescriptor, Vec<u8>)]>,
) -> String {
    let mut map = serde_json::Map::new();

    map.insert(
        "version".to_string(),
        serde_json::Value::Number(serde_json::Number::from(metadata.version)),
    );

    // Add base array (per-object metadata)
    if !metadata.base.is_empty() {
        let base_json: Vec<serde_json::Value> = metadata
            .base
            .iter()
            .map(|entry| {
                let m: serde_json::Map<String, serde_json::Value> = entry
                    .iter()
                    .map(|(k, v)| (k.to_string(), cbor_to_json(v)))
                    .collect();
                serde_json::Value::Object(m)
            })
            .collect();
        map.insert("base".to_string(), serde_json::Value::Array(base_json));
    }

    // Add extra (namespaced) keys
    if !metadata.extra.is_empty() {
        let extra_map: serde_json::Map<String, serde_json::Value> = metadata
            .extra
            .iter()
            .map(|(k, v)| (k.to_string(), cbor_to_json(v)))
            .collect();
        map.insert("extra".to_string(), serde_json::Value::Object(extra_map));
    }

    // Add objects summary if provided
    if let Some(objs) = objects {
        let objects_json: Vec<serde_json::Value> = objs
            .iter()
            .map(|(desc, _)| {
                let mut m = serde_json::Map::new();
                m.insert(
                    "type".to_string(),
                    serde_json::Value::String(desc.obj_type.clone()),
                );
                m.insert(
                    "dtype".to_string(),
                    serde_json::Value::String(desc.dtype.to_string()),
                );
                m.insert(
                    "shape".to_string(),
                    serde_json::Value::Array(
                        desc.shape
                            .iter()
                            .map(|&s| serde_json::Value::Number(s.into()))
                            .collect(),
                    ),
                );
                for (k, v) in &desc.params {
                    m.insert(k.to_string(), cbor_to_json(v));
                }
                serde_json::Value::Object(m)
            })
            .collect();
        map.insert(
            "objects".to_string(),
            serde_json::Value::Array(objects_json),
        );
    }

    if let Some(keys) = keys {
        // Filter to only requested keys
        let mut filtered = serde_json::Map::new();
        for key in keys {
            let key = key.as_ref();
            if let Some(value) = lookup_key(metadata, key) {
                filtered.insert(key.to_string(), serde_json::Value::String(value));
            }
        }
        serde_json::to_string_pretty(&serde_json::Value::Object(filtered))
            .unwrap_or_else(|_| "{}".to_string())
    } else {
        serde_json::to_string_pretty(&serde_json::Value::Object(map))
            .unwrap_or_else(|_| "{}".to_string())
    }
}

fn cbor_to_json(value: &ciborium::Value) -> serde_json::Value {
    match value {
        ciborium::Value::Text(s) => serde_json::Value::String(s.to_string()),
        ciborium::Value::Integer(i) => {
            let n: i128 = (*i).into();
            // Try i64 first (common case), fall back to string for very large integers
            if let Ok(v) = i64::try_from(n) {
                serde_json::Value::Number(serde_json::Number::from(v))
            } else if let Ok(v) = u64::try_from(n) {
                serde_json::Value::Number(serde_json::Number::from(v))
            } else {
                serde_json::Value::String(n.to_string())
            }
        }
        ciborium::Value::Float(f) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        ciborium::Value::Bool(b) => serde_json::Value::Bool(*b),
        ciborium::Value::Null => serde_json::Value::Null,
        ciborium::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(cbor_to_json).collect())
        }
        ciborium::Value::Map(entries) => {
            let mut map = serde_json::Map::new();
            for (k, v) in entries {
                let key = match k {
                    ciborium::Value::Text(s) => s.to_string(),
                    _ => format!("{k:?}"),
                };
                map.insert(key, cbor_to_json(v));
            }
            serde_json::Value::Object(map)
        }
        ciborium::Value::Bytes(b) => serde_json::Value::String(hex::encode(b)),
        ciborium::Value::Tag(_, inner) => cbor_to_json(inner),
        _ => serde_json::Value::Null,
    }
}

/// Format a single ciborium::Value as a display string.
pub fn format_json_value(value: &ciborium::Value) -> String {
    serde_json::to_string(&cbor_to_json(value)).unwrap_or_else(|_| format!("{value:?}"))
}

// Simple hex encoding since we may not want another dependency
mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn make_meta(extra: BTreeMap<String, ciborium::Value>) -> GlobalMetadata {
        GlobalMetadata {
            version: 2,
            extra,
            ..Default::default()
        }
    }

    #[test]
    fn table_row_basic() {
        let mut extra = BTreeMap::new();
        extra.insert("param".to_string(), ciborium::Value::Text("2t".to_string()));
        let meta = make_meta(extra);
        let row = format_table_row(&meta, &["version", "param"]);
        assert!(row.contains("2"));
        assert!(row.contains("2t"));
    }

    #[test]
    fn table_row_missing_key() {
        let meta = make_meta(BTreeMap::new());
        let row = format_table_row(&meta, &["missing"]);
        assert!(row.contains("N/A"));
    }

    #[test]
    fn json_full() {
        let mut extra = BTreeMap::new();
        extra.insert("key".to_string(), ciborium::Value::Text("val".to_string()));
        let meta = make_meta(extra);
        let json = format_json::<String>(&meta, None, None);
        assert!(json.contains("\"version\": 2"));
        assert!(json.contains("\"key\": \"val\""));
    }

    #[test]
    fn json_filtered_keys() {
        let mut extra = BTreeMap::new();
        extra.insert("a".to_string(), ciborium::Value::Text("1".to_string()));
        extra.insert("b".to_string(), ciborium::Value::Text("2".to_string()));
        let meta = make_meta(extra);
        let json = format_json(&meta, Some(&["a"]), None);
        assert!(json.contains("\"a\""));
        assert!(!json.contains("\"b\""));
    }

    #[test]
    fn json_with_objects() {
        let meta = make_meta(BTreeMap::new());
        let desc = DataObjectDescriptor {
            obj_type: "ntensor".into(),
            ndim: 1,
            shape: vec![4],
            strides: vec![1],
            dtype: tensogram_core::Dtype::Float32,
            byte_order: tensogram_core::ByteOrder::Big,
            encoding: "none".into(),
            filter: "none".into(),
            compression: "none".into(),
            params: Default::default(),
            hash: None,
        };
        let objects = vec![(desc, vec![0u8; 16])];
        let json = format_json::<String>(&meta, None, Some(&objects));
        assert!(json.contains("\"objects\""));
        assert!(json.contains("ntensor"));
        assert!(json.contains("float32"));
    }

    #[test]
    fn format_json_value_text() {
        let val = ciborium::Value::Text("hello".to_string());
        assert_eq!(format_json_value(&val), "\"hello\"");
    }

    #[test]
    fn format_json_value_integer() {
        let val = ciborium::Value::Integer(42.into());
        assert_eq!(format_json_value(&val), "42");
    }

    #[test]
    fn format_json_value_float() {
        let val = ciborium::Value::Float(3.5);
        assert_eq!(format_json_value(&val), "3.5");
    }

    #[test]
    fn format_json_value_bool() {
        assert_eq!(format_json_value(&ciborium::Value::Bool(true)), "true");
    }

    #[test]
    fn format_json_value_null() {
        assert_eq!(format_json_value(&ciborium::Value::Null), "null");
    }

    #[test]
    fn format_json_value_array() {
        let val = ciborium::Value::Array(vec![
            ciborium::Value::Integer(1.into()),
            ciborium::Value::Integer(2.into()),
        ]);
        assert_eq!(format_json_value(&val), "[1,2]");
    }

    #[test]
    fn format_json_value_map() {
        let val = ciborium::Value::Map(vec![(
            ciborium::Value::Text("k".to_string()),
            ciborium::Value::Text("v".to_string()),
        )]);
        let s = format_json_value(&val);
        assert!(s.contains("\"k\""));
        assert!(s.contains("\"v\""));
    }

    #[test]
    fn format_json_value_bytes() {
        let val = ciborium::Value::Bytes(vec![0xde, 0xad]);
        assert_eq!(format_json_value(&val), "\"dead\"");
    }

    #[test]
    fn format_json_value_tag() {
        let val = ciborium::Value::Tag(1, Box::new(ciborium::Value::Text("inner".to_string())));
        assert_eq!(format_json_value(&val), "\"inner\"");
    }

    #[test]
    fn hex_encode() {
        assert_eq!(hex::encode(&[0x00, 0xff, 0x42]), "00ff42");
        assert_eq!(hex::encode(&[]), "");
    }

    #[test]
    fn format_json_value_nan() {
        // NaN is not representable in JSON, should produce null
        let val = ciborium::Value::Float(f64::NAN);
        assert_eq!(format_json_value(&val), "null");
    }

    #[test]
    fn cbor_map_non_text_key() {
        // Non-text keys get Debug-formatted
        let val = ciborium::Value::Map(vec![(
            ciborium::Value::Integer(42.into()),
            ciborium::Value::Text("val".to_string()),
        )]);
        let s = format_json_value(&val);
        assert!(s.contains("val"));
    }

    // ── Additional coverage: base array in JSON output ──

    #[test]
    fn json_with_base_entries() {
        let mut entry = BTreeMap::new();
        entry.insert(
            "mars".to_string(),
            ciborium::Value::Map(vec![(
                ciborium::Value::Text("param".to_string()),
                ciborium::Value::Text("2t".to_string()),
            )]),
        );
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entry],
            ..Default::default()
        };
        let json = format_json::<String>(&meta, None, None);
        assert!(json.contains("\"base\""), "JSON should contain base array");
        assert!(json.contains("\"mars\""), "JSON should contain mars key");
        assert!(json.contains("\"param\""), "JSON should contain param key");
        assert!(json.contains("\"2t\""), "JSON should contain value 2t");
    }

    #[test]
    fn json_with_base_and_extra() {
        let mut entry = BTreeMap::new();
        entry.insert(
            "source".to_string(),
            ciborium::Value::Text("ecmwf".to_string()),
        );
        let mut extra = BTreeMap::new();
        extra.insert(
            "experiment".to_string(),
            ciborium::Value::Text("exp01".to_string()),
        );
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entry],
            extra,
            ..Default::default()
        };
        let json = format_json::<String>(&meta, None, None);
        assert!(json.contains("\"base\""));
        assert!(json.contains("\"extra\""));
        assert!(json.contains("\"ecmwf\""));
        assert!(json.contains("\"exp01\""));
    }

    #[test]
    fn json_empty_base_and_extra() {
        let meta = GlobalMetadata {
            version: 2,
            ..Default::default()
        };
        let json = format_json::<String>(&meta, None, None);
        // Should not contain base or extra sections when empty
        assert!(!json.contains("\"base\""));
        assert!(!json.contains("\"extra\""));
        assert!(json.contains("\"version\": 2"));
    }

    #[test]
    fn table_row_with_base_key() {
        let mut entry = BTreeMap::new();
        entry.insert("param".to_string(), ciborium::Value::Text("2t".to_string()));
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entry],
            ..Default::default()
        };
        let row = format_table_row(&meta, &["param", "version"]);
        assert!(row.contains("2t"));
        assert!(row.contains("2"));
    }

    #[test]
    fn cbor_to_json_large_integer() {
        // u64 max value should fall back through i64 → u64 path
        let val = ciborium::Value::Integer(ciborium::value::Integer::from(u64::MAX));
        let json = cbor_to_json(&val);
        // Should produce a number (u64 path) or string, not null
        assert!(!json.is_null());
    }
}
