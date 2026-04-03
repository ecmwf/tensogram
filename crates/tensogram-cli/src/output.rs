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

    // Add extra (namespaced) keys
    for (key, value) in &metadata.extra {
        map.insert(key.to_string(), cbor_to_json(value));
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
            serde_json::Value::Number(serde_json::Number::from(n as i64))
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
