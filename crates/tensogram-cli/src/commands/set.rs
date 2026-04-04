use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

use tensogram_core::{
    decode, decode_metadata, encode, DataObjectDescriptor, DecodeOptions, EncodeOptions,
    GlobalMetadata, TensogramFile,
};

use crate::filter;

/// Keys that cannot be modified via `set` (structural/integrity keys).
const IMMUTABLE_KEYS: &[&str] = &[
    "shape",
    "strides",
    "dtype",
    "ndim",
    "type",
    "encoding",
    "filter",
    "compression",
    "hash",
    "szip_rsi",
    "szip_block_size",
    "szip_flags",
    "szip_block_offsets",
    "reference_value",
    "binary_scale_factor",
    "decimal_scale_factor",
    "bits_per_value",
    "shuffle_element_size",
];

/// Update selected metadata keys in matching messages.
pub fn run(
    input: &PathBuf,
    output: &PathBuf,
    set_values: &str,
    where_clause: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let clause = where_clause
        .map(filter::parse_where)
        .transpose()
        .map_err(|e| format!("invalid where clause: {e}"))?;

    // Parse set values: "key=val,key2=val2"
    let mutations: Vec<(String, String)> = set_values
        .split(',')
        .map(|kv| {
            let (k, v) = kv
                .split_once('=')
                .ok_or_else(|| format!("invalid set value: {kv}"))?;
            Ok((k.trim().to_string(), v.trim().to_string()))
        })
        .collect::<Result<Vec<_>, String>>()?;

    // Validate no immutable keys
    for (key, _) in &mutations {
        let leaf = key.split('.').next_back().unwrap_or(key);
        if IMMUTABLE_KEYS.contains(&leaf) {
            return Err(format!("cannot modify immutable key: {key}").into());
        }
    }

    let mut in_file = TensogramFile::open(input)?;
    let mut out = std::fs::File::create(output)?;
    #[allow(deprecated)]
    let messages = in_file.messages()?;

    for msg in &messages {
        let metadata = decode_metadata(msg)?;
        let should_modify = clause
            .as_ref()
            .is_none_or(|c| filter::matches(&metadata, c));

        if should_modify {
            let (mut global_meta, mut objects) = decode(msg, &DecodeOptions::default())?;

            // Preserve original hashes per object before any re-encoding
            let original_hashes: Vec<_> =
                objects.iter().map(|(desc, _)| desc.hash.clone()).collect();

            for (key, value) in &mutations {
                apply_mutation(&mut global_meta, &mut objects, key, value)?;
            }

            // Restore hashes so re-encoding without hash computation preserves integrity
            for ((desc, _), original_hash) in objects.iter_mut().zip(original_hashes) {
                desc.hash = original_hash;
            }

            let descriptor_refs: Vec<(&DataObjectDescriptor, &[u8])> = objects
                .iter()
                .map(|(desc, data)| (desc, data.as_slice()))
                .collect();

            // Re-encode with no hash computation, preserving the original hash field.
            let options = EncodeOptions {
                hash_algorithm: None,
            };
            let encoded = encode(&global_meta, &descriptor_refs, &options)?;
            out.write_all(&encoded)?;
        } else {
            // Pass through unchanged — reuse the already-open output handle.
            out.write_all(msg)?;
        }
    }

    Ok(())
}

/// Apply a key=value mutation to either global metadata or a per-object descriptor.
///
/// Key paths:
/// - `"key"` or `"ns.key"` → mutate global metadata extra map
/// - `"objects.N.key"` or `"objects.N.ns.key"` → mutate object descriptor params
fn apply_mutation(
    global_meta: &mut GlobalMetadata,
    objects: &mut [(DataObjectDescriptor, Vec<u8>)],
    key: &str,
    value: &str,
) -> Result<(), String> {
    let parts: Vec<&str> = key.split('.').collect();

    if parts[0] == "objects" {
        if parts.len() < 3 {
            return Err(format!("invalid object key: {key}"));
        }
        let index = parts[1]
            .parse::<usize>()
            .map_err(|_| format!("invalid object index in key: {key}"))?;
        let (desc, _) = objects
            .get_mut(index)
            .ok_or_else(|| format!("object index out of range in key: {key}"))?;
        insert_nested_value(&mut desc.params, &parts[2..], value)
    } else {
        insert_nested_value(&mut global_meta.extra, &parts, value)
    }
}

fn insert_nested_value(
    map: &mut BTreeMap<String, ciborium::Value>,
    parts: &[&str],
    value: &str,
) -> Result<(), String> {
    if parts.is_empty() {
        return Err("empty mutation path".to_string());
    }

    if parts.len() == 1 {
        map.insert(
            parts[0].to_string(),
            ciborium::Value::Text(value.to_string()),
        );
        return Ok(());
    }

    let entry = map
        .entry(parts[0].to_string())
        .or_insert_with(|| ciborium::Value::Map(Vec::new()));
    insert_nested_cbor_value(entry, &parts[1..], value)
}

fn insert_nested_cbor_value(
    node: &mut ciborium::Value,
    parts: &[&str],
    value: &str,
) -> Result<(), String> {
    if parts.is_empty() {
        return Err("empty nested mutation path".to_string());
    }

    match node {
        ciborium::Value::Map(entries) => {
            let key = ciborium::Value::Text(parts[0].to_string());
            if parts.len() == 1 {
                if let Some((_, existing)) = entries.iter_mut().find(|(k, _)| *k == key) {
                    *existing = ciborium::Value::Text(value.to_string());
                } else {
                    entries.push((key, ciborium::Value::Text(value.to_string())));
                }
                Ok(())
            } else if let Some((_, child)) = entries.iter_mut().find(|(k, _)| *k == key) {
                insert_nested_cbor_value(child, &parts[1..], value)
            } else {
                let mut child = ciborium::Value::Map(Vec::new());
                insert_nested_cbor_value(&mut child, &parts[1..], value)?;
                entries.push((key, child));
                Ok(())
            }
        }
        _ => {
            *node = ciborium::Value::Map(Vec::new());
            insert_nested_cbor_value(node, parts, value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensogram_core::{encode, ByteOrder, DataObjectDescriptor, Dtype, GlobalMetadata};

    fn make_global_meta() -> GlobalMetadata {
        GlobalMetadata {
            version: 2,
            extra: BTreeMap::new(),
            ..Default::default()
        }
    }

    fn make_descriptor() -> DataObjectDescriptor {
        DataObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 1,
            shape: vec![4],
            strides: vec![1],
            dtype: Dtype::Float32,
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        }
    }

    fn unique_path(name: &str) -> std::path::PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "tensogram-cli-{name}-{stamp}-{}",
            std::process::id()
        ))
    }

    #[test]
    fn test_set_preserves_hash() {
        let input = unique_path("input.tgm");
        let output = unique_path("output.tgm");
        let data = vec![7u8; 16];

        let mut global_meta = make_global_meta();
        global_meta.extra.insert(
            "source".to_string(),
            ciborium::Value::Text("old".to_string()),
        );

        let desc = make_descriptor();
        let encoded = encode(&global_meta, &[(&desc, &data)], &EncodeOptions::default()).unwrap();
        std::fs::write(&input, encoded).unwrap();

        let original = decode_metadata(&std::fs::read(&input).unwrap()).unwrap();
        // Decode full message to get per-object hash
        let (_, original_objects) =
            decode(&std::fs::read(&input).unwrap(), &DecodeOptions::default()).unwrap();
        let original_hash = original_objects[0].0.hash.as_ref().unwrap();

        // Verify version is accessible from global metadata
        assert_eq!(original.version, 2);

        run(&input, &output, "source=new", None).unwrap();

        let updated = decode_metadata(&std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(
            updated.extra.get("source"),
            Some(&ciborium::Value::Text("new".to_string()))
        );

        let (_, updated_objects) =
            decode(&std::fs::read(&output).unwrap(), &DecodeOptions::default()).unwrap();
        let updated_hash = updated_objects[0].0.hash.as_ref().unwrap();
        assert_eq!(updated_hash.hash_type, original_hash.hash_type);
        assert_eq!(updated_hash.value, original_hash.value);

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }
}
