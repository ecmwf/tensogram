use std::path::PathBuf;

use tensogram_core::{decode, decode_metadata, DecodeOptions, EncodeOptions, TensogramFile};

use crate::filter;

/// Keys that cannot be modified via `set` (structural/integrity keys).
const IMMUTABLE_KEYS: &[&str] = &[
    "shape", "strides", "dtype", "ndim", "type",
    "encoding", "filter", "compression",
    "hash", "szip_rsi", "szip_block_size", "szip_flags", "szip_block_offsets",
    "reference_value", "binary_scale_factor", "decimal_scale_factor", "bits_per_value",
    "shuffle_element_size",
];

pub fn run(
    input: &PathBuf,
    output: &PathBuf,
    set_values: &str,
    where_clause: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let clause = where_clause.map(filter::parse_where).transpose()
        .map_err(|e| format!("invalid where clause: {e}"))?;

    // Parse set values: "key=val,key2=val2"
    let mutations: Vec<(String, String)> = set_values
        .split(',')
        .map(|kv| {
            let (k, v) = kv.split_once('=')
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
    let mut out_file = TensogramFile::create(output)?;
    let messages = in_file.messages()?;

    for msg in &messages {
        let metadata = decode_metadata(msg)?;
        let should_modify = clause
            .as_ref()
            .is_none_or(|c| filter::matches(&metadata, c));

        if should_modify {
            // Decode full message, modify metadata, re-encode
            let (mut meta, objects) = decode(msg, &DecodeOptions::default())?;
            for (key, value) in &mutations {
                apply_mutation(&mut meta, key, value);
            }
            let obj_refs: Vec<&[u8]> = objects.iter().map(|o| o.as_slice()).collect();
            // Re-encode with no hash (metadata changed, payload untouched)
            let options = EncodeOptions {
                hash_algorithm: None,
            };
            out_file.append(&meta, &obj_refs, &options)?;
        } else {
            // Pass through unchanged — write raw bytes
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(output)?;
            let mut f = std::fs::OpenOptions::new().append(true).open(output)?;
            use std::io::Write;
            f.write_all(msg)?;
        }
    }

    Ok(())
}

fn apply_mutation(metadata: &mut tensogram_core::Metadata, key: &str, value: &str) {
    let parts: Vec<&str> = key.split('.').collect();

    if parts.len() == 1 {
        // Top-level key
        metadata.extra.insert(
            key.to_string(),
            ciborium::Value::Text(value.to_string()),
        );
    } else if parts.len() == 2 {
        // Namespaced key like "mars.class"
        let ns = parts[0];
        let field = parts[1];

        let ns_val = metadata.extra.entry(ns.to_string()).or_insert_with(|| {
            ciborium::Value::Map(Vec::new())
        });

        if let ciborium::Value::Map(entries) = ns_val {
            // Update existing or insert new
            let mut found = false;
            for (k, v) in entries.iter_mut() {
                if let ciborium::Value::Text(k_str) = k {
                    if k_str == field {
                        *v = ciborium::Value::Text(value.to_string());
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                entries.push((
                    ciborium::Value::Text(field.to_string()),
                    ciborium::Value::Text(value.to_string()),
                ));
            }
        }
    }
}
