use std::path::PathBuf;

use tensogram_core::{decode, decode_metadata, DecodeOptions, TensogramFile};

use crate::filter;
use crate::output;

/// Dump full message metadata and object summaries.
pub fn run(
    files: &[PathBuf],
    where_clause: Option<&str>,
    keys: Option<&str>,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let clause = where_clause
        .map(filter::parse_where)
        .transpose()
        .map_err(|e| format!("invalid where clause: {e}"))?;

    let key_list: Option<Vec<String>> =
        keys.map(|k| k.split(',').map(|s| s.trim().to_string()).collect());

    for path in files {
        let mut file = TensogramFile::open(path)?;
        #[allow(deprecated)]
        let messages = file.messages()?;

        for (i, msg) in messages.iter().enumerate() {
            // Decode metadata first for cheap filtering
            let metadata = decode_metadata(msg)?;

            if let Some(ref clause) = clause {
                if !filter::matches(&metadata, clause) {
                    continue;
                }
            }

            // Decode full message to access per-object descriptors
            let (global_meta, objects) = decode(msg, &DecodeOptions::default())?;

            if json {
                println!(
                    "{}",
                    output::format_json(&global_meta, key_list.as_deref(), Some(&objects))
                );
            } else {
                println!("=== Message {i} ===");
                println!("version: {}", global_meta.version);
                println!("objects: {}", objects.len());
                for (j, (desc, _)) in objects.iter().enumerate() {
                    println!(
                        "  object[{j}]: type={}, dtype={}, shape={:?}",
                        desc.obj_type, desc.dtype, desc.shape
                    );
                }
                for (key, value) in &global_meta.extra {
                    println!("  {key}: {}", output::format_json_value(value));
                }
                println!();
            }
        }
    }

    Ok(())
}
