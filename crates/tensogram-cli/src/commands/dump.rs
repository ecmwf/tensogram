use std::path::PathBuf;

use tensogram_core::{decode_metadata, TensogramFile};

use crate::filter;
use crate::output;

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
        let messages = file.messages()?;

        for (i, msg) in messages.iter().enumerate() {
            let metadata = decode_metadata(msg)?;

            if let Some(ref clause) = clause {
                if !filter::matches(&metadata, clause) {
                    continue;
                }
            }

            if json {
                println!("{}", output::format_json(&metadata, key_list.as_deref()));
            } else {
                println!("=== Message {i} ===");
                println!("version: {}", metadata.version);
                println!("objects: {}", metadata.objects.len());
                for (j, obj) in metadata.objects.iter().enumerate() {
                    println!(
                        "  object[{j}]: type={}, dtype={}, shape={:?}",
                        obj.obj_type, obj.dtype, obj.shape
                    );
                }
                for (key, value) in &metadata.extra {
                    println!("  {key}: {}", output::format_json_value(value));
                }
                println!();
            }
        }
    }

    Ok(())
}
