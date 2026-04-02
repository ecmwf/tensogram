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

    let key_list: Vec<String> = keys
        .map(|k| k.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_else(|| vec!["version".to_string()]);

    // Print header for table mode
    if !json {
        println!("{}", key_list.join("\t"));
    }

    for path in files {
        let mut file = TensogramFile::open(path)?;
        let messages = file.messages()?;

        for msg in &messages {
            let metadata = decode_metadata(msg)?;

            if let Some(ref clause) = clause {
                if !filter::matches(&metadata, clause) {
                    continue;
                }
            }

            if json {
                println!("{}", output::format_json(&metadata, Some(&key_list)));
            } else {
                println!("{}", output::format_table_row(&metadata, &key_list));
            }
        }
    }

    Ok(())
}
