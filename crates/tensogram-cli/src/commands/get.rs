use std::path::PathBuf;

use tensogram_core::{decode_metadata, TensogramFile};

use crate::filter::{self, lookup_key};

pub fn run(
    files: &[PathBuf],
    where_clause: Option<&str>,
    keys: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let clause = where_clause
        .map(filter::parse_where)
        .transpose()
        .map_err(|e| format!("invalid where clause: {e}"))?;

    let key_list: Vec<String> = keys.split(',').map(|s| s.trim().to_string()).collect();

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

            // Strict: error on missing key
            let mut values = Vec::new();
            for key in &key_list {
                match lookup_key(&metadata, key) {
                    Some(val) => values.push(val),
                    None => {
                        return Err(format!("key not found: {key}").into());
                    }
                }
            }
            println!("{}", values.join(" "));
        }
    }

    Ok(())
}
