use std::io::Write;
use std::path::PathBuf;

use tensogram_core::{decode_metadata, TensogramFile};

use crate::filter::{self, lookup_key};

pub fn run(
    input: &PathBuf,
    output: &str,
    where_clause: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let clause = where_clause
        .map(filter::parse_where)
        .transpose()
        .map_err(|e| format!("invalid where clause: {e}"))?;

    let has_placeholders = output.contains('[') && output.contains(']');

    let mut in_file = TensogramFile::open(input)?;
    #[allow(deprecated)]
    let messages = in_file.messages()?;

    if !has_placeholders {
        // Simple copy: all matching messages to one output file
        let out_path = PathBuf::from(output);
        let mut out = std::fs::File::create(&out_path)?;

        for msg in &messages {
            let metadata = decode_metadata(msg)?;
            if let Some(ref clause) = clause {
                if !filter::matches(&metadata, clause) {
                    continue;
                }
            }
            out.write_all(msg)?;
        }
    } else {
        // Splitting: expand [key] placeholders per message
        for msg in &messages {
            let metadata = decode_metadata(msg)?;
            if let Some(ref clause) = clause {
                if !filter::matches(&metadata, clause) {
                    continue;
                }
            }

            let out_name = expand_placeholders(output, &metadata);
            let out_path = PathBuf::from(&out_name);

            // Append to output file (may already exist from prior messages)
            let mut out = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&out_path)?;
            out.write_all(msg)?;
        }
    }

    Ok(())
}

/// Expand `[keyName]` placeholders in a filename template using global metadata values.
///
/// For multi-object messages, each placeholder resolves to global metadata keys only.
/// That keeps split filenames stable.
fn expand_placeholders(template: &str, metadata: &tensogram_core::GlobalMetadata) -> String {
    let mut result = template.to_string();
    // Find all [key] patterns
    while let Some(start) = result.find('[') {
        if let Some(end) = result[start..].find(']') {
            let key = &result[start + 1..start + end];
            let value = lookup_key(metadata, key).unwrap_or_else(|| "unknown".to_string());
            result = format!(
                "{}{}{}",
                &result[..start],
                value,
                &result[start + end + 1..]
            );
        } else {
            break;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use tensogram_core::GlobalMetadata;

    fn metadata_without_param() -> GlobalMetadata {
        GlobalMetadata {
            version: 2,
            extra: BTreeMap::new(),
            ..Default::default()
        }
    }

    #[test]
    fn expand_placeholders_uses_unknown_for_missing_keys() {
        let metadata = metadata_without_param();
        let expanded = expand_placeholders("by_param/[mars.param].tgm", &metadata);
        assert_eq!(expanded, "by_param/unknown.tgm");
    }
}
