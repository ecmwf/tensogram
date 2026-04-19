// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::io::Write;
use std::path::PathBuf;

use tensogram::{TensogramFile, decode_metadata};

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

    let in_file = TensogramFile::open(input)?;
    let count = in_file.message_count()?;

    if !has_placeholders {
        // Simple copy: all matching messages to one output file
        let out_path = PathBuf::from(output);
        let mut out = std::fs::File::create(&out_path)?;

        for i in 0..count {
            let msg = in_file.read_message(i)?;
            let metadata = decode_metadata(&msg)?;
            if let Some(ref clause) = clause
                && !filter::matches(&metadata, clause)
            {
                continue;
            }
            out.write_all(&msg)?;
        }
    } else {
        // Splitting: expand [key] placeholders per message
        for i in 0..count {
            let msg = in_file.read_message(i)?;
            let metadata = decode_metadata(&msg)?;
            if let Some(ref clause) = clause
                && !filter::matches(&metadata, clause)
            {
                continue;
            }

            let out_name = expand_placeholders(output, &metadata);
            let out_path = PathBuf::from(&out_name);

            // Append to output file (may already exist from prior messages)
            let mut out = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&out_path)?;
            out.write_all(&msg)?;
        }
    }

    Ok(())
}

/// Expand `[keyName]` placeholders in a filename template using global metadata values.
///
/// For multi-object messages, each placeholder resolves to global metadata keys only.
/// That keeps split filenames stable.
fn expand_placeholders(template: &str, metadata: &tensogram::GlobalMetadata) -> String {
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
    use tensogram::GlobalMetadata;

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

    // ── Integration tests ──

    use tensogram::{ByteOrder, DataObjectDescriptor, Dtype, EncodeOptions};

    fn make_test_file(dir: &std::path::Path) -> PathBuf {
        let path = dir.join("copy_input.tgm");
        let mut f = tensogram::TensogramFile::create(&path).unwrap();
        let desc = DataObjectDescriptor {
            obj_type: "ntensor".into(),
            ndim: 1,
            shape: vec![4],
            strides: vec![1],
            dtype: Dtype::Float32,
            byte_order: ByteOrder::Big,
            encoding: "none".into(),
            filter: "none".into(),
            compression: "none".into(),
            params: Default::default(),
            masks: None,
            hash: None,
        };
        let data = vec![0u8; 16];
        let mut extra1 = BTreeMap::new();
        extra1.insert("param".to_string(), ciborium::Value::Text("2t".to_string()));
        let meta1 = GlobalMetadata {
            version: 2,
            extra: extra1,
            ..Default::default()
        };
        f.append(&meta1, &[(&desc, &data)], &EncodeOptions::default())
            .unwrap();
        let mut extra2 = BTreeMap::new();
        extra2.insert(
            "param".to_string(),
            ciborium::Value::Text("msl".to_string()),
        );
        let meta2 = GlobalMetadata {
            version: 2,
            extra: extra2,
            ..Default::default()
        };
        f.append(&meta2, &[(&desc, &data)], &EncodeOptions::default())
            .unwrap();
        path
    }

    #[test]
    fn copy_all_messages() {
        let dir = tempfile::tempdir().unwrap();
        let input = make_test_file(dir.path());
        let out = dir.path().join("copy_out.tgm");
        run(&input, out.to_str().unwrap(), None).unwrap();
        let f = tensogram::TensogramFile::open(&out).unwrap();
        assert_eq!(f.message_count().unwrap(), 2);
    }

    #[test]
    fn copy_with_where_filter() {
        let dir = tempfile::tempdir().unwrap();
        let input = make_test_file(dir.path());
        let out = dir.path().join("filtered.tgm");
        run(&input, out.to_str().unwrap(), Some("param=2t")).unwrap();
        let f = tensogram::TensogramFile::open(&out).unwrap();
        assert_eq!(f.message_count().unwrap(), 1);
    }

    #[test]
    fn copy_with_placeholder_split() {
        let dir = tempfile::tempdir().unwrap();
        let input = make_test_file(dir.path());
        let template = format!("{}/[param].tgm", dir.path().display());
        run(&input, &template, None).unwrap();
        // Should create 2t.tgm and msl.tgm
        assert!(dir.path().join("2t.tgm").exists());
        assert!(dir.path().join("msl.tgm").exists());
    }

    #[test]
    fn copy_where_no_match() {
        let dir = tempfile::tempdir().unwrap();
        let input = make_test_file(dir.path());
        let out = dir.path().join("empty.tgm");
        run(&input, out.to_str().unwrap(), Some("param=nonexistent")).unwrap();
        // Output file may be empty or not created
    }
}
