// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::path::PathBuf;

use tensogram_core::{decode_metadata, TensogramFile};

use crate::filter;
use crate::output;

/// List metadata rows for matching messages.
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
        let file = TensogramFile::open(path)?;
        let count = file.message_count()?;

        for i in 0..count {
            let msg = file.read_message(i)?;
            let metadata = decode_metadata(&msg)?;

            if let Some(ref clause) = clause {
                if !filter::matches(&metadata, clause) {
                    continue;
                }
            }

            if json {
                println!("{}", output::format_json(&metadata, Some(&key_list), None));
            } else {
                println!("{}", output::format_table_row(&metadata, &key_list));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use tensogram_core::{ByteOrder, DataObjectDescriptor, Dtype, EncodeOptions, GlobalMetadata};

    fn make_test_file(dir: &std::path::Path) -> PathBuf {
        let path = dir.join("ls_test.tgm");
        let mut f = tensogram_core::TensogramFile::create(&path).unwrap();
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
            hash: None,
        };
        let data = vec![0u8; 16];
        let mut extra = BTreeMap::new();
        extra.insert("param".to_string(), ciborium::Value::Text("2t".to_string()));
        let meta = GlobalMetadata {
            version: 2,
            extra,
            ..Default::default()
        };
        f.append(&meta, &[(&desc, &data)], &EncodeOptions::default())
            .unwrap();
        // Second message with different param
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
    fn ls_default_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path());
        run(&[path], None, None, false).unwrap();
    }

    #[test]
    fn ls_custom_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path());
        run(&[path], None, Some("version,param"), false).unwrap();
    }

    #[test]
    fn ls_json_output() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path());
        run(&[path], None, Some("param"), true).unwrap();
    }

    #[test]
    fn ls_with_where_filter() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path());
        run(&[path], Some("param=2t"), Some("version,param"), false).unwrap();
    }

    #[test]
    fn ls_where_no_match() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path());
        run(&[path], Some("param=nonexistent"), None, false).unwrap();
    }

    #[test]
    fn ls_invalid_where() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path());
        assert!(run(&[path], Some("bad-clause"), None, false).is_err());
    }
}
