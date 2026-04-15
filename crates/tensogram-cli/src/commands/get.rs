// (C) Copyright 2024- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::path::PathBuf;

use tensogram_core::{decode_metadata, TensogramFile};

use crate::filter::{self, lookup_key};

/// Print selected metadata values for matching messages.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use tensogram_core::{ByteOrder, DataObjectDescriptor, Dtype, EncodeOptions, GlobalMetadata};

    fn make_test_file(dir: &std::path::Path) -> PathBuf {
        let path = dir.join("get_test.tgm");
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
        path
    }

    #[test]
    fn get_existing_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path());
        run(&[path], None, "param").unwrap();
    }

    #[test]
    fn get_missing_key_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path());
        assert!(run(&[path], None, "nonexistent").is_err());
    }

    #[test]
    fn get_multiple_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path());
        run(&[path], None, "version,param").unwrap();
    }

    #[test]
    fn get_with_where() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path());
        run(&[path], Some("param=2t"), "param").unwrap();
    }

    #[test]
    fn get_where_filters_out() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path());
        // No messages match → no output, no error
        run(&[path], Some("param=xxx"), "param").unwrap();
    }
}
