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
        let count = file.message_count()?;

        for i in 0..count {
            let msg = file.read_message(i)?;

            // Decode metadata first for cheap filtering
            let metadata = decode_metadata(&msg)?;

            if let Some(ref clause) = clause {
                if !filter::matches(&metadata, clause) {
                    continue;
                }
            }

            // Decode full message to access per-object descriptors
            let (global_meta, objects) = decode(&msg, &DecodeOptions::default())?;

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
                for (j, entry) in global_meta.base.iter().enumerate() {
                    println!("  base[{j}]:");
                    for (key, value) in entry {
                        println!("    {key}: {}", output::format_json_value(value));
                    }
                }
                if !global_meta.extra.is_empty() {
                    println!("  extra:");
                    for (key, value) in &global_meta.extra {
                        println!("    {key}: {}", output::format_json_value(value));
                    }
                }
                println!();
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
        let path = dir.join("dump_test.tgm");
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
    fn dump_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path());
        run(&[path], None, None, false).unwrap();
    }

    #[test]
    fn dump_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path());
        run(&[path], None, None, true).unwrap();
    }

    #[test]
    fn dump_with_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path());
        run(&[path], None, Some("param"), true).unwrap();
    }

    #[test]
    fn dump_with_where() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path());
        run(&[path], Some("param=2t"), None, false).unwrap();
    }

    #[test]
    fn dump_where_no_match() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path());
        run(&[path], Some("param=xxx"), None, false).unwrap();
    }
}
