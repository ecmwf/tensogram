// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::io::Write;
use std::path::{Path, PathBuf};

use tensogram::{DecodeOptions, EncodeOptions, RESERVED_KEY, TensogramFile, decode, encode};

/// Split a multi-object message into separate single-object messages.
///
/// Each data object becomes its own message, inheriting the global metadata.
/// Output files are named using the template with `[index]` placeholder,
/// or sequentially numbered in the output directory.
pub fn run(
    input: &Path,
    output_template: &str,
    threads: u32,
    mask_cli: &super::MaskCliOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = TensogramFile::open(input)?;
    let count = file.message_count()?;

    let mut total_written = 0;

    for i in 0..count {
        let msg = file.read_message(i)?;
        // Wire byte order: preserve original byte layout for re-encoding.
        let wire_opts = DecodeOptions {
            native_byte_order: false,
            threads,
            ..Default::default()
        };
        let (meta, objects) = decode(&msg, &wire_opts)?;

        if objects.len() <= 1 {
            // Single-object message: write as-is
            let out_name = expand_index(output_template, total_written);
            let mut out = std::fs::File::create(&out_name)?;
            out.write_all(&msg)?;
            total_written += 1;
            continue;
        }

        for (idx, (desc, data)) in objects.iter().enumerate() {
            let out_name = expand_index(output_template, total_written);

            // Extract the base entry specific to this object so that
            // per-object metadata (e.g. mars keys) is preserved in the split.
            let mut split_meta = meta.clone();
            split_meta.base = if idx < meta.base.len() {
                let mut entry = meta.base[idx].clone();
                entry.remove(RESERVED_KEY); // encoder will regenerate
                vec![entry]
            } else {
                vec![]
            };
            // Clear reserved — the encoder will regenerate it.
            split_meta.reserved.clear();

            let mut encode_opts = EncodeOptions {
                threads,
                ..Default::default()
            };
            mask_cli.apply(&mut encode_opts)?;
            let encoded = encode(&split_meta, &[(desc, data.as_slice())], &encode_opts)?;

            let mut out = std::fs::File::create(&out_name)?;
            out.write_all(&encoded)?;
            total_written += 1;
        }
    }

    println!("Split into {total_written} message(s)");

    Ok(())
}

fn expand_index(template: &str, index: usize) -> String {
    if template.contains("[index]") {
        template.replace("[index]", &format!("{index:04}"))
    } else {
        // Append index before extension
        let path = PathBuf::from(template);
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "split".to_string());
        let ext = path
            .extension()
            .map(|s| format!(".{}", s.to_string_lossy()))
            .unwrap_or_default();
        let parent = path
            .parent()
            .map(|p| {
                let s = p.to_string_lossy().to_string();
                if s.is_empty() {
                    String::new()
                } else {
                    format!("{s}/")
                }
            })
            .unwrap_or_default();
        format!("{parent}{stem}_{index:04}{ext}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_index_with_placeholder() {
        assert_eq!(expand_index("out/msg_[index].tgm", 3), "out/msg_0003.tgm");
    }

    #[test]
    fn expand_index_without_placeholder() {
        assert_eq!(expand_index("output.tgm", 0), "output_0000.tgm");
        assert_eq!(expand_index("dir/output.tgm", 5), "dir/output_0005.tgm");
    }

    // ── Integration tests ──

    use tensogram::{ByteOrder, DataObjectDescriptor, Dtype, EncodeOptions, GlobalMetadata};

    fn make_multi_object_file(dir: &std::path::Path) -> PathBuf {
        let path = dir.join("split_input.tgm");
        let mut f = tensogram::TensogramFile::create(&path).unwrap();
        let desc1 = DataObjectDescriptor {
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
        };
        let desc2 = desc1.clone();
        let data = vec![0u8; 16];
        let meta = GlobalMetadata {
            ..Default::default()
        };
        // One message with 2 objects
        f.append(
            &meta,
            &[(&desc1, &data), (&desc2, &data)],
            &EncodeOptions::default(),
        )
        .unwrap();
        path
    }

    #[test]
    fn split_multi_object() {
        let dir = tempfile::tempdir().unwrap();
        let input = make_multi_object_file(dir.path());
        let template = format!("{}/split_[index].tgm", dir.path().display());
        run(
            &input,
            &template,
            0,
            &super::super::MaskCliOptions::default(),
        )
        .unwrap();
        assert!(dir.path().join("split_0000.tgm").exists());
        assert!(dir.path().join("split_0001.tgm").exists());
        // Verify each split file has 1 object
        let f0 = tensogram::TensogramFile::open(dir.path().join("split_0000.tgm")).unwrap();
        let msg = f0.read_message(0).unwrap();
        let (_, objs) = tensogram::decode(&msg, &tensogram::DecodeOptions::default()).unwrap();
        assert_eq!(objs.len(), 1);
    }

    #[test]
    fn split_auto_naming() {
        let dir = tempfile::tempdir().unwrap();
        let input = make_multi_object_file(dir.path());
        let template = format!("{}/out.tgm", dir.path().display());
        run(
            &input,
            &template,
            0,
            &super::super::MaskCliOptions::default(),
        )
        .unwrap();
        assert!(dir.path().join("out_0000.tgm").exists());
        assert!(dir.path().join("out_0001.tgm").exists());
    }

    #[test]
    fn split_single_object_passthrough() {
        // Single-object message should pass through as-is
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("single.tgm");
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
        };
        let data = vec![0u8; 16];
        let meta = GlobalMetadata {
            ..Default::default()
        };
        f.append(&meta, &[(&desc, &data)], &EncodeOptions::default())
            .unwrap();
        drop(f);

        let template = format!("{}/split_[index].tgm", dir.path().display());
        run(
            &path,
            &template,
            0,
            &super::super::MaskCliOptions::default(),
        )
        .unwrap();
        assert!(dir.path().join("split_0000.tgm").exists());
        // Should NOT have a second file
        assert!(!dir.path().join("split_0001.tgm").exists());

        // Verify the single file has 1 object
        let f = tensogram::TensogramFile::open(dir.path().join("split_0000.tgm")).unwrap();
        let msg = f.read_message(0).unwrap();
        let (_, objs) = tensogram::decode(&msg, &tensogram::DecodeOptions::default()).unwrap();
        assert_eq!(objs.len(), 1);
    }

    #[test]
    fn split_preserves_per_object_base_metadata() {
        // Multi-object message with per-object base metadata
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("split_meta.tgm");
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
        };
        let data = vec![0u8; 16];
        let mut base0 = std::collections::BTreeMap::new();
        base0.insert("param".into(), ciborium::Value::Text("2t".into()));
        let mut base1 = std::collections::BTreeMap::new();
        base1.insert("param".into(), ciborium::Value::Text("msl".into()));
        let meta = GlobalMetadata {
            base: vec![base0, base1],
            ..Default::default()
        };
        f.append(
            &meta,
            &[(&desc, &data), (&desc, &data)],
            &EncodeOptions::default(),
        )
        .unwrap();
        drop(f);

        let template = format!("{}/split_meta_[index].tgm", dir.path().display());
        run(
            &path,
            &template,
            0,
            &super::super::MaskCliOptions::default(),
        )
        .unwrap();

        // Verify first split has param=2t
        let msg0 = std::fs::read(dir.path().join("split_meta_0000.tgm")).unwrap();
        let meta0 = tensogram::decode_metadata(&msg0).unwrap();
        assert!(
            meta0
                .base
                .iter()
                .any(|e| e.get("param") == Some(&ciborium::Value::Text("2t".into())))
        );

        // Verify second split has param=msl
        let msg1 = std::fs::read(dir.path().join("split_meta_0001.tgm")).unwrap();
        let meta1 = tensogram::decode_metadata(&msg1).unwrap();
        assert!(
            meta1
                .base
                .iter()
                .any(|e| e.get("param") == Some(&ciborium::Value::Text("msl".into())))
        );
    }
}
