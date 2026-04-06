use std::path::PathBuf;

use tensogram_core::{decode_metadata, TensogramFile};

/// Print summary information for one or more Tensogram files.
pub fn run(files: &[PathBuf]) -> Result<(), Box<dyn std::error::Error>> {
    for path in files {
        let mut file = TensogramFile::open(path)?;
        let count = file.message_count()?;
        let file_size = std::fs::metadata(path)?.len();

        println!("File: {}", path.display());
        println!("  Messages: {count}");
        println!("  Size: {file_size} bytes");

        if count > 0 {
            let msg = file.read_message(0)?;
            let metadata = decode_metadata(&msg)?;
            println!("  Version: {}", metadata.version);
        }
        println!();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensogram_core::{ByteOrder, DataObjectDescriptor, EncodeOptions, GlobalMetadata};

    fn make_test_file(dir: &std::path::Path, n_messages: usize) -> PathBuf {
        let path = dir.join("test.tgm");
        let mut f = tensogram_core::TensogramFile::create(&path).unwrap();
        let desc = DataObjectDescriptor {
            obj_type: "ntensor".into(),
            ndim: 1,
            shape: vec![4],
            strides: vec![1],
            dtype: tensogram_core::Dtype::Float32,
            byte_order: ByteOrder::Big,
            encoding: "none".into(),
            filter: "none".into(),
            compression: "none".into(),
            params: Default::default(),
            hash: None,
        };
        let data = vec![0u8; 16]; // 4 × f32
        let meta = GlobalMetadata {
            version: 2,
            ..Default::default()
        };
        for _ in 0..n_messages {
            f.append(&meta, &[(&desc, &data)], &EncodeOptions::default())
                .unwrap();
        }
        path
    }

    #[test]
    fn info_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path(), 3);
        run(&[path]).unwrap();
    }

    #[test]
    fn info_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = make_test_file(dir.path(), 0);
        run(&[path]).unwrap();
    }

    #[test]
    fn info_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing.tgm");
        let result = run(&[missing]);
        assert!(result.is_err());
    }
}
