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
