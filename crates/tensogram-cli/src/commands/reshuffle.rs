use std::io::Write;
use std::path::Path;

use tensogram_core::{decode, encode, DecodeOptions, EncodeOptions, TensogramFile};

/// Reshuffle frames inside messages: move footer frames to header position.
///
/// Converts streaming-mode messages (footer-based index/hash) into
/// random-access-mode messages (header-based index/hash).
/// This is a decode → re-encode operation.
pub fn run(input: &Path, output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = TensogramFile::open(input)?;
    let count = file.message_count()?;

    let mut out = std::fs::File::create(output)?;

    for i in 0..count {
        let msg = file.read_message(i)?;
        let (meta, objects) = decode(&msg, &DecodeOptions::default())?;

        let refs: Vec<_> = objects.iter().map(|(d, b)| (d, b.as_slice())).collect();

        // Re-encode produces a header-mode message (non-streaming)
        let encoded = encode(&meta, &refs, &EncodeOptions::default())?;
        out.write_all(&encoded)?;
    }

    println!(
        "Reshuffled {count} message(s) from {} to {}",
        input.display(),
        output.display()
    );

    Ok(())
}
