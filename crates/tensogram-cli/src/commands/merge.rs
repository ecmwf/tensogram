use std::io::Write;
use std::path::Path;

use tensogram_core::{decode, encode, DecodeOptions, EncodeOptions, GlobalMetadata, TensogramFile};

/// Merge messages from one or more files into a single message.
///
/// All data objects are collected into one message. Global metadata is merged:
/// keys from the first message take precedence on conflict.
pub fn run(inputs: &[impl AsRef<Path>], output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if inputs.is_empty() {
        return Err("no input files specified".into());
    }

    let mut merged_meta: Option<GlobalMetadata> = None;
    let mut all_objects: Vec<(tensogram_core::DataObjectDescriptor, Vec<u8>)> = Vec::new();

    for input in inputs {
        let mut file = TensogramFile::open(input.as_ref())?;
        let count = file.message_count()?;

        for i in 0..count {
            let msg = file.read_message(i)?;
            let (meta, objects) = decode(&msg, &DecodeOptions::default())?;

            match &mut merged_meta {
                None => merged_meta = Some(meta),
                Some(existing) => {
                    for (k, v) in meta.common {
                        existing.common.entry(k).or_insert(v);
                    }
                    // Concatenate per-object payload entries from each message.
                    existing.payload.extend(meta.payload);
                    for (k, v) in meta.reserved {
                        existing.reserved.entry(k).or_insert(v);
                    }
                    for (k, v) in meta.extra {
                        existing.extra.entry(k).or_insert(v);
                    }
                }
            }

            for (desc, data) in objects {
                all_objects.push((desc, data));
            }
        }
    }

    let global_meta = merged_meta.unwrap_or_default();
    let refs: Vec<(&tensogram_core::DataObjectDescriptor, &[u8])> =
        all_objects.iter().map(|(d, b)| (d, b.as_slice())).collect();

    let encoded = encode(&global_meta, &refs, &EncodeOptions::default())?;

    let mut out = std::fs::File::create(output)?;
    out.write_all(&encoded)?;

    println!(
        "Merged {} objects from {} file(s) into {}",
        all_objects.len(),
        inputs.len(),
        output.display()
    );

    Ok(())
}
