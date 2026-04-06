use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use tensogram_core::{decode, encode, DecodeOptions, EncodeOptions, GlobalMetadata, TensogramFile};

/// Merge strategy for conflicting metadata keys.
#[derive(Debug, Clone, Copy)]
enum MergeStrategy {
    /// First value wins (default — existing behaviour).
    First,
    /// Last value wins (later files overwrite earlier ones).
    Last,
    /// Fail on conflict (any key clash is an error).
    Error,
}

impl MergeStrategy {
    fn parse(s: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match s {
            "first" => Ok(Self::First),
            "last" => Ok(Self::Last),
            "error" => Ok(Self::Error),
            other => Err(format!(
                "unknown merge strategy '{other}': expected first, last, or error"
            )
            .into()),
        }
    }
}

/// Insert `key → value` into `map` using the chosen strategy.
/// Returns Err only if strategy is Error and the key already exists with a different value.
fn merge_key(
    map: &mut BTreeMap<String, ciborium::Value>,
    key: String,
    value: ciborium::Value,
    strategy: MergeStrategy,
) -> Result<(), Box<dyn std::error::Error>> {
    match strategy {
        MergeStrategy::First => {
            map.entry(key).or_insert(value);
        }
        MergeStrategy::Last => {
            map.insert(key, value);
        }
        MergeStrategy::Error => {
            if let Some(existing) = map.get(&key) {
                if *existing != value {
                    return Err(format!(
                        "conflicting values for key '{key}' (use --strategy first or last to resolve)"
                    )
                    .into());
                }
            }
            map.insert(key, value);
        }
    }
    Ok(())
}

/// Merge messages from one or more files into a single message.
///
/// All data objects are collected into one message. Global metadata is merged
/// using the chosen strategy for key conflicts.
pub fn run(
    inputs: &[impl AsRef<Path>],
    output: &Path,
    strategy_str: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if inputs.is_empty() {
        return Err("no input files specified".into());
    }

    let strategy = MergeStrategy::parse(strategy_str)?;

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
                        merge_key(&mut existing.common, k, v, strategy)?;
                    }
                    // Concatenate per-object payload entries from each message.
                    existing.payload.extend(meta.payload);
                    for (k, v) in meta.reserved {
                        merge_key(&mut existing.reserved, k, v, strategy)?;
                    }
                    for (k, v) in meta.extra {
                        merge_key(&mut existing.extra, k, v, strategy)?;
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
        "Merged {} objects from {} file(s) into {} (strategy: {strategy_str})",
        all_objects.len(),
        inputs.len(),
        output.display()
    );

    Ok(())
}
