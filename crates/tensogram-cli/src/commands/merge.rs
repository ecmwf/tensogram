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
                if existing != &value {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn val(s: &str) -> ciborium::Value {
        ciborium::Value::Text(s.to_string())
    }

    #[test]
    fn first_strategy_keeps_first() {
        let mut map = BTreeMap::new();
        map.insert("k".to_string(), val("first"));
        merge_key(
            &mut map,
            "k".to_string(),
            val("second"),
            MergeStrategy::First,
        )
        .unwrap();
        assert_eq!(map["k"], val("first"));
    }

    #[test]
    fn first_strategy_inserts_new() {
        let mut map = BTreeMap::new();
        merge_key(&mut map, "k".to_string(), val("only"), MergeStrategy::First).unwrap();
        assert_eq!(map["k"], val("only"));
    }

    #[test]
    fn last_strategy_overwrites() {
        let mut map = BTreeMap::new();
        map.insert("k".to_string(), val("first"));
        merge_key(
            &mut map,
            "k".to_string(),
            val("second"),
            MergeStrategy::Last,
        )
        .unwrap();
        assert_eq!(map["k"], val("second"));
    }

    #[test]
    fn error_strategy_allows_identical() {
        let mut map = BTreeMap::new();
        map.insert("k".to_string(), val("same"));
        merge_key(&mut map, "k".to_string(), val("same"), MergeStrategy::Error).unwrap();
        assert_eq!(map["k"], val("same"));
    }

    #[test]
    fn error_strategy_rejects_conflict() {
        let mut map = BTreeMap::new();
        map.insert("k".to_string(), val("first"));
        let result = merge_key(
            &mut map,
            "k".to_string(),
            val("different"),
            MergeStrategy::Error,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("conflicting"));
    }

    #[test]
    fn parse_strategy_valid() {
        assert!(matches!(
            MergeStrategy::parse("first").unwrap(),
            MergeStrategy::First
        ));
        assert!(matches!(
            MergeStrategy::parse("last").unwrap(),
            MergeStrategy::Last
        ));
        assert!(matches!(
            MergeStrategy::parse("error").unwrap(),
            MergeStrategy::Error
        ));
    }

    #[test]
    fn parse_strategy_invalid() {
        assert!(MergeStrategy::parse("unknown").is_err());
    }

    // ── Integration tests ──

    fn make_test_file(dir: &std::path::Path, name: &str, param: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        let mut f = tensogram_core::TensogramFile::create(&path).unwrap();
        let desc = tensogram_core::DataObjectDescriptor {
            obj_type: "ntensor".into(),
            ndim: 1,
            shape: vec![4],
            strides: vec![1],
            dtype: tensogram_core::Dtype::Float32,
            byte_order: tensogram_core::ByteOrder::Big,
            encoding: "none".into(),
            filter: "none".into(),
            compression: "none".into(),
            params: Default::default(),
            hash: None,
        };
        let data = vec![0u8; 16];
        let mut extra = BTreeMap::new();
        extra.insert(
            "param".to_string(),
            ciborium::Value::Text(param.to_string()),
        );
        let meta = tensogram_core::GlobalMetadata {
            version: 2,
            extra,
            ..Default::default()
        };
        f.append(
            &meta,
            &[(&desc, &data)],
            &tensogram_core::EncodeOptions::default(),
        )
        .unwrap();
        path
    }

    #[test]
    fn merge_run_first_strategy() {
        let dir = tempfile::tempdir().unwrap();
        let a = make_test_file(dir.path(), "a.tgm", "2t");
        let b = make_test_file(dir.path(), "b.tgm", "msl");
        let out = dir.path().join("merged.tgm");
        run(&[a, b], &out, "first").unwrap();
        let mut f = tensogram_core::TensogramFile::open(&out).unwrap();
        assert_eq!(f.message_count().unwrap(), 1); // merged into 1 message
    }

    #[test]
    fn merge_run_last_strategy() {
        let dir = tempfile::tempdir().unwrap();
        let a = make_test_file(dir.path(), "a.tgm", "2t");
        let b = make_test_file(dir.path(), "b.tgm", "msl");
        let out = dir.path().join("merged.tgm");
        run(&[a, b], &out, "last").unwrap();
    }

    #[test]
    fn merge_run_no_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("merged.tgm");
        let empty: Vec<std::path::PathBuf> = vec![];
        assert!(run(&empty, &out, "first").is_err());
    }
}
