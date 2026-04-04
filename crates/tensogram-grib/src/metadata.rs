use std::collections::BTreeMap;

use ciborium::Value as CborValue;
use eccodes::{DynamicKeyType, RefMessage};

use crate::keys::MARS_KEYS;

/// Extracted key-value pairs from one GRIB message.
#[derive(Debug, Clone)]
pub(crate) struct GribKeySet {
    pub(crate) keys: BTreeMap<String, CborValue>,
}

/// Read all MARS namespace keys from a single ecCodes message handle.
///
/// Uses `read_key_dynamic` to auto-detect the value type (string, int, float).
/// Keys not present in the message are silently skipped.
pub(crate) fn extract_mars_keys(msg: &RefMessage) -> GribKeySet {
    let mut keys = BTreeMap::new();
    for &key_name in MARS_KEYS {
        if let Ok(val) = msg.read_key_dynamic(key_name) {
            match val {
                DynamicKeyType::Str(s) if s != "MISSING" && s != "not_found" => {
                    keys.insert(key_name.to_string(), CborValue::Text(s));
                }
                DynamicKeyType::Int(i) if i != 2147483647 && i != -2147483647 => {
                    keys.insert(key_name.to_string(), CborValue::Integer(i.into()));
                }
                DynamicKeyType::Float(f) if f.is_finite() => {
                    keys.insert(key_name.to_string(), CborValue::Float(f));
                }
                _ => {} // skip missing/sentinel/array values
            }
        }
    }
    GribKeySet { keys }
}

/// Given key sets from N messages, partition into common vs varying.
///
/// - `common`: keys present in ALL messages with identical values
/// - varying: for each message, the keys that differ from common
pub(crate) fn partition_keys(
    all_keys: &[&GribKeySet],
) -> (
    BTreeMap<String, CborValue>,
    Vec<BTreeMap<String, CborValue>>,
) {
    if all_keys.is_empty() {
        return (BTreeMap::new(), vec![]);
    }
    if all_keys.len() == 1 {
        return (all_keys[0].keys.clone(), vec![BTreeMap::new()]);
    }

    let mut common = BTreeMap::new();
    let first = &all_keys[0].keys;

    for (key, value) in first {
        let is_common = all_keys[1..].iter().all(|ks| {
            ks.keys
                .get(key)
                .is_some_and(|v| cbor_values_equal(v, value))
        });
        if is_common {
            common.insert(key.clone(), value.clone());
        }
    }

    let varying: Vec<_> = all_keys
        .iter()
        .map(|ks| {
            ks.keys
                .iter()
                .filter(|(k, _)| !common.contains_key(k.as_str()))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        })
        .collect();

    (common, varying)
}

fn cbor_values_equal(a: &CborValue, b: &CborValue) -> bool {
    match (a, b) {
        (CborValue::Text(a), CborValue::Text(b)) => a == b,
        (CborValue::Integer(a), CborValue::Integer(b)) => a == b,
        (CborValue::Float(a), CborValue::Float(b)) => a == b,
        (CborValue::Bool(a), CborValue::Bool(b)) => a == b,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_keyset(pairs: &[(&str, CborValue)]) -> GribKeySet {
        let keys = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        GribKeySet { keys }
    }

    #[test]
    fn partition_single_message_all_common() {
        let ks = make_keyset(&[
            ("class", CborValue::Text("od".into())),
            ("param", CborValue::Text("2t".into())),
        ]);
        let (common, varying) = partition_keys(&[&ks]);
        assert_eq!(common.len(), 2);
        assert_eq!(varying.len(), 1);
        assert!(varying[0].is_empty());
    }

    #[test]
    fn partition_two_messages_splits_correctly() {
        let ks1 = make_keyset(&[
            ("class", CborValue::Text("od".into())),
            ("param", CborValue::Text("2t".into())),
            ("date", CborValue::Integer(20260404.into())),
        ]);
        let ks2 = make_keyset(&[
            ("class", CborValue::Text("od".into())),
            ("param", CborValue::Text("10u".into())),
            ("date", CborValue::Integer(20260404.into())),
        ]);
        let (common, varying) = partition_keys(&[&ks1, &ks2]);

        assert!(common.contains_key("class"));
        assert!(common.contains_key("date"));
        assert!(!common.contains_key("param"));

        assert_eq!(varying[0].get("param"), Some(&CborValue::Text("2t".into())));
        assert_eq!(
            varying[1].get("param"),
            Some(&CborValue::Text("10u".into()))
        );
    }

    #[test]
    fn partition_empty_input() {
        let (common, varying) = partition_keys(&[]);
        assert!(common.is_empty());
        assert!(varying.is_empty());
    }
}
