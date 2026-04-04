use std::collections::BTreeMap;

use ciborium::Value as CborValue;
use eccodes::{DynamicKeyType, FallibleIterator, KeysIteratorFlags, RefMessage};

/// Non-mars GRIB namespaces to iterate when `preserve_all_keys` is enabled.
const GRIB_NAMESPACES: &[&str] = &[
    "ls",
    "geography",
    "time",
    "vertical",
    "parameter",
    "statistics",
];

/// Extracted key-value pairs from one GRIB message.
#[derive(Debug, Clone)]
pub(crate) struct GribKeySet {
    pub(crate) keys: BTreeMap<String, CborValue>,
}

// ── Namespace extraction ─────────────────────────────────────────────────────

/// Read all keys from a single ecCodes namespace.
///
/// Uses `new_keys_iterator(namespace)` to discover key names at runtime.
/// The iterator is collected into a `Vec<String>` and dropped (releasing
/// the mutable borrow on `msg`) before reading values.
fn read_namespace_keys(
    msg: &mut RefMessage,
    namespace: &str,
) -> Result<BTreeMap<String, CborValue>, eccodes::errors::CodesError> {
    // Phase 1: collect key names.
    let key_names: Vec<String> = {
        let mut iter = msg.new_keys_iterator(
            &[
                KeysIteratorFlags::AllKeys,
                KeysIteratorFlags::SkipDuplicates,
            ],
            namespace,
        )?;
        let mut names = Vec::new();
        while let Some(name) = iter.next()? {
            names.push(name);
        }
        names
        // `iter` dropped here, releasing `&mut msg`.
    };

    // Phase 2: read each key's value dynamically.
    let mut keys = BTreeMap::new();
    for key_name in &key_names {
        if let Ok(val) = msg.read_key_dynamic(key_name) {
            if let Some(cbor) = dynamic_to_cbor(val) {
                keys.insert(key_name.clone(), cbor);
            }
        }
    }

    Ok(keys)
}

/// Convert an ecCodes `DynamicKeyType` to a CBOR value, filtering out
/// missing/sentinel/array values.  Returns `None` for values to skip.
fn dynamic_to_cbor(val: DynamicKeyType) -> Option<CborValue> {
    match val {
        DynamicKeyType::Str(s) if s != "MISSING" && s != "not_found" => Some(CborValue::Text(s)),
        DynamicKeyType::Int(i) if i != 2147483647 && i != -2147483647 => {
            Some(CborValue::Integer(i.into()))
        }
        DynamicKeyType::Float(f) if f.is_finite() => Some(CborValue::Float(f)),
        _ => None, // missing, sentinel, array, or other
    }
}

/// Dynamically read all MARS namespace keys from a single ecCodes message.
pub(crate) fn extract_mars_keys(
    msg: &mut RefMessage,
) -> Result<GribKeySet, eccodes::errors::CodesError> {
    let keys = read_namespace_keys(msg, "mars")?;
    Ok(GribKeySet { keys })
}

/// Read all non-mars GRIB namespace keys from a single ecCodes message.
///
/// Returns `{ namespace_name → { key → value } }` for each of the
/// standard GRIB namespaces (ls, geography, time, vertical, parameter,
/// statistics).  Empty namespaces are omitted.
pub(crate) fn extract_all_namespace_keys(
    msg: &mut RefMessage,
) -> Result<BTreeMap<String, BTreeMap<String, CborValue>>, eccodes::errors::CodesError> {
    let mut result = BTreeMap::new();
    for &ns in GRIB_NAMESPACES {
        let keys = read_namespace_keys(msg, ns)?;
        if !keys.is_empty() {
            result.insert(ns.to_string(), keys);
        }
    }
    Ok(result)
}

// ── Partitioning ─────────────────────────────────────────────────────────────

/// Partition flat key/value maps from N messages into common vs varying.
///
/// - `common`: keys present in ALL maps with identical values.
/// - `varying`: for each map, the keys that differ from common.
pub(crate) fn partition_flat_keys(
    all: &[&BTreeMap<String, CborValue>],
) -> (
    BTreeMap<String, CborValue>,
    Vec<BTreeMap<String, CborValue>>,
) {
    if all.is_empty() {
        return (BTreeMap::new(), vec![]);
    }
    if all.len() == 1 {
        return (all[0].clone(), vec![BTreeMap::new()]);
    }

    // A key is common iff it appears in ALL maps with the same value.
    // We check from the first map outward (keys unique to non-first maps
    // are not common by definition, and they appear in varying naturally).
    let mut common = BTreeMap::new();
    let first = all[0];

    for (key, value) in first {
        let is_common = all[1..]
            .iter()
            .all(|m| m.get(key).is_some_and(|v| cbor_values_equal(v, value)));
        if is_common {
            common.insert(key.clone(), value.clone());
        }
    }

    let varying: Vec<_> = all
        .iter()
        .map(|m| {
            m.iter()
                .filter(|(k, _)| !common.contains_key(k.as_str()))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        })
        .collect();

    (common, varying)
}

/// Partition MARS key sets using [`partition_flat_keys`].
pub(crate) fn partition_keys(
    all_keys: &[&GribKeySet],
) -> (
    BTreeMap<String, CborValue>,
    Vec<BTreeMap<String, CborValue>>,
) {
    let maps: Vec<&BTreeMap<String, CborValue>> = all_keys.iter().map(|ks| &ks.keys).collect();
    partition_flat_keys(&maps)
}

/// Partition namespaced GRIB keys from N messages into common vs varying.
///
/// For each namespace independently, runs [`partition_flat_keys`].
/// Returns `(common_grib, varying_per_object)` where each is
/// `{ namespace → { key → value } }`.  Empty namespaces are omitted.
pub(crate) fn partition_grib_keys(
    all: &[&BTreeMap<String, BTreeMap<String, CborValue>>],
) -> (
    BTreeMap<String, BTreeMap<String, CborValue>>,
    Vec<BTreeMap<String, BTreeMap<String, CborValue>>>,
) {
    if all.is_empty() {
        return (BTreeMap::new(), vec![]);
    }

    let n = all.len();
    let mut common_grib: BTreeMap<String, BTreeMap<String, CborValue>> = BTreeMap::new();
    let mut varying_grib: Vec<BTreeMap<String, BTreeMap<String, CborValue>>> =
        (0..n).map(|_| BTreeMap::new()).collect();

    // Collect the union of all namespace names across all messages.
    let mut all_ns: Vec<&str> = Vec::new();
    for msg_grib in all {
        for ns in msg_grib.keys() {
            if !all_ns.contains(&ns.as_str()) {
                all_ns.push(ns.as_str());
            }
        }
    }
    all_ns.sort_unstable();

    let empty = BTreeMap::new();
    for ns in all_ns {
        // Collect the flat key map for this namespace from each message.
        let ns_maps: Vec<&BTreeMap<String, CborValue>> = all
            .iter()
            .map(|msg_grib| msg_grib.get(ns).unwrap_or(&empty))
            .collect();

        let (common_ns, varying_ns) = partition_flat_keys(&ns_maps);

        if !common_ns.is_empty() {
            common_grib.insert(ns.to_string(), common_ns);
        }
        for (i, v) in varying_ns.into_iter().enumerate() {
            if !v.is_empty() {
                varying_grib[i].insert(ns.to_string(), v);
            }
        }
    }

    (common_grib, varying_grib)
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

    #[test]
    fn partition_grib_keys_single_message() {
        let mut msg = BTreeMap::new();
        let mut geo = BTreeMap::new();
        geo.insert("Ni".to_string(), CborValue::Integer(1440.into()));
        msg.insert("geography".to_string(), geo);

        let (common, varying) = partition_grib_keys(&[&msg]);
        assert!(common.contains_key("geography"));
        assert_eq!(varying.len(), 1);
        assert!(varying[0].is_empty());
    }

    #[test]
    fn partition_grib_keys_splits_across_namespaces() {
        let mut msg1 = BTreeMap::new();
        let mut msg2 = BTreeMap::new();

        // geography: Ni same, gridType same
        let mut geo1 = BTreeMap::new();
        geo1.insert("Ni".to_string(), CborValue::Integer(1440.into()));
        geo1.insert("gridType".to_string(), CborValue::Text("regular_ll".into()));
        msg1.insert("geography".to_string(), geo1.clone());
        msg2.insert("geography".to_string(), geo1);

        // parameter: shortName differs
        let mut par1 = BTreeMap::new();
        par1.insert("shortName".to_string(), CborValue::Text("2t".into()));
        msg1.insert("parameter".to_string(), par1);

        let mut par2 = BTreeMap::new();
        par2.insert("shortName".to_string(), CborValue::Text("lsm".into()));
        msg2.insert("parameter".to_string(), par2);

        let (common, varying) = partition_grib_keys(&[&msg1, &msg2]);

        // geography is fully common
        assert!(common.contains_key("geography"));
        assert_eq!(common["geography"].len(), 2);

        // parameter.shortName differs → goes to varying
        assert!(!common.contains_key("parameter"));
        assert!(varying[0].contains_key("parameter"));
        assert!(varying[1].contains_key("parameter"));
    }
}
