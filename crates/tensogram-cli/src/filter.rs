use std::collections::BTreeMap;

use tensogram_core::GlobalMetadata;

/// A parsed where-clause filter.
#[derive(Debug)]
pub struct WhereClause {
    pub key: String,
    pub op: FilterOp,
    pub values: Vec<String>,
}

#[derive(Debug, PartialEq)]
pub enum FilterOp {
    Eq,
    NotEq,
}

/// Parse a where-clause string like "mars.param=2t/10u" or "mars.class!=od".
pub fn parse_where(input: &str) -> Result<WhereClause, String> {
    if let Some((key, rest)) = input.split_once("!=") {
        let values = rest.split('/').map(|s| s.to_string()).collect();
        Ok(WhereClause {
            key: key.to_string(),
            op: FilterOp::NotEq,
            values,
        })
    } else if let Some((key, rest)) = input.split_once('=') {
        let values = rest.split('/').map(|s| s.to_string()).collect();
        Ok(WhereClause {
            key: key.to_string(),
            op: FilterOp::Eq,
            values,
        })
    } else {
        Err(format!(
            "invalid where-clause: {input} (expected key=value or key!=value)"
        ))
    }
}

/// Look up a dot-notation key in global metadata, returning the FIRST matching value.
///
/// Supports arbitrary nesting depth:
///   - `version` → struct field, then `common`, then `extra`
///   - `mars.param` → `common["mars"]["param"]`, …
///   - `grib.geography.Ni` → `common["grib"]["geography"]["Ni"]`, …
///   - `a.b.c.d.e` → walks as many nested CBOR maps as needed
///
/// Search order: `common` → `payload[i]` (first match) → `extra`.
pub fn lookup_key(metadata: &GlobalMetadata, key: &str) -> Option<String> {
    let parts: Vec<&str> = key.split('.').collect();

    if parts == ["version"] {
        return Some(metadata.version.to_string());
    }

    // Search: common → payload entries → extra
    if let Some(val) = resolve_path_in_btree(&metadata.common, &parts) {
        return Some(val);
    }
    for entry in &metadata.payload {
        if let Some(val) = resolve_path_in_btree(entry, &parts) {
            return Some(val);
        }
    }
    resolve_path_in_btree(&metadata.extra, &parts)
}

/// Walk a dot-path starting from a `BTreeMap` root.
///
/// The first segment indexes the `BTreeMap` by string key.
/// Remaining segments navigate nested `CborValue::Map` layers.
fn resolve_path_in_btree(
    map: &BTreeMap<String, ciborium::Value>,
    parts: &[&str],
) -> Option<String> {
    let (first, rest) = parts.split_first()?;
    let value = map.get(*first)?;
    resolve_in_cbor(value, rest)
}

/// Recursively walk remaining path segments into a `CborValue`.
///
/// When no segments remain, stringify the leaf value.
/// When segments remain, the current value must be a `CborValue::Map`
/// to navigate further; otherwise returns `None`.
fn resolve_in_cbor(value: &ciborium::Value, remaining: &[&str]) -> Option<String> {
    if remaining.is_empty() {
        return Some(cbor_value_to_string(value));
    }
    if let ciborium::Value::Map(entries) = value {
        for (k, v) in entries {
            if matches!(k, ciborium::Value::Text(s) if s == remaining[0]) {
                return resolve_in_cbor(v, &remaining[1..]);
            }
        }
    }
    None
}

/// Test if global metadata matches a where-clause.
pub fn matches(metadata: &GlobalMetadata, clause: &WhereClause) -> bool {
    match lookup_key(metadata, &clause.key) {
        Some(actual) => match clause.op {
            FilterOp::Eq => clause.values.iter().any(|v| v == &actual),
            FilterOp::NotEq => clause.values.iter().all(|v| v != &actual),
        },
        None => matches!(clause.op, FilterOp::NotEq), // missing key: Eq fails, NotEq passes
    }
}

fn cbor_value_to_string(value: &ciborium::Value) -> String {
    match value {
        ciborium::Value::Text(s) => s.to_string(),
        ciborium::Value::Integer(i) => {
            let n: i128 = (*i).into();
            n.to_string()
        }
        ciborium::Value::Float(f) => f.to_string(),
        ciborium::Value::Bool(b) => b.to_string(),
        ciborium::Value::Null => "null".to_string(),
        _ => format!("{value:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ciborium::Value;

    #[test]
    fn test_parse_eq() {
        let clause = parse_where("mars.param=2t/10u").unwrap();
        assert_eq!(clause.key, "mars.param");
        assert_eq!(clause.op, FilterOp::Eq);
        assert_eq!(clause.values, vec!["2t", "10u"]);
    }

    #[test]
    fn test_parse_neq() {
        let clause = parse_where("mars.class!=od").unwrap();
        assert_eq!(clause.key, "mars.class");
        assert_eq!(clause.op, FilterOp::NotEq);
        assert_eq!(clause.values, vec!["od"]);
    }

    #[test]
    fn test_parse_invalid() {
        assert!(parse_where("no_operator").is_err());
    }

    // ── lookup_key tests ────────────────────────────────────────────────

    #[test]
    fn test_lookup_depth_1() {
        let mut common = BTreeMap::new();
        common.insert("centre".into(), Value::Text("ecmwf".into()));
        let meta = GlobalMetadata {
            version: 2,
            common,
            ..Default::default()
        };
        assert_eq!(lookup_key(&meta, "centre"), Some("ecmwf".into()));
    }

    #[test]
    fn test_lookup_depth_2() {
        let mars = Value::Map(vec![(
            Value::Text("param".into()),
            Value::Text("2t".into()),
        )]);
        let mut common = BTreeMap::new();
        common.insert("mars".into(), mars);
        let meta = GlobalMetadata {
            version: 2,
            common,
            ..Default::default()
        };
        assert_eq!(lookup_key(&meta, "mars.param"), Some("2t".into()));
    }

    #[test]
    fn test_lookup_depth_3() {
        let geo = Value::Map(vec![(
            Value::Text("Ni".into()),
            Value::Integer(1440.into()),
        )]);
        let grib = Value::Map(vec![(Value::Text("geography".into()), geo)]);
        let mut common = BTreeMap::new();
        common.insert("grib".into(), grib);
        let meta = GlobalMetadata {
            version: 2,
            common,
            ..Default::default()
        };
        assert_eq!(lookup_key(&meta, "grib.geography.Ni"), Some("1440".into()));
    }

    #[test]
    fn test_lookup_missing_path() {
        let meta = GlobalMetadata::default();
        assert_eq!(lookup_key(&meta, "no.such.path"), None);
    }

    #[test]
    fn test_lookup_payload_fallback() {
        // Key in payload[0], not in common.
        let mars = Value::Map(vec![(
            Value::Text("param".into()),
            Value::Text("lsm".into()),
        )]);
        let mut entry = BTreeMap::new();
        entry.insert("mars".into(), mars);
        let meta = GlobalMetadata {
            version: 2,
            payload: vec![entry],
            ..Default::default()
        };
        assert_eq!(lookup_key(&meta, "mars.param"), Some("lsm".into()));
    }
}
