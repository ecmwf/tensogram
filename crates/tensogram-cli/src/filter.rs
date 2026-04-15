// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::collections::BTreeMap;

use tensogram_core::{GlobalMetadata, RESERVED_KEY};

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
///   - `version` → struct field
///   - `mars.param` → `base[i]["mars"]["param"]`, …
///   - `grib.geography.Ni` → `base[i]["grib"]["geography"]["Ni"]`, …
///   - `a.b.c.d.e` → walks as many nested CBOR maps as needed
///
/// Search order: `base[i]` (skip `_reserved_` key, first match across entries) → `extra`.
pub fn lookup_key(metadata: &GlobalMetadata, key: &str) -> Option<String> {
    if key.is_empty() {
        return None;
    }
    let parts: Vec<&str> = key.split('.').collect();

    if parts.is_empty() || parts[0].is_empty() {
        return None;
    }

    if parts == ["version"] {
        return Some(metadata.version.to_string());
    }

    // Explicit _extra_.key prefix targets the extra map directly
    if parts[0] == "_extra_" || parts[0] == "extra" {
        if parts.len() > 1 {
            return resolve_path_in_btree(&metadata.extra, &parts[1..]);
        }
        return None;
    }

    // Search base entries (skip _reserved_ key within each entry)
    for entry in &metadata.base {
        if let Some(val) = resolve_path_in_btree_skip_reserved(entry, &parts) {
            return Some(val);
        }
    }
    // Fall back to extra
    resolve_path_in_btree(&metadata.extra, &parts)
}

/// Walk a dot-path starting from a `BTreeMap` root, skipping `_reserved_` keys.
fn resolve_path_in_btree_skip_reserved(
    map: &BTreeMap<String, ciborium::Value>,
    parts: &[&str],
) -> Option<String> {
    let (first, rest) = parts.split_first()?;
    if *first == RESERVED_KEY {
        return None;
    }
    let value = map.get(*first)?;
    resolve_in_cbor(value, rest)
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
        let mut entry = BTreeMap::new();
        entry.insert("centre".into(), Value::Text("ecmwf".into()));
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entry],
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
        let mut entry = BTreeMap::new();
        entry.insert("mars".into(), mars);
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entry],
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
        let mut entry = BTreeMap::new();
        entry.insert("grib".into(), grib);
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entry],
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
    fn test_lookup_base_entry_fallback() {
        // Key in base[0]
        let mars = Value::Map(vec![(
            Value::Text("param".into()),
            Value::Text("lsm".into()),
        )]);
        let mut entry = BTreeMap::new();
        entry.insert("mars".into(), mars);
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entry],
            ..Default::default()
        };
        assert_eq!(lookup_key(&meta, "mars.param"), Some("lsm".into()));
    }

    #[test]
    fn test_lookup_skips_reserved() {
        // _reserved_ key should be skipped during lookup
        let mut entry = BTreeMap::new();
        entry.insert(
            "_reserved_".into(),
            Value::Map(vec![(
                Value::Text("tensor".into()),
                Value::Text("internal".into()),
            )]),
        );
        entry.insert("param".into(), Value::Text("2t".into()));
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entry],
            ..Default::default()
        };
        // Should find param but not _reserved_
        assert_eq!(lookup_key(&meta, "param"), Some("2t".into()));
        assert_eq!(lookup_key(&meta, "_reserved_.tensor"), None);
    }

    // ── Edge case tests ──

    #[test]
    fn test_lookup_empty_key() {
        let meta = GlobalMetadata::default();
        assert_eq!(lookup_key(&meta, ""), None);
    }

    #[test]
    fn test_lookup_dot_only() {
        let meta = GlobalMetadata::default();
        assert_eq!(lookup_key(&meta, "."), None);
    }

    #[test]
    fn test_lookup_base_first_match() {
        // base[0] has mars.param=2t, base[1] has mars.param=msl
        // Should return first match: 2t
        let mars0 = Value::Map(vec![(
            Value::Text("param".into()),
            Value::Text("2t".into()),
        )]);
        let mars1 = Value::Map(vec![(
            Value::Text("param".into()),
            Value::Text("msl".into()),
        )]);
        let mut entry0 = BTreeMap::new();
        entry0.insert("mars".into(), mars0);
        let mut entry1 = BTreeMap::new();
        entry1.insert("mars".into(), mars1);
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entry0, entry1],
            ..Default::default()
        };
        assert_eq!(lookup_key(&meta, "mars.param"), Some("2t".into()));
    }

    #[test]
    fn test_lookup_base_wins_over_extra() {
        let mut entry = BTreeMap::new();
        entry.insert("shared".into(), Value::Text("from_base".into()));
        let mut extra = BTreeMap::new();
        extra.insert("shared".into(), Value::Text("from_extra".into()));
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entry],
            extra,
            ..Default::default()
        };
        // base should win (first-match order)
        assert_eq!(lookup_key(&meta, "shared"), Some("from_base".into()));
    }

    #[test]
    fn test_lookup_extra_prefix() {
        let mut extra = BTreeMap::new();
        extra.insert("custom".into(), Value::Text("val".into()));
        let meta = GlobalMetadata {
            version: 2,
            extra,
            ..Default::default()
        };
        // Explicit _extra_.custom prefix should resolve in extra
        assert_eq!(lookup_key(&meta, "_extra_.custom"), Some("val".into()));
        // Same with "extra." prefix
        assert_eq!(lookup_key(&meta, "extra.custom"), Some("val".into()));
    }

    #[test]
    fn test_lookup_deeply_nested() {
        // a.b.c.d.e
        let e_val = Value::Map(vec![(Value::Text("e".into()), Value::Text("deep".into()))]);
        let d_val = Value::Map(vec![(Value::Text("d".into()), e_val)]);
        let c_val = Value::Map(vec![(Value::Text("c".into()), d_val)]);
        let b_val = Value::Map(vec![(Value::Text("b".into()), c_val)]);
        let mut entry = BTreeMap::new();
        entry.insert("a".into(), b_val);
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entry],
            ..Default::default()
        };
        assert_eq!(lookup_key(&meta, "a.b.c.d.e"), Some("deep".into()));
        // a.b.c.d is a Map — resolve_in_cbor stringifies it via Debug
        assert!(lookup_key(&meta, "a.b.c.d").is_some());
    }

    #[test]
    fn test_lookup_base_partial_match() {
        // base[0] has mars.param but base[1] doesn't have mars at all
        let mars = Value::Map(vec![(
            Value::Text("param".into()),
            Value::Text("2t".into()),
        )]);
        let mut entry0 = BTreeMap::new();
        entry0.insert("mars".into(), mars);
        let mut entry1 = BTreeMap::new();
        entry1.insert("other".into(), Value::Text("val".into()));
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entry0, entry1],
            ..Default::default()
        };
        // Returns 2t from entry0, entry1 is skipped (no "mars" key)
        assert_eq!(lookup_key(&meta, "mars.param"), Some("2t".into()));
    }

    #[test]
    fn test_matches_missing_key_eq_fails() {
        let clause = parse_where("nonexistent=val").unwrap();
        let meta = GlobalMetadata::default();
        assert!(!matches(&meta, &clause));
    }

    #[test]
    fn test_matches_missing_key_neq_passes() {
        let clause = parse_where("nonexistent!=val").unwrap();
        let meta = GlobalMetadata::default();
        assert!(matches(&meta, &clause));
    }

    #[test]
    fn test_matches_multi_base_first_match() {
        // Filter matches on first-match from base entries
        let mut entry0 = BTreeMap::new();
        entry0.insert("param".into(), Value::Text("2t".into()));
        let mut entry1 = BTreeMap::new();
        entry1.insert("param".into(), Value::Text("msl".into()));
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entry0, entry1],
            ..Default::default()
        };
        let clause = parse_where("param=2t").unwrap();
        // First match is "2t" → matches
        assert!(matches(&meta, &clause));
        let clause_msl = parse_where("param=msl").unwrap();
        // First match is "2t" not "msl" → doesn't match
        assert!(!matches(&meta, &clause_msl));
    }

    // ── Additional coverage: OR separator and != ──

    #[test]
    fn test_parse_eq_or_separator() {
        let clause = parse_where("param=2t/msl/10u").unwrap();
        assert_eq!(clause.op, FilterOp::Eq);
        assert_eq!(clause.values, vec!["2t", "msl", "10u"]);
    }

    #[test]
    fn test_parse_neq_or_separator() {
        let clause = parse_where("class!=od/rd").unwrap();
        assert_eq!(clause.op, FilterOp::NotEq);
        assert_eq!(clause.values, vec!["od", "rd"]);
    }

    #[test]
    fn test_matches_eq_or_any_match() {
        let mut entry = BTreeMap::new();
        entry.insert("param".into(), Value::Text("msl".into()));
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entry],
            ..Default::default()
        };
        let clause = parse_where("param=2t/msl/10u").unwrap();
        // "msl" is one of the OR values → matches
        assert!(matches(&meta, &clause));
    }

    #[test]
    fn test_matches_eq_or_no_match() {
        let mut entry = BTreeMap::new();
        entry.insert("param".into(), Value::Text("sp".into()));
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entry],
            ..Default::default()
        };
        let clause = parse_where("param=2t/msl/10u").unwrap();
        // "sp" not in OR values → no match
        assert!(!matches(&meta, &clause));
    }

    #[test]
    fn test_matches_neq_or_all_different() {
        let mut entry = BTreeMap::new();
        entry.insert("class".into(), Value::Text("ea".into()));
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entry],
            ..Default::default()
        };
        let clause = parse_where("class!=od/rd").unwrap();
        // "ea" != "od" AND "ea" != "rd" → matches
        assert!(matches(&meta, &clause));
    }

    #[test]
    fn test_matches_neq_or_one_matches() {
        let mut entry = BTreeMap::new();
        entry.insert("class".into(), Value::Text("od".into()));
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entry],
            ..Default::default()
        };
        let clause = parse_where("class!=od/rd").unwrap();
        // "od" == "od" so NotEq fails (all values must differ)
        assert!(!matches(&meta, &clause));
    }

    #[test]
    fn test_lookup_version() {
        let meta = GlobalMetadata {
            version: 2,
            ..Default::default()
        };
        assert_eq!(lookup_key(&meta, "version"), Some("2".into()));
    }

    #[test]
    fn test_matches_version_eq() {
        let meta = GlobalMetadata {
            version: 2,
            ..Default::default()
        };
        let clause = parse_where("version=2").unwrap();
        assert!(matches(&meta, &clause));
        let clause_wrong = parse_where("version=1").unwrap();
        assert!(!matches(&meta, &clause_wrong));
    }

    #[test]
    fn test_matches_version_neq() {
        let meta = GlobalMetadata {
            version: 2,
            ..Default::default()
        };
        let clause = parse_where("version!=1").unwrap();
        assert!(matches(&meta, &clause));
        let clause_same = parse_where("version!=2").unwrap();
        assert!(!matches(&meta, &clause_same));
    }

    #[test]
    fn test_lookup_extra_only() {
        // Key only in extra, no base entries
        let mut extra = BTreeMap::new();
        extra.insert("source".into(), Value::Text("test".into()));
        let meta = GlobalMetadata {
            version: 2,
            extra,
            ..Default::default()
        };
        assert_eq!(lookup_key(&meta, "source"), Some("test".into()));
    }

    #[test]
    fn test_lookup_extra_prefix_no_subkey() {
        // Bare "_extra_" without subkey
        let meta = GlobalMetadata::default();
        assert_eq!(lookup_key(&meta, "_extra_"), None);
        assert_eq!(lookup_key(&meta, "extra"), None);
    }

    #[test]
    fn test_cbor_value_to_string_types() {
        assert_eq!(cbor_value_to_string(&Value::Null), "null");
        assert_eq!(cbor_value_to_string(&Value::Bool(false)), "false");
        assert_eq!(cbor_value_to_string(&Value::Float(2.5)), "2.5");
    }
}
