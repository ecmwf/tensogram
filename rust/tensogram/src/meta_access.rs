// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Unified metadata access — the single source of truth for reading the CBOR
//! metadata frame.
//!
//! This module hosts the dot-path lookup semantics (first-match across
//! `base[i]`, `_extra_` fallback, `_reserved_` hiding) that were previously
//! re-implemented in `tensogram-ffi` and `tensogram-cli`, plus a borrowing
//! value cursor ([`MetaValue`]) that the FFI value handle and every language
//! binding mirror.
//!
//! # Access model
//!
//! [`GlobalMetadata::get`] / [`GlobalMetadata::get_at`] return
//! `Option<MetaValue>` — present ⇔ a value handle, absent ⇔ `None`. This makes
//! *absent* distinguishable from *present-but-wrong-type* and from a real value
//! that happens to equal a caller's default (e.g. `0` / `""`), which the
//! sentinel-returning legacy getters cannot do.
//!
//! The precise accessors do **no** cross-type coercion: [`MetaValue::as_str`]
//! is text-only, [`MetaValue::as_i64`] is integer-only, and
//! [`MetaValue::as_f64`] accepts a float or an integer widened to `f64`.
//!
//! # Scoping
//!
//! - *Message-level* ([`get`](GlobalMetadata::get) / [`contains`](GlobalMetadata::contains)):
//!   first match across `base[0..]` (skipping `_reserved_` at the first
//!   segment), then fall back to `_extra_`. An explicit `extra.` / `_extra_.`
//!   prefix targets `_extra_` directly.
//! - *Per-object* ([`get_at`](GlobalMetadata::get_at) /
//!   [`contains_at`](GlobalMetadata::contains_at)): scoped to `base[obj]` only —
//!   no cross-object first-match, no `_extra_` fallback, `_reserved_` hidden at
//!   the first segment.
//!
//! The `"version"` pseudo-key is **not** special-cased here (the wire version
//! lives in the preamble, not the CBOR frame); consumers that want the legacy
//! `version` shortcut handle it themselves and use [`crate::WIRE_VERSION`].

use std::collections::BTreeMap;

use ciborium::Value;

use crate::metadata::RESERVED_KEY;
use crate::types::GlobalMetadata;

/// The kind of a metadata value, mirrored across every binding as an enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaType {
    /// CBOR null.
    Null,
    /// Boolean.
    Bool,
    /// Integer (CBOR integers are i128-backed; use [`MetaValue::as_i64`] /
    /// [`MetaValue::as_u64`] to extract with range checking).
    Int,
    /// Floating point.
    Float,
    /// UTF-8 text string.
    String,
    /// Byte string.
    Bytes,
    /// Array of values.
    Array,
    /// Map (nested object, or a `base[i]` / `_extra_` / `_reserved_` section).
    Map,
}

/// A borrowing cursor over a single metadata value.
///
/// A `MetaValue` is a zero-copy view into the parsed [`GlobalMetadata`]; it is
/// valid for as long as the metadata it was obtained from. Scalars are read via
/// the `as_*` accessors, arrays via [`len`](Self::len) + [`get_index`](Self::get_index),
/// and maps via [`get_key`](Self::get_key) or positional
/// [`key_at`](Self::key_at) / [`value_at`](Self::value_at).
#[derive(Debug, Clone, Copy)]
pub struct MetaValue<'a> {
    inner: Ref<'a>,
}

/// A `base[i]` / `_extra_` / `_reserved_` section is a Rust `BTreeMap`, not a
/// `ciborium::Value::Map`; wrapping either as a borrowed reference keeps the
/// value cursor zero-copy (no synthesised `Value::Map` clone).
#[derive(Debug, Clone, Copy)]
enum Ref<'a> {
    Cbor(&'a Value),
    Section(&'a BTreeMap<String, Value>),
}

impl<'a> MetaValue<'a> {
    pub(crate) fn cbor(value: &'a Value) -> Self {
        MetaValue {
            inner: Ref::Cbor(value),
        }
    }

    pub(crate) fn section(map: &'a BTreeMap<String, Value>) -> Self {
        MetaValue {
            inner: Ref::Section(map),
        }
    }

    /// The kind of this value.
    #[must_use]
    pub fn value_type(&self) -> MetaType {
        match self.inner {
            Ref::Section(_) => MetaType::Map,
            Ref::Cbor(v) => match v {
                Value::Null => MetaType::Null,
                Value::Bool(_) => MetaType::Bool,
                Value::Integer(_) => MetaType::Int,
                Value::Float(_) => MetaType::Float,
                Value::Text(_) => MetaType::String,
                Value::Bytes(_) => MetaType::Bytes,
                Value::Array(_) => MetaType::Array,
                Value::Map(_) => MetaType::Map,
                // Tags / other exotic kinds are not produced by the metadata
                // encoder; treat them as null for stable behaviour.
                _ => MetaType::Null,
            },
        }
    }

    /// `true` if this value is CBOR null.
    #[must_use]
    pub fn is_null(&self) -> bool {
        matches!(self.inner, Ref::Cbor(Value::Null))
    }

    /// Extract a boolean (no coercion).
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self.inner {
            Ref::Cbor(Value::Bool(b)) => Some(*b),
            _ => None,
        }
    }

    /// Extract a signed integer (integer values only; `None` on overflow).
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match self.inner {
            Ref::Cbor(Value::Integer(i)) => {
                let n: i128 = (*i).into();
                i64::try_from(n).ok()
            }
            _ => None,
        }
    }

    /// Extract an unsigned integer (integer values only; `None` if negative or
    /// out of range).
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self.inner {
            Ref::Cbor(Value::Integer(i)) => {
                let n: i128 = (*i).into();
                u64::try_from(n).ok()
            }
            _ => None,
        }
    }

    /// Extract a float. Accepts a float, or an integer widened to `f64` (which
    /// may lose precision for very large magnitudes).
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self.inner {
            Ref::Cbor(Value::Float(f)) => Some(*f),
            Ref::Cbor(Value::Integer(i)) => {
                let n: i128 = (*i).into();
                Some(n as f64)
            }
            _ => None,
        }
    }

    /// Extract a borrowed UTF-8 string (text values only). May contain interior
    /// NUL bytes.
    #[must_use]
    pub fn as_str(&self) -> Option<&'a str> {
        match self.inner {
            Ref::Cbor(Value::Text(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Extract a borrowed byte string (byte-string values only).
    #[must_use]
    pub fn as_bytes(&self) -> Option<&'a [u8]> {
        match self.inner {
            Ref::Cbor(Value::Bytes(b)) => Some(b.as_slice()),
            _ => None,
        }
    }

    /// Number of elements for an array or map (including a section map); `0`
    /// for scalars.
    #[must_use]
    pub fn len(&self) -> usize {
        match self.inner {
            Ref::Section(m) => m.len(),
            Ref::Cbor(Value::Array(a)) => a.len(),
            Ref::Cbor(Value::Map(m)) => m.len(),
            _ => 0,
        }
    }

    /// `true` if this array/map is empty (or the value is a scalar).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Index into an array; `None` if not an array or out of range.
    #[must_use]
    pub fn get_index(&self, index: usize) -> Option<MetaValue<'a>> {
        match self.inner {
            Ref::Cbor(Value::Array(a)) => a.get(index).map(MetaValue::cbor),
            _ => None,
        }
    }

    /// Look up a single (non-dotted) key in a map or section; `None` if not a
    /// map or the key is absent.
    #[must_use]
    pub fn get_key(&self, key: &str) -> Option<MetaValue<'a>> {
        match self.inner {
            Ref::Section(m) => m.get(key).map(MetaValue::cbor),
            Ref::Cbor(Value::Map(entries)) => entries.iter().find_map(|(k, v)| match k {
                Value::Text(s) if s == key => Some(MetaValue::cbor(v)),
                _ => None,
            }),
            _ => None,
        }
    }

    /// `true` if a map or section contains the given single key.
    #[must_use]
    pub fn contains_key(&self, key: &str) -> bool {
        self.get_key(key).is_some()
    }

    /// The `index`-th key of a map or section, in iteration order (sections and
    /// canonical CBOR maps are key-sorted). `None` if not a map, out of range,
    /// or the key is non-text.
    #[must_use]
    pub fn key_at(&self, index: usize) -> Option<&'a str> {
        match self.inner {
            Ref::Section(m) => m.keys().nth(index).map(String::as_str),
            Ref::Cbor(Value::Map(entries)) => entries.get(index).and_then(|(k, _)| match k {
                Value::Text(s) => Some(s.as_str()),
                _ => None,
            }),
            _ => None,
        }
    }

    /// The `index`-th value of a map or section, in iteration order. `None` if
    /// not a map or out of range.
    #[must_use]
    pub fn value_at(&self, index: usize) -> Option<MetaValue<'a>> {
        match self.inner {
            Ref::Section(m) => m.values().nth(index).map(MetaValue::cbor),
            Ref::Cbor(Value::Map(entries)) => entries.get(index).map(|(_, v)| MetaValue::cbor(v)),
            _ => None,
        }
    }
}

impl GlobalMetadata {
    /// Number of data objects (length of `base`).
    #[must_use]
    pub fn num_objects(&self) -> usize {
        self.base.len()
    }

    /// Message-level dot-path lookup returning the borrowed raw CBOR value.
    ///
    /// First match across `base[i]` (skipping `_reserved_` at the first
    /// segment), then `_extra_`. This is the low-level accessor the legacy
    /// coercing getters build on; most callers want [`get`](Self::get).
    #[must_use]
    pub fn get_value(&self, path: &str) -> Option<&Value> {
        message_level_lookup(self, path)
    }

    /// Per-object dot-path lookup returning the borrowed raw CBOR value,
    /// scoped to `base[obj]` (no cross-object match, no `_extra_` fallback,
    /// `_reserved_` hidden at the first segment).
    #[must_use]
    pub fn get_value_at(&self, obj: usize, path: &str) -> Option<&Value> {
        per_object_lookup(self, obj, path)
    }

    /// Message-level lookup returning a [`MetaValue`] cursor (present ⇔ `Some`).
    #[must_use]
    pub fn get(&self, path: &str) -> Option<MetaValue<'_>> {
        self.get_value(path).map(MetaValue::cbor)
    }

    /// Per-object lookup returning a [`MetaValue`] cursor.
    #[must_use]
    pub fn get_at(&self, obj: usize, path: &str) -> Option<MetaValue<'_>> {
        self.get_value_at(obj, path).map(MetaValue::cbor)
    }

    /// `true` if a message-level dot-path resolves to a value (of any type).
    #[must_use]
    pub fn contains(&self, path: &str) -> bool {
        self.get_value(path).is_some()
    }

    /// `true` if a per-object dot-path resolves to a value (of any type).
    #[must_use]
    pub fn contains_at(&self, obj: usize, path: &str) -> bool {
        self.get_value_at(obj, path).is_some()
    }

    /// View object `obj`'s full metadata map (`base[obj]`) as a [`MetaValue`]
    /// for enumeration. **Includes** `_reserved_` (matching the JSON export and
    /// Python's `meta.base[i]`); path getters still hide `_reserved_`.
    #[must_use]
    pub fn object(&self, obj: usize) -> Option<MetaValue<'_>> {
        self.base.get(obj).map(MetaValue::section)
    }

    /// View the `_extra_` section as a [`MetaValue`] map for enumeration.
    #[must_use]
    pub fn extra_view(&self) -> MetaValue<'_> {
        MetaValue::section(&self.extra)
    }

    /// View the `_reserved_` section as a [`MetaValue`] map for enumeration.
    #[must_use]
    pub fn reserved_view(&self) -> MetaValue<'_> {
        MetaValue::section(&self.reserved)
    }
}

// ---------------------------------------------------------------------------
// Path walkers (single source of truth; FFI + CLI delegate here)
// ---------------------------------------------------------------------------

fn message_level_lookup<'a>(meta: &'a GlobalMetadata, path: &str) -> Option<&'a Value> {
    if path.is_empty() {
        return None;
    }
    let parts: Vec<&str> = path.split('.').collect();
    if parts[0].is_empty() {
        return None;
    }
    // The wire version is a preamble field, not a CBOR key — consumers add the
    // legacy `version` shortcut themselves.
    if parts[0] == "version" && parts.len() == 1 {
        return None;
    }

    // Explicit `_extra_` / `extra` prefix targets the extra map directly.
    if parts[0] == "_extra_" || parts[0] == "extra" {
        if parts.len() > 1 {
            return resolve_in_btree(&meta.extra, &parts[1..]);
        }
        return None;
    }

    // First match across base entries (skipping `_reserved_` at the first
    // segment), then fall back to `_extra_`.
    for entry in &meta.base {
        if let Some(v) = resolve_in_btree_skip_reserved(entry, &parts) {
            return Some(v);
        }
    }
    resolve_in_btree(&meta.extra, &parts)
}

fn per_object_lookup<'a>(meta: &'a GlobalMetadata, obj: usize, path: &str) -> Option<&'a Value> {
    if path.is_empty() {
        return None;
    }
    let parts: Vec<&str> = path.split('.').collect();
    if parts[0].is_empty() {
        return None;
    }
    let entry = meta.base.get(obj)?;
    resolve_in_btree_skip_reserved(entry, &parts)
}

/// Walk a dot-path in a `BTreeMap`, refusing `_reserved_` at the first segment.
fn resolve_in_btree_skip_reserved<'a>(
    map: &'a BTreeMap<String, Value>,
    parts: &[&str],
) -> Option<&'a Value> {
    let (first, rest) = parts.split_first()?;
    if *first == RESERVED_KEY {
        return None;
    }
    let value = map.get(*first)?;
    resolve_cbor_path(value, rest)
}

/// Walk a dot-path in a `BTreeMap` (no `_reserved_` filtering).
fn resolve_in_btree<'a>(map: &'a BTreeMap<String, Value>, parts: &[&str]) -> Option<&'a Value> {
    let (first, rest) = parts.split_first()?;
    let value = map.get(*first)?;
    resolve_cbor_path(value, rest)
}

/// Recursively walk remaining path segments into a CBOR value. When segments
/// remain, the current value must be a `Map` to navigate further.
fn resolve_cbor_path<'a>(value: &'a Value, remaining: &[&str]) -> Option<&'a Value> {
    if remaining.is_empty() {
        return Some(value);
    }
    if let Value::Map(entries) = value {
        for (k, v) in entries {
            if matches!(k, Value::Text(s) if s == remaining[0]) {
                return resolve_cbor_path(v, &remaining[1..]);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    fn map(pairs: &[(&str, Value)]) -> Value {
        Value::Map(
            pairs
                .iter()
                .map(|(k, v)| (Value::Text(k.to_string()), v.clone()))
                .collect(),
        )
    }

    #[test]
    fn get_present_vs_absent_is_distinguishable() {
        let meta = GlobalMetadata {
            base: vec![entry(&[("count", Value::Integer(0.into()))])],
            ..Default::default()
        };
        // A stored 0 is present (not confused with "absent"), and a missing
        // key is None — the whole point of the optional API.
        assert!(meta.contains("count"));
        assert_eq!(meta.get("count").unwrap().as_i64(), Some(0));
        assert!(!meta.contains("missing"));
        assert!(meta.get("missing").is_none());
    }

    #[test]
    fn empty_string_is_present_not_absent() {
        let meta = GlobalMetadata {
            base: vec![entry(&[("name", Value::Text(String::new()))])],
            ..Default::default()
        };
        assert!(meta.contains("name"));
        assert_eq!(meta.get("name").unwrap().as_str(), Some(""));
    }

    #[test]
    fn typed_accessors_are_precise_no_coercion() {
        let meta = GlobalMetadata {
            base: vec![entry(&[
                ("s", Value::Text("2t".into())),
                ("i", Value::Integer(42.into())),
                ("f", Value::Float(2.5)),
                ("b", Value::Bool(true)),
            ])],
            ..Default::default()
        };
        // Right type extracts, wrong type returns None (distinct from absent).
        assert_eq!(meta.get("i").unwrap().as_i64(), Some(42));
        assert_eq!(meta.get("i").unwrap().as_f64(), Some(42.0)); // int widened
        assert_eq!(meta.get("i").unwrap().as_str(), None); // no stringify
        assert_eq!(meta.get("s").unwrap().as_i64(), None);
        assert_eq!(meta.get("f").unwrap().as_f64(), Some(2.5));
        assert_eq!(meta.get("f").unwrap().as_i64(), None);
        assert_eq!(meta.get("b").unwrap().as_bool(), Some(true));
        assert_eq!(meta.get("i").unwrap().value_type(), MetaType::Int);
        assert_eq!(meta.get("s").unwrap().value_type(), MetaType::String);
    }

    #[test]
    fn nested_map_navigation() {
        let meta = GlobalMetadata {
            base: vec![entry(&[(
                "mars",
                map(&[("class", Value::Text("od".into()))]),
            )])],
            ..Default::default()
        };
        // Dot-path and cursor navigation agree.
        assert_eq!(meta.get("mars.class").unwrap().as_str(), Some("od"));
        let mars = meta.get("mars").unwrap();
        assert_eq!(mars.value_type(), MetaType::Map);
        assert_eq!(mars.get_key("class").unwrap().as_str(), Some("od"));
        assert!(mars.contains_key("class"));
        assert_eq!(mars.key_at(0), Some("class"));
    }

    #[test]
    fn array_navigation() {
        let arr = Value::Array(vec![
            Value::Integer(1.into()),
            Value::Integer(2.into()),
            Value::Integer(3.into()),
        ]);
        let meta = GlobalMetadata {
            base: vec![entry(&[("shape", arr)])],
            ..Default::default()
        };
        let shape = meta.get("shape").unwrap();
        assert_eq!(shape.value_type(), MetaType::Array);
        assert_eq!(shape.len(), 3);
        assert_eq!(shape.get_index(1).unwrap().as_i64(), Some(2));
        assert!(shape.get_index(3).is_none());
    }

    #[test]
    fn first_match_then_extra_fallback() {
        let meta = GlobalMetadata {
            base: vec![
                entry(&[("p", Value::Text("first".into()))]),
                entry(&[("p", Value::Text("second".into()))]),
            ],
            extra: entry(&[("only_extra", Value::Text("x".into()))]),
            ..Default::default()
        };
        assert_eq!(meta.get("p").unwrap().as_str(), Some("first"));
        assert_eq!(meta.get("only_extra").unwrap().as_str(), Some("x"));
        assert_eq!(meta.get("extra.only_extra").unwrap().as_str(), Some("x"));
    }

    #[test]
    fn per_object_scoping_and_reserved_rules() {
        let meta = GlobalMetadata {
            base: vec![
                entry(&[("p", Value::Text("a".into()))]),
                entry(&[
                    ("p", Value::Text("b".into())),
                    (RESERVED_KEY, map(&[("tensor", Value::Text("t".into()))])),
                ]),
            ],
            ..Default::default()
        };
        // Scoped to the object, no cross-object first-match.
        assert_eq!(meta.get_at(0, "p").unwrap().as_str(), Some("a"));
        assert_eq!(meta.get_at(1, "p").unwrap().as_str(), Some("b"));
        assert!(meta.get_at(2, "p").is_none()); // out of range
        // Path getters hide `_reserved_`...
        assert!(!meta.contains_at(1, "_reserved_.tensor"));
        // ...but enumeration via object(i) exposes it (parity with Python/JSON).
        let obj1 = meta.object(1).unwrap();
        assert!(obj1.contains_key(RESERVED_KEY));
        assert!(obj1.contains_key("p"));
    }

    #[test]
    fn section_views_enumerate() {
        let meta = GlobalMetadata {
            extra: entry(&[
                ("a", Value::Integer(1.into())),
                ("b", Value::Integer(2.into())),
            ]),
            ..Default::default()
        };
        let extra = meta.extra_view();
        assert_eq!(extra.value_type(), MetaType::Map);
        assert_eq!(extra.len(), 2);
        // BTreeMap iteration is key-sorted.
        assert_eq!(extra.key_at(0), Some("a"));
        assert_eq!(extra.value_at(1).unwrap().as_i64(), Some(2));
        assert_eq!(extra.get_key("b").unwrap().as_i64(), Some(2));
    }

    #[test]
    fn version_pseudo_key_not_special_cased_in_core() {
        let meta = GlobalMetadata::default();
        assert!(meta.get("version").is_none());
        assert!(!meta.contains("version"));
    }
}
