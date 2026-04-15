// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

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

// Note: The old partition_flat_keys / partition_keys / partition_grib_keys
// functions were removed during the metadata-major-refactor.  The new model
// stores ALL metadata per object in `base[i]` independently — no
// common/varying split is needed at encode time.  Use
// `tensogram_core::compute_common()` at display/merge time if you need to
// extract shared keys from base entries.

// Tests for extract_mars_keys / extract_all_namespace_keys require real
// ecCodes handles and live in the integration test suite (tests/integration.rs).
