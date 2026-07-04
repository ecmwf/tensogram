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
        if let Ok(val) = msg.read_key_dynamic(key_name)
            && let Some(cbor) = dynamic_to_cbor(val)
        {
            keys.insert(key_name.clone(), cbor);
        }
    }

    Ok(keys)
}

/// Upper bound on array length kept in metadata. Small *definition* arrays
/// (`pl` for reduced grids, `pv` for hybrid levels, `scaledValues`) are needed
/// to reconstruct a message and must be preserved; the huge *data* arrays
/// (`values`/`codedValues`, ~10^6 elements) are the payload and are read
/// separately — never through this path — but the guard keeps the CBOR bounded
/// against any pathological namespace key.
const MAX_ARRAY_LEN: usize = 8192;

/// Convert an ecCodes `DynamicKeyType` to a CBOR value, filtering out
/// missing/sentinel values and oversized arrays. Returns `None` to skip.
///
/// Small arrays and byte strings (local-section blobs) ARE kept — they are
/// required for lossless round-trip reconstruction (`plans/GRIB_NETCDF_ROUNDTRIP.md`).
fn dynamic_to_cbor(val: DynamicKeyType) -> Option<CborValue> {
    match val {
        DynamicKeyType::Str(s) if s != "MISSING" && s != "not_found" => Some(CborValue::Text(s)),
        DynamicKeyType::Int(i) if i != 2147483647 && i != -2147483647 => {
            Some(CborValue::Integer(i.into()))
        }
        DynamicKeyType::Float(f) if f.is_finite() => Some(CborValue::Float(f)),
        DynamicKeyType::IntArray(a) if a.len() <= MAX_ARRAY_LEN => Some(CborValue::Array(
            a.into_iter().map(|i| CborValue::Integer(i.into())).collect(),
        )),
        DynamicKeyType::FloatArray(a)
            if a.len() <= MAX_ARRAY_LEN && a.iter().all(|f| f.is_finite()) =>
        {
            Some(CborValue::Array(a.into_iter().map(CborValue::Float).collect()))
        }
        DynamicKeyType::Bytes(b) => Some(CborValue::Bytes(b)),
        _ => None, // missing, sentinel, or oversized/non-finite array
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

// ── Reconstruct key-set (for lossless round-trip / `to-grib`) ────────────────

/// WMO namespaces whose keys define the message geometry, product, time, and
/// vertical level and are settable on a cloned ecCodes sample. Together with
/// the scalar keys below they let `to-grib` rebuild a message from scratch —
/// ecCodes computes the remaining (read-only/derived) keys.
const RECONSTRUCT_NAMESPACES: &[&str] = &["geography", "parameter", "time", "vertical"];

/// Extra scalar keys, outside the namespaces above, needed to select the grid
/// template, product template, packing, identification section, and the ECMWF
/// local-use section on reconstruction.
const RECONSTRUCT_SCALAR_KEYS: &[&str] = &[
    "edition",
    "discipline",
    "gridType",
    "packingType",
    "bitsPerValue",
    "decimalScaleFactor",
    // Identification (section 1) + processing metadata.
    "tablesVersion",
    "centre",
    "subCentre",
    "productionStatusOfProcessedData",
    "typeOfProcessedData",
    "typeOfGeneratingProcess",
    "generatingProcessIdentifier",
    // ECMWF local-use section (section 2). `setLocalDefinition` must be set
    // first on export (see `export::PRIORITY_KEYS`) before these can exist.
    "setLocalDefinition",
    "grib2LocalSectionNumber",
    "marsClass",
    "marsType",
    "marsStream",
    "experimentVersionNumber",
];

/// Capture the key-set needed to rebuild this message from an ecCodes sample.
///
/// Stored flat under `base[i]["grib_repro"]`; consumed by the `to-grib`
/// exporter. See `plans/GRIB_NETCDF_ROUNDTRIP.md`.
pub(crate) fn extract_reconstruct_keys(
    msg: &mut RefMessage,
) -> Result<BTreeMap<String, CborValue>, eccodes::errors::CodesError> {
    let mut keys = BTreeMap::new();
    for &ns in RECONSTRUCT_NAMESPACES {
        for (k, v) in read_namespace_keys(msg, ns)? {
            keys.insert(k, v);
        }
    }
    for &k in RECONSTRUCT_SCALAR_KEYS {
        if let Ok(val) = msg.read_key_dynamic(k)
            && let Some(cbor) = dynamic_to_cbor(val)
        {
            keys.insert(k.to_string(), cbor);
        }
    }
    Ok(keys)
}

// Note: The old partition_flat_keys / partition_keys / partition_grib_keys
// functions were removed during the metadata-major-refactor.  The new model
// stores ALL metadata per object in `base[i]` independently — no
// common/varying split is needed at encode time.  Use
// `tensogram::compute_common()` at display/merge time if you need to
// extract shared keys from base entries.

// Tests for extract_mars_keys / extract_all_namespace_keys require real
// ecCodes handles and live in the integration test suite (tests/integration.rs).

#[cfg(test)]
mod tests {
    //! Unit tests for [`dynamic_to_cbor`].
    //!
    //! The happy-path branches (`Str`, `Int`, `Float`) are already exercised
    //! end-to-end by `tests/integration.rs` on real GRIB fixtures; these
    //! cases fire so often that coverage shows high hit counts.  What
    //! integration tests *cannot* reach are the skip branches — sentinel
    //! strings, sentinel integers, NaN floats, and the array variants —
    //! because none of our fixtures contain such keys.  This module pokes
    //! those branches directly with synthesised `DynamicKeyType` values.

    use super::*;

    #[test]
    fn str_non_sentinel_becomes_text() {
        assert_eq!(
            dynamic_to_cbor(DynamicKeyType::Str("2t".to_string())),
            Some(CborValue::Text("2t".to_string())),
        );
    }

    #[test]
    fn str_missing_sentinel_returns_none() {
        assert_eq!(
            dynamic_to_cbor(DynamicKeyType::Str("MISSING".to_string())),
            None,
        );
    }

    #[test]
    fn str_not_found_sentinel_returns_none() {
        assert_eq!(
            dynamic_to_cbor(DynamicKeyType::Str("not_found".to_string())),
            None,
        );
    }

    #[test]
    fn int_non_sentinel_becomes_integer() {
        assert_eq!(
            dynamic_to_cbor(DynamicKeyType::Int(42)),
            Some(CborValue::Integer(42_i64.into())),
        );
    }

    #[test]
    fn int_positive_sentinel_returns_none() {
        // ecCodes' i32::MAX sentinel for "missing" integer keys.
        assert_eq!(dynamic_to_cbor(DynamicKeyType::Int(2147483647)), None);
    }

    #[test]
    fn int_negative_sentinel_returns_none() {
        assert_eq!(dynamic_to_cbor(DynamicKeyType::Int(-2147483647)), None);
    }

    #[test]
    fn float_finite_becomes_float() {
        assert_eq!(
            dynamic_to_cbor(DynamicKeyType::Float(1.5)),
            Some(CborValue::Float(1.5)),
        );
    }

    #[test]
    fn float_nan_returns_none() {
        assert_eq!(dynamic_to_cbor(DynamicKeyType::Float(f64::NAN)), None);
    }

    #[test]
    fn float_positive_infinity_returns_none() {
        assert_eq!(dynamic_to_cbor(DynamicKeyType::Float(f64::INFINITY)), None);
    }

    #[test]
    fn float_negative_infinity_returns_none() {
        assert_eq!(
            dynamic_to_cbor(DynamicKeyType::Float(f64::NEG_INFINITY)),
            None,
        );
    }

    #[test]
    fn small_float_array_is_kept() {
        // Small definition arrays (e.g. `pv`, `scaledValues`) are needed for
        // lossless round-trip reconstruction, so they are preserved.
        assert_eq!(
            dynamic_to_cbor(DynamicKeyType::FloatArray(vec![1.0, 2.0, 3.0])),
            Some(CborValue::Array(vec![
                CborValue::Float(1.0),
                CborValue::Float(2.0),
                CborValue::Float(3.0),
            ])),
        );
    }

    #[test]
    fn small_int_array_is_kept() {
        // Small definition arrays (e.g. `pl` for reduced grids) are preserved.
        assert_eq!(
            dynamic_to_cbor(DynamicKeyType::IntArray(vec![1, 2, 3])),
            Some(CborValue::Array(vec![
                CborValue::Integer(1_i64.into()),
                CborValue::Integer(2_i64.into()),
                CborValue::Integer(3_i64.into()),
            ])),
        );
    }

    #[test]
    fn oversized_array_returns_none() {
        // The data payload never routes through here, but a pathological
        // oversized namespace key is dropped to keep the CBOR bounded.
        let big = vec![0_i64; MAX_ARRAY_LEN + 1];
        assert_eq!(dynamic_to_cbor(DynamicKeyType::IntArray(big)), None);
    }

    #[test]
    fn non_finite_float_array_returns_none() {
        assert_eq!(
            dynamic_to_cbor(DynamicKeyType::FloatArray(vec![1.0, f64::NAN])),
            None,
        );
    }

    #[test]
    fn bytes_are_kept() {
        // Bytes is the "binary section" variant (local-section blobs) — kept
        // verbatim as a CBOR byte string for reconstruction.
        assert_eq!(
            dynamic_to_cbor(DynamicKeyType::Bytes(vec![0xDE, 0xAD, 0xBE, 0xEF])),
            Some(CborValue::Bytes(vec![0xDE, 0xAD, 0xBE, 0xEF])),
        );
    }
}
