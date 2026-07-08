// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! GRIB export: reconstruct GRIB messages from a Tensogram message.
//!
//! Reads each object's `base[i]["grib_repro"]` reconstruct key-set and its
//! decoded `f64` values, clones an ecCodes sample, sets the captured keys +
//! values, and re-emits GRIB.  This is the reverse of
//! [`crate::convert_grib_file`].  See `plans/GRIB_NETCDF_ROUNDTRIP.md`.
//!
//! Scope: `regular_ll` geometry with `grid_simple` / `grid_ieee` / `grid_ccsds`
//! packing.  The WMO geometry / product / packing / level keys, the
//! identification section, and the ECMWF local-use section (section 2) are
//! restored — `grib_compare` on all keys is clean for the real ECMWF field
//! `2t.grib2`.  Other geometries and exotic templates remain follow-ups.

use std::collections::BTreeMap;
use std::path::PathBuf;

use ciborium::Value as CborValue;
use eccodes::{BufMessage, CodesFile, FallibleIterator, KeyWrite, ProductKind};

use tensogram::types::{ByteOrder, GlobalMetadata};
use tensogram::{DecodeOptions, Dtype, decode};

use crate::error::GribError;

/// ecCodes sample used as the reconstruction template for `regular_ll` surface
/// fields.  (Milestone 1: `regular_ll` only; other geometries select other
/// samples in a follow-up.)
const SAMPLE_PATH: &str = "/usr/share/eccodes/samples/regular_ll_sfc_grib2.tmpl";

/// Keys set *first*, in this order, because they select templates / sizing /
/// sections.  ecCodes is order-sensitive: the identification and grid template
/// must precede the local section, which must precede the mars keys it defines,
/// and packing/precision must precede the values.  Remaining keys are set after
/// (tolerant of read-only rejects).
const PRIORITY_KEYS: &[&str] = &[
    "edition",
    "tablesVersion",
    "centre",
    "subCentre",
    "gridType",
    "setLocalDefinition",
    "grib2LocalSectionNumber",
    "packingType",
    "bitsPerValue",
];

/// Reconstruct a GRIB file (one message per data object) from a Tensogram
/// message produced by [`crate::convert_grib_file`] / [`crate::convert_grib_buffer`].
///
/// # Errors
///
/// - [`GribError::InvalidData`] — the message is not decodable, an object is
///   not `float64`, or an object lacks the `grib_repro` key-set (i.e. was not
///   produced by `convert-grib`).
/// - [`GribError::EcCodes`] — ecCodes rejected a key or failed to encode.
/// - [`GribError::Io`] — the temporary encode file could not be written/read.
pub fn to_grib(message: &[u8]) -> Result<Vec<u8>, GribError> {
    let (meta, objects) = decode(message, &DecodeOptions::default())
        .map_err(|e| GribError::InvalidData(format!("decode tensogram message: {e}")))?;

    let mut out = Vec::new();
    for (i, (desc, payload)) in objects.iter().enumerate() {
        let repro = repro_keys(&meta, i)?;
        let values = payload_to_f64(desc.dtype, desc.byte_order, payload)?;
        out.extend_from_slice(&reconstruct_message(&repro, &values)?);
    }
    Ok(out)
}

/// Pull the flat `grib_repro` key map for object `i` out of the metadata.
fn repro_keys(meta: &GlobalMetadata, i: usize) -> Result<BTreeMap<String, CborValue>, GribError> {
    let entry = meta
        .base
        .get(i)
        .ok_or_else(|| GribError::InvalidData(format!("no base metadata for object {i}")))?;
    let Some(CborValue::Map(m)) = entry.get("grib_repro") else {
        return Err(GribError::InvalidData(format!(
            "object {i} has no 'grib_repro' key-set; was it produced by convert-grib?"
        )));
    };
    let mut keys = BTreeMap::new();
    for (k, v) in m {
        if let CborValue::Text(name) = k {
            keys.insert(name.clone(), v.clone());
        }
    }
    Ok(keys)
}

/// Decode the object payload into `f64` grid-point values.
fn payload_to_f64(dtype: Dtype, order: ByteOrder, bytes: &[u8]) -> Result<Vec<f64>, GribError> {
    if dtype != Dtype::Float64 {
        return Err(GribError::InvalidData(format!(
            "to-grib expects float64 objects, got {dtype}"
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        let mut b = [0u8; 8];
        b.copy_from_slice(chunk);
        out.push(match order {
            ByteOrder::Little => f64::from_le_bytes(b),
            ByteOrder::Big => f64::from_be_bytes(b),
        });
    }
    Ok(out)
}

/// Clone the sample, set the captured keys + values, and return the encoded
/// GRIB message bytes.
fn reconstruct_message(
    repro: &BTreeMap<String, CborValue>,
    values: &[f64],
) -> Result<Vec<u8>, GribError> {
    let mut handle = CodesFile::new_from_file(SAMPLE_PATH, ProductKind::GRIB)?;
    let sample = handle
        .ref_message_iter()
        .next()?
        .ok_or_else(|| GribError::InvalidData(format!("sample {SAMPLE_PATH} has no message")))?;
    let mut msg = sample.try_clone()?;

    // `setLocalDefinition` is a write-only *trigger*: it always reads back `0`,
    // so the captured value cannot recreate the section.  When the source
    // carried an ECMWF local section (evidenced by `grib2LocalSectionNumber`),
    // fire the trigger with a literal `1` to allocate section 2 before the mars
    // keys that live inside it are set.
    let has_local_section = repro.contains_key("grib2LocalSectionNumber");

    // Priority keys first (order-sensitive template/sizing selectors).
    for &k in PRIORITY_KEYS {
        if k == "setLocalDefinition" {
            if has_local_section {
                let _ = set_key(&mut msg, k, &CborValue::Integer(1_i64.into()));
            }
            continue;
        }
        if let Some(v) = repro.get(k) {
            // Tolerant: a priority key may be read-only in some templates.
            let _ = set_key(&mut msg, k, v);
        }
    }
    // Remaining keys, tolerant of read-only / derived keys that reject writes.
    for (k, v) in repro {
        if PRIORITY_KEYS.contains(&k.as_str()) {
            continue;
        }
        let _ = set_key(&mut msg, k, v);
    }
    // Values last — this sizes the data section (`numberOfValues`).
    msg.write_key_unchecked("values", values)?;

    // The eccodes crate can only emit to a file path; use a unique temp file.
    let tmp = temp_path();
    msg.write_to_file(&tmp, false)?;
    let bytes = std::fs::read(&tmp)?;
    let _ = std::fs::remove_file(&tmp);
    Ok(bytes)
}

/// Set one CBOR-typed key on the message, mapping to the ecCodes setter.
fn set_key(msg: &mut BufMessage, key: &str, value: &CborValue) -> Result<(), GribError> {
    match value {
        CborValue::Integer(i) => {
            let v = i64::try_from(*i).map_err(|_| {
                GribError::InvalidData(format!("key {key}: integer out of i64 range"))
            })?;
            msg.write_key_unchecked(key, v)?;
        }
        CborValue::Float(f) => {
            msg.write_key_unchecked(key, *f)?;
        }
        CborValue::Text(s) => {
            msg.write_key_unchecked(key, s.as_str())?;
        }
        CborValue::Bytes(b) => {
            msg.write_key_unchecked(key, b.as_slice())?;
        }
        // Array-valued keys (e.g. `pv`, `pl`) are captured but not yet replayed.
        _ => {}
    }
    Ok(())
}

/// A unique temporary file path for the ecCodes encode step.
fn temp_path() -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CTR: AtomicU64 = AtomicU64::new(0);
    let n = CTR.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("tensogram_to_grib_{pid}_{nanos}_{n}.grib2"))
}

#[cfg(test)]
mod tests {
    //! Error-path unit tests for the pure (ecCodes-free) helpers.  The happy
    //! path (sample clone → set keys → encode) is covered end-to-end against
    //! real fixtures in `tests/roundtrip.rs`; here we pin the `InvalidData`
    //! contracts that a caller relies on when a message is not `to-grib` shaped.

    use super::*;

    #[test]
    fn payload_to_f64_rejects_non_float64() {
        let err = payload_to_f64(Dtype::Int32, ByteOrder::Little, &[0u8; 4]).unwrap_err();
        assert!(
            matches!(err, GribError::InvalidData(m) if m.contains("float64")),
            "non-float64 payload must be a float64 InvalidData error"
        );
    }

    #[test]
    fn payload_to_f64_reads_both_endiannesses() {
        let le = payload_to_f64(Dtype::Float64, ByteOrder::Little, &1.5_f64.to_le_bytes()).unwrap();
        let be = payload_to_f64(Dtype::Float64, ByteOrder::Big, &1.5_f64.to_be_bytes()).unwrap();
        assert_eq!(le, vec![1.5]);
        assert_eq!(be, vec![1.5]);
    }

    #[test]
    fn repro_keys_errors_when_object_absent() {
        let meta = GlobalMetadata::default();
        assert!(
            repro_keys(&meta, 0).is_err(),
            "no base entry for object 0 must error"
        );
    }

    #[test]
    fn repro_keys_errors_without_grib_repro() {
        let meta = GlobalMetadata {
            base: vec![BTreeMap::new()],
            ..Default::default()
        };
        let err = repro_keys(&meta, 0).unwrap_err();
        assert!(
            matches!(err, GribError::InvalidData(m) if m.contains("grib_repro")),
            "an object without grib_repro must name the missing key-set"
        );
    }

    #[test]
    fn to_grib_rejects_undecodable_message() {
        assert!(
            to_grib(b"not a tensogram message").is_err(),
            "garbage input must not be decoded as a message"
        );
    }
}
