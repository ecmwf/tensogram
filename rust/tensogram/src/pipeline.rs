// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Shared encoding/filter/compression pipeline helpers for importers.
//!
//! Both `tensogram-grib` and `tensogram-netcdf` accept the same set of CLI
//! flags (`--encoding` / `--bits` / `--filter` / `--compression` /
//! `--compression-level`) and translate them into
//! [`DataObjectDescriptor`] fields + `params` map entries in exactly the
//! same way. This module centralises that translation so the two
//! importers cannot drift out of sync.
//!
//! ## Usage
//!
//! ```no_run
//! use tensogram::pipeline::{apply_pipeline, DataPipeline};
//! use tensogram::types::{ByteOrder, DataObjectDescriptor};
//! use tensogram::Dtype;
//! use std::collections::BTreeMap;
//!
//! let mut desc = DataObjectDescriptor {
//!     obj_type: "ntensor".to_string(),
//!     ndim: 1,
//!     shape: vec![4],
//!     strides: vec![1],
//!     dtype: Dtype::Float64,
//!     byte_order: ByteOrder::Little,
//!     encoding: "none".to_string(),
//!     filter: "none".to_string(),
//!     compression: "none".to_string(),
//!     params: BTreeMap::new(),
//!     hash: None,
//! };
//!
//! let pipeline = DataPipeline {
//!     compression: "zstd".to_string(),
//!     ..Default::default()
//! };
//!
//! let values = [1.0_f64, 2.0, 3.0, 4.0];
//! apply_pipeline(&mut desc, Some(&values), &pipeline, "my_var").unwrap();
//! assert_eq!(desc.compression, "zstd");
//! ```

use ciborium::Value as CborValue;
use tensogram_encodings::simple_packing;

use crate::types::DataObjectDescriptor;

/// Encoding/filter/compression configuration for data objects.
///
/// Defaults to all `"none"` — produces uncompressed raw little-endian
/// payloads identical to the pre-pipeline behaviour. This is the shared
/// type used by both `tensogram-grib` and `tensogram-netcdf`; the two
/// crates re-export it from their own `lib.rs` for convenience.
#[derive(Debug, Clone)]
pub struct DataPipeline {
    /// Encoding stage: `"none"` (default) or `"simple_packing"`.
    pub encoding: String,
    /// Bits per value for `simple_packing`. Defaults to 16 when `None`.
    pub bits: Option<u32>,
    /// Filter stage: `"none"` (default) or `"shuffle"`.
    pub filter: String,
    /// Compression codec: `"none"` (default), `"zstd"`, `"lz4"`,
    /// `"blosc2"`, or `"szip"`.
    pub compression: String,
    /// Optional compression level (used by `zstd` and `blosc2`; ignored
    /// by other codecs).
    pub compression_level: Option<i32>,
}

impl Default for DataPipeline {
    fn default() -> Self {
        Self {
            encoding: "none".to_string(),
            bits: None,
            filter: "none".to_string(),
            compression: "none".to_string(),
            compression_level: None,
        }
    }
}

/// Apply a [`DataPipeline`] to a [`DataObjectDescriptor`] by setting its
/// `encoding` / `filter` / `compression` fields and populating `params`.
///
/// `values` carries the float64 payload when available — `simple_packing`
/// is a float64-only encoding, so this parameter is `Some(&[f64])` when
/// the caller has typed f64 values and `None` otherwise (e.g. integer
/// variables in a mixed NetCDF file). When `pipeline.encoding ==
/// "simple_packing"` but `values` is `None`, the encoding stage is
/// skipped with a stderr warning and the conversion continues with
/// `encoding = "none"`.
///
/// `var_label` is embedded in warning/error messages for human-readable
/// diagnostics — typically the variable name (`"temperature"`) or
/// something like `"GRIB message"`.
///
/// # Errors
///
/// Returns a human-readable error string when `pipeline.encoding`,
/// `pipeline.filter`, or `pipeline.compression` is not one of the
/// recognised values. Callers wrap this string into their own
/// importer-specific error type (`GribError::InvalidData` /
/// `NetcdfError::InvalidData`).
///
/// Soft failures (`simple_packing` rejecting `NaN`-containing data, or
/// `simple_packing` requested on a non-f64 variable) are reported as
/// stderr warnings and do NOT return an error — the variable falls
/// back to `encoding = "none"` and the conversion continues.
pub fn apply_pipeline(
    desc: &mut DataObjectDescriptor,
    values: Option<&[f64]>,
    pipeline: &DataPipeline,
    var_label: &str,
) -> Result<(), String> {
    // ── Encoding stage ─────────────────────────────────────────────────
    let mut applied_simple_packing = false;
    match pipeline.encoding.as_str() {
        "none" => {}
        "simple_packing" => match values {
            None => {
                // Non-f64 payloads cannot be simple-packed.  The dtype
                // is the caller's choice and this branch is a sanity
                // check, not a NaN/Inf failure — leave the stderr
                // warning + fallback to `encoding="none"` so the rest
                // of the conversion still succeeds.
                eprintln!(
                    "warning: skipping simple_packing for {var_label} \
                     (not a float64 payload)"
                );
            }
            Some(values) => {
                let bits = pipeline.bits.unwrap_or(16);
                // Hard-fail on any `compute_params` error — most often
                // NaN or Inf in the input data.  Pre-0.17 behaviour
                // soft-downgraded silently to `encoding="none"` with
                // only a stderr warning, which hid data-quality
                // problems from callers.  The remedy hint is
                // appropriate for every failure mode: fix the data
                // (NaN/Inf) or pick a different encoding (which also
                // covers `BitsPerValueTooLarge` via `--bits`).
                let params = simple_packing::compute_params(values, bits, 0).map_err(|e| {
                    format!(
                        "simple_packing failed for {var_label}: {e}. \
                         Pre-process the data or choose a different \
                         encoding (e.g. encoding=\"none\")."
                    )
                })?;
                desc.encoding = "simple_packing".to_string();
                desc.params.insert(
                    "reference_value".to_string(),
                    CborValue::Float(params.reference_value),
                );
                desc.params.insert(
                    "binary_scale_factor".to_string(),
                    CborValue::Integer((i64::from(params.binary_scale_factor)).into()),
                );
                desc.params.insert(
                    "decimal_scale_factor".to_string(),
                    CborValue::Integer((i64::from(params.decimal_scale_factor)).into()),
                );
                desc.params.insert(
                    "bits_per_value".to_string(),
                    CborValue::Integer((i64::from(params.bits_per_value)).into()),
                );
                applied_simple_packing = true;
            }
        },
        other => {
            return Err(format!(
                "unknown encoding '{other}'; expected 'none' or 'simple_packing'"
            ));
        }
    }

    // ── Filter stage ───────────────────────────────────────────────────
    match pipeline.filter.as_str() {
        "none" => {}
        "shuffle" => {
            desc.filter = "shuffle".to_string();
            // shuffle is run AFTER encoding by the pipeline, so the
            // element size is the *post-encoding* byte width:
            //   - simple_packing applied → ⌈bpv/8⌉
            //   - otherwise → native dtype byte width
            let element_size = if applied_simple_packing {
                let bpv = pipeline.bits.unwrap_or(16) as usize;
                bpv.div_ceil(8).max(1)
            } else {
                desc.dtype.byte_width()
            };
            desc.params.insert(
                "shuffle_element_size".to_string(),
                CborValue::Integer((element_size as i64).into()),
            );
        }
        other => {
            return Err(format!(
                "unknown filter '{other}'; expected 'none' or 'shuffle'"
            ));
        }
    }

    // ── Compression stage ──────────────────────────────────────────────
    match pipeline.compression.as_str() {
        "none" => {}
        "zstd" => {
            desc.compression = "zstd".to_string();
            let level = pipeline.compression_level.unwrap_or(3);
            desc.params.insert(
                "zstd_level".to_string(),
                CborValue::Integer((i64::from(level)).into()),
            );
        }
        "lz4" => {
            desc.compression = "lz4".to_string();
        }
        "blosc2" => {
            desc.compression = "blosc2".to_string();
            let clevel = pipeline.compression_level.unwrap_or(5);
            desc.params.insert(
                "blosc2_clevel".to_string(),
                CborValue::Integer((i64::from(clevel)).into()),
            );
            // Default sub-codec — users wanting a different one should
            // construct a `DataObjectDescriptor` manually.
            desc.params.insert(
                "blosc2_codec".to_string(),
                CborValue::Text("lz4".to_string()),
            );
        }
        "szip" => {
            desc.compression = "szip".to_string();
            // Sensible szip defaults consistent with the rest of the
            // codebase (see `tensogram` tests + examples).
            desc.params
                .insert("szip_rsi".to_string(), CborValue::Integer(128.into()));
            desc.params
                .insert("szip_block_size".to_string(), CborValue::Integer(16.into()));
            desc.params
                .insert("szip_flags".to_string(), CborValue::Integer(8.into()));
        }
        other => {
            return Err(format!(
                "unknown compression '{other}'; expected one of: none, zstd, lz4, blosc2, szip"
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::Dtype;
    use crate::types::ByteOrder;

    fn mk_desc() -> DataObjectDescriptor {
        DataObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 1,
            shape: vec![4],
            strides: vec![1],
            dtype: Dtype::Float64,
            byte_order: ByteOrder::Little,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        }
    }

    fn int_param(desc: &DataObjectDescriptor, key: &str) -> i64 {
        match desc.params.get(key) {
            Some(CborValue::Integer(i)) => {
                let n: i128 = (*i).into();
                n as i64
            }
            other => panic!("{key} not an integer: {other:?}"),
        }
    }

    // ── Defaults ────────────────────────────────────────────────────

    #[test]
    fn default_pipeline_is_all_none() {
        let p = DataPipeline::default();
        assert_eq!(p.encoding, "none");
        assert_eq!(p.filter, "none");
        assert_eq!(p.compression, "none");
        assert!(p.bits.is_none());
        assert!(p.compression_level.is_none());
    }

    #[test]
    fn default_pipeline_leaves_descriptor_unchanged() {
        let mut desc = mk_desc();
        let values = [1.0, 2.0, 3.0, 4.0];
        apply_pipeline(&mut desc, Some(&values), &DataPipeline::default(), "x").unwrap();
        assert_eq!(desc.encoding, "none");
        assert_eq!(desc.filter, "none");
        assert_eq!(desc.compression, "none");
        assert!(desc.params.is_empty());
    }

    // ── Encoding ────────────────────────────────────────────────────

    #[test]
    fn simple_packing_populates_four_params() {
        let mut desc = mk_desc();
        let p = DataPipeline {
            encoding: "simple_packing".to_string(),
            bits: Some(16),
            ..Default::default()
        };
        let values = [0.0_f64, 1.0, 2.0, 3.0];
        apply_pipeline(&mut desc, Some(&values), &p, "test").unwrap();
        assert_eq!(desc.encoding, "simple_packing");
        assert_eq!(int_param(&desc, "bits_per_value"), 16);
        assert_eq!(int_param(&desc, "decimal_scale_factor"), 0);
        assert!(desc.params.contains_key("reference_value"));
        assert!(desc.params.contains_key("binary_scale_factor"));
    }

    #[test]
    fn simple_packing_with_no_values_skips_with_warning() {
        let mut desc = mk_desc();
        let p = DataPipeline {
            encoding: "simple_packing".to_string(),
            ..Default::default()
        };
        apply_pipeline(&mut desc, None, &p, "int_var").unwrap();
        assert_eq!(desc.encoding, "none", "should skip, not set");
        assert!(desc.params.is_empty(), "no params should be inserted");
    }

    #[test]
    fn simple_packing_with_nan_values_fails_hard() {
        // Post-0.17: NaN in input to simple_packing is a hard failure.
        // The converter no longer silently downgrades to encoding="none"
        // because that hid data-quality problems from callers.
        let mut desc = mk_desc();
        let p = DataPipeline {
            encoding: "simple_packing".to_string(),
            ..Default::default()
        };
        let values = [1.0_f64, f64::NAN, 3.0];
        let err = apply_pipeline(&mut desc, Some(&values), &p, "nan_var").unwrap_err();
        assert!(
            err.contains("simple_packing"),
            "error should name the encoding: {err}"
        );
        assert!(
            err.contains("nan_var"),
            "error should name the variable: {err}"
        );
        assert!(
            err.contains("NaN"),
            "error should name the trigger kind: {err}"
        );
    }

    #[test]
    fn simple_packing_with_inf_values_fails_hard() {
        // Sibling of the NaN hard-fail.  `simple_packing::compute_params`
        // rejects Inf in input (added in 0.16); this pins that the
        // converter propagates rather than silently downgrading.
        let mut desc = mk_desc();
        let p = DataPipeline {
            encoding: "simple_packing".to_string(),
            ..Default::default()
        };
        let values = [1.0_f64, f64::INFINITY, 3.0];
        let err = apply_pipeline(&mut desc, Some(&values), &p, "inf_var").unwrap_err();
        assert!(
            err.contains("simple_packing"),
            "error should name the encoding: {err}"
        );
        assert!(
            err.contains("inf_var"),
            "error should name the variable: {err}"
        );
        // The underlying PackingError::InfiniteValue stringifies as "Inf value at index N".
        assert!(
            err.to_lowercase().contains("inf"),
            "error should name the trigger kind: {err}"
        );
    }

    #[test]
    fn simple_packing_with_finite_data_still_succeeds() {
        // Regression guard: the hard-fail must not break the happy path.
        let mut desc = mk_desc();
        let p = DataPipeline {
            encoding: "simple_packing".to_string(),
            bits: Some(12),
            ..Default::default()
        };
        let values = [1.0_f64, 2.0, 3.0, 4.0];
        apply_pipeline(&mut desc, Some(&values), &p, "clean_var").unwrap();
        assert_eq!(desc.encoding, "simple_packing");
        assert!(desc.params.contains_key("reference_value"));
    }

    #[test]
    fn simple_packing_with_non_f64_payload_still_skips_with_warning() {
        // The non-f64 branch is NOT a NaN/Inf failure — it's "variable
        // is not float64, skip simple_packing".  That branch keeps its
        // soft behaviour (stderr warning + encoding="none") because it
        // reflects a structural mismatch, not a data-quality problem.
        let mut desc = mk_desc();
        let p = DataPipeline {
            encoding: "simple_packing".to_string(),
            ..Default::default()
        };
        apply_pipeline(&mut desc, None, &p, "int_var").unwrap();
        assert_eq!(desc.encoding, "none");
    }

    #[test]
    fn unknown_encoding_errors() {
        let mut desc = mk_desc();
        let p = DataPipeline {
            encoding: "magic_packing".to_string(),
            ..Default::default()
        };
        let err = apply_pipeline(&mut desc, None, &p, "x").unwrap_err();
        assert!(err.contains("magic_packing"));
        assert!(err.contains("simple_packing"));
    }

    // ── Filter ──────────────────────────────────────────────────────

    #[test]
    fn shuffle_on_raw_f64_uses_native_byte_width() {
        let mut desc = mk_desc(); // f64 → 8 bytes
        let p = DataPipeline {
            filter: "shuffle".to_string(),
            ..Default::default()
        };
        apply_pipeline(&mut desc, None, &p, "x").unwrap();
        assert_eq!(desc.filter, "shuffle");
        assert_eq!(int_param(&desc, "shuffle_element_size"), 8);
    }

    #[test]
    fn shuffle_on_simple_packed_uses_post_pack_byte_width() {
        let mut desc = mk_desc();
        let p = DataPipeline {
            encoding: "simple_packing".to_string(),
            bits: Some(16),
            filter: "shuffle".to_string(),
            ..Default::default()
        };
        let values = [0.0_f64, 1.0, 2.0, 3.0];
        apply_pipeline(&mut desc, Some(&values), &p, "x").unwrap();
        assert_eq!(desc.filter, "shuffle");
        assert_eq!(
            int_param(&desc, "shuffle_element_size"),
            2,
            "16-bit packed → 2-byte elements"
        );
    }

    #[test]
    fn shuffle_with_24bit_packing_rounds_up() {
        let mut desc = mk_desc();
        let p = DataPipeline {
            encoding: "simple_packing".to_string(),
            bits: Some(24),
            filter: "shuffle".to_string(),
            ..Default::default()
        };
        let values = [0.0_f64, 1.0, 2.0, 3.0];
        apply_pipeline(&mut desc, Some(&values), &p, "x").unwrap();
        assert_eq!(int_param(&desc, "shuffle_element_size"), 3);
    }

    #[test]
    fn unknown_filter_errors() {
        let mut desc = mk_desc();
        let p = DataPipeline {
            filter: "wibble".to_string(),
            ..Default::default()
        };
        let err = apply_pipeline(&mut desc, None, &p, "x").unwrap_err();
        assert!(err.contains("wibble"));
    }

    // ── Compression ─────────────────────────────────────────────────

    #[test]
    fn zstd_with_default_level() {
        let mut desc = mk_desc();
        let p = DataPipeline {
            compression: "zstd".to_string(),
            ..Default::default()
        };
        apply_pipeline(&mut desc, None, &p, "x").unwrap();
        assert_eq!(desc.compression, "zstd");
        assert_eq!(int_param(&desc, "zstd_level"), 3);
    }

    #[test]
    fn zstd_with_custom_level() {
        let mut desc = mk_desc();
        let p = DataPipeline {
            compression: "zstd".to_string(),
            compression_level: Some(9),
            ..Default::default()
        };
        apply_pipeline(&mut desc, None, &p, "x").unwrap();
        assert_eq!(int_param(&desc, "zstd_level"), 9);
    }

    #[test]
    fn lz4_has_no_params() {
        let mut desc = mk_desc();
        let p = DataPipeline {
            compression: "lz4".to_string(),
            ..Default::default()
        };
        apply_pipeline(&mut desc, None, &p, "x").unwrap();
        assert_eq!(desc.compression, "lz4");
        assert!(desc.params.is_empty());
    }

    #[test]
    fn blosc2_with_custom_level() {
        let mut desc = mk_desc();
        let p = DataPipeline {
            compression: "blosc2".to_string(),
            compression_level: Some(7),
            ..Default::default()
        };
        apply_pipeline(&mut desc, None, &p, "x").unwrap();
        assert_eq!(desc.compression, "blosc2");
        assert_eq!(int_param(&desc, "blosc2_clevel"), 7);
        match desc.params.get("blosc2_codec") {
            Some(CborValue::Text(s)) => assert_eq!(s, "lz4"),
            other => panic!("blosc2_codec should be lz4: {other:?}"),
        }
    }

    #[test]
    fn szip_sets_defaults() {
        let mut desc = mk_desc();
        let p = DataPipeline {
            compression: "szip".to_string(),
            ..Default::default()
        };
        apply_pipeline(&mut desc, None, &p, "x").unwrap();
        assert_eq!(desc.compression, "szip");
        assert_eq!(int_param(&desc, "szip_rsi"), 128);
        assert_eq!(int_param(&desc, "szip_block_size"), 16);
        assert_eq!(int_param(&desc, "szip_flags"), 8);
    }

    #[test]
    fn unknown_compression_errors() {
        let mut desc = mk_desc();
        let p = DataPipeline {
            compression: "bogus".to_string(),
            ..Default::default()
        };
        let err = apply_pipeline(&mut desc, None, &p, "x").unwrap_err();
        assert!(err.contains("bogus"));
    }

    // ── Combined ────────────────────────────────────────────────────

    #[test]
    fn full_pipeline_simple_packing_shuffle_zstd() {
        let mut desc = mk_desc();
        let p = DataPipeline {
            encoding: "simple_packing".to_string(),
            bits: Some(24),
            filter: "shuffle".to_string(),
            compression: "zstd".to_string(),
            compression_level: Some(5),
        };
        let values = [1.0_f64, 2.0, 3.0, 4.0];
        apply_pipeline(&mut desc, Some(&values), &p, "x").unwrap();
        assert_eq!(desc.encoding, "simple_packing");
        assert_eq!(desc.filter, "shuffle");
        assert_eq!(desc.compression, "zstd");
        assert_eq!(int_param(&desc, "bits_per_value"), 24);
        assert_eq!(int_param(&desc, "shuffle_element_size"), 3);
        assert_eq!(int_param(&desc, "zstd_level"), 5);
    }
}
