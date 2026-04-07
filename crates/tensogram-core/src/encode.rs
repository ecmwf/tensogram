use std::collections::BTreeMap;

use crate::dtype::Dtype;
use crate::error::{Result, TensogramError};
use crate::framing::{self, EncodedObject};
use crate::hash::{compute_hash, HashAlgorithm};
use crate::metadata::RESERVED_KEY;
use crate::types::{DataObjectDescriptor, GlobalMetadata, HashDescriptor};
#[cfg(feature = "blosc2")]
use tensogram_encodings::pipeline::Blosc2Codec;
#[cfg(feature = "sz3")]
use tensogram_encodings::pipeline::Sz3ErrorBound;
#[cfg(feature = "zfp")]
use tensogram_encodings::pipeline::ZfpMode;
use tensogram_encodings::pipeline::{
    self, CompressionType, EncodingType, FilterType, PipelineConfig,
};
use tensogram_encodings::simple_packing::SimplePackingParams;

/// Options for encoding.
#[derive(Debug, Clone)]
pub struct EncodeOptions {
    /// Hash algorithm to use for payload integrity. None = no hashing.
    pub hash_algorithm: Option<HashAlgorithm>,
    /// Reserved for future buffered-mode preceder support.
    ///
    /// Currently, setting this to `true` in buffered mode (`encode()`)
    /// returns an error — use [`StreamingEncoder::write_preceder`](crate::streaming::StreamingEncoder::write_preceder) instead.
    /// The streaming encoder ignores this field; it emits preceders only
    /// when `write_preceder()` is called explicitly.
    pub emit_preceders: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            hash_algorithm: Some(HashAlgorithm::Xxh3),
            emit_preceders: false,
        }
    }
}

pub(crate) fn validate_object(desc: &DataObjectDescriptor, data_len: usize) -> Result<()> {
    if desc.obj_type.is_empty() {
        return Err(TensogramError::Metadata(
            "obj_type must not be empty".to_string(),
        ));
    }
    if desc.ndim as usize != desc.shape.len() {
        return Err(TensogramError::Metadata(format!(
            "ndim {} does not match shape.len() {}",
            desc.ndim,
            desc.shape.len()
        )));
    }
    if desc.strides.len() != desc.shape.len() {
        return Err(TensogramError::Metadata(format!(
            "strides.len() {} does not match shape.len() {}",
            desc.strides.len(),
            desc.shape.len()
        )));
    }
    if desc.encoding == "none" && desc.dtype.byte_width() > 0 {
        let product = desc
            .shape
            .iter()
            .try_fold(1u64, |acc, &x| acc.checked_mul(x))
            .ok_or_else(|| TensogramError::Metadata("shape product overflow".to_string()))?;
        let expected_bytes = product
            .checked_mul(desc.dtype.byte_width() as u64)
            .ok_or_else(|| TensogramError::Metadata("shape product overflow".to_string()))?;
        if expected_bytes != data_len as u64 {
            return Err(TensogramError::Metadata(format!(
                "data_len {data_len} does not match expected {expected_bytes} bytes from shape and dtype"
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum EncodeMode {
    Raw,
    #[allow(dead_code)] // Will be fully wired in Task 2
    PreEncoded,
}

fn encode_inner(
    global_metadata: &GlobalMetadata,
    descriptors: &[(&DataObjectDescriptor, &[u8])],
    options: &EncodeOptions,
    mode: EncodeMode,
) -> Result<Vec<u8>> {
    // Buffered encode does not support emit_preceders — use StreamingEncoder
    // with write_preceder() instead.
    if options.emit_preceders {
        return Err(TensogramError::Encoding(
            "emit_preceders is not supported in buffered mode; use StreamingEncoder::write_preceder() instead".to_string(),
        ));
    }

    let mut encoded_objects = Vec::with_capacity(descriptors.len());

    for (desc, data) in descriptors {
        validate_object(desc, data.len())?;

        let shape_product = desc
            .shape
            .iter()
            .try_fold(1u64, |acc, &x| acc.checked_mul(x))
            .ok_or_else(|| TensogramError::Metadata("shape product overflow".to_string()))?;
        let num_elements = usize::try_from(shape_product)
            .map_err(|_| TensogramError::Metadata("element count overflows usize".to_string()))?;
        let dtype = desc.dtype;

        let config = build_pipeline_config(desc, num_elements, dtype)?;
        // For Raw: run the full encoding pipeline.
        // For PreEncoded: data bytes are already encoded — pipeline call is a
        // placeholder that will be replaced in Task 2.
        let result = match mode {
            EncodeMode::Raw | EncodeMode::PreEncoded => pipeline::encode_pipeline(data, &config)
                .map_err(|e| TensogramError::Encoding(e.to_string()))?,
        };

        // Build the final descriptor with computed fields
        let mut final_desc = (*desc).clone();

        // Store szip block offsets if produced
        if let Some(offsets) = &result.block_offsets {
            final_desc.params.insert(
                "szip_block_offsets".to_string(),
                ciborium::Value::Array(
                    offsets
                        .iter()
                        .map(|&o| ciborium::Value::Integer(o.into()))
                        .collect(),
                ),
            );
        }

        // Compute hash
        if let Some(algorithm) = options.hash_algorithm {
            let hash_value = compute_hash(&result.encoded_bytes, algorithm);
            final_desc.hash = Some(HashDescriptor {
                hash_type: algorithm.as_str().to_string(),
                value: hash_value,
            });
        }

        encoded_objects.push(EncodedObject {
            descriptor: final_desc,
            encoded_payload: result.encoded_bytes,
        });
    }

    // Validate that the caller hasn't written to _reserved_ at any level.
    validate_no_client_reserved(global_metadata)?;

    // Validate base/descriptor count: base may be shorter (auto-extended) or
    // equal, but having MORE base entries than descriptors is an error —
    // the extra entries would be silently discarded.
    if global_metadata.base.len() > descriptors.len() {
        return Err(TensogramError::Metadata(format!(
            "metadata base has {} entries but only {} descriptors provided; \
             extra base entries would be discarded",
            global_metadata.base.len(),
            descriptors.len()
        )));
    }

    // Populate per-object base entries with _reserved_.tensor (ndim/shape/strides/dtype).
    // Pre-existing application keys (e.g. "mars") are preserved.
    let mut enriched_meta = global_metadata.clone();
    populate_base_entries(&mut enriched_meta.base, &encoded_objects);
    populate_reserved_provenance(&mut enriched_meta.reserved);

    framing::encode_message(&enriched_meta, &encoded_objects)
}

/// Encode a complete Tensogram message.
///
/// `global_metadata` is the message-level metadata (version, MARS keys, etc.).
/// `descriptors` is a list of (DataObjectDescriptor, raw_data) pairs.
/// Returns the complete wire-format message.
#[tracing::instrument(skip(global_metadata, descriptors, options), fields(objects = descriptors.len()))]
pub fn encode(
    global_metadata: &GlobalMetadata,
    descriptors: &[(&DataObjectDescriptor, &[u8])],
    options: &EncodeOptions,
) -> Result<Vec<u8>> {
    encode_inner(global_metadata, descriptors, options, EncodeMode::Raw)
}

/// Validate that the caller hasn't written to `_reserved_` at any level.
///
/// The `_reserved_` namespace is library-managed.  Client code must not
/// set it in the message-level metadata or in any `base[i]` entry.
fn validate_no_client_reserved(meta: &GlobalMetadata) -> Result<()> {
    if !meta.reserved.is_empty() {
        return Err(TensogramError::Metadata(format!(
            "client code must not write to '{RESERVED_KEY}' at message level; \
             this field is populated by the library"
        )));
    }
    for (i, entry) in meta.base.iter().enumerate() {
        if entry.contains_key(RESERVED_KEY) {
            return Err(TensogramError::Metadata(format!(
                "client code must not write to '{RESERVED_KEY}' in base[{i}]; \
                 this field is populated by the library"
            )));
        }
    }
    Ok(())
}

/// Populate per-object base entries with tensor metadata under `_reserved_.tensor`.
///
/// Resizes `base` to match the object count, then inserts a `_reserved_`
/// map containing `tensor: {ndim, shape, strides, dtype}` into each entry.
/// Pre-existing application keys (e.g. `"mars"`) are preserved.
pub(crate) fn populate_base_entries(
    base: &mut Vec<BTreeMap<String, ciborium::Value>>,
    encoded_objects: &[crate::framing::EncodedObject],
) {
    use ciborium::Value;

    // Ensure base has exactly one entry per object.
    base.resize_with(encoded_objects.len(), BTreeMap::new);

    for (entry, obj) in base.iter_mut().zip(encoded_objects.iter()) {
        let desc = &obj.descriptor;

        let tensor_map = Value::Map(vec![
            (
                Value::Text("ndim".to_string()),
                Value::Integer(desc.ndim.into()),
            ),
            (
                Value::Text("shape".to_string()),
                Value::Array(
                    desc.shape
                        .iter()
                        .map(|&d| Value::Integer(d.into()))
                        .collect(),
                ),
            ),
            (
                Value::Text("strides".to_string()),
                Value::Array(
                    desc.strides
                        .iter()
                        .map(|&s| Value::Integer(s.into()))
                        .collect(),
                ),
            ),
            (
                Value::Text("dtype".to_string()),
                Value::Text(desc.dtype.to_string()),
            ),
        ]);

        let reserved_map = Value::Map(vec![(Value::Text("tensor".to_string()), tensor_map)]);

        entry.insert(RESERVED_KEY.to_string(), reserved_map);
    }
}

/// Populate the `reserved` section with provenance fields as specified in
/// `WIRE_FORMAT.md`:
///
/// - `encoder.name` — `"tensogram"`
/// - `encoder.version` — library version at encode time
/// - `time` — UTC ISO 8601 timestamp
/// - `uuid` — RFC 4122 v4 UUID
///
/// Pre-existing keys in `reserved` are preserved; only these four are
/// set (or overwritten).
pub(crate) fn populate_reserved_provenance(reserved: &mut BTreeMap<String, ciborium::Value>) {
    use ciborium::Value;
    use std::time::SystemTime;

    // encoder.name + encoder.version
    let version_str = env!("CARGO_PKG_VERSION");
    let encoder_map = Value::Map(vec![
        (
            Value::Text("name".to_string()),
            Value::Text("tensogram".to_string()),
        ),
        (
            Value::Text("version".to_string()),
            Value::Text(version_str.to_string()),
        ),
    ]);
    reserved.insert("encoder".to_string(), encoder_map);

    // time — ISO 8601 UTC
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    // Simple UTC format: YYYY-MM-DDThh:mm:ssZ
    // We compute from epoch seconds to avoid adding a datetime crate.
    let days = secs / 86400;
    let day_secs = secs % 86400;
    let hours = day_secs / 3600;
    let minutes = (day_secs % 3600) / 60;
    let seconds = day_secs % 60;
    // Civil date from days since 1970-01-01 (Howard Hinnant algorithm)
    let (y, m, d) = civil_from_days(days as i64);
    let timestamp = format!("{y:04}-{m:02}-{d:02}T{hours:02}:{minutes:02}:{seconds:02}Z");
    reserved.insert("time".to_string(), Value::Text(timestamp));

    // uuid — RFC 4122 v4
    let id = uuid::Uuid::new_v4();
    reserved.insert("uuid".to_string(), Value::Text(id.to_string()));
}

/// Convert days since 1970-01-01 to (year, month, day).
/// Howard Hinnant's algorithm (public domain).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    // doe (day of era) is guaranteed in [0, 146096] by the era computation,
    // so the u32 cast cannot truncate.
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

pub(crate) fn build_pipeline_config(
    desc: &DataObjectDescriptor,
    num_values: usize,
    dtype: Dtype,
) -> Result<PipelineConfig> {
    let encoding = match desc.encoding.as_str() {
        "none" => EncodingType::None,
        "simple_packing" => {
            if dtype.byte_width() != 8 {
                return Err(TensogramError::Encoding(
                    "simple_packing only supports float64 dtype".to_string(),
                ));
            }
            let params = extract_simple_packing_params(&desc.params)?;
            EncodingType::SimplePacking(params)
        }
        other => {
            return Err(TensogramError::Encoding(format!(
                "unknown encoding: {other}"
            )))
        }
    };

    let filter = match desc.filter.as_str() {
        "none" => FilterType::None,
        "shuffle" => {
            let element_size = usize::try_from(get_u64_param(
                &desc.params,
                "shuffle_element_size",
            )?)
            .map_err(|_| {
                TensogramError::Metadata("shuffle_element_size out of usize range".to_string())
            })?;
            FilterType::Shuffle { element_size }
        }
        other => return Err(TensogramError::Encoding(format!("unknown filter: {other}"))),
    };

    let compression = match desc.compression.as_str() {
        "none" => CompressionType::None,
        #[cfg(feature = "szip")]
        "szip" => {
            let rsi = u32::try_from(get_u64_param(&desc.params, "szip_rsi")?)
                .map_err(|_| TensogramError::Metadata("szip_rsi out of u32 range".to_string()))?;
            let block_size = u32::try_from(get_u64_param(&desc.params, "szip_block_size")?)
                .map_err(|_| {
                    TensogramError::Metadata("szip_block_size out of u32 range".to_string())
                })?;
            let flags = u32::try_from(get_u64_param(&desc.params, "szip_flags")?)
                .map_err(|_| TensogramError::Metadata("szip_flags out of u32 range".to_string()))?;
            let bits_per_sample = match (&encoding, &filter) {
                (EncodingType::SimplePacking(params), _) => params.bits_per_value,
                (EncodingType::None, FilterType::Shuffle { .. }) => 8,
                (EncodingType::None, FilterType::None) => (dtype.byte_width() * 8) as u32,
            };
            CompressionType::Szip {
                rsi,
                block_size,
                flags,
                bits_per_sample,
            }
        }
        #[cfg(feature = "zstd")]
        "zstd" => {
            let level_i64 = get_i64_param(&desc.params, "zstd_level").unwrap_or(3);
            let level = i32::try_from(level_i64).map_err(|_| {
                TensogramError::Metadata(format!("zstd_level value {level_i64} out of i32 range"))
            })?;
            CompressionType::Zstd { level }
        }
        #[cfg(feature = "lz4")]
        "lz4" => CompressionType::Lz4,
        #[cfg(feature = "blosc2")]
        "blosc2" => {
            let codec_str = match desc.params.get("blosc2_codec") {
                Some(ciborium::Value::Text(s)) => s.as_str(),
                _ => "lz4",
            };
            let codec = match codec_str {
                "blosclz" => Blosc2Codec::Blosclz,
                "lz4" => Blosc2Codec::Lz4,
                "lz4hc" => Blosc2Codec::Lz4hc,
                "zlib" => Blosc2Codec::Zlib,
                "zstd" => Blosc2Codec::Zstd,
                other => {
                    return Err(TensogramError::Encoding(format!(
                        "unknown blosc2 codec: {other}"
                    )))
                }
            };
            let clevel_i64 = get_i64_param(&desc.params, "blosc2_clevel").unwrap_or(5);
            let clevel = i32::try_from(clevel_i64).map_err(|_| {
                TensogramError::Metadata(format!(
                    "blosc2_clevel value {clevel_i64} out of i32 range"
                ))
            })?;
            let typesize = match (&encoding, &filter) {
                (EncodingType::SimplePacking(params), _) => {
                    (params.bits_per_value as usize).div_ceil(8)
                }
                (EncodingType::None, FilterType::Shuffle { .. }) => 1,
                (EncodingType::None, FilterType::None) => dtype.byte_width(),
            };
            CompressionType::Blosc2 {
                codec,
                clevel,
                typesize,
            }
        }
        #[cfg(feature = "zfp")]
        "zfp" => {
            let mode_str = match desc.params.get("zfp_mode") {
                Some(ciborium::Value::Text(s)) => s.clone(),
                _ => {
                    return Err(TensogramError::Metadata(
                        "missing required parameter: zfp_mode".to_string(),
                    ))
                }
            };
            let mode = match mode_str.as_str() {
                "fixed_rate" => {
                    let rate = get_f64_param(&desc.params, "zfp_rate")?;
                    ZfpMode::FixedRate { rate }
                }
                "fixed_precision" => {
                    let precision = u32::try_from(get_u64_param(&desc.params, "zfp_precision")?)
                        .map_err(|_| {
                            TensogramError::Metadata("zfp_precision out of u32 range".to_string())
                        })?;
                    ZfpMode::FixedPrecision { precision }
                }
                "fixed_accuracy" => {
                    let tolerance = get_f64_param(&desc.params, "zfp_tolerance")?;
                    ZfpMode::FixedAccuracy { tolerance }
                }
                other => {
                    return Err(TensogramError::Encoding(format!(
                        "unknown zfp_mode: {other}"
                    )))
                }
            };
            CompressionType::Zfp { mode }
        }
        #[cfg(feature = "sz3")]
        "sz3" => {
            let mode_str = match desc.params.get("sz3_error_bound_mode") {
                Some(ciborium::Value::Text(s)) => s.clone(),
                _ => {
                    return Err(TensogramError::Metadata(
                        "missing required parameter: sz3_error_bound_mode".to_string(),
                    ))
                }
            };
            let bound_val = get_f64_param(&desc.params, "sz3_error_bound")?;
            let error_bound = match mode_str.as_str() {
                "abs" => Sz3ErrorBound::Absolute(bound_val),
                "rel" => Sz3ErrorBound::Relative(bound_val),
                "psnr" => Sz3ErrorBound::Psnr(bound_val),
                other => {
                    return Err(TensogramError::Encoding(format!(
                        "unknown sz3_error_bound_mode: {other}"
                    )))
                }
            };
            CompressionType::Sz3 { error_bound }
        }
        other => {
            return Err(TensogramError::Encoding(format!(
                "unknown compression: {other}"
            )))
        }
    };

    Ok(PipelineConfig {
        encoding,
        filter,
        compression,
        num_values,
        byte_order: desc.byte_order,
        dtype_byte_width: dtype.byte_width(),
    })
}

fn extract_simple_packing_params(
    params: &BTreeMap<String, ciborium::Value>,
) -> Result<SimplePackingParams> {
    let reference_value = get_f64_param(params, "reference_value")?;
    if reference_value.is_nan() || reference_value.is_infinite() {
        return Err(TensogramError::Metadata(format!(
            "reference_value must be finite, got {reference_value}"
        )));
    }
    Ok(SimplePackingParams {
        reference_value,
        binary_scale_factor: i32::try_from(get_i64_param(params, "binary_scale_factor")?).map_err(
            |_| TensogramError::Metadata("binary_scale_factor out of i32 range".to_string()),
        )?,
        decimal_scale_factor: i32::try_from(get_i64_param(params, "decimal_scale_factor")?)
            .map_err(|_| {
                TensogramError::Metadata("decimal_scale_factor out of i32 range".to_string())
            })?,
        bits_per_value: u32::try_from(get_u64_param(params, "bits_per_value")?)
            .map_err(|_| TensogramError::Metadata("bits_per_value out of u32 range".to_string()))?,
    })
}

pub(crate) fn get_f64_param(params: &BTreeMap<String, ciborium::Value>, key: &str) -> Result<f64> {
    match params.get(key) {
        Some(ciborium::Value::Float(f)) => Ok(*f),
        Some(ciborium::Value::Integer(i)) => {
            // i128 → f64 may lose precision for very large integers (> 2^53),
            // but this is acceptable for a float accessor on an integer value.
            let n: i128 = (*i).into();
            Ok(n as f64)
        }
        Some(other) => Err(TensogramError::Metadata(format!(
            "expected number for {key}, got {other:?}"
        ))),
        None => Err(TensogramError::Metadata(format!(
            "missing required parameter: {key}"
        ))),
    }
}

pub(crate) fn get_i64_param(params: &BTreeMap<String, ciborium::Value>, key: &str) -> Result<i64> {
    match params.get(key) {
        Some(ciborium::Value::Integer(i)) => {
            let n: i128 = (*i).into();
            i64::try_from(n).map_err(|_| {
                TensogramError::Metadata(format!("integer value {n} out of i64 range for {key}"))
            })
        }
        Some(other) => Err(TensogramError::Metadata(format!(
            "expected integer for {key}, got {other:?}"
        ))),
        None => Err(TensogramError::Metadata(format!(
            "missing required parameter: {key}"
        ))),
    }
}

pub(crate) fn get_u64_param(params: &BTreeMap<String, ciborium::Value>, key: &str) -> Result<u64> {
    match params.get(key) {
        Some(ciborium::Value::Integer(i)) => {
            let n: i128 = (*i).into();
            u64::try_from(n).map_err(|_| {
                TensogramError::Metadata(format!("integer value {n} out of u64 range for {key}"))
            })
        }
        Some(other) => Err(TensogramError::Metadata(format!(
            "expected integer for {key}, got {other:?}"
        ))),
        None => Err(TensogramError::Metadata(format!(
            "missing required parameter: {key}"
        ))),
    }
}

// ── Edge case tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{decode, DecodeOptions};
    use crate::types::{ByteOrder, GlobalMetadata};
    use std::collections::BTreeMap;

    fn make_descriptor(shape: Vec<u64>) -> DataObjectDescriptor {
        let strides = {
            let mut s = vec![1u64; shape.len()];
            for i in (0..shape.len().saturating_sub(1)).rev() {
                s[i] = s[i + 1] * shape[i + 1];
            }
            s
        };
        DataObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: shape.len() as u64,
            shape,
            strides,
            dtype: Dtype::Float32,
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        }
    }

    // ── Category 1: base array mismatches ────────────────────────────────

    #[test]
    fn test_base_more_entries_than_descriptors_rejected() {
        // base has 5 entries but only 2 descriptors — should error.
        let meta = GlobalMetadata {
            version: 2,
            base: vec![
                BTreeMap::new(),
                BTreeMap::new(),
                BTreeMap::new(),
                BTreeMap::new(),
                BTreeMap::new(),
            ],
            ..Default::default()
        };
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        let result = encode(
            &meta,
            &[(&desc, data.as_slice()), (&desc, data.as_slice())],
            &options,
        );
        assert!(
            result.is_err(),
            "5 base entries with 2 descriptors should fail"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("5") && err.contains("2"),
            "error should mention counts: {err}"
        );
    }

    #[test]
    fn test_base_fewer_entries_than_descriptors_auto_extended() {
        // base has 0 entries but 3 descriptors — auto-extends, _reserved_ inserted.
        let meta = GlobalMetadata {
            version: 2,
            base: vec![],
            ..Default::default()
        };
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        let msg = encode(
            &meta,
            &[
                (&desc, data.as_slice()),
                (&desc, data.as_slice()),
                (&desc, data.as_slice()),
            ],
            &options,
        )
        .unwrap();

        let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();
        assert_eq!(decoded.base.len(), 3);
        // Each entry should have _reserved_ with tensor info
        for entry in &decoded.base {
            assert!(
                entry.contains_key("_reserved_"),
                "auto-extended base entry should have _reserved_"
            );
        }
    }

    #[test]
    fn test_base_entry_with_top_level_key_names_no_collision() {
        // base[0] contains a key named "version" — no collision with top-level version.
        let mut entry = BTreeMap::new();
        entry.insert(
            "version".to_string(),
            ciborium::Value::Text("my-version".to_string()),
        );
        entry.insert(
            "base".to_string(),
            ciborium::Value::Text("not-the-real-base".to_string()),
        );
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entry],
            ..Default::default()
        };
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        let msg = encode(&meta, &[(&desc, data.as_slice())], &options).unwrap();
        let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();

        // Top-level version is still 2
        assert_eq!(decoded.version, 2);
        // base[0] should have both custom keys preserved
        assert_eq!(
            decoded.base[0].get("version"),
            Some(&ciborium::Value::Text("my-version".to_string()))
        );
        assert_eq!(
            decoded.base[0].get("base"),
            Some(&ciborium::Value::Text("not-the-real-base".to_string()))
        );
    }

    #[test]
    fn test_base_entry_with_deeply_nested_reserved_allowed() {
        // Only top-level _reserved_ in base[i] should be rejected.
        // Deeply nested _reserved_ (like {"foo": {"_reserved_": ...}}) is fine.
        let nested = ciborium::Value::Map(vec![(
            ciborium::Value::Text("_reserved_".to_string()),
            ciborium::Value::Text("nested-is-ok".to_string()),
        )]);
        let mut entry = BTreeMap::new();
        entry.insert("foo".to_string(), nested);
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entry],
            ..Default::default()
        };
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        // Should succeed — only top-level _reserved_ is rejected
        let msg = encode(&meta, &[(&desc, data.as_slice())], &options).unwrap();
        let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();
        // The nested _reserved_ should survive
        let foo = decoded.base[0].get("foo").unwrap();
        if let ciborium::Value::Map(pairs) = foo {
            assert_eq!(pairs.len(), 1);
        } else {
            panic!("expected map for foo");
        }
    }

    // ── Category 2: _reserved_ edge cases ────────────────────────────────

    #[test]
    fn test_reserved_rejected_at_message_level() {
        let mut reserved = BTreeMap::new();
        reserved.insert(
            "rogue".to_string(),
            ciborium::Value::Text("bad".to_string()),
        );
        let meta = GlobalMetadata {
            version: 2,
            reserved,
            ..Default::default()
        };
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let result = encode(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("_reserved_") && err.contains("message level"),
            "error: {err}"
        );
    }

    #[test]
    fn test_reserved_rejected_in_base_entry() {
        let mut entry = BTreeMap::new();
        entry.insert("_reserved_".to_string(), ciborium::Value::Map(vec![]));
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entry],
            ..Default::default()
        };
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let result = encode(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("_reserved_") && err.contains("base[0]"),
            "error: {err}"
        );
    }

    #[test]
    fn test_reserved_tensor_has_four_keys_after_encode() {
        let meta = GlobalMetadata::default();
        let desc = make_descriptor(vec![3, 4]);
        let data = vec![0u8; 3 * 4 * 4]; // 3*4 float32
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        let msg = encode(&meta, &[(&desc, data.as_slice())], &options).unwrap();
        let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();

        let reserved = decoded.base[0]
            .get("_reserved_")
            .expect("_reserved_ missing");
        if let ciborium::Value::Map(pairs) = reserved {
            // Should have "tensor" key
            let tensor_entry = pairs
                .iter()
                .find(|(k, _)| *k == ciborium::Value::Text("tensor".to_string()));
            assert!(tensor_entry.is_some(), "missing tensor key in _reserved_");
            if let Some((_, ciborium::Value::Map(tensor_pairs))) = tensor_entry {
                let keys: Vec<String> = tensor_pairs
                    .iter()
                    .filter_map(|(k, _)| {
                        if let ciborium::Value::Text(s) = k {
                            Some(s.clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                assert_eq!(keys.len(), 4, "tensor should have 4 keys, got: {keys:?}");
                assert!(keys.contains(&"ndim".to_string()));
                assert!(keys.contains(&"shape".to_string()));
                assert!(keys.contains(&"strides".to_string()));
                assert!(keys.contains(&"dtype".to_string()));
            } else {
                panic!("tensor is not a map");
            }
        } else {
            panic!("_reserved_ is not a map");
        }
    }

    #[test]
    fn test_reserved_tensor_scalar_ndim_zero() {
        // Scalar: ndim=0, shape=[], strides=[]
        let desc = DataObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 0,
            shape: vec![],
            strides: vec![],
            dtype: Dtype::Float32,
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        };
        let data = vec![0u8; 4]; // 1 float32
        let meta = GlobalMetadata::default();
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        let msg = encode(&meta, &[(&desc, data.as_slice())], &options).unwrap();
        let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();

        let reserved = decoded.base[0]
            .get("_reserved_")
            .expect("_reserved_ missing");
        if let ciborium::Value::Map(pairs) = reserved {
            let tensor_entry = pairs
                .iter()
                .find(|(k, _)| *k == ciborium::Value::Text("tensor".to_string()));
            if let Some((_, ciborium::Value::Map(tensor_pairs))) = tensor_entry {
                // ndim should be 0
                let ndim = tensor_pairs
                    .iter()
                    .find(|(k, _)| *k == ciborium::Value::Text("ndim".to_string()));
                assert!(
                    matches!(ndim, Some((_, ciborium::Value::Integer(i))) if i128::from(*i) == 0),
                    "ndim should be 0 for scalar"
                );
                // shape should be empty array
                let shape = tensor_pairs
                    .iter()
                    .find(|(k, _)| *k == ciborium::Value::Text("shape".to_string()));
                assert!(
                    matches!(shape, Some((_, ciborium::Value::Array(a))) if a.is_empty()),
                    "shape should be [] for scalar"
                );
            } else {
                panic!("tensor missing or not a map");
            }
        } else {
            panic!("_reserved_ is not a map");
        }
    }

    // ── Category 3: _extra_ edge cases ───────────────────────────────────

    #[test]
    fn test_extra_with_keys_colliding_with_base_entry_keys() {
        // _extra_ has key "mars", base[0] also has key "mars" — different scopes, both survive
        let mut entry = BTreeMap::new();
        entry.insert(
            "mars".to_string(),
            ciborium::Value::Text("base-mars".to_string()),
        );
        let mut extra = BTreeMap::new();
        extra.insert(
            "mars".to_string(),
            ciborium::Value::Text("extra-mars".to_string()),
        );
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entry],
            extra,
            ..Default::default()
        };
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        let msg = encode(&meta, &[(&desc, data.as_slice())], &options).unwrap();
        let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();

        assert_eq!(
            decoded.base[0].get("mars"),
            Some(&ciborium::Value::Text("base-mars".to_string()))
        );
        assert_eq!(
            decoded.extra.get("mars"),
            Some(&ciborium::Value::Text("extra-mars".to_string()))
        );
    }

    #[test]
    fn test_empty_extra_omitted_from_cbor() {
        let meta = GlobalMetadata {
            version: 2,
            extra: BTreeMap::new(),
            ..Default::default()
        };
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        let msg = encode(&meta, &[(&desc, data.as_slice())], &options).unwrap();
        let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();
        assert!(decoded.extra.is_empty());
    }

    #[test]
    fn test_extra_with_nested_maps_round_trips() {
        let nested = ciborium::Value::Map(vec![
            (
                ciborium::Value::Text("key1".to_string()),
                ciborium::Value::Integer(42.into()),
            ),
            (
                ciborium::Value::Text("key2".to_string()),
                ciborium::Value::Map(vec![(
                    ciborium::Value::Text("deep".to_string()),
                    ciborium::Value::Bool(true),
                )]),
            ),
        ]);
        let mut extra = BTreeMap::new();
        extra.insert("nested".to_string(), nested.clone());
        let meta = GlobalMetadata {
            version: 2,
            extra,
            ..Default::default()
        };
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        let msg = encode(&meta, &[(&desc, data.as_slice())], &options).unwrap();
        let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();
        // Nested maps should round-trip
        assert!(decoded.extra.contains_key("nested"));
    }

    // ── Category 4: Serde deserialization ────────────────────────────────

    #[test]
    fn test_old_common_payload_keys_silently_ignored() {
        // Simulate an old v2 message with "common" and "payload" keys at top level.
        // GlobalMetadata uses `deny_unknown_fields` is NOT set (serde default),
        // so unknown keys should be silently ignored.
        use ciborium::Value;
        let cbor = Value::Map(vec![
            (Value::Text("version".to_string()), Value::Integer(2.into())),
            (Value::Text("common".to_string()), Value::Map(vec![])),
            (Value::Text("payload".to_string()), Value::Array(vec![])),
        ]);
        let mut bytes = Vec::new();
        ciborium::into_writer(&cbor, &mut bytes).unwrap();

        let decoded: GlobalMetadata = crate::metadata::cbor_to_global_metadata(&bytes).unwrap();
        assert_eq!(decoded.version, 2);
        assert!(decoded.base.is_empty());
        assert!(decoded.extra.is_empty());
        assert!(decoded.reserved.is_empty());
    }

    #[test]
    fn test_old_reserved_key_name_ignored() {
        // "reserved" (old name) should be ignored, only "_reserved_" is captured.
        use ciborium::Value;
        let cbor = Value::Map(vec![
            (Value::Text("version".to_string()), Value::Integer(2.into())),
            (
                Value::Text("reserved".to_string()),
                Value::Map(vec![(
                    Value::Text("rogue".to_string()),
                    Value::Text("value".to_string()),
                )]),
            ),
        ]);
        let mut bytes = Vec::new();
        ciborium::into_writer(&cbor, &mut bytes).unwrap();

        let decoded: GlobalMetadata = crate::metadata::cbor_to_global_metadata(&bytes).unwrap();
        assert!(
            decoded.reserved.is_empty(),
            "old 'reserved' key should be ignored"
        );
    }

    // ── Category 4b: validate_no_client_reserved — multi-entry base ────

    #[test]
    fn test_reserved_rejected_in_second_base_entry_only() {
        // base[0] is clean, base[1] has _reserved_ → should fail, mentioning base[1]
        let mut entry0 = BTreeMap::new();
        entry0.insert("clean".to_string(), ciborium::Value::Text("ok".to_string()));
        let mut entry1 = BTreeMap::new();
        entry1.insert("_reserved_".to_string(), ciborium::Value::Map(vec![]));
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entry0, entry1],
            ..Default::default()
        };
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let result = encode(
            &meta,
            &[(&desc, data.as_slice()), (&desc, data.as_slice())],
            &EncodeOptions::default(),
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("base[1]"),
            "error should mention base[1]: {err}"
        );
    }

    #[test]
    fn test_reserved_accepted_when_all_base_entries_clean() {
        // Multiple base entries, none have _reserved_ → should succeed
        let mut e0 = BTreeMap::new();
        e0.insert(
            "key0".to_string(),
            ciborium::Value::Text("val0".to_string()),
        );
        let mut e1 = BTreeMap::new();
        e1.insert(
            "key1".to_string(),
            ciborium::Value::Text("val1".to_string()),
        );
        let meta = GlobalMetadata {
            version: 2,
            base: vec![e0, e1],
            ..Default::default()
        };
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        let msg = encode(
            &meta,
            &[(&desc, data.as_slice()), (&desc, data.as_slice())],
            &options,
        )
        .unwrap();
        let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();
        assert_eq!(decoded.base.len(), 2);
        assert!(decoded.base[0].contains_key("key0"));
        assert!(decoded.base[1].contains_key("key1"));
    }

    // ── Category 5: populate_base_entries — all dtypes ───────────────────

    #[test]
    fn test_reserved_tensor_dtype_strings_for_all_dtypes() {
        // Verify that _reserved_.tensor.dtype string is correct for every Dtype variant
        let dtypes_and_expected = [
            (Dtype::Float16, "float16"),
            (Dtype::Bfloat16, "bfloat16"),
            (Dtype::Float32, "float32"),
            (Dtype::Float64, "float64"),
            (Dtype::Complex64, "complex64"),
            (Dtype::Complex128, "complex128"),
            (Dtype::Int8, "int8"),
            (Dtype::Int16, "int16"),
            (Dtype::Int32, "int32"),
            (Dtype::Int64, "int64"),
            (Dtype::Uint8, "uint8"),
            (Dtype::Uint16, "uint16"),
            (Dtype::Uint32, "uint32"),
            (Dtype::Uint64, "uint64"),
        ];

        for (dtype, expected_str) in dtypes_and_expected {
            let byte_width = dtype.byte_width();
            let num_elements: u64 = 4;
            let data_len = num_elements as usize * byte_width;

            let desc = DataObjectDescriptor {
                obj_type: "ntensor".to_string(),
                ndim: 1,
                shape: vec![num_elements],
                strides: vec![1],
                dtype,
                byte_order: ByteOrder::Big,
                encoding: "none".to_string(),
                filter: "none".to_string(),
                compression: "none".to_string(),
                params: BTreeMap::new(),
                hash: None,
            };
            let data = vec![0u8; data_len];
            let meta = GlobalMetadata::default();
            let options = EncodeOptions {
                hash_algorithm: None,
                ..Default::default()
            };
            let msg = encode(&meta, &[(&desc, data.as_slice())], &options).unwrap();
            let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();

            let reserved = decoded.base[0]
                .get("_reserved_")
                .unwrap_or_else(|| panic!("_reserved_ missing for dtype {dtype}"));
            if let ciborium::Value::Map(pairs) = reserved {
                let tensor_entry = pairs
                    .iter()
                    .find(|(k, _)| *k == ciborium::Value::Text("tensor".to_string()));
                if let Some((_, ciborium::Value::Map(tensor_pairs))) = tensor_entry {
                    let dtype_val = tensor_pairs
                        .iter()
                        .find(|(k, _)| *k == ciborium::Value::Text("dtype".to_string()));
                    assert!(
                        matches!(
                            dtype_val,
                            Some((_, ciborium::Value::Text(s))) if s == expected_str
                        ),
                        "dtype for {dtype} should be '{expected_str}', got: {dtype_val:?}"
                    );
                } else {
                    panic!("tensor missing or not a map for dtype {dtype}");
                }
            } else {
                panic!("_reserved_ is not a map for dtype {dtype}");
            }
        }
    }

    // ── Category 6: GlobalMetadata serde with all fields ─────────────────

    #[test]
    fn test_global_metadata_serde_all_fields_populated() {
        // base + reserved + extra all populated — verify CBOR round-trip
        use ciborium::Value;

        let mut base_entry = BTreeMap::new();
        base_entry.insert("key".to_string(), Value::Text("base_val".to_string()));
        let mut reserved = BTreeMap::new();
        reserved.insert("encoder".to_string(), Value::Text("test".to_string()));
        let mut extra = BTreeMap::new();
        extra.insert("custom".to_string(), Value::Integer(42.into()));

        let meta = GlobalMetadata {
            version: 2,
            base: vec![base_entry],
            reserved,
            extra,
        };

        // Serialize to CBOR and back
        let cbor_bytes = crate::metadata::global_metadata_to_cbor(&meta).unwrap();
        let decoded: GlobalMetadata =
            crate::metadata::cbor_to_global_metadata(&cbor_bytes).unwrap();

        assert_eq!(decoded.version, 2);
        assert_eq!(decoded.base.len(), 1);
        assert_eq!(
            decoded.base[0].get("key"),
            Some(&Value::Text("base_val".to_string()))
        );
        assert!(decoded.reserved.contains_key("encoder"));
        assert_eq!(
            decoded.extra.get("custom"),
            Some(&Value::Integer(42.into()))
        );
    }

    // ── Category 7: populate_reserved_provenance ─────────────────────────

    #[test]
    fn test_provenance_fields_present_after_encode() {
        let meta = GlobalMetadata::default();
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        let msg = encode(&meta, &[(&desc, data.as_slice())], &options).unwrap();
        let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();

        // Message-level reserved should have encoder, time, uuid
        assert!(decoded.reserved.contains_key("encoder"));
        assert!(decoded.reserved.contains_key("time"));
        assert!(decoded.reserved.contains_key("uuid"));

        // encoder should contain name and version
        if let ciborium::Value::Map(pairs) = decoded.reserved.get("encoder").unwrap() {
            let has_name = pairs
                .iter()
                .any(|(k, _)| *k == ciborium::Value::Text("name".to_string()));
            let has_version = pairs
                .iter()
                .any(|(k, _)| *k == ciborium::Value::Text("version".to_string()));
            assert!(has_name, "encoder map should have 'name' key");
            assert!(has_version, "encoder map should have 'version' key");
        } else {
            panic!("encoder should be a map");
        }

        // uuid should be a valid UUID string (36 chars with hyphens)
        if let ciborium::Value::Text(uuid_str) = decoded.reserved.get("uuid").unwrap() {
            assert_eq!(uuid_str.len(), 36, "UUID should be 36 chars: {uuid_str}");
            assert_eq!(
                uuid_str.chars().filter(|c| *c == '-').count(),
                4,
                "UUID should have 4 hyphens: {uuid_str}"
            );
        } else {
            panic!("uuid should be a text");
        }

        // time should be an ISO 8601 timestamp ending with Z
        if let ciborium::Value::Text(time_str) = decoded.reserved.get("time").unwrap() {
            assert!(
                time_str.ends_with('Z'),
                "time should end with Z: {time_str}"
            );
            assert!(
                time_str.contains('T'),
                "time should contain T separator: {time_str}"
            );
        } else {
            panic!("time should be a text");
        }
    }

    #[test]
    fn test_both_reserved_and_reserved_underscore_only_new_captured() {
        // Both "reserved" and "_reserved_" present — only "_reserved_" should be captured.
        use ciborium::Value;
        let cbor = Value::Map(vec![
            (
                Value::Text("_reserved_".to_string()),
                Value::Map(vec![(
                    Value::Text("encoder".to_string()),
                    Value::Text("tensogram".to_string()),
                )]),
            ),
            (
                Value::Text("reserved".to_string()),
                Value::Map(vec![(
                    Value::Text("old".to_string()),
                    Value::Text("ignored".to_string()),
                )]),
            ),
            (Value::Text("version".to_string()), Value::Integer(2.into())),
        ]);
        let mut bytes = Vec::new();
        ciborium::into_writer(&cbor, &mut bytes).unwrap();

        let decoded: GlobalMetadata = crate::metadata::cbor_to_global_metadata(&bytes).unwrap();
        assert!(decoded.reserved.contains_key("encoder"));
        assert!(!decoded.reserved.contains_key("old"));
    }
}
