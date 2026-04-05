use std::collections::BTreeMap;

use crate::dtype::Dtype;
use crate::error::{Result, TensogramError};
use crate::framing::{self, EncodedObject};
use crate::hash::{compute_hash, HashAlgorithm};
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
    /// returns an error — use [`StreamingEncoder::write_preceder`] instead.
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

/// Encode a complete Tensogram message.
///
/// `global_metadata` is the message-level metadata (version, MARS keys, etc.).
/// `descriptors` is a list of (DataObjectDescriptor, raw_data) pairs.
/// Returns the complete wire-format message.
pub fn encode(
    global_metadata: &GlobalMetadata,
    descriptors: &[(&DataObjectDescriptor, &[u8])],
    options: &EncodeOptions,
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

        let num_elements = usize::try_from(desc.shape.iter().product::<u64>())
            .map_err(|_| TensogramError::Metadata("element count overflows usize".to_string()))?;
        let dtype = desc.dtype;

        let config = build_pipeline_config(desc, num_elements, dtype)?;
        let result = pipeline::encode_pipeline(data, &config)
            .map_err(|e| TensogramError::Encoding(e.to_string()))?;

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

    // Populate the per-object payload entries with ndim/shape/strides/dtype.
    // Pre-existing application keys (e.g. "mars") are preserved.
    // Structural keys (ndim/shape/strides/dtype) are authoritative and always
    // overwritten by the encoder to match the actual encoded objects.
    let mut enriched_meta = global_metadata.clone();
    populate_payload_entries(&mut enriched_meta.payload, &encoded_objects);

    framing::encode_message(&enriched_meta, &encoded_objects)
}

/// Populate per-object payload entries with structural tensor metadata.
///
/// Resizes `payload` to match the object count, then inserts `ndim`,
/// `shape`, `strides`, and `dtype` into each entry.  These structural
/// keys are authoritative and always overwritten.  Pre-existing
/// application keys (e.g. `"mars"`) are preserved.
pub(crate) fn populate_payload_entries(
    payload: &mut Vec<BTreeMap<String, ciborium::Value>>,
    encoded_objects: &[crate::framing::EncodedObject],
) {
    // Ensure payload has exactly one entry per object.
    payload.resize_with(encoded_objects.len(), BTreeMap::new);

    for (entry, obj) in payload.iter_mut().zip(encoded_objects.iter()) {
        let desc = &obj.descriptor;
        entry.insert(
            "ndim".to_string(),
            ciborium::Value::Integer(desc.ndim.into()),
        );
        entry.insert(
            "shape".to_string(),
            ciborium::Value::Array(
                desc.shape
                    .iter()
                    .map(|&d| ciborium::Value::Integer(d.into()))
                    .collect(),
            ),
        );
        entry.insert(
            "strides".to_string(),
            ciborium::Value::Array(
                desc.strides
                    .iter()
                    .map(|&s| ciborium::Value::Integer(s.into()))
                    .collect(),
            ),
        );
        entry.insert(
            "dtype".to_string(),
            ciborium::Value::Text(desc.dtype.to_string()),
        );
    }
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
            let level = get_i64_param(&desc.params, "zstd_level").unwrap_or(3) as i32;
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
            let clevel = get_i64_param(&desc.params, "blosc2_clevel").unwrap_or(5) as i32;
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
