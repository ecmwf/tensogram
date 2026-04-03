use std::collections::BTreeMap;

use crate::dtype::Dtype;
use crate::error::{Result, TensogramError};
use crate::framing;
use crate::hash::{compute_hash, HashAlgorithm};
use crate::metadata::metadata_to_cbor;
use crate::types::{HashDescriptor, Metadata, ObjectDescriptor, PayloadDescriptor};
use tensogram_encodings::pipeline::{
    self, CompressionType, EncodingType, FilterType, PipelineConfig,
};
use tensogram_encodings::simple_packing::SimplePackingParams;

/// Options for encoding.
#[derive(Debug, Clone)]
pub struct EncodeOptions {
    /// Hash algorithm to use for payload integrity. None = no hashing.
    pub hash_algorithm: Option<HashAlgorithm>,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            hash_algorithm: Some(HashAlgorithm::Xxh3),
        }
    }
}

fn validate_object(
    obj: &ObjectDescriptor,
    payload: &PayloadDescriptor,
    data_len: usize,
) -> Result<()> {
    if obj.obj_type.is_empty() {
        return Err(TensogramError::Metadata(
            "obj_type must not be empty".to_string(),
        ));
    }
    if obj.ndim as usize != obj.shape.len() {
        return Err(TensogramError::Metadata(format!(
            "ndim {} does not match shape.len() {}",
            obj.ndim,
            obj.shape.len()
        )));
    }
    if obj.strides.len() != obj.shape.len() {
        return Err(TensogramError::Metadata(format!(
            "strides.len() {} does not match shape.len() {}",
            obj.strides.len(),
            obj.shape.len()
        )));
    }
    if payload.encoding == "none" && obj.dtype.byte_width() > 0 {
        let product = obj
            .shape
            .iter()
            .try_fold(1u64, |acc, &x| acc.checked_mul(x))
            .ok_or_else(|| TensogramError::Metadata("shape product overflow".to_string()))?;
        let expected_bytes = product
            .checked_mul(obj.dtype.byte_width() as u64)
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
/// `metadata` describes all objects and their encoding pipeline.
/// `data_objects` is a list of raw data byte slices, one per object.
/// Returns the complete wire-format message.
pub fn encode(
    metadata: &Metadata,
    data_objects: &[&[u8]],
    options: &EncodeOptions,
) -> Result<Vec<u8>> {
    // Validate array lengths
    if metadata.objects.len() != metadata.payload.len() {
        return Err(TensogramError::Metadata(format!(
            "objects.len ({}) != payload.len ({})",
            metadata.objects.len(),
            metadata.payload.len()
        )));
    }
    if metadata.objects.len() != data_objects.len() {
        return Err(TensogramError::Metadata(format!(
            "objects.len ({}) != data_objects.len ({})",
            metadata.objects.len(),
            data_objects.len()
        )));
    }

    // Clone metadata so we can update hash and szip_block_offsets
    let mut metadata = metadata.clone();

    // Encode each data object through its pipeline
    let mut encoded_payloads = Vec::with_capacity(data_objects.len());
    for (i, &data) in data_objects.iter().enumerate() {
        validate_object(&metadata.objects[i], &metadata.payload[i], data.len())?;

        let payload_desc = &metadata.payload[i];
        let num_elements = metadata.objects[i].shape.iter().product::<u64>() as usize;
        let dtype = metadata.objects[i].dtype;

        let config = build_pipeline_config(payload_desc, num_elements, dtype)?;
        let result = pipeline::encode_pipeline(data, &config)
            .map_err(|e| TensogramError::Encoding(e.to_string()))?;

        // Store szip block offsets if produced
        if let Some(offsets) = &result.szip_block_offsets {
            metadata.payload[i].params.insert(
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
            metadata.payload[i].hash = Some(HashDescriptor {
                hash_type: algorithm.as_str().to_string(),
                value: hash_value,
            });
        }

        encoded_payloads.push(result.encoded_bytes);
    }

    // Serialize metadata to CBOR
    let cbor_bytes = metadata_to_cbor(&metadata)?;

    // Build the wire-format frame
    Ok(framing::encode_frame(&cbor_bytes, &encoded_payloads))
}

pub(crate) fn build_pipeline_config(
    desc: &PayloadDescriptor,
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
        "szip" => {
            let rsi = u32::try_from(get_u64_param(&desc.params, "szip_rsi")?)
                .map_err(|_| TensogramError::Metadata("szip_rsi out of u32 range".to_string()))?;
            let block_size = u32::try_from(get_u64_param(&desc.params, "szip_block_size")?)
                .map_err(|_| {
                    TensogramError::Metadata("szip_block_size out of u32 range".to_string())
                })?;
            let flags = u32::try_from(get_u64_param(&desc.params, "szip_flags")?)
                .map_err(|_| TensogramError::Metadata("szip_flags out of u32 range".to_string()))?;
            // Determine bits_per_sample based on what precedes compression:
            // - simple_packing: bits_per_value from packing params
            // - shuffle: 8 (shuffled bytes are uint8)
            // - none: dtype byte width * 8
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
    Ok(SimplePackingParams {
        reference_value: get_f64_param(params, "reference_value")?,
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

fn get_f64_param(params: &BTreeMap<String, ciborium::Value>, key: &str) -> Result<f64> {
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

fn get_i64_param(params: &BTreeMap<String, ciborium::Value>, key: &str) -> Result<i64> {
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

fn get_u64_param(params: &BTreeMap<String, ciborium::Value>, key: &str) -> Result<u64> {
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
