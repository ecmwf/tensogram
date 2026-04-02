use std::collections::BTreeMap;

use crate::error::{Result, TensogramError};
use crate::framing;
use crate::hash::{compute_hash, HashAlgorithm};
use crate::metadata::metadata_to_cbor;
use crate::types::{HashDescriptor, Metadata, PayloadDescriptor};
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
        let payload_desc = &metadata.payload[i];
        let num_elements = metadata.objects[i].shape.iter().product::<u64>() as usize;

        let config = build_pipeline_config(payload_desc, num_elements)?;
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

/// Build a PipelineConfig from a PayloadDescriptor.
pub(crate) fn build_pipeline_config(
    desc: &PayloadDescriptor,
    num_values: usize,
) -> Result<PipelineConfig> {
    let encoding = match desc.encoding.as_str() {
        "none" => EncodingType::None,
        "simple_packing" => {
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
            let element_size = get_u64_param(&desc.params, "shuffle_element_size")? as usize;
            FilterType::Shuffle { element_size }
        }
        other => return Err(TensogramError::Encoding(format!("unknown filter: {other}"))),
    };

    let compression = match desc.compression.as_str() {
        "none" => CompressionType::None,
        "szip" => {
            let rsi = get_u64_param(&desc.params, "szip_rsi")? as u32;
            let block_size = get_u64_param(&desc.params, "szip_block_size")? as u32;
            let flags = get_u64_param(&desc.params, "szip_flags")? as u32;
            CompressionType::Szip {
                rsi,
                block_size,
                flags,
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
    })
}

fn extract_simple_packing_params(
    params: &BTreeMap<String, ciborium::Value>,
) -> Result<SimplePackingParams> {
    Ok(SimplePackingParams {
        reference_value: get_f64_param(params, "reference_value")?,
        binary_scale_factor: get_i64_param(params, "binary_scale_factor")? as i32,
        decimal_scale_factor: get_i64_param(params, "decimal_scale_factor")? as i32,
        bits_per_value: get_u64_param(params, "bits_per_value")? as u32,
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
            Ok(n as i64)
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
            Ok(n as u64)
        }
        Some(other) => Err(TensogramError::Metadata(format!(
            "expected integer for {key}, got {other:?}"
        ))),
        None => Err(TensogramError::Metadata(format!(
            "missing required parameter: {key}"
        ))),
    }
}
