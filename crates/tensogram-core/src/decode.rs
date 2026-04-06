use crate::encode::build_pipeline_config;
use crate::error::{Result, TensogramError};
use crate::framing;
use crate::hash;
use crate::types::{DataObjectDescriptor, DecodedObject, GlobalMetadata};
use tensogram_encodings::pipeline;

fn extract_block_offsets(
    params: &std::collections::BTreeMap<String, ciborium::Value>,
) -> Result<Vec<u64>> {
    match params.get("szip_block_offsets") {
        Some(ciborium::Value::Array(arr)) => arr
            .iter()
            .map(|v| match v {
                ciborium::Value::Integer(i) => {
                    let n: i128 = (*i).into();
                    u64::try_from(n).map_err(|_| {
                        TensogramError::Metadata("szip_block_offset out of u64 range".to_string())
                    })
                }
                _ => Err(TensogramError::Metadata(
                    "szip_block_offsets must contain integers".to_string(),
                )),
            })
            .collect(),
        Some(_) => Err(TensogramError::Metadata(
            "szip_block_offsets must be an array".to_string(),
        )),
        None => Err(TensogramError::Compression(
            "missing szip_block_offsets in payload metadata (required for partial range decode)"
                .to_string(),
        )),
    }
}

/// Options for decoding.
#[derive(Debug, Clone, Default)]
pub struct DecodeOptions {
    /// Whether to verify payload hashes during decode.
    pub verify_hash: bool,
}

/// Decode all objects from a message buffer.
/// Returns (global_metadata, list of (descriptor, decoded_data)).
#[tracing::instrument(skip(buf, options), fields(buf_len = buf.len()))]
pub fn decode(buf: &[u8], options: &DecodeOptions) -> Result<(GlobalMetadata, Vec<DecodedObject>)> {
    let msg = framing::decode_message(buf)?;

    let mut data_objects = Vec::with_capacity(msg.objects.len());
    for (desc, payload_bytes, _offset) in &msg.objects {
        let decoded = decode_single_object(desc, payload_bytes, options)?;
        data_objects.push((desc.clone(), decoded));
    }

    Ok((msg.global_metadata, data_objects))
}

/// Decode only global metadata from a message buffer, skipping payloads.
pub fn decode_metadata(buf: &[u8]) -> Result<GlobalMetadata> {
    framing::decode_metadata_only(buf)
}

/// Decode global metadata **and** per-object descriptors without decoding
/// any payload data.
///
/// This is cheaper than [`decode`] because the pipeline (decompression,
/// filter reversal, endian swap) is never executed.  Use it when you only
/// need shapes, dtypes, and metadata — e.g. for building xarray Datasets
/// at open time.
pub fn decode_descriptors(buf: &[u8]) -> Result<(GlobalMetadata, Vec<DataObjectDescriptor>)> {
    let msg = framing::decode_message(buf)?;
    let descriptors = msg.objects.into_iter().map(|(desc, _, _)| desc).collect();
    Ok((msg.global_metadata, descriptors))
}

/// Decode a single object by index (O(1) access via index frame).
/// Returns (global_metadata, descriptor, decoded_data).
pub fn decode_object(
    buf: &[u8],
    index: usize,
    options: &DecodeOptions,
) -> Result<(GlobalMetadata, DataObjectDescriptor, Vec<u8>)> {
    let msg = framing::decode_message(buf)?;

    if index >= msg.objects.len() {
        return Err(TensogramError::Object(format!(
            "object index {} out of range (num_objects={})",
            index,
            msg.objects.len()
        )));
    }

    let (desc, payload_bytes, _) = &msg.objects[index];
    let decoded = decode_single_object(desc, payload_bytes, options)?;

    Ok((msg.global_metadata, desc.clone(), decoded))
}

/// Decode partial ranges from a data object.
///
/// `ranges` is a list of (element_offset, element_count) pairs.
///
/// Returns one `Vec<u8>` per range (split results).  Callers that need a
/// single concatenated buffer can flatten with `results.into_iter().flatten()`.
pub fn decode_range(
    buf: &[u8],
    object_index: usize,
    ranges: &[(u64, u64)],
    options: &DecodeOptions,
) -> Result<Vec<Vec<u8>>> {
    let msg = framing::decode_message(buf)?;

    if object_index >= msg.objects.len() {
        return Err(TensogramError::Object(format!(
            "object index {} out of range (num_objects={})",
            object_index,
            msg.objects.len()
        )));
    }

    let (desc, payload_bytes, _) = &msg.objects[object_index];

    // Reject shuffle filter
    if desc.filter != "none" {
        return Err(TensogramError::Encoding(
            "decode_range is not supported when a filter (e.g. shuffle) is applied".to_string(),
        ));
    }

    if desc.dtype.byte_width() == 0 {
        return Err(TensogramError::Encoding(
            "partial range decode not supported for bitmask dtype".to_string(),
        ));
    }

    if options.verify_hash {
        if let Some(ref hash_desc) = desc.hash {
            hash::verify_hash(payload_bytes, hash_desc)?;
        }
    }

    let shape_product = desc
        .shape
        .iter()
        .try_fold(1u64, |acc, &x| acc.checked_mul(x))
        .ok_or_else(|| TensogramError::Metadata("shape product overflow".to_string()))?;
    let num_elements = usize::try_from(shape_product)
        .map_err(|_| TensogramError::Metadata("element count overflows usize".to_string()))?;
    let config = build_pipeline_config(desc, num_elements, desc.dtype)?;

    let block_offsets = if desc.compression == "szip" {
        extract_block_offsets(&desc.params)?
    } else {
        Vec::new()
    };

    let mut results = Vec::with_capacity(ranges.len());
    for &(offset, count) in ranges {
        let range_bytes =
            pipeline::decode_range_pipeline(payload_bytes, &config, &block_offsets, offset, count)
                .map_err(|e| {
                    TensogramError::Encoding(format!("range (offset={offset}, count={count}): {e}"))
                })?;
        results.push(range_bytes);
    }

    Ok(results)
}

/// Decode a single object through the full pipeline.
fn decode_single_object(
    desc: &DataObjectDescriptor,
    payload_bytes: &[u8],
    options: &DecodeOptions,
) -> Result<Vec<u8>> {
    if options.verify_hash {
        if let Some(ref hash_desc) = desc.hash {
            hash::verify_hash(payload_bytes, hash_desc)?;
        }
    }

    let shape_product = desc
        .shape
        .iter()
        .try_fold(1u64, |acc, &x| acc.checked_mul(x))
        .ok_or_else(|| TensogramError::Metadata("shape product overflow".to_string()))?;
    let num_elements = usize::try_from(shape_product)
        .map_err(|_| TensogramError::Metadata("element count overflows usize".to_string()))?;
    let config = build_pipeline_config(desc, num_elements, desc.dtype)?;
    let decoded = pipeline::decode_pipeline(payload_bytes, &config)
        .map_err(|e| TensogramError::Encoding(e.to_string()))?;

    Ok(decoded)
}
