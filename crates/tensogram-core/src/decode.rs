use crate::encode::build_pipeline_config;
use crate::error::{Result, TensogramError};
use crate::framing;
use crate::hash;
use crate::metadata::cbor_to_metadata;
use crate::types::Metadata;
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
/// Returns (metadata, decoded_data_objects).
pub fn decode(buf: &[u8], options: &DecodeOptions) -> Result<(Metadata, Vec<Vec<u8>>)> {
    let frame = framing::decode_frame(buf)?;
    let metadata = cbor_to_metadata(frame.cbor_bytes)?;

    let mut data_objects = Vec::with_capacity(metadata.objects.len());
    for (i, _) in metadata.objects.iter().enumerate() {
        let payload_bytes = framing::extract_object_payload(buf, &frame, i)?;
        verify_and_decode_object(&metadata, i, payload_bytes, options, &mut data_objects)?;
    }

    Ok((metadata, data_objects))
}

/// Decode only metadata from a message buffer, skipping payloads.
pub fn decode_metadata(buf: &[u8]) -> Result<Metadata> {
    let frame = framing::decode_frame(buf)?;
    cbor_to_metadata(frame.cbor_bytes)
}

/// Decode a single object by index (O(1) access via binary header).
/// Returns (metadata, decoded_data).
pub fn decode_object(
    buf: &[u8],
    index: usize,
    options: &DecodeOptions,
) -> Result<(Metadata, Vec<u8>)> {
    let frame = framing::decode_frame(buf)?;
    let metadata = cbor_to_metadata(frame.cbor_bytes)?;

    if index >= metadata.objects.len() {
        return Err(TensogramError::Object(format!(
            "object index {} out of range (num_objects={})",
            index,
            metadata.objects.len()
        )));
    }

    let payload_bytes = framing::extract_object_payload(buf, &frame, index)?;

    // Verify hash if requested
    if options.verify_hash {
        if let Some(ref hash_desc) = metadata.payload[index].hash {
            hash::verify_hash(payload_bytes, hash_desc)?;
        }
    }

    let num_elements = usize::try_from(metadata.objects[index].shape.iter().product::<u64>())
        .map_err(|_| TensogramError::Metadata("element count overflows usize".to_string()))?;
    let dtype = metadata.objects[index].dtype;
    let config = build_pipeline_config(&metadata.payload[index], num_elements, dtype)?;
    let decoded = pipeline::decode_pipeline(payload_bytes, &config)
        .map_err(|e| TensogramError::Encoding(e.to_string()))?;

    Ok((metadata, decoded))
}

/// Decode a partial range from a data object.
///
/// Supports uncompressed data (direct byte slicing), simple_packing (bit extraction),
/// szip-compressed data (RSI block seeking via libaec), and combinations thereof.
/// Shuffle filter is NOT supported with range decode (returns error).
///
/// `ranges` is a list of (element_offset, element_count) pairs.
pub fn decode_range(
    buf: &[u8],
    object_index: usize,
    ranges: &[(u64, u64)],
    options: &DecodeOptions,
) -> Result<Vec<u8>> {
    let frame = framing::decode_frame(buf)?;
    let metadata = cbor_to_metadata(frame.cbor_bytes)?;

    if object_index >= metadata.objects.len() {
        return Err(TensogramError::Object(format!(
            "object index {} out of range (num_objects={})",
            object_index,
            metadata.objects.len()
        )));
    }

    let payload_desc = &metadata.payload[object_index];
    let obj_desc = &metadata.objects[object_index];

    // Reject shuffle filter — byte rearrangement makes RSI block boundaries
    // not correspond to contiguous sample ranges (documented limitation)
    if payload_desc.filter != "none" {
        return Err(TensogramError::Encoding(
            "decode_range is not supported when a filter (e.g. shuffle) is applied".to_string(),
        ));
    }

    if obj_desc.dtype.byte_width() == 0 {
        return Err(TensogramError::Encoding(
            "partial range decode not supported for bitmask dtype".to_string(),
        ));
    }

    let payload_bytes = framing::extract_object_payload(buf, &frame, object_index)?;

    if options.verify_hash {
        if let Some(ref hash_desc) = payload_desc.hash {
            hash::verify_hash(payload_bytes, hash_desc)?;
        }
    }

    let num_elements = usize::try_from(obj_desc.shape.iter().product::<u64>())
        .map_err(|_| TensogramError::Metadata("element count overflows usize".to_string()))?;
    let config = build_pipeline_config(payload_desc, num_elements, obj_desc.dtype)?;

    // Extract block offsets if compression is szip
    let block_offsets = if payload_desc.compression == "szip" {
        extract_block_offsets(&payload_desc.params)?
    } else {
        Vec::new()
    };

    let mut result = Vec::new();
    for &(offset, count) in ranges {
        let range_bytes =
            pipeline::decode_range_pipeline(payload_bytes, &config, &block_offsets, offset, count)
                .map_err(|e| TensogramError::Encoding(e.to_string()))?;
        result.extend_from_slice(&range_bytes);
    }

    Ok(result)
}

fn verify_and_decode_object(
    metadata: &Metadata,
    index: usize,
    payload_bytes: &[u8],
    options: &DecodeOptions,
    out: &mut Vec<Vec<u8>>,
) -> Result<()> {
    // Verify hash if requested
    if options.verify_hash {
        if let Some(ref hash_desc) = metadata.payload[index].hash {
            hash::verify_hash(payload_bytes, hash_desc)?;
        }
    }

    let num_elements = usize::try_from(metadata.objects[index].shape.iter().product::<u64>())
        .map_err(|_| TensogramError::Metadata("element count overflows usize".to_string()))?;
    let dtype = metadata.objects[index].dtype;
    let config = build_pipeline_config(&metadata.payload[index], num_elements, dtype)?;
    let decoded = pipeline::decode_pipeline(payload_bytes, &config)
        .map_err(|e| TensogramError::Encoding(e.to_string()))?;
    out.push(decoded);
    Ok(())
}
