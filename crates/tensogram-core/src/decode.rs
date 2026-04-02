use crate::encode::build_pipeline_config;
use crate::error::{Result, TensogramError};
use crate::framing;
use crate::hash;
use crate::metadata::cbor_to_metadata;
use crate::types::Metadata;
use tensogram_encodings::pipeline;

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
    for i in 0..metadata.objects.len() {
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

    let num_elements = metadata.objects[index].shape.iter().product::<u64>() as usize;
    let config = build_pipeline_config(&metadata.payload[index], num_elements)?;
    let decoded = pipeline::decode_pipeline(payload_bytes, &config)
        .map_err(|e| TensogramError::Encoding(e.to_string()))?;

    Ok((metadata, decoded))
}

/// Decode a partial range from a data object (uncompressed path only for now).
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

    // Partial range decode only works for uncompressed, unencoded data
    if payload_desc.encoding != "none" || payload_desc.compression != "none" {
        if payload_desc.filter == "shuffle" {
            return Err(TensogramError::Encoding(
                "partial range decode not supported with shuffle filter".to_string(),
            ));
        }
        // For simple_packing + szip: would need RSI block seeking (future)
        if payload_desc.compression != "none" {
            return Err(TensogramError::Encoding(
                "partial range decode with compression not yet implemented".to_string(),
            ));
        }
    }

    let payload_bytes = framing::extract_object_payload(buf, &frame, object_index)?;

    if options.verify_hash {
        if let Some(ref hash_desc) = payload_desc.hash {
            hash::verify_hash(payload_bytes, hash_desc)?;
        }
    }

    // For uncompressed data: direct byte offset
    let element_size = obj_desc.dtype.byte_width();
    if element_size == 0 {
        return Err(TensogramError::Encoding(
            "partial range decode not supported for bitmask dtype".to_string(),
        ));
    }

    let mut result = Vec::new();
    for &(offset, count) in ranges {
        let byte_start = offset as usize * element_size;
        let byte_end = byte_start + count as usize * element_size;
        if byte_end > payload_bytes.len() {
            return Err(TensogramError::Object(format!(
                "range ({offset}, {count}) exceeds payload size"
            )));
        }
        result.extend_from_slice(&payload_bytes[byte_start..byte_end]);
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

    let num_elements = metadata.objects[index].shape.iter().product::<u64>() as usize;
    let config = build_pipeline_config(&metadata.payload[index], num_elements)?;
    let decoded = pipeline::decode_pipeline(payload_bytes, &config)
        .map_err(|e| TensogramError::Encoding(e.to_string()))?;

    out.push(decoded);
    Ok(())
}
