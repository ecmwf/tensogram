use crate::error::{Result, TensogramError};
use crate::metadata;
use crate::types::{DataObjectDescriptor, GlobalMetadata, HashFrame, IndexFrame};
use crate::wire::{
    DataObjectFlags, FrameHeader, FrameType, MessageFlags, Postamble, Preamble,
    DATA_OBJECT_FOOTER_SIZE, FRAME_END, FRAME_HEADER_SIZE, MAGIC, POSTAMBLE_SIZE, PREAMBLE_SIZE,
};

// ── Frame-level primitives ───────────────────────────────────────────────────

/// Write a complete frame: frame_header + payload + ENDF.
/// Optionally pads to 8-byte alignment after ENDF.
fn write_frame(
    out: &mut Vec<u8>,
    frame_type: FrameType,
    version: u16,
    flags: u16,
    payload: &[u8],
    align: bool,
) {
    let total_length = (FRAME_HEADER_SIZE + payload.len() + FRAME_END.len()) as u64;

    let fh = FrameHeader {
        frame_type,
        version,
        flags,
        total_length,
    };
    fh.write_to(out);
    out.extend_from_slice(payload);
    out.extend_from_slice(FRAME_END);

    if align {
        let pad = (8 - (out.len() % 8)) % 8;
        out.extend(std::iter::repeat_n(0u8, pad));
    }
}

/// Read one frame from a buffer. Returns (FrameHeader, payload_slice, total_bytes_consumed).
/// `total_bytes_consumed` includes any padding to the next 8-byte boundary.
fn read_frame(buf: &[u8]) -> Result<(FrameHeader, &[u8], usize)> {
    let fh = FrameHeader::read_from(buf)?;
    let frame_total = fh.total_length as usize;

    if frame_total > buf.len() {
        return Err(TensogramError::Framing(format!(
            "frame total_length {} exceeds buffer: {}",
            frame_total,
            buf.len()
        )));
    }

    // Validate ENDF marker
    let endf_start = frame_total - FRAME_END.len();
    if &buf[endf_start..frame_total] != FRAME_END {
        return Err(TensogramError::Framing(format!(
            "missing ENDF marker at offset {endf_start}"
        )));
    }

    let payload = &buf[FRAME_HEADER_SIZE..endf_start];

    // Skip padding after ENDF to next 8-byte boundary
    let mut consumed = frame_total;
    let aligned = (consumed + 7) & !7;
    if aligned <= buf.len() {
        consumed = aligned;
    }

    Ok((fh, payload, consumed))
}

// ── Data Object Frame encode/decode ──────────────────────────────────────────

/// Encode a data object frame with CBOR descriptor after payload (default).
///
/// Layout: FrameHeader(16) | payload_bytes | cbor_bytes | cbor_offset(8) + ENDF(4)
///
/// The cbor_offset in the footer is the byte offset from frame start to the
/// start of the CBOR descriptor.
pub fn encode_data_object_frame(
    descriptor: &DataObjectDescriptor,
    payload: &[u8],
    cbor_before: bool,
) -> Result<Vec<u8>> {
    let cbor_bytes = metadata::object_descriptor_to_cbor(descriptor)?;
    let flags = if cbor_before {
        0
    } else {
        DataObjectFlags::CBOR_AFTER_PAYLOAD
    };

    // Calculate the total frame length:
    // frame_header(16) + cbor_bytes + payload + cbor_offset(8) + ENDF(4)
    let frame_body_len = cbor_bytes.len() + payload.len() + DATA_OBJECT_FOOTER_SIZE;
    let total_length = (FRAME_HEADER_SIZE + frame_body_len) as u64;

    let mut out = Vec::with_capacity(total_length as usize);

    // Frame header
    let fh = FrameHeader {
        frame_type: FrameType::DataObject,
        version: 1,
        flags,
        total_length,
    };
    fh.write_to(&mut out);

    if cbor_before {
        // CBOR descriptor first, then payload
        let cbor_offset = FRAME_HEADER_SIZE as u64;
        out.extend_from_slice(&cbor_bytes);
        out.extend_from_slice(payload);
        out.extend_from_slice(&cbor_offset.to_be_bytes());
    } else {
        // Payload first, then CBOR descriptor (default)
        let cbor_offset = (FRAME_HEADER_SIZE + payload.len()) as u64;
        out.extend_from_slice(payload);
        out.extend_from_slice(&cbor_bytes);
        out.extend_from_slice(&cbor_offset.to_be_bytes());
    }

    out.extend_from_slice(FRAME_END);

    debug_assert_eq!(out.len(), total_length as usize);
    Ok(out)
}

/// Decode a data object frame, returning the descriptor and payload slice.
///
/// `buf` must start at the frame header.
pub fn decode_data_object_frame(buf: &[u8]) -> Result<(DataObjectDescriptor, &[u8], usize)> {
    let fh = FrameHeader::read_from(buf)?;
    if fh.frame_type != FrameType::DataObject {
        return Err(TensogramError::Framing(format!(
            "expected DataObject frame, got {:?}",
            fh.frame_type
        )));
    }

    let frame_total = fh.total_length as usize;
    // Minimum: frame_header(16) + cbor_offset(8) + ENDF(4) = 28
    let min_frame_size = FRAME_HEADER_SIZE + DATA_OBJECT_FOOTER_SIZE;
    if frame_total < min_frame_size {
        return Err(TensogramError::Framing(format!(
            "data object frame too small: {} < {}",
            frame_total, min_frame_size
        )));
    }
    if frame_total > buf.len() {
        return Err(TensogramError::Framing(format!(
            "data object frame total_length {} exceeds buffer: {}",
            frame_total,
            buf.len()
        )));
    }

    // Validate ENDF marker
    let endf_start = frame_total - FRAME_END.len();
    if &buf[endf_start..frame_total] != FRAME_END {
        return Err(TensogramError::Framing(
            "missing ENDF marker in data object frame".to_string(),
        ));
    }

    // Read cbor_offset from the data object footer (8 bytes before ENDF)
    let cbor_offset_pos = endf_start - 8;
    let cbor_offset = crate::wire::read_u64_be(buf, cbor_offset_pos) as usize;

    let cbor_after = fh.flags & DataObjectFlags::CBOR_AFTER_PAYLOAD != 0;

    let (cbor_slice, payload_slice) = if cbor_after {
        // Layout: header(16) | payload | cbor | cbor_offset(8) | ENDF(4)
        let payload_start = FRAME_HEADER_SIZE;
        let cbor_start = cbor_offset;
        let cbor_end = cbor_offset_pos;
        let payload_end = cbor_start;
        (&buf[cbor_start..cbor_end], &buf[payload_start..payload_end])
    } else {
        // Layout: header(16) | cbor | payload | cbor_offset(8) | ENDF(4)
        let cbor_start = cbor_offset;
        // Parse CBOR to find where it ends (self-delimiting)
        let cbor_value: ciborium::Value = ciborium::from_reader(&buf[cbor_start..cbor_offset_pos])
            .map_err(|e| {
                TensogramError::Metadata(format!("failed to parse object descriptor CBOR: {e}"))
            })?;
        let cbor_bytes = {
            let mut tmp = Vec::new();
            ciborium::into_writer(&cbor_value, &mut tmp)
                .map_err(|e| TensogramError::Metadata(format!("failed to re-encode CBOR: {e}")))?;
            tmp
        };
        let cbor_end = cbor_start + cbor_bytes.len();
        let payload_start = cbor_end;
        let payload_end = cbor_offset_pos;
        (&buf[cbor_start..cbor_end], &buf[payload_start..payload_end])
    };

    let descriptor = metadata::cbor_to_object_descriptor(cbor_slice)?;

    // Bytes consumed, including padding
    let mut consumed = frame_total;
    let aligned = (consumed + 7) & !7;
    if aligned <= buf.len() {
        consumed = aligned;
    }

    Ok((descriptor, payload_slice, consumed))
}

// ── Message-level encode (buffered mode) ─────────────────────────────────────

/// Encoded data object: descriptor + encoded payload bytes.
pub struct EncodedObject {
    pub descriptor: DataObjectDescriptor,
    pub encoded_payload: Vec<u8>,
}

/// Encode a complete message in buffered mode.
///
/// All objects are known upfront. Header contains metadata + index + hashes.
/// Footer has only the postamble (no footer frames).
///
/// Strategy: build the message in two passes.
/// Pass 1: serialize all pieces, compute sizes/offsets.
/// Pass 2: assemble into final buffer.
pub fn encode_message(global_meta: &GlobalMetadata, objects: &[EncodedObject]) -> Result<Vec<u8>> {
    // 1. Serialize metadata CBOR
    let meta_cbor = metadata::global_metadata_to_cbor(global_meta)?;

    // 2. Pre-encode all data object frames
    let mut object_frames: Vec<Vec<u8>> = Vec::with_capacity(objects.len());
    for obj in objects {
        let frame = encode_data_object_frame(&obj.descriptor, &obj.encoded_payload, false)?;
        object_frames.push(frame);
    }

    // 3. Build hash frame CBOR (if any objects have hashes)
    let has_hashes = objects.iter().any(|o| o.descriptor.hash.is_some());
    let hash_cbor = if has_hashes {
        let hash_type = objects
            .iter()
            .find_map(|o| o.descriptor.hash.as_ref())
            .map(|h| h.hash_type.clone())
            .unwrap_or_default();
        let hashes: Vec<String> = objects
            .iter()
            .map(|o| {
                o.descriptor
                    .hash
                    .as_ref()
                    .map(|h| h.value.clone())
                    .unwrap_or_default()
            })
            .collect();
        let hf = HashFrame {
            object_count: objects.len() as u64,
            hash_type,
            hashes,
        };
        Some(metadata::hash_frame_to_cbor(&hf)?)
    } else {
        None
    };

    // 4. Two-pass index construction.
    //    We need to know where data objects start in order to build the index,
    //    but the index size affects where data objects start.
    //    Solve by: build header without index → compute offsets → build index
    //    → if index size causes shift, adjust offsets once more.

    // Build a temporary header buffer (without index) to measure its size
    let mut header_no_index = Vec::new();
    // Preamble placeholder (24 bytes, will be overwritten)
    header_no_index.extend_from_slice(&[0u8; PREAMBLE_SIZE]);
    // Metadata frame
    write_frame(
        &mut header_no_index,
        FrameType::HeaderMetadata,
        1,
        0,
        &meta_cbor,
        true,
    );
    // Hash frame (if present)
    if let Some(ref h_cbor) = hash_cbor {
        write_frame(
            &mut header_no_index,
            FrameType::HeaderHash,
            1,
            0,
            h_cbor,
            true,
        );
    }
    let header_size_no_index = header_no_index.len();

    // Build index CBOR if we have objects
    let index_frame_bytes = if !objects.is_empty() {
        // First estimate: assume index goes right after meta frame
        // Use dummy offsets of 0 to estimate CBOR size
        let dummy_idx = IndexFrame {
            object_count: objects.len() as u64,
            offsets: vec![0u64; objects.len()],
            lengths: object_frames.iter().map(|f| f.len() as u64).collect(),
        };
        let dummy_cbor = metadata::index_to_cbor(&dummy_idx)?;
        let dummy_frame_size = aligned_frame_total_size(dummy_cbor.len());
        let data_cursor = header_size_no_index + dummy_frame_size;

        // Compute object offsets
        let mut offsets = Vec::with_capacity(objects.len());
        let mut c = data_cursor;
        for frame in &object_frames {
            offsets.push(c as u64);
            c += frame.len();
            // Align between frames
            c = (c + 7) & !7;
        }

        // Build real index
        let real_idx = IndexFrame {
            object_count: objects.len() as u64,
            offsets: offsets.clone(),
            lengths: dummy_idx.lengths.clone(),
        };
        let real_cbor = metadata::index_to_cbor(&real_idx)?;

        // Check if size changed (larger offsets = more CBOR bytes)
        let final_cbor = if real_cbor.len() != dummy_cbor.len() {
            let real_frame_size = aligned_frame_total_size(real_cbor.len());
            let new_data_cursor = header_size_no_index + real_frame_size;
            let mut new_offsets = Vec::with_capacity(objects.len());
            let mut nc = new_data_cursor;
            for frame in &object_frames {
                new_offsets.push(nc as u64);
                nc += frame.len();
                nc = (nc + 7) & !7;
            }
            let final_idx = IndexFrame {
                object_count: objects.len() as u64,
                offsets: new_offsets,
                lengths: dummy_idx.lengths.clone(),
            };
            let third_cbor = metadata::index_to_cbor(&final_idx)?;
            // Guard: a third size change would mean offsets crossed another
            // CBOR integer encoding tier, invalidating the layout.
            if aligned_frame_total_size(third_cbor.len()) != real_frame_size {
                return Err(TensogramError::Framing(
                    "index CBOR size changed unexpectedly on third pass".to_string(),
                ));
            }
            third_cbor
        } else {
            real_cbor
        };

        // Serialize the index frame
        let mut idx_frame = Vec::new();
        write_frame(
            &mut idx_frame,
            FrameType::HeaderIndex,
            1,
            0,
            &final_cbor,
            true,
        );
        Some(idx_frame)
    } else {
        None
    };

    // 5. Compute flags
    let mut flags = MessageFlags::default();
    flags.set(MessageFlags::HEADER_METADATA);
    if index_frame_bytes.is_some() {
        flags.set(MessageFlags::HEADER_INDEX);
    }
    if has_hashes {
        flags.set(MessageFlags::HEADER_HASHES);
    }

    // 6. Assemble the full message
    let mut out = Vec::new();

    // Preamble placeholder (patched after we know total_length)
    let preamble_pos = out.len();
    out.extend_from_slice(&[0u8; PREAMBLE_SIZE]);

    // Header metadata frame
    write_frame(&mut out, FrameType::HeaderMetadata, 1, 0, &meta_cbor, true);

    // Header index frame (between metadata and hash, per spec ordering)
    if let Some(ref idx_bytes) = index_frame_bytes {
        out.extend_from_slice(idx_bytes);
    }

    // Header hash frame
    if let Some(ref h_cbor) = hash_cbor {
        write_frame(&mut out, FrameType::HeaderHash, 1, 0, h_cbor, true);
    }

    // Data object frames with inter-frame alignment
    for (i, frame) in object_frames.iter().enumerate() {
        out.extend_from_slice(frame);
        if i + 1 < object_frames.len() {
            let pad = (8 - (out.len() % 8)) % 8;
            out.extend(std::iter::repeat_n(0u8, pad));
        }
    }

    // Postamble (no footer frames in buffered mode)
    let postamble_offset = out.len();
    let postamble = Postamble {
        first_footer_offset: postamble_offset as u64,
    };
    postamble.write_to(&mut out);

    let total_length = out.len() as u64;

    // Patch the preamble with the real values
    let preamble = Preamble {
        version: 2,
        flags,
        reserved: 0,
        total_length,
    };
    let mut preamble_bytes = Vec::new();
    preamble.write_to(&mut preamble_bytes);
    out[preamble_pos..preamble_pos + PREAMBLE_SIZE].copy_from_slice(&preamble_bytes);

    Ok(out)
}

// ── Message-level decode ─────────────────────────────────────────────────────

/// A decoded message with all components.
pub struct DecodedMessage<'a> {
    pub preamble: Preamble,
    pub global_metadata: GlobalMetadata,
    pub index: Option<IndexFrame>,
    pub hash_frame: Option<HashFrame>,
    /// (descriptor, payload_slice, frame_offset_in_message)
    pub objects: Vec<(DataObjectDescriptor, &'a [u8], usize)>,
}

/// Decode a complete message from a buffer.
pub fn decode_message(buf: &[u8]) -> Result<DecodedMessage<'_>> {
    let preamble = Preamble::read_from(buf)?;

    // Validate total_length if non-zero
    if preamble.total_length > 0 {
        if (preamble.total_length as usize) > buf.len() {
            return Err(TensogramError::Framing(format!(
                "total_length {} exceeds buffer size {}",
                preamble.total_length,
                buf.len()
            )));
        }

        // Validate postamble
        let pa_offset = preamble.total_length as usize - POSTAMBLE_SIZE;
        let _postamble = Postamble::read_from(&buf[pa_offset..])?;
    }

    let mut pos = PREAMBLE_SIZE;
    let msg_end = if preamble.total_length > 0 {
        preamble.total_length as usize - POSTAMBLE_SIZE
    } else {
        buf.len() - POSTAMBLE_SIZE
    };

    let mut global_metadata: Option<GlobalMetadata> = None;
    let mut index: Option<IndexFrame> = None;
    let mut hash_frame: Option<HashFrame> = None;
    let mut objects: Vec<(DataObjectDescriptor, &[u8], usize)> = Vec::new();

    // Parse frames sequentially
    while pos < msg_end {
        // Check if we've reached a frame header or the postamble
        if pos + 2 > buf.len() {
            break;
        }

        // Skip padding bytes (zeros between ENDF and next FR)
        if &buf[pos..pos + 2] != b"FR" {
            pos += 1;
            continue;
        }

        let frame_start = pos;

        // Peek at frame type
        if pos + FRAME_HEADER_SIZE > buf.len() {
            break;
        }
        let fh = FrameHeader::read_from(&buf[pos..])?;

        match fh.frame_type {
            FrameType::HeaderMetadata | FrameType::FooterMetadata => {
                let (_, payload, consumed) = read_frame(&buf[pos..])?;
                let meta = metadata::cbor_to_global_metadata(payload)?;
                global_metadata = Some(meta);
                pos += consumed;
            }
            FrameType::HeaderIndex | FrameType::FooterIndex => {
                let (_, payload, consumed) = read_frame(&buf[pos..])?;
                let idx = metadata::cbor_to_index(payload)?;
                index = Some(idx);
                pos += consumed;
            }
            FrameType::HeaderHash | FrameType::FooterHash => {
                let (_, payload, consumed) = read_frame(&buf[pos..])?;
                let hf = metadata::cbor_to_hash_frame(payload)?;
                hash_frame = Some(hf);
                pos += consumed;
            }
            FrameType::DataObject => {
                let (desc, payload, consumed) = decode_data_object_frame(&buf[pos..])?;
                objects.push((desc, payload, frame_start));
                pos += consumed;
            }
        }
    }

    let global_metadata = global_metadata.ok_or_else(|| {
        TensogramError::Metadata("no metadata frame found in message".to_string())
    })?;

    Ok(DecodedMessage {
        preamble,
        global_metadata,
        index,
        hash_frame,
        objects,
    })
}

/// Decode only global metadata from a message, skipping data frames.
pub fn decode_metadata_only(buf: &[u8]) -> Result<GlobalMetadata> {
    let preamble = Preamble::read_from(buf)?;

    let mut pos = PREAMBLE_SIZE;
    let msg_end = if preamble.total_length > 0 {
        preamble.total_length as usize - POSTAMBLE_SIZE
    } else {
        buf.len() - POSTAMBLE_SIZE
    };

    while pos < msg_end {
        if pos + 2 > buf.len() {
            break;
        }
        if &buf[pos..pos + 2] != b"FR" {
            pos += 1;
            continue;
        }
        if pos + FRAME_HEADER_SIZE > buf.len() {
            break;
        }
        let fh = FrameHeader::read_from(&buf[pos..])?;
        match fh.frame_type {
            FrameType::HeaderMetadata | FrameType::FooterMetadata => {
                let (_, payload, _) = read_frame(&buf[pos..])?;
                return metadata::cbor_to_global_metadata(payload);
            }
            _ => {
                // Skip this frame
                let frame_total = fh.total_length as usize;
                pos += frame_total;
                pos = (pos + 7) & !7; // align
            }
        }
    }

    Err(TensogramError::Metadata(
        "no metadata frame found in message".to_string(),
    ))
}

// ── Scan ─────────────────────────────────────────────────────────────────────

/// Scan a multi-message buffer for message boundaries.
/// Returns (offset, length) of each message found.
pub fn scan(buf: &[u8]) -> Vec<(usize, usize)> {
    let mut messages = Vec::new();
    let mut pos = 0;

    while pos + PREAMBLE_SIZE + POSTAMBLE_SIZE <= buf.len() {
        if &buf[pos..pos + MAGIC.len()] == MAGIC {
            // Try to read preamble
            if let Ok(preamble) = Preamble::read_from(&buf[pos..]) {
                if preamble.total_length > 0 {
                    let total = preamble.total_length as usize;
                    if pos + total <= buf.len() {
                        // Validate end magic
                        let end_magic_offset = pos + total - 8;
                        if &buf[end_magic_offset..end_magic_offset + 8] == crate::wire::END_MAGIC {
                            messages.push((pos, total));
                            pos += total;
                            continue;
                        }
                    }
                } else {
                    // Streaming mode: scan forward to find end magic
                    // Look for the next 39277777 pattern
                    let mut end_pos = pos + PREAMBLE_SIZE;
                    let mut found = false;
                    while end_pos + 8 <= buf.len() {
                        if &buf[end_pos..end_pos + 8] == crate::wire::END_MAGIC {
                            let msg_len = end_pos + 8 - pos;
                            messages.push((pos, msg_len));
                            pos = end_pos + 8;
                            found = true;
                            break;
                        }
                        end_pos += 1;
                    }
                    if found {
                        continue;
                    }
                }
            }
        }
        pos += 1;
    }

    messages
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Total frame size: header + payload + ENDF
fn frame_total_size(payload_len: usize) -> usize {
    FRAME_HEADER_SIZE + payload_len + FRAME_END.len()
}

/// Frame total size aligned to 8 bytes
fn aligned_frame_total_size(payload_len: usize) -> usize {
    let raw = frame_total_size(payload_len);
    (raw + 7) & !7
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;
    use crate::types::ByteOrder;
    use std::collections::BTreeMap;

    fn make_global_meta() -> GlobalMetadata {
        GlobalMetadata {
            version: 2,
            extra: BTreeMap::new(),
        }
    }

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
            byte_order: ByteOrder::Little,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        }
    }

    #[test]
    fn test_data_object_frame_round_trip_cbor_after() {
        let desc = make_descriptor(vec![4]);
        let payload = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

        let frame = encode_data_object_frame(&desc, &payload, false).unwrap();

        let (decoded_desc, decoded_payload, consumed) = decode_data_object_frame(&frame).unwrap();
        assert_eq!(decoded_desc.shape, vec![4]);
        assert_eq!(decoded_desc.dtype, Dtype::Float32);
        assert_eq!(decoded_payload, &payload[..]);
        assert!(consumed >= frame.len());
    }

    #[test]
    fn test_data_object_frame_round_trip_cbor_before() {
        let desc = make_descriptor(vec![2, 3]);
        let payload = vec![0xABu8; 24]; // 2*3*4 = 24 bytes for float32

        let frame = encode_data_object_frame(&desc, &payload, true).unwrap();

        let (decoded_desc, decoded_payload, _) = decode_data_object_frame(&frame).unwrap();
        assert_eq!(decoded_desc.shape, vec![2, 3]);
        assert_eq!(decoded_payload, &payload[..]);
    }

    #[test]
    fn test_empty_message_round_trip() {
        let meta = make_global_meta();
        let msg = encode_message(&meta, &[]).unwrap();

        // Check magic and end magic
        assert_eq!(&msg[0..8], MAGIC);
        assert_eq!(&msg[msg.len() - 8..], crate::wire::END_MAGIC);

        let decoded = decode_message(&msg).unwrap();
        assert_eq!(decoded.global_metadata.version, 2);
        assert_eq!(decoded.objects.len(), 0);
        assert!(decoded.index.is_none()); // no objects = no index
    }

    #[test]
    fn test_single_object_message_round_trip() {
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let payload = vec![42u8; 16]; // 4 * float32

        let objects = vec![EncodedObject {
            descriptor: desc,
            encoded_payload: payload.clone(),
        }];

        let msg = encode_message(&meta, &objects).unwrap();
        let decoded = decode_message(&msg).unwrap();

        assert_eq!(decoded.global_metadata.version, 2);
        assert_eq!(decoded.objects.len(), 1);
        assert_eq!(decoded.objects[0].0.shape, vec![4]);
        assert_eq!(decoded.objects[0].1, &payload[..]);
        assert!(decoded.index.is_some());
        assert_eq!(decoded.index.as_ref().unwrap().object_count, 1);
    }

    #[test]
    fn test_multi_object_message_round_trip() {
        let meta = make_global_meta();
        let desc0 = make_descriptor(vec![4]);
        let desc1 = make_descriptor(vec![2, 3]);
        let payload0 = vec![10u8; 16];
        let payload1 = vec![20u8; 24];

        let objects = vec![
            EncodedObject {
                descriptor: desc0,
                encoded_payload: payload0.clone(),
            },
            EncodedObject {
                descriptor: desc1,
                encoded_payload: payload1.clone(),
            },
        ];

        let msg = encode_message(&meta, &objects).unwrap();
        let decoded = decode_message(&msg).unwrap();

        assert_eq!(decoded.objects.len(), 2);
        assert_eq!(decoded.objects[0].0.shape, vec![4]);
        assert_eq!(decoded.objects[0].1, &payload0[..]);
        assert_eq!(decoded.objects[1].0.shape, vec![2, 3]);
        assert_eq!(decoded.objects[1].1, &payload1[..]);

        let idx = decoded.index.as_ref().unwrap();
        assert_eq!(idx.object_count, 2);
        assert_eq!(idx.offsets.len(), 2);
    }

    #[test]
    fn test_scan_multi_message() {
        let meta = make_global_meta();
        let msg1 = encode_message(
            &meta,
            &[EncodedObject {
                descriptor: make_descriptor(vec![4]),
                encoded_payload: vec![1u8; 16],
            }],
        )
        .unwrap();
        let msg2 = encode_message(
            &meta,
            &[EncodedObject {
                descriptor: make_descriptor(vec![2]),
                encoded_payload: vec![2u8; 8],
            }],
        )
        .unwrap();

        let mut buf = msg1.clone();
        buf.extend_from_slice(&msg2);

        let offsets = scan(&buf);
        assert_eq!(offsets.len(), 2);
        assert_eq!(offsets[0], (0, msg1.len()));
        assert_eq!(offsets[1], (msg1.len(), msg2.len()));
    }

    #[test]
    fn test_scan_with_garbage() {
        let meta = make_global_meta();
        let msg = encode_message(
            &meta,
            &[EncodedObject {
                descriptor: make_descriptor(vec![4]),
                encoded_payload: vec![1u8; 16],
            }],
        )
        .unwrap();

        let mut buf = vec![0xFF; 10];
        buf.extend_from_slice(&msg);
        buf.extend_from_slice(&[0xAA; 5]);

        let offsets = scan(&buf);
        assert_eq!(offsets.len(), 1);
        assert_eq!(offsets[0], (10, msg.len()));
    }

    #[test]
    fn test_decode_metadata_only() {
        let mut meta = make_global_meta();
        meta.extra.insert(
            "test_key".to_string(),
            ciborium::Value::Text("test_value".to_string()),
        );

        let msg = encode_message(
            &meta,
            &[EncodedObject {
                descriptor: make_descriptor(vec![4]),
                encoded_payload: vec![0u8; 16],
            }],
        )
        .unwrap();

        let decoded_meta = decode_metadata_only(&msg).unwrap();
        assert_eq!(decoded_meta.version, 2);
        assert!(decoded_meta.extra.contains_key("test_key"));
    }
}
