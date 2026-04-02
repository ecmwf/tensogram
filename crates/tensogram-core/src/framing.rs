use crate::error::{Result, TensogramError};
use crate::wire::{BinaryHeader, MAGIC, OBJE, OBJS, TERMINATOR};

/// Encode a complete message frame.
/// `cbor_bytes` is the serialized CBOR metadata.
/// `encoded_payloads` is a list of already-encoded payload byte vectors (one per data object).
/// Returns the complete wire-format message as `Vec<u8>`.
pub fn encode_frame(cbor_bytes: &[u8], encoded_payloads: &[Vec<u8>]) -> Vec<u8> {
    let num_objects = encoded_payloads.len() as u64;
    let header_size = BinaryHeader::header_size(num_objects);

    // metadata immediately follows the header (no padding in v1)
    let metadata_offset = header_size as u64;
    let metadata_length = cbor_bytes.len() as u64;

    // Compute object offsets and total size
    let mut object_offsets = Vec::with_capacity(encoded_payloads.len());
    let mut cursor = header_size + cbor_bytes.len();
    for payload in encoded_payloads {
        object_offsets.push(cursor as u64);
        cursor += 4 + payload.len() + 4; // OBJS + payload + OBJE
    }
    let total_length = (cursor + TERMINATOR.len()) as u64;

    // Build the message
    let mut out = Vec::with_capacity(total_length as usize);

    // Binary header
    let header = BinaryHeader {
        total_length,
        metadata_offset,
        metadata_length,
        num_objects,
        object_offsets,
    };
    header.write_to(&mut out);

    // CBOR metadata
    out.extend_from_slice(cbor_bytes);

    // Data objects
    for payload in encoded_payloads {
        out.extend_from_slice(OBJS);
        out.extend_from_slice(payload);
        out.extend_from_slice(OBJE);
    }

    // Terminator
    out.extend_from_slice(TERMINATOR);

    debug_assert_eq!(out.len(), total_length as usize);
    out
}

/// Parsed frame: the binary header, the CBOR slice, and the start of the payload region.
pub struct DecodedFrame<'a> {
    pub header: BinaryHeader,
    pub cbor_bytes: &'a [u8],
}

/// Validate and parse a message from a buffer.
/// The buffer must start at a message boundary (TENSOGRM magic).
pub fn decode_frame(buf: &[u8]) -> Result<DecodedFrame<'_>> {
    let header = BinaryHeader::read_from(buf)?;

    // Validate total_length (0 = streaming mode)
    if header.total_length > 0 {
        if (header.total_length as usize) > buf.len() {
            return Err(TensogramError::Framing(format!(
                "total_length {} exceeds buffer size {}",
                header.total_length,
                buf.len()
            )));
        }

        // Validate terminator
        let term_offset = header.total_length as usize - TERMINATOR.len();
        if &buf[term_offset..term_offset + TERMINATOR.len()] != TERMINATOR {
            return Err(TensogramError::Framing("invalid terminator".to_string()));
        }
    }

    // Extract CBOR slice
    let meta_start = header.metadata_offset as usize;
    let meta_end = meta_start + header.metadata_length as usize;
    if meta_end > buf.len() {
        return Err(TensogramError::Framing(format!(
            "metadata extends beyond buffer: {}+{} > {}",
            meta_start,
            header.metadata_length,
            buf.len()
        )));
    }
    let cbor_bytes = &buf[meta_start..meta_end];

    // Validate OBJS/OBJE markers for each object
    for (i, &obj_offset) in header.object_offsets.iter().enumerate() {
        let off = obj_offset as usize;
        if off + 4 > buf.len() {
            return Err(TensogramError::Object(format!(
                "object {i} OBJS marker out of bounds"
            )));
        }
        if &buf[off..off + 4] != OBJS {
            return Err(TensogramError::Object(format!(
                "object {i} missing OBJS marker at offset {off}"
            )));
        }

        // Find OBJE: for object i, end is at object_offsets[i+1] or terminator
        let obj_end = if i + 1 < header.object_offsets.len() {
            header.object_offsets[i + 1] as usize
        } else if header.total_length > 0 {
            header.total_length as usize - TERMINATOR.len()
        } else {
            buf.len() - TERMINATOR.len()
        };

        if obj_end < 4 {
            return Err(TensogramError::Object(format!(
                "object {i} region too small"
            )));
        }
        let obje_off = obj_end - 4;
        if &buf[obje_off..obje_off + 4] != OBJE {
            return Err(TensogramError::Object(format!(
                "object {i} missing OBJE marker at offset {obje_off}"
            )));
        }
    }

    Ok(DecodedFrame { header, cbor_bytes })
}

/// Extract the raw payload bytes for object at `index` from a decoded frame.
/// Strips OBJS/OBJE markers.
pub fn extract_object_payload<'a>(
    buf: &'a [u8],
    frame: &DecodedFrame,
    index: usize,
) -> Result<&'a [u8]> {
    if index >= frame.header.num_objects as usize {
        return Err(TensogramError::Object(format!(
            "object index {} out of range (num_objects={})",
            index, frame.header.num_objects
        )));
    }

    let obj_start = frame.header.object_offsets[index] as usize + 4; // skip OBJS
    let obj_end = if index + 1 < frame.header.object_offsets.len() {
        frame.header.object_offsets[index + 1] as usize
    } else if frame.header.total_length > 0 {
        frame.header.total_length as usize - TERMINATOR.len()
    } else {
        buf.len() - TERMINATOR.len()
    };
    let payload_end = obj_end - 4; // before OBJE

    Ok(&buf[obj_start..payload_end])
}

/// Scan a multi-message buffer for message boundaries.
/// Returns (offset, length) of each message found.
pub fn scan(buf: &[u8]) -> Vec<(usize, usize)> {
    let mut messages = Vec::new();
    let mut pos = 0;

    while pos + MAGIC.len() <= buf.len() {
        // Look for MAGIC
        if &buf[pos..pos + MAGIC.len()] == MAGIC {
            // Try to read total_length
            if pos + 16 <= buf.len() {
                let total_length =
                    u64::from_be_bytes(buf[pos + 8..pos + 16].try_into().unwrap()) as usize;

                if total_length > 0 && pos + total_length <= buf.len() {
                    // Validate terminator
                    let term_pos = pos + total_length - TERMINATOR.len();
                    if &buf[term_pos..term_pos + TERMINATOR.len()] == TERMINATOR {
                        messages.push((pos, total_length));
                        pos += total_length;
                        continue;
                    }
                }
            }
        }
        pos += 1;
    }

    messages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_message_round_trip() {
        let cbor = vec![0xA0]; // empty CBOR map {}
        let msg = encode_frame(&cbor, &[]);
        assert_eq!(&msg[0..8], MAGIC);
        assert_eq!(&msg[msg.len() - 8..], TERMINATOR);

        let frame = decode_frame(&msg).unwrap();
        assert_eq!(frame.header.num_objects, 0);
        assert_eq!(frame.cbor_bytes, &[0xA0]);
    }

    #[test]
    fn test_single_object_round_trip() {
        let cbor = vec![0xA0];
        let payload = vec![1u8, 2, 3, 4, 5];
        let msg = encode_frame(&cbor, std::slice::from_ref(&payload));

        let frame = decode_frame(&msg).unwrap();
        assert_eq!(frame.header.num_objects, 1);

        let extracted = extract_object_payload(&msg, &frame, 0).unwrap();
        assert_eq!(extracted, &payload[..]);
    }

    #[test]
    fn test_multi_object_round_trip() {
        let cbor = vec![0xA0];
        let p1 = vec![10u8, 20, 30];
        let p2 = vec![40u8, 50, 60, 70, 80];
        let msg = encode_frame(&cbor, &[p1.clone(), p2.clone()]);

        let frame = decode_frame(&msg).unwrap();
        assert_eq!(frame.header.num_objects, 2);

        let e1 = extract_object_payload(&msg, &frame, 0).unwrap();
        assert_eq!(e1, &p1[..]);
        let e2 = extract_object_payload(&msg, &frame, 1).unwrap();
        assert_eq!(e2, &p2[..]);
    }

    #[test]
    fn test_scan_multi_message() {
        let cbor = vec![0xA0];
        let msg1 = encode_frame(&cbor, &[vec![1, 2]]);
        let msg2 = encode_frame(&cbor, &[vec![3, 4, 5]]);
        let msg3 = encode_frame(&cbor, &[]);

        let mut buf = Vec::new();
        buf.extend_from_slice(&msg1);
        buf.extend_from_slice(&msg2);
        buf.extend_from_slice(&msg3);

        let offsets = scan(&buf);
        assert_eq!(offsets.len(), 3);
        assert_eq!(offsets[0], (0, msg1.len()));
        assert_eq!(offsets[1], (msg1.len(), msg2.len()));
        assert_eq!(offsets[2], (msg1.len() + msg2.len(), msg3.len()));
    }

    #[test]
    fn test_scan_with_garbage() {
        let cbor = vec![0xA0];
        let msg = encode_frame(&cbor, &[vec![1, 2]]);

        let mut buf = vec![0xFF; 10]; // garbage
        buf.extend_from_slice(&msg);
        buf.extend_from_slice(&[0xAA; 5]); // more garbage

        let offsets = scan(&buf);
        assert_eq!(offsets.len(), 1);
        assert_eq!(offsets[0], (10, msg.len()));
    }

    #[test]
    fn test_invalid_terminator() {
        let cbor = vec![0xA0];
        let mut msg = encode_frame(&cbor, &[]);
        // Corrupt terminator
        let len = msg.len();
        msg[len - 1] = 0xFF;
        assert!(decode_frame(&msg).is_err());
    }
}
