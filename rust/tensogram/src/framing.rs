// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::collections::BTreeMap;

use crate::error::{Result, TensogramError};
use crate::metadata::{self, RESERVED_KEY};
use crate::types::{DataObjectDescriptor, GlobalMetadata, HashFrame, IndexFrame};
use crate::wire::{
    DATA_OBJECT_FOOTER_SIZE, DataObjectFlags, FRAME_END, FRAME_HEADER_SIZE, FrameHeader, FrameType,
    MAGIC, MessageFlags, POSTAMBLE_SIZE, PREAMBLE_SIZE, Postamble, Preamble,
};

/// Compute the byte offset in `buf` where the encoded payload ends
/// and the (optional) mask region begins, given the payload start,
/// the region end (the start of CBOR), and the parsed descriptor.
///
/// When `descriptor.masks` is `None` — the common case — the payload
/// fills the entire region, so `payload_end == region_end`.  When
/// masks are present, the payload ends at the smallest offset among
/// the present mask descriptors (offsets are relative to the
/// payload-region start, so the absolute end is
/// `payload_start + smallest_offset`).
///
/// Returns [`TensogramError::Framing`] when a mask descriptor's
/// offset + length exceeds the region size — a corrupted or
/// malicious frame.
fn mask_aware_payload_end(
    payload_start: usize,
    region_end: usize,
    desc: &DataObjectDescriptor,
) -> Result<usize> {
    let Some(masks) = desc.masks.as_ref() else {
        return Ok(region_end);
    };
    let region_len = region_end.saturating_sub(payload_start);

    // Validate every present mask fits inside the region.
    let mut smallest_offset: Option<usize> = None;
    let mut validate_one =
        |md: Option<&crate::types::MaskDescriptor>, kind: &'static str| -> Result<()> {
            let Some(md) = md else { return Ok(()) };
            let offset = usize::try_from(md.offset).map_err(|_| {
                TensogramError::Framing(format!("mask_{kind}.offset {} overflows usize", md.offset))
            })?;
            let length = usize::try_from(md.length).map_err(|_| {
                TensogramError::Framing(format!("mask_{kind}.length {} overflows usize", md.length))
            })?;
            let end = offset.checked_add(length).ok_or_else(|| {
                TensogramError::Framing(format!(
                    "mask_{kind}.offset + length overflow (offset={offset}, length={length})"
                ))
            })?;
            if end > region_len {
                return Err(TensogramError::Framing(format!(
                    "mask_{kind}.offset + length ({end}) exceeds payload region size ({region_len})"
                )));
            }
            smallest_offset = Some(smallest_offset.map_or(offset, |s| s.min(offset)));
            Ok(())
        };
    validate_one(masks.nan.as_ref(), "nan")?;
    validate_one(masks.pos_inf.as_ref(), "inf+")?;
    validate_one(masks.neg_inf.as_ref(), "inf-")?;

    // Every kind absent (masks == Some(empty)) — treat like no masks.
    let relative_end = smallest_offset.unwrap_or(region_len);
    Ok(payload_start + relative_end)
}

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
    let frame_total = usize::try_from(fh.total_length).map_err(|_| {
        TensogramError::Framing(format!(
            "frame total_length {} overflows usize",
            fh.total_length
        ))
    })?;

    // Minimum frame: header(16) + ENDF(4) = 20 bytes
    let min_frame_size = FRAME_HEADER_SIZE + FRAME_END.len();
    if frame_total < min_frame_size {
        return Err(TensogramError::Framing(format!(
            "frame total_length {} is smaller than minimum {min_frame_size}",
            frame_total
        )));
    }

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

    // Frame header — 0.17+ emits `NTensorMaskedFrame` (type 9) for
    // every new data-object frame.  When the descriptor carries no
    // `masks` sub-map the on-wire layout matches pre-0.17 type 4
    // byte-for-byte except for the type number.  Mask sections
    // (when present) live between the payload and the CBOR
    // descriptor, located by offsets in `descriptor.masks`.
    let fh = FrameHeader {
        frame_type: FrameType::NTensorMaskedFrame,
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

/// Decode a data object frame, returning the descriptor, the
/// (trimmed) data-payload slice, the raw mask-region slice, and the
/// total bytes consumed (including alignment padding).
///
/// The mask-region slice contains the bytes between the data payload
/// and the CBOR descriptor — non-empty only when
/// `descriptor.masks.is_some()`, per the `NTensorMaskedFrame` layout
/// in `plans/BITMASK_FRAME.md` §3.2.  Callers that only care about
/// the data payload can ignore it.
///
/// `buf` must start at the frame header.
pub fn decode_data_object_frame(buf: &[u8]) -> Result<(DataObjectDescriptor, &[u8], &[u8], usize)> {
    let fh = FrameHeader::read_from(buf)?;
    // Accept both legacy `NTensorFrame` (type 4) and the 0.17+
    // `NTensorMaskedFrame` (type 9).  Type 4 frames always yield an
    // empty mask region; type 9 frames may carry mask blobs between
    // the payload and the CBOR descriptor located via
    // `descriptor.masks`.
    if !fh.frame_type.is_data_object() {
        return Err(TensogramError::Framing(format!(
            "expected data-object frame (type 4 or 9), got {:?}",
            fh.frame_type
        )));
    }

    let frame_total = usize::try_from(fh.total_length).map_err(|_| {
        TensogramError::Framing(format!(
            "data object frame total_length {} overflows usize",
            fh.total_length
        ))
    })?;
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
    if endf_start < 8 {
        return Err(TensogramError::Framing(format!(
            "data object frame too small for cbor_offset: endf_start={endf_start} < 8"
        )));
    }
    let cbor_offset_pos = endf_start - 8;
    // cbor_offset_pos is guaranteed >= 0 (checked endf_start >= 8 above),
    // and cbor_offset_pos + 8 <= endf_start <= frame_total <= buf.len(),
    // so read_u64_be is safe.
    let cbor_offset_raw = crate::wire::read_u64_be(buf, cbor_offset_pos);
    let cbor_offset = usize::try_from(cbor_offset_raw).map_err(|_| {
        TensogramError::Framing(format!("cbor_offset {cbor_offset_raw} overflows usize"))
    })?;

    // Validate cbor_offset points within the frame body
    if cbor_offset < FRAME_HEADER_SIZE || cbor_offset > cbor_offset_pos {
        return Err(TensogramError::Framing(format!(
            "cbor_offset {cbor_offset} out of valid range [{FRAME_HEADER_SIZE}, {cbor_offset_pos}]"
        )));
    }

    let cbor_after = fh.flags & DataObjectFlags::CBOR_AFTER_PAYLOAD != 0;

    let (descriptor, payload_slice, mask_region) = if cbor_after {
        // Layout: header(16) | payload_region | cbor | cbor_offset(8) | ENDF(4)
        //
        // `payload_region` = [encoded_payload][mask_nan][mask_inf+][mask_inf-]
        // when descriptor.masks is Some; just [encoded_payload] otherwise.
        // See `plans/BITMASK_FRAME.md` §3.2.
        let payload_start = FRAME_HEADER_SIZE;
        let cbor_start = cbor_offset;
        let cbor_end = cbor_offset_pos;
        let cbor_slice = &buf[cbor_start..cbor_end];
        let desc = metadata::cbor_to_object_descriptor(cbor_slice)?;
        let payload_end = mask_aware_payload_end(payload_start, cbor_start, &desc)?;
        (
            desc,
            &buf[payload_start..payload_end],
            &buf[payload_end..cbor_start],
        )
    } else {
        // Layout: header(16) | cbor | payload_region | cbor_offset(8) | ENDF(4)
        // Use Cursor to measure exact consumed CBOR bytes on the wire.
        // Re-serialization would produce different lengths for non-canonical CBOR.
        let cbor_start = cbor_offset;
        let region = &buf[cbor_start..cbor_offset_pos];
        let mut cursor = std::io::Cursor::new(region);
        let cbor_value: ciborium::Value = ciborium::from_reader(&mut cursor).map_err(|e| {
            TensogramError::Metadata(format!("failed to parse object descriptor CBOR: {e}"))
        })?;
        let cbor_len = usize::try_from(cursor.position()).map_err(|_| {
            TensogramError::Metadata("CBOR descriptor length overflows usize".to_string())
        })?;
        let payload_start = cbor_start + cbor_len;
        // Deserialize directly from parsed Value — avoids a second CBOR parse
        let desc: DataObjectDescriptor = cbor_value.deserialized().map_err(|e| {
            TensogramError::Metadata(format!("failed to deserialize descriptor: {e}"))
        })?;
        let payload_end = mask_aware_payload_end(payload_start, cbor_offset_pos, &desc)?;
        (
            desc,
            &buf[payload_start..payload_end],
            &buf[payload_end..cbor_offset_pos],
        )
    };

    // Bytes consumed, including padding
    let mut consumed = frame_total;
    let aligned = (consumed + 7) & !7;
    if aligned <= buf.len() {
        consumed = aligned;
    }

    Ok((descriptor, payload_slice, mask_region, consumed))
}

// ── Message-level encode (buffered mode) ─────────────────────────────────────

/// Encoded data object: descriptor + encoded payload bytes.
pub struct EncodedObject {
    pub descriptor: DataObjectDescriptor,
    pub encoded_payload: Vec<u8>,
}

/// Build hash frame CBOR if any objects carry hashes.
fn build_hash_frame_cbor(objects: &[EncodedObject]) -> Result<Option<Vec<u8>>> {
    let has_hashes = objects.iter().any(|o| o.descriptor.hash.is_some());
    if !has_hashes {
        return Ok(None);
    }

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
    Ok(Some(metadata::hash_frame_to_cbor(&hf)?))
}

/// Two-pass index construction: compute data object offsets accounting for
/// the index frame's own size, which depends on the offsets.
fn build_index_frame(
    header_size_no_index: usize,
    object_frames: &[Vec<u8>],
) -> Result<Option<Vec<u8>>> {
    if object_frames.is_empty() {
        return Ok(None);
    }

    let frame_lengths: Vec<u64> = object_frames.iter().map(|f| f.len() as u64).collect();

    // First estimate: use dummy offsets of 0 to estimate CBOR size
    let dummy_idx = IndexFrame {
        object_count: object_frames.len() as u64,
        offsets: vec![0u64; object_frames.len()],
        lengths: frame_lengths.clone(),
    };
    let dummy_cbor = metadata::index_to_cbor(&dummy_idx)?;
    let dummy_frame_size = aligned_frame_total_size(dummy_cbor.len());
    let data_cursor = header_size_no_index + dummy_frame_size;

    // Compute object offsets with the estimated index size
    let offsets = compute_object_offsets(data_cursor, object_frames);

    // Build real index with actual offsets
    let real_idx = IndexFrame {
        object_count: object_frames.len() as u64,
        offsets,
        lengths: frame_lengths.clone(),
    };
    let real_cbor = metadata::index_to_cbor(&real_idx)?;

    // If offset values changed CBOR size, recompute once more
    let final_cbor = if real_cbor.len() != dummy_cbor.len() {
        let real_frame_size = aligned_frame_total_size(real_cbor.len());
        let new_data_cursor = header_size_no_index + real_frame_size;
        let new_offsets = compute_object_offsets(new_data_cursor, object_frames);
        let final_idx = IndexFrame {
            object_count: object_frames.len() as u64,
            offsets: new_offsets,
            lengths: frame_lengths,
        };
        let third_cbor = metadata::index_to_cbor(&final_idx)?;
        // Guard: a third size change means offsets crossed another CBOR
        // integer encoding tier, invalidating the layout.
        if aligned_frame_total_size(third_cbor.len()) != real_frame_size {
            return Err(TensogramError::Framing(
                "index CBOR size changed unexpectedly on third pass".to_string(),
            ));
        }
        third_cbor
    } else {
        real_cbor
    };

    let mut idx_frame = Vec::new();
    write_frame(
        &mut idx_frame,
        FrameType::HeaderIndex,
        1,
        0,
        &final_cbor,
        true,
    );
    Ok(Some(idx_frame))
}

/// Compute byte offsets for each object frame, accounting for 8-byte alignment.
fn compute_object_offsets(start: usize, object_frames: &[Vec<u8>]) -> Vec<u64> {
    let mut offsets = Vec::with_capacity(object_frames.len());
    let mut cursor = start;
    for frame in object_frames {
        offsets.push(cursor as u64);
        cursor += frame.len();
        cursor = (cursor + 7) & !7;
    }
    offsets
}

/// Compute message flags from the presence of optional frames.
fn compute_message_flags(has_index: bool, has_hashes: bool) -> MessageFlags {
    let mut flags = MessageFlags::default();
    flags.set(MessageFlags::HEADER_METADATA);
    if has_index {
        flags.set(MessageFlags::HEADER_INDEX);
    }
    if has_hashes {
        flags.set(MessageFlags::HEADER_HASHES);
    }
    flags
}

/// Assemble the final message buffer from pre-computed components.
fn assemble_message(
    flags: MessageFlags,
    meta_cbor: &[u8],
    index_frame_bytes: Option<&[u8]>,
    hash_cbor: Option<&[u8]>,
    object_frames: &[Vec<u8>],
) -> Vec<u8> {
    let mut out = Vec::new();

    // Preamble placeholder (patched after we know total_length)
    let preamble_pos = out.len();
    out.extend_from_slice(&[0u8; PREAMBLE_SIZE]);

    // Header metadata frame
    write_frame(&mut out, FrameType::HeaderMetadata, 1, 0, meta_cbor, true);

    // Header index frame (between metadata and hash, per spec ordering)
    if let Some(idx_bytes) = index_frame_bytes {
        out.extend_from_slice(idx_bytes);
    }

    // Header hash frame
    if let Some(h_cbor) = hash_cbor {
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

    out
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
    // Serialize metadata CBOR
    let meta_cbor = metadata::global_metadata_to_cbor(global_meta)?;

    // Pre-encode all data object frames
    let mut object_frames: Vec<Vec<u8>> = Vec::with_capacity(objects.len());
    for obj in objects {
        let frame = encode_data_object_frame(&obj.descriptor, &obj.encoded_payload, false)?;
        object_frames.push(frame);
    }

    // Build hash frame CBOR (if any objects have hashes)
    let hash_cbor = build_hash_frame_cbor(objects)?;

    // Measure header size without index to feed the two-pass index builder
    let mut header_no_index = Vec::new();
    header_no_index.extend_from_slice(&[0u8; PREAMBLE_SIZE]);
    write_frame(
        &mut header_no_index,
        FrameType::HeaderMetadata,
        1,
        0,
        &meta_cbor,
        true,
    );
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

    // Two-pass index construction
    let index_frame_bytes = build_index_frame(header_no_index.len(), &object_frames)?;

    // Compute flags and assemble
    let flags = compute_message_flags(index_frame_bytes.is_some(), hash_cbor.is_some());

    Ok(assemble_message(
        flags,
        &meta_cbor,
        index_frame_bytes.as_deref(),
        hash_cbor.as_deref(),
        &object_frames,
    ))
}

// ── Message-level decode ─────────────────────────────────────────────────────

/// A decoded message with all components.
#[derive(Debug)]
pub struct DecodedMessage<'a> {
    pub preamble: Preamble,
    pub global_metadata: GlobalMetadata,
    pub index: Option<IndexFrame>,
    pub hash_frame: Option<HashFrame>,
    /// (descriptor, data_payload_slice, mask_region_slice, frame_offset_in_message)
    ///
    /// `mask_region_slice` is empty when the frame has no masks;
    /// otherwise it holds the raw compressed mask bytes.  See
    /// `plans/BITMASK_FRAME.md` §3.2 for the region layout.
    pub objects: Vec<(DataObjectDescriptor, &'a [u8], &'a [u8], usize)>,
    /// Per-object preceder metadata, parallel to `objects`.
    /// `Some(map)` if a PrecederMetadata frame preceded that object.
    /// After decode, these entries are merged into `global_metadata.base`
    /// (preceder wins over footer entries for the same object index).
    pub preceder_payloads: Vec<Option<BTreeMap<String, ciborium::Value>>>,
}

/// Decode phase tracks expected frame ordering.
/// Valid order: Headers → DataObjects → Footers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum DecodePhase {
    Headers = 0,
    DataObjects = 1,
    Footers = 2,
}

/// Returns the decode phase a frame type belongs to.
fn frame_phase(ft: FrameType) -> DecodePhase {
    match ft {
        FrameType::HeaderMetadata | FrameType::HeaderIndex | FrameType::HeaderHash => {
            DecodePhase::Headers
        }
        // PrecederMetadata lives alongside data-object frames — it must appear
        // immediately before the data-object frame it describes, within the
        // data phase.  Both NTensorFrame (legacy) and NTensorMaskedFrame
        // count as data-object frames.
        FrameType::NTensorFrame | FrameType::NTensorMaskedFrame | FrameType::PrecederMetadata => {
            DecodePhase::DataObjects
        }
        FrameType::FooterHash | FrameType::FooterIndex | FrameType::FooterMetadata => {
            DecodePhase::Footers
        }
    }
}

/// Decode a complete message from a buffer.
///
/// Validates that frames appear in the expected order:
/// header frames first, then data objects, then footer frames.
pub fn decode_message(buf: &[u8]) -> Result<DecodedMessage<'_>> {
    let preamble = Preamble::read_from(buf)?;

    // Validate total_length if non-zero
    if preamble.total_length > 0 {
        let total_len = usize::try_from(preamble.total_length).map_err(|_| {
            TensogramError::Framing(format!(
                "total_length {} overflows usize",
                preamble.total_length
            ))
        })?;
        if total_len > buf.len() {
            return Err(TensogramError::Framing(format!(
                "total_length {} exceeds buffer size {}",
                preamble.total_length,
                buf.len()
            )));
        }

        // Validate postamble
        let pa_offset = total_len - POSTAMBLE_SIZE;
        let _postamble = Postamble::read_from(&buf[pa_offset..])?;
    }

    let mut pos = PREAMBLE_SIZE;
    let msg_end = if preamble.total_length > 0 {
        // Safe: validated above that total_length fits in usize
        preamble.total_length as usize - POSTAMBLE_SIZE
    } else {
        buf.len().checked_sub(POSTAMBLE_SIZE).ok_or_else(|| {
            TensogramError::Framing(format!(
                "buffer too short for postamble: {} < {POSTAMBLE_SIZE}",
                buf.len()
            ))
        })?
    };

    let mut global_metadata: Option<GlobalMetadata> = None;
    let mut index: Option<IndexFrame> = None;
    let mut hash_frame: Option<HashFrame> = None;
    let mut objects: Vec<(DataObjectDescriptor, &[u8], &[u8], usize)> = Vec::new();
    let mut preceder_payloads: Vec<Option<BTreeMap<String, ciborium::Value>>> = Vec::new();
    let mut current_phase = DecodePhase::Headers;

    // Tracks a PrecederMetadata payload waiting for its DataObject.
    // Two consecutive preceders (without an intervening DataObject) are invalid.
    let mut pending_preceder: Option<BTreeMap<String, ciborium::Value>> = None;

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

        // Peek at frame type
        if pos + FRAME_HEADER_SIZE > buf.len() {
            break;
        }
        let fh = FrameHeader::read_from(&buf[pos..])?;

        // Validate frame ordering
        let phase = frame_phase(fh.frame_type);
        if phase < current_phase {
            return Err(TensogramError::Framing(format!(
                "unexpected {:?} frame after {:?} phase — frames must appear in order: headers, data objects, footers",
                fh.frame_type, current_phase
            )));
        }

        // A pending preceder must be followed by a DataObject, not a footer
        // or another preceder.
        if pending_preceder.is_some() && !fh.frame_type.is_data_object() {
            return Err(TensogramError::Framing(format!(
                "PrecederMetadata must be followed by a data-object frame, got {:?}",
                fh.frame_type
            )));
        }

        current_phase = phase;
        let frame_start = pos;

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
            FrameType::PrecederMetadata => {
                let (_, payload, consumed) = read_frame(&buf[pos..])?;
                let preceder_meta = metadata::cbor_to_global_metadata(payload)?;
                // Preceder base must have exactly one entry
                let n = preceder_meta.base.len();
                if n != 1 {
                    return Err(TensogramError::Metadata(format!(
                        "PrecederMetadata base must have exactly 1 entry, got {n}"
                    )));
                }
                // Safe: we just checked len == 1
                let mut entry = preceder_meta.base.into_iter().next().unwrap_or_default();
                // Strip _reserved_ from preceder — it's library-managed and
                // would collide with the encoder's auto-populated _reserved_.tensor.
                // The decoder is permissive: strip rather than reject, since the
                // data may come from a non-standard producer.
                entry.remove(RESERVED_KEY);
                pending_preceder = Some(entry);
                pos += consumed;
            }
            FrameType::NTensorFrame | FrameType::NTensorMaskedFrame => {
                // Both frame types share the same payload + CBOR layout;
                // `decode_data_object_frame` returns the trimmed data
                // payload and the (possibly empty) mask region slice.
                // Type 4 frames always have an empty mask region.
                let (desc, payload, mask_region, consumed) = decode_data_object_frame(&buf[pos..])?;
                objects.push((desc, payload, mask_region, frame_start));
                // Consume the pending preceder (if any) for this object
                preceder_payloads.push(pending_preceder.take());
                pos += consumed;
            }
        }
    }

    // A preceder at the end of the stream with no following data-object frame is invalid
    if pending_preceder.is_some() {
        return Err(TensogramError::Framing(
            "dangling PrecederMetadata: no data-object frame followed".to_string(),
        ));
    }

    let mut global_metadata = global_metadata.ok_or_else(|| {
        TensogramError::Metadata("no metadata frame found in message".to_string())
    })?;

    // Merge preceder payloads into global_metadata.base (preceder wins).
    // Key-level merge: preceder keys override footer keys on conflict,
    // but footer-only keys (e.g. _reserved_.tensor) are preserved when
    // absent from the preceder.
    let obj_count = objects.len();
    if global_metadata.base.len() > obj_count {
        return Err(TensogramError::Metadata(format!(
            "metadata base has {} entries but message contains {} objects",
            global_metadata.base.len(),
            obj_count
        )));
    }
    if global_metadata.base.len() < obj_count {
        global_metadata.base.resize_with(obj_count, BTreeMap::new);
    }
    for (i, preceder) in preceder_payloads.iter().enumerate() {
        if let Some(prec_map) = preceder {
            for (k, v) in prec_map {
                global_metadata.base[i].insert(k.clone(), v.clone());
            }
        }
    }

    Ok(DecodedMessage {
        preamble,
        global_metadata,
        index,
        hash_frame,
        objects,
        preceder_payloads,
    })
}

/// Decode only global metadata from a message, skipping data frames.
pub fn decode_metadata_only(buf: &[u8]) -> Result<GlobalMetadata> {
    let preamble = Preamble::read_from(buf)?;

    let mut pos = PREAMBLE_SIZE;
    let msg_end = if preamble.total_length > 0 {
        // Safe: on 64-bit usize == u64; on 32-bit the message would already
        // fail to fit in memory, so this truncation is acceptable.
        let total_len = usize::try_from(preamble.total_length).map_err(|_| {
            TensogramError::Framing(format!(
                "total_length {} overflows usize",
                preamble.total_length
            ))
        })?;
        total_len.checked_sub(POSTAMBLE_SIZE).ok_or_else(|| {
            TensogramError::Framing(format!(
                "total_length {} too small for postamble",
                preamble.total_length
            ))
        })?
    } else {
        buf.len().checked_sub(POSTAMBLE_SIZE).ok_or_else(|| {
            TensogramError::Framing(format!(
                "buffer too short for postamble: {} < {POSTAMBLE_SIZE}",
                buf.len()
            ))
        })?
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
                let frame_total = usize::try_from(fh.total_length).map_err(|_| {
                    TensogramError::Framing(format!(
                        "frame total_length {} overflows usize",
                        fh.total_length
                    ))
                })?;
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
#[tracing::instrument(skip(buf), fields(buf_len = buf.len()))]
pub fn scan(buf: &[u8]) -> Vec<(usize, usize)> {
    let mut messages = Vec::new();
    let mut pos = 0;

    while pos + PREAMBLE_SIZE + POSTAMBLE_SIZE <= buf.len() {
        if &buf[pos..pos + MAGIC.len()] == MAGIC {
            // Try to read preamble
            if let Ok(preamble) = Preamble::read_from(&buf[pos..]) {
                if preamble.total_length > 0 {
                    let Ok(total) = usize::try_from(preamble.total_length) else {
                        pos += 1;
                        continue;
                    };
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

// ── File-based scan ──────────────────────────────────────────────────────────

/// Scan a file for message boundaries without loading the entire file into memory.
///
/// Reads preamble-sized chunks and seeks forward, avoiding full-file reads
/// for large files. Returns the same `(offset, length)` pairs as `scan()`.
pub fn scan_file(file: &mut (impl std::io::Read + std::io::Seek)) -> Result<Vec<(usize, usize)>> {
    use std::io::SeekFrom;

    let file_len_u64 = file.seek(SeekFrom::End(0))?;
    let file_len = usize::try_from(file_len_u64).map_err(|_| {
        TensogramError::Framing(format!("file size {file_len_u64} overflows usize"))
    })?;
    file.seek(SeekFrom::Start(0))?;

    let mut messages = Vec::new();
    let mut pos: usize = 0;

    let mut preamble_buf = [0u8; PREAMBLE_SIZE];

    while pos + PREAMBLE_SIZE + POSTAMBLE_SIZE <= file_len {
        file.seek(SeekFrom::Start(pos as u64))?;
        if file.read_exact(&mut preamble_buf).is_err() {
            break;
        }

        if &preamble_buf[..MAGIC.len()] == MAGIC
            && let Ok(preamble) = Preamble::read_from(&preamble_buf)
        {
            if preamble.total_length > 0 {
                let Ok(total) = usize::try_from(preamble.total_length) else {
                    pos += 1;
                    continue;
                };
                if pos + total <= file_len {
                    // Read end magic to validate
                    let end_magic_offset = pos + total - 8;
                    file.seek(SeekFrom::Start(end_magic_offset as u64))?;
                    let mut end_buf = [0u8; 8];
                    if file.read_exact(&mut end_buf).is_ok() && &end_buf == crate::wire::END_MAGIC {
                        messages.push((pos, total));
                        pos += total;
                        continue;
                    }
                }
            } else {
                // Streaming mode: scan forward for END_MAGIC
                // Read in chunks to find the terminator
                let mut search_pos = pos + PREAMBLE_SIZE;
                let mut found = false;
                let chunk_size = 4096;
                let mut chunk = vec![0u8; chunk_size];

                while search_pos + 8 <= file_len {
                    file.seek(SeekFrom::Start(search_pos as u64))?;
                    let to_read = (file_len - search_pos).min(chunk_size);
                    let buf = &mut chunk[..to_read];
                    if file.read_exact(buf).is_err() {
                        break;
                    }

                    // Search for END_MAGIC in this chunk
                    for i in 0..to_read.saturating_sub(7) {
                        if &buf[i..i + 8] == crate::wire::END_MAGIC {
                            let end_pos = search_pos + i;
                            let msg_len = end_pos + 8 - pos;
                            messages.push((pos, msg_len));
                            pos = end_pos + 8;
                            found = true;
                            break;
                        }
                    }
                    if found {
                        break;
                    }
                    // Overlap by 7 bytes to catch END_MAGIC spanning chunks
                    search_pos += to_read.saturating_sub(7);
                }
                if found {
                    continue;
                }
            }
        }
        pos += 1;
    }

    Ok(messages)
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
            ..Default::default()
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
            masks: None,
            hash: None,
        }
    }

    #[test]
    fn test_data_object_frame_round_trip_cbor_after() {
        let desc = make_descriptor(vec![4]);
        let payload = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

        let frame = encode_data_object_frame(&desc, &payload, false).unwrap();

        let (decoded_desc, decoded_payload, _mask_region, consumed) =
            decode_data_object_frame(&frame).unwrap();
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

        let (decoded_desc, decoded_payload, _mask_region, _) =
            decode_data_object_frame(&frame).unwrap();
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
        // _extra_ is the CBOR key name (via serde rename)
        assert!(decoded_meta.extra.contains_key("test_key"));
    }

    // ── Phase 2: Frame ordering validation tests ─────────────────────────

    /// Helper: build a raw message from manually ordered frames.
    fn build_raw_message(frames: &[(&[u8], FrameType)]) -> Vec<u8> {
        let meta = make_global_meta();
        let meta_cbor = crate::metadata::global_metadata_to_cbor(&meta).unwrap();
        let desc = make_descriptor(vec![4]);
        let payload = vec![0u8; 16];

        let mut out = Vec::new();
        // Preamble placeholder
        out.extend_from_slice(&[0u8; PREAMBLE_SIZE]);

        for (content, frame_type) in frames {
            match frame_type {
                FrameType::NTensorFrame => {
                    let frame = encode_data_object_frame(&desc, &payload, false).unwrap();
                    out.extend_from_slice(&frame);
                    let pad = (8 - (out.len() % 8)) % 8;
                    out.extend(std::iter::repeat_n(0u8, pad));
                }
                _ => {
                    let data = if content.is_empty() {
                        &meta_cbor
                    } else {
                        *content
                    };
                    write_frame(&mut out, *frame_type, 1, 0, data, true);
                }
            }
        }

        // Postamble
        let postamble_offset = out.len();
        let postamble = Postamble {
            first_footer_offset: postamble_offset as u64,
        };
        postamble.write_to(&mut out);

        let total_length = out.len() as u64;
        let mut flags = MessageFlags::default();
        flags.set(MessageFlags::HEADER_METADATA);
        let preamble = Preamble {
            version: 2,
            flags,
            reserved: 0,
            total_length,
        };
        let mut preamble_bytes = Vec::new();
        preamble.write_to(&mut preamble_bytes);
        out[0..PREAMBLE_SIZE].copy_from_slice(&preamble_bytes);

        out
    }

    #[test]
    fn test_decode_rejects_header_after_data_object() {
        // DataObject before HeaderMetadata — should fail
        let msg = build_raw_message(&[
            (&[], FrameType::NTensorFrame),
            (&[], FrameType::HeaderMetadata),
        ]);
        let result = decode_message(&msg);
        assert!(
            result.is_err(),
            "header frame after data object should be rejected"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("order") || err.contains("unexpected"),
            "error should mention ordering: {err}"
        );
    }

    #[test]
    fn test_decode_rejects_data_object_after_footer() {
        // HeaderMetadata, FooterHash, then DataObject — should fail
        let meta = make_global_meta();
        let meta_cbor = crate::metadata::global_metadata_to_cbor(&meta).unwrap();
        let hf = HashFrame {
            object_count: 0,
            hash_type: "xxh3".to_string(),
            hashes: vec![],
        };
        let hash_cbor = crate::metadata::hash_frame_to_cbor(&hf).unwrap();

        let msg = build_raw_message(&[
            (&meta_cbor, FrameType::HeaderMetadata),
            (&hash_cbor, FrameType::FooterHash),
            (&[], FrameType::NTensorFrame),
        ]);
        let result = decode_message(&msg);
        assert!(
            result.is_err(),
            "data object after footer should be rejected"
        );
    }

    #[test]
    fn test_decode_accepts_valid_frame_order() {
        // HeaderMetadata → DataObject — canonical order
        let msg = build_raw_message(&[
            (&[], FrameType::HeaderMetadata),
            (&[], FrameType::NTensorFrame),
        ]);
        let result = decode_message(&msg);
        assert!(
            result.is_ok(),
            "valid frame order should be accepted: {:?}",
            result.err()
        );
    }

    // ── Phase 3: Streaming scan_file tests ─────────────────────────────

    #[test]
    fn test_scan_file_matches_scan_buffer() {
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

        // Compare scan() (buffer) vs scan_file() (seeking)
        let buffer_offsets = scan(&buf);

        let mut cursor = std::io::Cursor::new(&buf);
        let file_offsets = scan_file(&mut cursor).unwrap();

        assert_eq!(buffer_offsets, file_offsets);
    }

    #[test]
    fn test_scan_file_with_garbage() {
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

        let mut cursor = std::io::Cursor::new(&buf);
        let offsets = scan_file(&mut cursor).unwrap();
        assert_eq!(offsets.len(), 1);
        assert_eq!(offsets[0], (10, msg.len()));
    }

    #[test]
    fn test_scan_file_empty() {
        let buf: Vec<u8> = Vec::new();
        let mut cursor = std::io::Cursor::new(&buf);
        let offsets = scan_file(&mut cursor).unwrap();
        assert!(offsets.is_empty());
    }

    #[test]
    fn test_decode_accepts_footer_after_data_objects() {
        // HeaderMetadata → DataObject → FooterMetadata — valid streaming layout
        let meta = make_global_meta();
        let meta_cbor = crate::metadata::global_metadata_to_cbor(&meta).unwrap();

        let msg = build_raw_message(&[
            (&meta_cbor, FrameType::HeaderMetadata),
            (&[], FrameType::NTensorFrame),
            (&meta_cbor, FrameType::FooterMetadata),
        ]);
        let result = decode_message(&msg);
        assert!(
            result.is_ok(),
            "footer after data objects should be accepted: {:?}",
            result.err()
        );
    }

    // ── Phase 4: PrecederMetadata frame tests ────────────────────────────

    /// Helper: build a preceder metadata CBOR blob with a single base entry.
    fn make_preceder_cbor(entries: std::collections::BTreeMap<String, ciborium::Value>) -> Vec<u8> {
        let meta = GlobalMetadata {
            version: 2,
            base: vec![entries],
            ..Default::default()
        };
        crate::metadata::global_metadata_to_cbor(&meta).unwrap()
    }

    #[test]
    fn test_decode_preceder_before_data_object() {
        let mut entries = BTreeMap::new();
        entries.insert(
            "mars".to_string(),
            ciborium::Value::Map(vec![(
                ciborium::Value::Text("param".to_string()),
                ciborium::Value::Text("2t".to_string()),
            )]),
        );
        let preceder_cbor = make_preceder_cbor(entries);

        let msg = build_raw_message(&[
            (&[], FrameType::HeaderMetadata),
            (&preceder_cbor, FrameType::PrecederMetadata),
            (&[], FrameType::NTensorFrame),
        ]);
        let decoded = decode_message(&msg).unwrap();

        assert_eq!(decoded.objects.len(), 1);
        assert_eq!(decoded.preceder_payloads.len(), 1);
        assert!(decoded.preceder_payloads[0].is_some());

        // Verify preceder merged into global_metadata.base
        assert_eq!(decoded.global_metadata.base.len(), 1);
        assert!(decoded.global_metadata.base[0].contains_key("mars"));
    }

    #[test]
    fn test_decode_consecutive_preceders_rejected() {
        let entries = BTreeMap::new();
        let preceder_cbor = make_preceder_cbor(entries);

        let msg = build_raw_message(&[
            (&[], FrameType::HeaderMetadata),
            (&preceder_cbor, FrameType::PrecederMetadata),
            (&preceder_cbor, FrameType::PrecederMetadata),
            (&[], FrameType::NTensorFrame),
        ]);
        let result = decode_message(&msg);
        assert!(result.is_err(), "consecutive preceders should be rejected");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("PrecederMetadata") && err.contains("data-object"),
            "error should explain preceder must precede a data-object frame: {err}"
        );
    }

    #[test]
    fn test_decode_dangling_preceder_rejected() {
        let entries = BTreeMap::new();
        let preceder_cbor = make_preceder_cbor(entries);

        // Preceder at end of message without a following DataObject
        let msg = build_raw_message(&[
            (&[], FrameType::HeaderMetadata),
            (&[], FrameType::NTensorFrame),
            (&preceder_cbor, FrameType::PrecederMetadata),
        ]);
        let result = decode_message(&msg);
        assert!(result.is_err(), "dangling preceder should be rejected");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("dangling"),
            "error should mention dangling: {err}"
        );
    }

    #[test]
    fn test_decode_preceder_with_multiple_base_entries_rejected() {
        // Preceder with 2 base entries — should be rejected (must have exactly 1)
        let meta = GlobalMetadata {
            version: 2,
            base: vec![BTreeMap::new(), BTreeMap::new()],
            ..Default::default()
        };
        let bad_cbor = crate::metadata::global_metadata_to_cbor(&meta).unwrap();

        let msg = build_raw_message(&[
            (&[], FrameType::HeaderMetadata),
            (&bad_cbor, FrameType::PrecederMetadata),
            (&[], FrameType::NTensorFrame),
        ]);
        let result = decode_message(&msg);
        assert!(
            result.is_err(),
            "preceder with 2 payload entries should be rejected"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("exactly 1"),
            "error should mention 'exactly 1': {err}"
        );
    }

    #[test]
    fn test_decode_preceder_with_zero_base_entries_rejected() {
        // Preceder with 0 base entries — should be rejected
        let meta = GlobalMetadata {
            version: 2,
            base: vec![],
            ..Default::default()
        };
        let bad_cbor = crate::metadata::global_metadata_to_cbor(&meta).unwrap();

        let msg = build_raw_message(&[
            (&[], FrameType::HeaderMetadata),
            (&bad_cbor, FrameType::PrecederMetadata),
            (&[], FrameType::NTensorFrame),
        ]);
        let result = decode_message(&msg);
        assert!(
            result.is_err(),
            "preceder with 0 payload entries should be rejected"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("exactly 1") && err.contains("got 0"),
            "error should mention 'exactly 1' and 'got 0': {err}"
        );
    }

    #[test]
    fn test_decode_preceder_followed_by_footer_rejected() {
        let entries = BTreeMap::new();
        let preceder_cbor = make_preceder_cbor(entries);
        let meta = make_global_meta();
        let meta_cbor = crate::metadata::global_metadata_to_cbor(&meta).unwrap();

        let msg = build_raw_message(&[
            (&meta_cbor, FrameType::HeaderMetadata),
            (&preceder_cbor, FrameType::PrecederMetadata),
            (&meta_cbor, FrameType::FooterMetadata),
        ]);
        let result = decode_message(&msg);
        assert!(
            result.is_err(),
            "preceder followed by footer should be rejected"
        );
    }

    #[test]
    fn test_decode_mixed_preceder_and_no_preceder() {
        // Object 0: has preceder, Object 1: no preceder
        let mut entries = BTreeMap::new();
        entries.insert(
            "note".to_string(),
            ciborium::Value::Text("from preceder".to_string()),
        );
        let preceder_cbor = make_preceder_cbor(entries);

        let msg = build_raw_message(&[
            (&[], FrameType::HeaderMetadata),
            (&preceder_cbor, FrameType::PrecederMetadata),
            (&[], FrameType::NTensorFrame),
            (&[], FrameType::NTensorFrame),
        ]);
        let decoded = decode_message(&msg).unwrap();

        assert_eq!(decoded.objects.len(), 2);
        assert_eq!(decoded.preceder_payloads.len(), 2);
        assert!(decoded.preceder_payloads[0].is_some());
        assert!(decoded.preceder_payloads[1].is_none());

        // base[0] should have preceder entry, base[1] should be empty
        assert!(decoded.global_metadata.base[0].contains_key("note"));
        assert!(!decoded.global_metadata.base[1].contains_key("note"));
    }

    #[test]
    fn test_decode_preceder_wins_over_footer_payload() {
        // Build a message where both footer metadata and preceder provide
        // payload[0] — preceder should win.
        let mut prec_entries = BTreeMap::new();
        prec_entries.insert(
            "source".to_string(),
            ciborium::Value::Text("preceder".to_string()),
        );
        let preceder_cbor = make_preceder_cbor(prec_entries);

        // Footer metadata with different base[0]
        let mut footer_base = BTreeMap::new();
        footer_base.insert(
            "source".to_string(),
            ciborium::Value::Text("footer".to_string()),
        );
        let footer_meta = GlobalMetadata {
            version: 2,
            base: vec![footer_base],
            ..Default::default()
        };
        let footer_cbor = crate::metadata::global_metadata_to_cbor(&footer_meta).unwrap();

        let msg = build_raw_message(&[
            (&[], FrameType::HeaderMetadata),
            (&preceder_cbor, FrameType::PrecederMetadata),
            (&[], FrameType::NTensorFrame),
            (&footer_cbor, FrameType::FooterMetadata),
        ]);
        let decoded = decode_message(&msg).unwrap();

        // Footer metadata is parsed last, so global_metadata would have
        // footer base. But after merging, preceder wins.
        let source = decoded.global_metadata.base[0]
            .get("source")
            .and_then(|v| match v {
                ciborium::Value::Text(s) => Some(s.as_str()),
                _ => None,
            });
        assert_eq!(source, Some("preceder"), "preceder should win over footer");
    }

    #[test]
    fn test_decode_rejects_base_count_exceeding_objects() {
        // Footer metadata with 3 base entries but only 1 data object
        // should be rejected (base.len > obj_count).
        let footer_meta = GlobalMetadata {
            version: 2,
            base: vec![BTreeMap::new(), BTreeMap::new(), BTreeMap::new()],
            ..Default::default()
        };
        let footer_cbor = crate::metadata::global_metadata_to_cbor(&footer_meta).unwrap();

        let msg = build_raw_message(&[
            (&footer_cbor, FrameType::HeaderMetadata),
            (&[], FrameType::NTensorFrame),
        ]);
        let result = decode_message(&msg);
        assert!(
            result.is_err(),
            "base with more entries than objects should be rejected"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("3") && err.contains("1"),
            "error should mention counts: {err}"
        );
    }
}
