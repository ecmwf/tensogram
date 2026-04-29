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
    DATA_OBJECT_FOOTER_SIZE, DataObjectFlags, FRAME_COMMON_FOOTER_SIZE, FRAME_END,
    FRAME_HEADER_SIZE, FrameHeader, FrameType, MAGIC, MessageFlags, POSTAMBLE_SIZE, PREAMBLE_SIZE,
    Postamble, Preamble, WIRE_VERSION,
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

/// Write a non-data-object frame: frame_header + payload + hash_slot + ENDF.
///
/// The 8-byte hash slot is always written (new in v3, see
/// `plans/WIRE_FORMAT.md` §2.2).  When `hash_algorithm` is
/// `Some(_)` the slot is populated with the xxh3-64 digest of the
/// payload; when `None` it is written as zeros.  The preamble-level
/// `HASHES_PRESENT` flag tells readers which mode was used.
///
/// Optionally pads to 8-byte alignment after ENDF.
fn write_frame(
    out: &mut Vec<u8>,
    frame_type: FrameType,
    version: u16,
    flags: u16,
    payload: &[u8],
    hashing: bool,
    align: bool,
) {
    // Data-object frames have their own encoder; this helper only
    // handles frames with the common 12-byte footer.
    debug_assert!(
        !frame_type.is_data_object(),
        "write_frame is for non-data-object frames only"
    );
    // Non-data-object frame payloads are bounded CBOR (metadata,
    // index, hash) at most a few KiB; `u64` overflow here is
    // genuinely unreachable.  The debug_assert catches accidental
    // misuse on tiny-pointer-size targets (e.g. 16-bit test
    // platforms that this crate doesn't support but clippy still
    // checks against).
    debug_assert!(
        payload.len() <= u64::MAX as usize - FRAME_HEADER_SIZE - FRAME_COMMON_FOOTER_SIZE,
        "non-data-object frame payload too large: {}",
        payload.len()
    );
    let total_length = (FRAME_HEADER_SIZE + payload.len() + FRAME_COMMON_FOOTER_SIZE) as u64;

    let fh = FrameHeader {
        frame_type,
        version,
        flags,
        total_length,
    };
    fh.write_to(out);
    out.extend_from_slice(payload);

    // Inline hash slot (8 bytes) — hashes the body (just `payload`
    // for non-data-object frames).
    let hash_value: u64 = if hashing {
        xxhash_rust::xxh3::xxh3_64(payload)
    } else {
        0
    };
    out.extend_from_slice(&hash_value.to_be_bytes());

    // End marker.
    out.extend_from_slice(FRAME_END);

    if align {
        let pad = (8 - (out.len() % 8)) % 8;
        out.extend(std::iter::repeat_n(0u8, pad));
    }
}

/// Read one non-data-object frame from a buffer.
///
/// Returns `(FrameHeader, payload_slice, total_bytes_consumed)`
/// where `payload_slice` is the CBOR body *without* the inline
/// hash slot or ENDF marker.  `total_bytes_consumed` includes any
/// padding to the next 8-byte boundary.
///
/// Data-object frames go through [`decode_data_object_frame`]
/// instead — their footer has an additional `cbor_offset` field.
fn read_frame(buf: &[u8]) -> Result<(FrameHeader, &[u8], usize)> {
    let fh = FrameHeader::read_from(buf)?;
    let frame_total = usize::try_from(fh.total_length).map_err(|_| {
        TensogramError::Framing(format!(
            "frame total_length {} overflows usize",
            fh.total_length
        ))
    })?;

    // Minimum frame: header(16) + hash(8) + ENDF(4) = 28 bytes
    let min_frame_size = FRAME_HEADER_SIZE + FRAME_COMMON_FOOTER_SIZE;
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

    // Validate ENDF marker at the very end.
    let endf_start = frame_total - FRAME_END.len();
    if &buf[endf_start..frame_total] != FRAME_END {
        return Err(TensogramError::Framing(format!(
            "missing ENDF marker at offset {endf_start}"
        )));
    }

    // Payload excludes the 12-byte common footer (hash slot + ENDF).
    let payload_end = frame_total - FRAME_COMMON_FOOTER_SIZE;
    let payload = &buf[FRAME_HEADER_SIZE..payload_end];

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
/// v3 layout (see `plans/WIRE_FORMAT.md` §6.5):
///   `FrameHeader(16) | payload | cbor | cbor_offset(8) | hash(8) | ENDF(4)`
///
/// The `cbor_offset` in the footer is the byte offset from frame
/// start to the start of the CBOR descriptor.  The `hash` slot is
/// populated with xxh3-64 of the body when `hash_algorithm` is
/// `Some(_)`, or `0x0000000000000000` when `None` (see hash-scope
/// rule in `plans/WIRE_FORMAT.md` §2.4 — scope excludes the header,
/// `cbor_offset`, hash slot, and ENDF).
pub fn encode_data_object_frame(
    descriptor: &DataObjectDescriptor,
    payload: &[u8],
    cbor_before: bool,
    hashing: bool,
) -> Result<Vec<u8>> {
    let cbor_bytes = metadata::object_descriptor_to_cbor(descriptor)?;
    let flags = if cbor_before {
        0
    } else {
        DataObjectFlags::CBOR_AFTER_PAYLOAD
    };

    // Calculate the total frame length (v3 data-object footer = 20 B):
    //   header(16) + body + footer(20)
    // where body = payload + cbor_bytes and footer =
    // cbor_offset(8) + hash(8) + ENDF(4).
    //
    // Checked arithmetic here guards against pathological payloads
    // (huge CBOR + tensor bytes) from silently wrapping the
    // `total_length` field or panicking the process.  The
    // resulting error is mapped to `TensogramError::Framing` so the
    // caller sees a clean message rather than a debug-build panic.
    let total_length: u64 = [
        FRAME_HEADER_SIZE,
        payload.len(),
        cbor_bytes.len(),
        DATA_OBJECT_FOOTER_SIZE,
    ]
    .into_iter()
    .try_fold(0u64, |acc, part| acc.checked_add(part as u64))
    .ok_or_else(|| {
        TensogramError::Framing(format!(
            "data-object frame total_length overflows u64 \
             (payload {} bytes, CBOR {} bytes, framing {} bytes)",
            payload.len(),
            cbor_bytes.len(),
            FRAME_HEADER_SIZE + DATA_OBJECT_FOOTER_SIZE
        ))
    })?;

    let mut out = Vec::with_capacity(total_length as usize);

    // Frame header — v3 emits `NTensorFrame` (type 9) for every new
    // data-object frame.  When the descriptor carries no `masks`
    // sub-map the payload region holds only the encoded tensor
    // bytes.  When masks are present they live between the payload
    // and the CBOR descriptor, located by offsets in
    // `descriptor.masks`.
    let fh = FrameHeader {
        frame_type: FrameType::NTensorFrame,
        version: 1,
        flags,
        total_length,
    };
    fh.write_to(&mut out);

    if cbor_before {
        // CBOR descriptor first, then payload
        out.extend_from_slice(&cbor_bytes);
        out.extend_from_slice(payload);
    } else {
        // Payload first, then CBOR descriptor (default)
        out.extend_from_slice(payload);
        out.extend_from_slice(&cbor_bytes);
    }

    // Compute the inline hash now, *before* writing the footer
    // fields.  The hash scope is exactly the body bytes we've
    // appended since the header end; cbor_offset and the hash slot
    // itself live in the footer and are not in scope
    // (`plans/WIRE_FORMAT.md` §2.4).
    let hash_value: u64 = if hashing {
        xxhash_rust::xxh3::xxh3_64(&out[FRAME_HEADER_SIZE..])
    } else {
        0
    };

    // Footer: [cbor_offset u64][hash u64][ENDF 4].
    let cbor_offset = if cbor_before {
        FRAME_HEADER_SIZE as u64
    } else {
        (FRAME_HEADER_SIZE + payload.len()) as u64
    };
    out.extend_from_slice(&cbor_offset.to_be_bytes());
    out.extend_from_slice(&hash_value.to_be_bytes());
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
/// `descriptor.masks.is_some()`, per the `NTensorFrame` layout in
/// `plans/WIRE_FORMAT.md` §6.5.  Callers that only care about the
/// data payload can ignore it.
///
/// `buf` must start at the frame header.
pub fn decode_data_object_frame(buf: &[u8]) -> Result<(DataObjectDescriptor, &[u8], &[u8], usize)> {
    let fh = FrameHeader::read_from(buf)?;
    // Only `NTensorFrame` (type 9) is a valid data-object frame in
    // v3; other types hit the `is_data_object() == false` branch
    // below.  Type 4 (obsolete v2 NTensorFrame) never reaches here
    // because `FrameHeader::read_from` rejects it at the registry
    // lookup.
    if !fh.frame_type.is_data_object() {
        return Err(TensogramError::Framing(format!(
            "expected data-object frame (type 9 NTensorFrame in v3), got {:?}",
            fh.frame_type
        )));
    }

    let frame_total = usize::try_from(fh.total_length).map_err(|_| {
        TensogramError::Framing(format!(
            "data object frame total_length {} overflows usize",
            fh.total_length
        ))
    })?;
    // v3 minimum: header(16) + cbor_offset(8) + hash(8) + ENDF(4) = 36
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

    // Validate ENDF marker at the very end.
    let endf_start = frame_total - FRAME_END.len();
    if &buf[endf_start..frame_total] != FRAME_END {
        return Err(TensogramError::Framing(
            "missing ENDF marker in data object frame".to_string(),
        ));
    }

    // v3 footer layout (data-object): `[cbor_offset u64][hash u64][ENDF 4]`.
    //
    // The two u64 slots precede the 4-byte ENDF.  Using a named
    // constant here keeps the offset math linked to the single
    // source of truth in `wire::DATA_OBJECT_FOOTER_SIZE`: a future
    // footer-layout change triggers compile errors at every read
    // site rather than letting them silently drift.
    const DATA_OBJECT_FOOTER_PRE_ENDF: usize = DATA_OBJECT_FOOTER_SIZE - FRAME_END.len();
    if endf_start < DATA_OBJECT_FOOTER_PRE_ENDF {
        return Err(TensogramError::Framing(format!(
            "data object frame too small for v3 footer: \
             endf_start ({endf_start}) < footer-fields-size ({DATA_OBJECT_FOOTER_PRE_ENDF})"
        )));
    }
    // `cbor_offset` sits at the very start of the type-specific
    // footer; `hash` is 8 bytes later.  Derived layout, not magic
    // numbers.
    let cbor_offset_pos = endf_start - DATA_OBJECT_FOOTER_PRE_ENDF;
    let cbor_offset_raw = crate::wire::read_u64_be(buf, cbor_offset_pos);
    let cbor_offset = usize::try_from(cbor_offset_raw).map_err(|_| {
        TensogramError::Framing(format!("cbor_offset {cbor_offset_raw} overflows usize"))
    })?;

    // Validate cbor_offset points within the frame body (before
    // the 16-byte type-specific footer portion).
    if cbor_offset < FRAME_HEADER_SIZE || cbor_offset > cbor_offset_pos {
        return Err(TensogramError::Framing(format!(
            "cbor_offset {cbor_offset} out of valid range [{FRAME_HEADER_SIZE}, {cbor_offset_pos}]"
        )));
    }

    let cbor_after = fh.flags & DataObjectFlags::CBOR_AFTER_PAYLOAD != 0;

    let (descriptor, payload_slice, mask_region) = if cbor_after {
        // v3 layout:
        //   header(16) | payload_region | cbor | cbor_offset(8) | hash(8) | ENDF(4)
        //
        // `payload_region` = [encoded_payload][mask_nan][mask_inf+][mask_inf-]
        // when descriptor.masks is Some; just [encoded_payload] otherwise.
        // See `plans/WIRE_FORMAT.md` §6.5.
        let payload_start = FRAME_HEADER_SIZE;
        let cbor_start = cbor_offset;
        let cbor_end = cbor_offset_pos; // v3: end-of-CBOR = start-of-footer (end-16)
        let cbor_slice = &buf[cbor_start..cbor_end];
        let desc = metadata::cbor_to_object_descriptor(cbor_slice)?;
        let payload_end = mask_aware_payload_end(payload_start, cbor_start, &desc)?;
        (
            desc,
            &buf[payload_start..payload_end],
            &buf[payload_end..cbor_start],
        )
    } else {
        // v3 layout:
        //   header(16) | cbor | payload_region | cbor_offset(8) | hash(8) | ENDF(4)
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

/// Build the aggregate `HeaderHash` / `FooterHash` CBOR from a
/// list of pre-computed data-object frames.
///
/// Walks each frame, reads its inline hash slot at
/// `frame_end − 12`, and renders the 8-byte digest as a lowercase
/// 16-character hex string.  When `hash_algorithm` is `None` or
/// no frames have populated hash slots, returns `Ok(None)`.
///
/// The helper is cheap (no hashing) because the inline slot was
/// already computed at frame-encode time.
fn build_hash_frame_cbor(object_frames: &[Vec<u8>], hashing: bool) -> Result<Option<Vec<u8>>> {
    use crate::wire::{FRAME_COMMON_FOOTER_SIZE, read_u64_be};

    if !hashing {
        return Ok(None);
    }
    if object_frames.is_empty() {
        return Ok(None);
    }

    let mut hashes = Vec::with_capacity(object_frames.len());
    for frame in object_frames {
        if frame.len() < FRAME_COMMON_FOOTER_SIZE {
            return Err(TensogramError::Framing(format!(
                "data object frame too small to read inline hash slot: {} < {}",
                frame.len(),
                FRAME_COMMON_FOOTER_SIZE
            )));
        }
        let slot = frame.len() - FRAME_COMMON_FOOTER_SIZE;
        let digest = read_u64_be(frame, slot);
        hashes.push(crate::hash::format_xxh3_digest(digest));
    }

    let hf = HashFrame {
        algorithm: crate::hash::HASH_ALGORITHM_NAME.to_string(),
        hashes,
    };
    Ok(Some(metadata::hash_frame_to_cbor(&hf)?))
}

/// Two-pass index construction: compute data object offsets accounting for
/// the index frame's own size, which depends on the offsets.
fn build_index_frame(
    header_size_no_index: usize,
    object_frames: &[Vec<u8>],
    hashing: bool,
) -> Result<Option<Vec<u8>>> {
    if object_frames.is_empty() {
        return Ok(None);
    }

    let frame_lengths: Vec<u64> = object_frames.iter().map(|f| f.len() as u64).collect();

    // First estimate: use dummy offsets of 0 to estimate CBOR size
    let dummy_idx = IndexFrame {
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
        hashing,
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
#[allow(clippy::too_many_arguments)]
fn assemble_message(
    flags: MessageFlags,
    meta_cbor: &[u8],
    index_frame_bytes: Option<&[u8]>,
    header_hash_cbor: Option<&[u8]>,
    footer_hash_cbor: Option<&[u8]>,
    object_frames: &[Vec<u8>],
    hashing: bool,
) -> Vec<u8> {
    let mut out = Vec::new();

    // Preamble placeholder (patched after we know total_length)
    let preamble_pos = out.len();
    out.extend_from_slice(&[0u8; PREAMBLE_SIZE]);

    // Header metadata frame
    write_frame(
        &mut out,
        FrameType::HeaderMetadata,
        1,
        0,
        meta_cbor,
        hashing,
        true,
    );

    // Header index frame (between metadata and hash, per spec ordering)
    if let Some(idx_bytes) = index_frame_bytes {
        out.extend_from_slice(idx_bytes);
    }

    // Header hash frame
    if let Some(h_cbor) = header_hash_cbor {
        write_frame(&mut out, FrameType::HeaderHash, 1, 0, h_cbor, hashing, true);
    }

    // Data object frames with inter-frame alignment
    for (i, frame) in object_frames.iter().enumerate() {
        out.extend_from_slice(frame);
        if i + 1 < object_frames.len() {
            let pad = (8 - (out.len() % 8)) % 8;
            out.extend(std::iter::repeat_n(0u8, pad));
        }
    }
    // Align to 8 bytes after the last data object, before any footer
    // frames / postamble.
    let pad = (8 - (out.len() % 8)) % 8;
    out.extend(std::iter::repeat_n(0u8, pad));

    // Footer frames region (v3 supports buffered-mode FooterHash
    // for callers who opt in).  `first_footer_offset` points at
    // the first footer frame, or at the postamble itself when no
    // footer frames exist.
    let footer_start_offset = out.len();
    if let Some(h_cbor) = footer_hash_cbor {
        write_frame(&mut out, FrameType::FooterHash, 1, 0, h_cbor, hashing, true);
    }

    // Postamble.  `first_footer_offset` = the footer-frames region
    // start (or the postamble offset if there were no footer frames).
    let postamble_offset = out.len();
    let first_footer_offset = if footer_hash_cbor.is_some() {
        footer_start_offset as u64
    } else {
        postamble_offset as u64
    };
    let postamble_placeholder = Postamble {
        first_footer_offset,
        total_length: 0,
    };
    postamble_placeholder.write_to(&mut out);

    let total_length = out.len() as u64;

    // Patch the preamble with the real values.
    let preamble = Preamble {
        version: WIRE_VERSION,
        flags,
        reserved: 0,
        total_length,
    };
    let mut preamble_bytes = Vec::new();
    preamble.write_to(&mut preamble_bytes);
    out[preamble_pos..preamble_pos + PREAMBLE_SIZE].copy_from_slice(&preamble_bytes);

    // Patch the postamble's `total_length` — buffered mode always
    // knows the final size, so this field is always non-zero here.
    // The mirrored value enables O(1) backward scan (v3, §9.2).
    let total_length_bytes = total_length.to_be_bytes();
    out[postamble_offset + 8..postamble_offset + 16].copy_from_slice(&total_length_bytes);

    out
}

/// Hash-frame-emission policy passed through `encode_message`.
///
/// Buffered mode supports both header and footer hash frames; the
/// streaming encoder (which has no access to future bytes when it
/// writes the header) only emits the footer frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct HashFramePolicy {
    /// Emit a `HeaderHash` aggregate frame (buffered mode only).
    pub header: bool,
    /// Emit a `FooterHash` aggregate frame.
    pub footer: bool,
}

/// Encode a complete message in buffered mode.
///
/// All objects are known upfront. Header contains metadata + index + optional
/// aggregate HashHeader.  Footer may carry an aggregate HashFooter.
///
/// When `hash_algorithm` is `Some(_)` every frame in the message
/// gets its inline hash slot populated and the preamble's
/// `HASHES_PRESENT` flag is set.  When `None`, the slot is written
/// as zeros and the flag is clear (see v3 §2.4).  The hash-frame
/// `policy` controls whether aggregate Header/Footer HashFrames are
/// emitted; it's ignored when `hash_algorithm.is_none()`.
///
/// Strategy: build the message in two passes.
/// Pass 1: serialize all pieces, compute sizes/offsets.
/// Pass 2: assemble into final buffer.
pub fn encode_message(
    global_meta: &GlobalMetadata,
    objects: &[EncodedObject],
    hashing: bool,
    hash_policy: HashFramePolicy,
) -> Result<Vec<u8>> {
    // Serialize metadata CBOR
    let meta_cbor = metadata::global_metadata_to_cbor(global_meta)?;

    // Pre-encode all data object frames
    let mut object_frames: Vec<Vec<u8>> = Vec::with_capacity(objects.len());
    for obj in objects {
        let frame =
            encode_data_object_frame(&obj.descriptor, &obj.encoded_payload, false, hashing)?;
        object_frames.push(frame);
    }

    // Build aggregate HashFrame CBOR from the inline slots.  The
    // same bytes are used for both header and footer aggregates.
    let aggregate_hash_cbor = if hashing && (hash_policy.header || hash_policy.footer) {
        build_hash_frame_cbor(&object_frames, hashing)?
    } else {
        None
    };
    let header_hash_cbor: Option<&[u8]> = if hash_policy.header {
        aggregate_hash_cbor.as_deref()
    } else {
        None
    };
    let footer_hash_cbor: Option<&[u8]> = if hash_policy.footer {
        aggregate_hash_cbor.as_deref()
    } else {
        None
    };

    // Measure header size without index to feed the two-pass index builder
    let mut header_no_index = Vec::new();
    header_no_index.extend_from_slice(&[0u8; PREAMBLE_SIZE]);
    write_frame(
        &mut header_no_index,
        FrameType::HeaderMetadata,
        1,
        0,
        &meta_cbor,
        hashing,
        true,
    );
    if let Some(h_cbor) = header_hash_cbor {
        write_frame(
            &mut header_no_index,
            FrameType::HeaderHash,
            1,
            0,
            h_cbor,
            hashing,
            true,
        );
    }

    // Two-pass index construction
    let index_frame_bytes = build_index_frame(header_no_index.len(), &object_frames, hashing)?;

    // Compute flags and assemble.  `HASHES_PRESENT` is set whenever
    // hashing is on — the per-frame hash slots are populated
    // uniformly for every frame in the message.
    let mut flags = compute_message_flags(index_frame_bytes.is_some(), header_hash_cbor.is_some());
    if footer_hash_cbor.is_some() {
        flags.set(MessageFlags::FOOTER_HASHES);
    }
    if hashing {
        flags.set(MessageFlags::HASHES_PRESENT);
    }

    Ok(assemble_message(
        flags,
        &meta_cbor,
        index_frame_bytes.as_deref(),
        header_hash_cbor,
        footer_hash_cbor,
        &object_frames,
        hashing,
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
    /// `plans/WIRE_FORMAT.md` §6.5 for the region layout.
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
        // PrecederMetadata lives alongside data-object frames — it
        // must appear immediately before the data-object frame it
        // describes, within the data phase.  In v3 the only concrete
        // data-object type is NTensorFrame (type 9); new types will
        // join this match arm without a wire-format version bump.
        FrameType::NTensorFrame | FrameType::PrecederMetadata => DecodePhase::DataObjects,
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

        // Validate postamble.  In v3 the postamble carries a mirrored
        // `total_length` that — when non-zero — must match the
        // preamble's value.  Zero in the postamble means a
        // non-seekable streaming producer couldn't back-fill it;
        // that's always valid, just loses the backward-scan property.
        let pa_offset = total_len - POSTAMBLE_SIZE;
        let postamble = Postamble::read_from(&buf[pa_offset..])?;
        if postamble.total_length != 0 && postamble.total_length != preamble.total_length {
            return Err(TensogramError::Framing(format!(
                "postamble total_length ({}) does not match preamble total_length ({})",
                postamble.total_length, preamble.total_length
            )));
        }
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
            FrameType::NTensorFrame => {
                // `decode_data_object_frame` returns the trimmed data
                // payload and the (possibly empty) mask region slice.
                // Frames without masks have an empty mask region.
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

/// Walk the frame chain of a single message and collect the
/// inline hash slot of every data-object frame, in emission order.
///
/// Returns `Ok(Vec<Option<u64>>)` with one entry per
/// `NTensorFrame`:
///
/// - `Some(digest)` when the frame's hash slot holds a non-zero
///   xxh3-64 digest (the common case when `HASHES_PRESENT = 1`).
/// - `None` when the slot is `0x0000000000000000` — either
///   because the message's preamble flag clears
///   `HASHES_PRESENT` or because a future selective-hashing
///   policy opted-out this specific frame.
///
/// This is the cheap specialization of [`decode_message`] for
/// callers that only want the per-object hashes and don't care
/// about CBOR descriptors or payload bytes — the walker visits
/// only the 16-byte frame headers plus 8 bytes per hash slot,
/// skipping CBOR parsing entirely.  Useful for FFI inline-hash
/// surfaces and fast integrity scans.
///
/// # Errors
///
/// Returns `TensogramError::Framing` on any structural problem
/// (buffer too short, bad frame header, total_length overflow,
/// unknown frame type).  Callers that can tolerate a partial
/// result should fall back to `decode_message` and recover from
/// that function's fields instead.
pub fn data_object_inline_hashes(buf: &[u8]) -> Result<Vec<Option<u64>>> {
    let preamble = Preamble::read_from(buf)?;

    let msg_end = if preamble.total_length > 0 {
        let total_len = usize::try_from(preamble.total_length).map_err(|_| {
            TensogramError::Framing(format!(
                "total_length {} overflows usize",
                preamble.total_length
            ))
        })?;
        total_len.checked_sub(POSTAMBLE_SIZE).ok_or_else(|| {
            TensogramError::Framing(format!(
                "total_length {} too small for postamble ({POSTAMBLE_SIZE} bytes)",
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

    let mut hashes = Vec::new();
    let mut pos = PREAMBLE_SIZE;
    while pos < msg_end {
        if pos + FRAME_HEADER_SIZE > buf.len() || &buf[pos..pos + 2] != b"FR" {
            // Ragged tail; forward scanner would byte-walk here —
            // for an inline-hash collector we stop cleanly at the
            // first mis-alignment.
            break;
        }
        let fh = FrameHeader::read_from(&buf[pos..])?;
        let frame_total = usize::try_from(fh.total_length).map_err(|_| {
            TensogramError::Framing(format!(
                "frame total_length {} overflows usize",
                fh.total_length
            ))
        })?;
        if pos + frame_total > buf.len() {
            return Err(TensogramError::Framing(format!(
                "frame at offset {pos} extends past buffer end ({} > {})",
                pos + frame_total,
                buf.len()
            )));
        }
        if fh.frame_type.is_data_object() {
            // Hash slot sits at `frame_end - 12` regardless of
            // frame type (v3 §2.2 common tail).
            let slot_start = pos + frame_total - crate::wire::FRAME_COMMON_FOOTER_SIZE;
            let slot = crate::wire::read_u64_be(buf, slot_start);
            hashes.push(if slot == 0 { None } else { Some(slot) });
        }
        pos += frame_total;
        pos = (pos + 7) & !7; // 8-byte alignment padding
    }
    Ok(hashes)
}

// ── Scan ─────────────────────────────────────────────────────────────────────

/// Options controlling the in-memory `scan` and file-level
/// `scan_file` algorithms.
///
/// `bidirectional = true` (default) enables a meet-in-the-middle
/// walker pair — one forward from offset 0, one backward from EOF —
/// that can halve the hop count on multi-message files.  The
/// backward walker uses the postamble's mirrored `total_length`
/// field (v3 §7) to jump directly to the preceding message's
/// start without a byte-by-byte search.  If the backward walker
/// hits a message whose postamble has `total_length = 0`
/// (non-seekable streaming producer) or any structural anomaly,
/// it yields and the forward walker completes the scan alone.
///
/// `max_message_size` caps the apparent message length advertised
/// by a postamble's `total_length` field.  Any value larger than
/// this cap is treated as corruption and the backward walker
/// yields to the forward walker.  Default 4 GiB.  Lower this for
/// producers that guarantee smaller messages to make corruption
/// detection tighter.
#[derive(Debug, Clone)]
pub struct ScanOptions {
    /// Enable meet-in-the-middle walking.  When `false`, the
    /// scanner walks forward from offset 0 only (pre-v3 behaviour
    /// — still correct, just without the hop-count halving).
    pub bidirectional: bool,
    /// Upper bound on the `total_length` value advertised by a
    /// postamble.  Anything larger is treated as corruption and
    /// the backward walker yields to the forward walker.
    /// Default 4 GiB.
    pub max_message_size: u64,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            bidirectional: true,
            max_message_size: 4 * 1024 * 1024 * 1024,
        }
    }
}

/// Scan a multi-message buffer for message boundaries.
///
/// Delegates to [`scan_with_options`] with the default options
/// (bidirectional walk enabled).  Returns `(offset, length)` for
/// each message found.
#[tracing::instrument(skip(buf), fields(buf_len = buf.len()))]
pub fn scan(buf: &[u8]) -> Vec<(usize, usize)> {
    scan_with_options(buf, &ScanOptions::default())
}

/// Scan a multi-message buffer with explicit options.
///
/// When `opts.bidirectional` is `true`, uses a meet-in-the-middle
/// walker pair.  When `false`, falls back to pure forward scanning
/// (identical to pre-v3 behaviour).
#[tracing::instrument(skip(buf, opts), fields(buf_len = buf.len(), bidir = opts.bidirectional))]
pub fn scan_with_options(buf: &[u8], opts: &ScanOptions) -> Vec<(usize, usize)> {
    if !opts.bidirectional {
        return scan_forward_all(buf, 0, buf.len());
    }
    scan_bidirectional(buf, opts)
}

/// Try a single forward hop starting at `pos`.
///
/// Returns `Some((start, length))` when a full message was located;
/// returns `None` when the preamble does not start at `pos` (caller
/// should advance one byte and retry).  Zero is returned through
/// `Some` as well for streaming messages (their length is the
/// offset to the next `END_MAGIC`).
///
/// `bound_end` is the exclusive upper bound on the scan — the
/// forward walker will not return a message whose end would exceed
/// `bound_end`.  This is the mechanism by which the bidirectional
/// walker limits forward progress once the backward walker has
/// staked out the tail of the buffer.
fn try_forward_hop(buf: &[u8], pos: usize, bound_end: usize) -> Option<(usize, usize)> {
    if pos + PREAMBLE_SIZE + POSTAMBLE_SIZE > bound_end {
        return None;
    }
    if &buf[pos..pos + MAGIC.len()] != MAGIC {
        return None;
    }
    let preamble = Preamble::read_from(&buf[pos..]).ok()?;
    if preamble.total_length > 0 {
        let total = usize::try_from(preamble.total_length).ok()?;
        if pos + total > bound_end {
            return None;
        }
        // Validate END_MAGIC lives where the preamble claims.
        let end_magic_offset = pos + total - 8;
        if &buf[end_magic_offset..end_magic_offset + 8] == crate::wire::END_MAGIC {
            return Some((pos, total));
        }
        return None;
    }
    // Streaming message — scan forward for END_MAGIC bounded by
    // `bound_end`.
    let mut end_pos = pos + PREAMBLE_SIZE;
    while end_pos + 8 <= bound_end {
        if &buf[end_pos..end_pos + 8] == crate::wire::END_MAGIC {
            let msg_len = end_pos + 8 - pos;
            return Some((pos, msg_len));
        }
        end_pos += 1;
    }
    None
}

/// Forward-only scan over `buf[range_start..range_end)`.
fn scan_forward_all(buf: &[u8], range_start: usize, range_end: usize) -> Vec<(usize, usize)> {
    let mut messages = Vec::new();
    let mut pos = range_start;
    while pos + PREAMBLE_SIZE + POSTAMBLE_SIZE <= range_end {
        match try_forward_hop(buf, pos, range_end) {
            Some((start, total)) => {
                messages.push((start, total));
                pos = start + total;
            }
            None => pos += 1,
        }
    }
    messages
}

/// Outcome of a single backward hop.
enum BackwardHop {
    /// Found a message at `(start, length)`; the caller should
    /// continue backward from `start`.
    Hit(usize, usize),
    /// A postamble's `total_length` field was zero — this message
    /// was produced by a streaming non-seekable sink and its start
    /// cannot be determined in O(1).  The backward walker must stop;
    /// the forward walker will handle the remaining region.
    StreamingStop,
    /// No further postamble found in the bounded backward search.
    None,
}

/// Try a single backward hop ending at `bound_end` (exclusive).
///
/// Reads the fixed-size postamble at `[bound_end - POSTAMBLE_SIZE,
/// bound_end)` and uses its mirrored `total_length` field to
/// compute the message start.  Returns `None` on any structural
/// mismatch (missing END_MAGIC, bad preamble at the computed
/// start, total_length > `max_message_size`) so the caller falls
/// back to forward scanning.
///
/// `max_message_size` caps the apparent message length to guard
/// against corruption that could otherwise cause the backward
/// walker to jump to a nonsense offset.
fn try_backward_hop(
    buf: &[u8],
    range_start: usize,
    bound_end: usize,
    max_message_size: u64,
) -> BackwardHop {
    if bound_end < range_start + PREAMBLE_SIZE + POSTAMBLE_SIZE {
        return BackwardHop::None;
    }
    if &buf[bound_end - 8..bound_end] != crate::wire::END_MAGIC {
        return BackwardHop::None;
    }
    // Parse the full postamble at `bound_end - POSTAMBLE_SIZE`.
    let pa_start = bound_end - POSTAMBLE_SIZE;
    let postamble = match Postamble::read_from(&buf[pa_start..bound_end]) {
        Ok(p) => p,
        Err(_) => return BackwardHop::None,
    };
    if postamble.total_length == 0 {
        // Streaming non-seekable: can't jump backward.
        return BackwardHop::StreamingStop;
    }
    if postamble.total_length > max_message_size {
        // Implausibly large — likely corruption.  Bail back to the
        // forward walker which does its own per-step validation.
        return BackwardHop::None;
    }
    let total = match usize::try_from(postamble.total_length) {
        Ok(t) => t,
        Err(_) => return BackwardHop::None,
    };
    if total > bound_end - range_start {
        return BackwardHop::None;
    }
    let msg_start = bound_end - total;
    if &buf[msg_start..msg_start + MAGIC.len()] != MAGIC {
        return BackwardHop::None;
    }
    // Sanity: validate the preamble itself.
    if Preamble::read_from(&buf[msg_start..]).is_err() {
        return BackwardHop::None;
    }
    BackwardHop::Hit(msg_start, total)
}

/// Meet-in-the-middle bidirectional scan.
///
/// Alternates forward and backward hops from the two ends of the
/// buffer until they meet, falling back to pure forward scan of
/// the residual middle once the walkers cross (or when the
/// backward walker yields on a streaming non-seekable message).
fn scan_bidirectional(buf: &[u8], opts: &ScanOptions) -> Vec<(usize, usize)> {
    let mut fwd: Vec<(usize, usize)> = Vec::new();
    let mut bwd: Vec<(usize, usize)> = Vec::new();
    let mut fwd_pos: usize = 0;
    let mut bwd_end: usize = buf.len();

    loop {
        // Stop when walkers would overlap.
        if fwd_pos + PREAMBLE_SIZE + POSTAMBLE_SIZE > bwd_end {
            break;
        }

        // One forward hop: advance one byte on miss, one message
        // on hit.  Caps the forward range at `bwd_end` so we don't
        // step into the region already claimed by the backward
        // walker.
        match try_forward_hop(buf, fwd_pos, bwd_end) {
            Some((start, total)) => {
                fwd.push((start, total));
                fwd_pos = start + total;
            }
            None => {
                fwd_pos += 1;
                continue;
            }
        }

        // Stop when walkers would overlap.
        if fwd_pos + PREAMBLE_SIZE + POSTAMBLE_SIZE > bwd_end {
            break;
        }

        // One backward hop.  On a streaming-non-seekable yield or
        // any structural anomaly, drop to forward-only for the
        // residual region and return the merged result.
        match try_backward_hop(buf, fwd_pos, bwd_end, opts.max_message_size) {
            BackwardHop::Hit(start, total) => {
                bwd.push((start, total));
                bwd_end = start;
            }
            BackwardHop::StreamingStop | BackwardHop::None => {
                let residual = scan_forward_all(buf, fwd_pos, bwd_end);
                return merge_bidirectional_results(fwd, bwd, residual);
            }
        }
    }

    // Walkers crossed cleanly: scan the (usually empty) residual
    // and merge.
    let residual = scan_forward_all(buf, fwd_pos, bwd_end);
    merge_bidirectional_results(fwd, bwd, residual)
}

// ── File-based scan ──────────────────────────────────────────────────────────

/// Scan a file for message boundaries without loading the entire file into memory.
///
/// Reads preamble-sized chunks and seeks forward, avoiding full-file reads
/// for large files. Returns the same `(offset, length)` pairs as `scan()`.
///
/// Delegates to [`scan_file_with_options`] with the default
/// options (bidirectional walk enabled).
pub fn scan_file(file: &mut (impl std::io::Read + std::io::Seek)) -> Result<Vec<(usize, usize)>> {
    scan_file_with_options(file, &ScanOptions::default())
}

/// Scan a file with explicit scan options.
pub fn scan_file_with_options(
    file: &mut (impl std::io::Read + std::io::Seek),
    opts: &ScanOptions,
) -> Result<Vec<(usize, usize)>> {
    use std::io::SeekFrom;

    let file_len_u64 = file.seek(SeekFrom::End(0))?;
    let file_len = usize::try_from(file_len_u64).map_err(|_| {
        TensogramError::Framing(format!("file size {file_len_u64} overflows usize"))
    })?;
    file.seek(SeekFrom::Start(0))?;

    if !opts.bidirectional {
        return scan_file_forward(file, 0, file_len);
    }
    scan_file_bidirectional(file, file_len, opts.max_message_size)
}

/// Forward-only scan over `file[range_start..range_end)` using
/// seek-and-read I/O.  Factored out of the old `scan_file` so the
/// bidirectional entry point can reuse it for the residual region
/// between the two walkers.
fn scan_file_forward(
    file: &mut (impl std::io::Read + std::io::Seek),
    range_start: usize,
    range_end: usize,
) -> Result<Vec<(usize, usize)>> {
    use std::io::SeekFrom;

    let mut messages = Vec::new();
    let mut pos: usize = range_start;
    let mut preamble_buf = [0u8; PREAMBLE_SIZE];

    while pos + PREAMBLE_SIZE + POSTAMBLE_SIZE <= range_end {
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
                if pos + total <= range_end {
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
                let mut search_pos = pos + PREAMBLE_SIZE;
                let mut found = false;
                let chunk_size = 4096;
                let mut chunk = vec![0u8; chunk_size];

                while search_pos + 8 <= range_end {
                    file.seek(SeekFrom::Start(search_pos as u64))?;
                    let to_read = (range_end - search_pos).min(chunk_size);
                    let buf = &mut chunk[..to_read];
                    if file.read_exact(buf).is_err() {
                        break;
                    }

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

/// Merge a forward-walker prefix and a backward-walker suffix
/// into the final boundary list — the backward results are
/// reversed (since they were collected EOF-first) and appended.
fn merge_bidirectional_results(
    mut fwd: Vec<(usize, usize)>,
    bwd: Vec<(usize, usize)>,
    residual: Vec<(usize, usize)>,
) -> Vec<(usize, usize)> {
    fwd.extend(residual);
    let mut bwd_reversed = bwd;
    bwd_reversed.reverse();
    fwd.extend(bwd_reversed);
    fwd
}

/// Meet-in-the-middle bidirectional scan over a seekable stream.
///
/// Each backward hop reads `POSTAMBLE_SIZE` bytes ending at the
/// current backward cursor, parses the postamble, and jumps by
/// `postamble.total_length`.  Yields to the forward walker on a
/// zero `total_length` (streaming non-seekable sink), on any
/// structural mismatch (missing TENSOGRM at the computed start),
/// or when the claimed length exceeds `max_message_size`.
fn scan_file_bidirectional(
    file: &mut (impl std::io::Read + std::io::Seek),
    file_len: usize,
    max_message_size: u64,
) -> Result<Vec<(usize, usize)>> {
    use std::io::SeekFrom;

    let mut fwd: Vec<(usize, usize)> = Vec::new();
    let mut bwd: Vec<(usize, usize)> = Vec::new();
    let mut fwd_pos: usize = 0;
    let mut bwd_end: usize = file_len;

    let mut preamble_buf = [0u8; PREAMBLE_SIZE];
    let mut postamble_buf = [0u8; POSTAMBLE_SIZE];

    // Helper: hand off to the forward-only walker for the
    // remaining region between `fwd_pos` and `bwd_end` and return
    // the merged boundary list.  Used at every fall-back point in
    // the walker below to avoid copy-paste drift.
    macro_rules! fall_back_to_forward {
        () => {{
            let residual = scan_file_forward(file, fwd_pos, bwd_end)?;
            return Ok(merge_bidirectional_results(fwd, bwd, residual));
        }};
    }

    loop {
        if fwd_pos + PREAMBLE_SIZE + POSTAMBLE_SIZE > bwd_end {
            break;
        }

        // ── Forward hop ────────────────────────────────────────
        file.seek(SeekFrom::Start(fwd_pos as u64))?;
        let read_fwd = file.read_exact(&mut preamble_buf).is_ok();
        let mut advanced = false;
        if read_fwd
            && &preamble_buf[..MAGIC.len()] == MAGIC
            && let Ok(preamble) = Preamble::read_from(&preamble_buf)
        {
            if preamble.total_length > 0 {
                if let Ok(total) = usize::try_from(preamble.total_length)
                    && fwd_pos + total <= bwd_end
                {
                    let end_magic_offset = fwd_pos + total - 8;
                    file.seek(SeekFrom::Start(end_magic_offset as u64))?;
                    let mut em = [0u8; 8];
                    if file.read_exact(&mut em).is_ok() && &em == crate::wire::END_MAGIC {
                        fwd.push((fwd_pos, total));
                        fwd_pos += total;
                        advanced = true;
                    }
                }
            } else {
                // Streaming message — forward-only walker handles
                // the remaining region.
                fall_back_to_forward!();
            }
        }
        if !advanced {
            fwd_pos += 1;
            continue;
        }

        if fwd_pos + PREAMBLE_SIZE + POSTAMBLE_SIZE > bwd_end {
            break;
        }

        // ── Backward hop ───────────────────────────────────────
        let pa_start = bwd_end - POSTAMBLE_SIZE;
        file.seek(SeekFrom::Start(pa_start as u64))?;
        if file.read_exact(&mut postamble_buf).is_err() {
            fall_back_to_forward!();
        }
        match Postamble::read_from(&postamble_buf) {
            Ok(postamble) if postamble.total_length != 0 => {
                if postamble.total_length > max_message_size {
                    // Postamble claims a message larger than the
                    // per-reader cap — treat as corruption.
                    fall_back_to_forward!();
                }
                let Ok(total) = usize::try_from(postamble.total_length) else {
                    // Exceeds usize — fall back.
                    fall_back_to_forward!();
                };
                if total > bwd_end - fwd_pos {
                    // Claimed length exceeds the un-scanned region —
                    // overlaps the forward walker's claims.
                    fall_back_to_forward!();
                }
                let msg_start = bwd_end - total;
                file.seek(SeekFrom::Start(msg_start as u64))?;
                if file.read_exact(&mut preamble_buf).is_err()
                    || &preamble_buf[..MAGIC.len()] != MAGIC
                    || Preamble::read_from(&preamble_buf).is_err()
                {
                    // TENSOGRM magic absent at the computed start —
                    // the postamble's `total_length` lies.
                    fall_back_to_forward!();
                }
                bwd.push((msg_start, total));
                bwd_end = msg_start;
            }
            Ok(_) | Err(_) => {
                // Postamble said `total_length = 0` (streaming
                // non-seekable) or was corrupt — back off.
                fall_back_to_forward!();
            }
        }
    }

    // Walkers crossed without a fallback trigger: scan the residual
    // middle (usually empty) and merge the two results.
    let residual = scan_file_forward(file, fwd_pos, bwd_end)?;
    Ok(merge_bidirectional_results(fwd, bwd, residual))
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Total frame size for a non-data-object frame in v3:
/// `header(16) + payload + hash(8) + ENDF(4)`.
fn frame_total_size(payload_len: usize) -> usize {
    FRAME_HEADER_SIZE + payload_len + FRAME_COMMON_FOOTER_SIZE
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
        }
    }

    #[test]
    fn mask_aware_payload_end_rejects_mask_past_region() {
        // Defense-in-depth: a corrupted / malicious descriptor with
        // mask.offset + length exceeding the region must produce a
        // clean Framing error at parse time, BEFORE the mask codec
        // runs.  Exercises mask_aware_payload_end directly since
        // round-tripping a tampered frame at the integration layer
        // is fragile (CBOR byte-shifting).
        let mut desc = make_descriptor(vec![2]);
        desc.masks = Some(crate::types::MasksMetadata {
            nan: Some(crate::types::MaskDescriptor {
                method: "none".to_string(),
                offset: 4,
                length: 100, // way past any reasonable region
                params: BTreeMap::new(),
            }),
            ..Default::default()
        });
        // Region covers [payload_start=16 .. cbor_start=24] — only
        // 8 bytes available, so the claimed mask (offset=4, len=100)
        // exceeds the region (end=104 > 8).
        let err = mask_aware_payload_end(16, 24, &desc).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("exceeds payload region size"),
            "expected region-size-exceeded error, got: {msg}"
        );
    }

    #[test]
    fn mask_aware_payload_end_rejects_offset_length_overflow() {
        // u64::MAX + u64::MAX overflows checked_add and should
        // surface a clear overflow error.
        let mut desc = make_descriptor(vec![2]);
        desc.masks = Some(crate::types::MasksMetadata {
            nan: Some(crate::types::MaskDescriptor {
                method: "none".to_string(),
                offset: usize::MAX as u64,
                length: usize::MAX as u64,
                params: BTreeMap::new(),
            }),
            ..Default::default()
        });
        let err = mask_aware_payload_end(16, 24, &desc).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("overflow") || msg.contains("exceeds"),
            "expected overflow / exceeds error, got: {msg}"
        );
    }

    #[test]
    fn mask_aware_payload_end_no_masks_uses_region_end() {
        // Baseline: when descriptor.masks is None, the payload fills
        // the entire region (pre-0.17 compat path).
        let desc = make_descriptor(vec![4]);
        let end = mask_aware_payload_end(16, 32, &desc).unwrap();
        assert_eq!(end, 32);
    }

    #[test]
    fn test_data_object_frame_round_trip_cbor_after() {
        let desc = make_descriptor(vec![4]);
        let payload = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

        let frame = encode_data_object_frame(&desc, &payload, false, false).unwrap();

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

        let frame = encode_data_object_frame(&desc, &payload, true, false).unwrap();

        let (decoded_desc, decoded_payload, _mask_region, _) =
            decode_data_object_frame(&frame).unwrap();
        assert_eq!(decoded_desc.shape, vec![2, 3]);
        assert_eq!(decoded_payload, &payload[..]);
    }

    #[test]
    fn test_empty_message_round_trip() {
        let meta = make_global_meta();
        let msg = encode_message(&meta, &[], false, Default::default()).unwrap();

        // Check magic and end magic
        assert_eq!(&msg[0..8], MAGIC);
        assert_eq!(&msg[msg.len() - 8..], crate::wire::END_MAGIC);

        let decoded = decode_message(&msg).unwrap();
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

        let msg = encode_message(&meta, &objects, false, Default::default()).unwrap();
        let decoded = decode_message(&msg).unwrap();
        assert_eq!(decoded.objects.len(), 1);
        assert_eq!(decoded.objects[0].0.shape, vec![4]);
        assert_eq!(decoded.objects[0].1, &payload[..]);
        assert!(decoded.index.is_some());
        assert_eq!(decoded.index.as_ref().unwrap().offsets.len(), 1);
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

        let msg = encode_message(&meta, &objects, false, Default::default()).unwrap();
        let decoded = decode_message(&msg).unwrap();

        assert_eq!(decoded.objects.len(), 2);
        assert_eq!(decoded.objects[0].0.shape, vec![4]);
        assert_eq!(decoded.objects[0].1, &payload0[..]);
        assert_eq!(decoded.objects[1].0.shape, vec![2, 3]);
        assert_eq!(decoded.objects[1].1, &payload1[..]);

        let idx = decoded.index.as_ref().unwrap();
        assert_eq!(idx.offsets.len(), 2);
        assert_eq!(idx.lengths.len(), 2);
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
            false,
            Default::default(),
        )
        .unwrap();
        let msg2 = encode_message(
            &meta,
            &[EncodedObject {
                descriptor: make_descriptor(vec![4]),
                encoded_payload: vec![1u8; 16],
            }],
            false,
            Default::default(),
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
            false,
            Default::default(),
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
                encoded_payload: vec![1u8; 16],
            }],
            false,
            Default::default(),
        )
        .unwrap();

        let decoded_meta = decode_metadata_only(&msg).unwrap(); // _extra_ is the CBOR key name (via serde rename)
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
                    let frame = encode_data_object_frame(&desc, &payload, false, false).unwrap();
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
                    write_frame(&mut out, *frame_type, 1, 0, data, false, true);
                }
            }
        }

        // Postamble — first_footer_offset is known; total_length is
        // patched after the final length is known.
        let postamble_offset = out.len();
        let postamble = Postamble {
            first_footer_offset: postamble_offset as u64,
            total_length: 0,
        };
        postamble.write_to(&mut out);

        let total_length = out.len() as u64;
        let mut flags = MessageFlags::default();
        flags.set(MessageFlags::HEADER_METADATA);
        let preamble = Preamble {
            version: WIRE_VERSION,
            flags,
            reserved: 0,
            total_length,
        };
        let mut preamble_bytes = Vec::new();
        preamble.write_to(&mut preamble_bytes);
        out[0..PREAMBLE_SIZE].copy_from_slice(&preamble_bytes);
        // Patch the postamble's total_length now that it is known.
        let total_length_bytes = total_length.to_be_bytes();
        out[postamble_offset + 8..postamble_offset + 16].copy_from_slice(&total_length_bytes);

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
            algorithm: "xxh3".to_string(),
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
            false,
            Default::default(),
        )
        .unwrap();
        let msg2 = encode_message(
            &meta,
            &[EncodedObject {
                descriptor: make_descriptor(vec![4]),
                encoded_payload: vec![1u8; 16],
            }],
            false,
            Default::default(),
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
            false,
            Default::default(),
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
