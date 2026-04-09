//! Level 1: Structure validation — raw byte walking.

use crate::wire::{
    self, DataObjectFlags, FrameHeader, FrameType, MessageFlags, Postamble, Preamble,
    DATA_OBJECT_FOOTER_SIZE, FRAME_END, FRAME_HEADER_SIZE, FRAME_MAGIC, MAGIC, POSTAMBLE_SIZE,
    PREAMBLE_SIZE,
};

use super::types::*;

/// Phases for frame ordering enforcement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Phase {
    Headers,
    DataObjects,
    Footers,
}

fn frame_phase(ft: FrameType) -> Phase {
    match ft {
        FrameType::HeaderMetadata | FrameType::HeaderIndex | FrameType::HeaderHash => {
            Phase::Headers
        }
        FrameType::DataObject | FrameType::PrecederMetadata => Phase::DataObjects,
        FrameType::FooterHash | FrameType::FooterIndex | FrameType::FooterMetadata => {
            Phase::Footers
        }
    }
}

/// Info collected during Level 1 frame walk, reused by Level 2/3.
pub(crate) struct FrameWalkResult<'a> {
    /// (frame_type, payload_bytes) for non-DataObject frames.
    pub meta_frames: Vec<(FrameType, &'a [u8])>,
    /// (descriptor_cbor_slice, payload_bytes, frame_start_offset) per data object.
    pub data_objects: Vec<(&'a [u8], &'a [u8], usize)>,
}

/// Walk the raw bytes of a message, collecting structural issues.
///
/// Returns `Some(FrameWalkResult)` when a frame walk can be performed, even
/// if structural issues were recorded along the way (e.g. early loop exit).
/// Returns `None` only for unrecoverable early failures (bad magic, bad
/// preamble, bad postamble) that prevent the walk from starting at all.
/// The caller uses `hash_verified = false` when any errors exist to avoid
/// claiming verification on partial walks.
pub(crate) fn validate_structure<'a>(
    buf: &'a [u8],
    issues: &mut Vec<ValidationIssue>,
) -> Option<FrameWalkResult<'a>> {
    // --- Preamble ---
    if buf.len() < PREAMBLE_SIZE {
        issues.push(err(
            IssueCode::BufferTooShort,
            ValidationLevel::Structure,
            None,
            Some(0),
            format!(
                "buffer too short for preamble: {} < {PREAMBLE_SIZE}",
                buf.len()
            ),
        ));
        return None;
    }

    if &buf[0..8] != MAGIC {
        issues.push(err(
            IssueCode::InvalidMagic,
            ValidationLevel::Structure,
            None,
            Some(0),
            format!(
                "invalid magic bytes: expected TENSOGRM, got {:?}",
                String::from_utf8_lossy(&buf[0..8])
            ),
        ));
        return None;
    }

    let preamble = match Preamble::read_from(buf) {
        Ok(p) => p,
        Err(e) => {
            issues.push(err(
                IssueCode::PreambleParseFailed,
                ValidationLevel::Structure,
                None,
                Some(0),
                format!("preamble parse failed: {e}"),
            ));
            return None;
        }
    };

    // --- Total length / postamble ---
    // Parse postamble once, capture both msg_end and first_footer_offset.
    let (msg_end, declared_ffo) = if preamble.total_length > 0 {
        let total = match usize::try_from(preamble.total_length) {
            Ok(t) => t,
            Err(_) => {
                issues.push(err(
                    IssueCode::TotalLengthOverflow,
                    ValidationLevel::Structure,
                    None,
                    Some(16),
                    format!("total_length {} overflows usize", preamble.total_length),
                ));
                return None;
            }
        };
        if total > buf.len() {
            issues.push(err(
                IssueCode::TotalLengthExceedsBuffer,
                ValidationLevel::Structure,
                None,
                Some(16),
                format!("total_length {} exceeds buffer size {}", total, buf.len()),
            ));
            return None;
        }
        let min_msg_size = PREAMBLE_SIZE + POSTAMBLE_SIZE;
        if total < min_msg_size {
            issues.push(err(
                IssueCode::TotalLengthTooSmall,
                ValidationLevel::Structure,
                None,
                Some(16),
                format!(
                    "total_length {} is smaller than minimum message size {min_msg_size}",
                    total
                ),
            ));
            return None;
        }
        let pa_offset = total - POSTAMBLE_SIZE;
        match Postamble::read_from(&buf[pa_offset..]) {
            Ok(pa) => {
                let ffo = usize::try_from(pa.first_footer_offset).ok();
                if let Some(ffo_val) = ffo {
                    if ffo_val < PREAMBLE_SIZE || ffo_val > pa_offset {
                        issues.push(err(
                            IssueCode::FooterOffsetOutOfRange,
                            ValidationLevel::Structure,
                            None,
                            Some(pa_offset),
                            format!(
                                "first_footer_offset {} out of range [{PREAMBLE_SIZE}, {pa_offset}]",
                                ffo_val
                            ),
                        ));
                    }
                } else {
                    issues.push(err(
                        IssueCode::FooterOffsetOutOfRange,
                        ValidationLevel::Structure,
                        None,
                        Some(pa_offset),
                        format!(
                            "first_footer_offset {} does not fit in usize",
                            pa.first_footer_offset
                        ),
                    ));
                }
                (pa_offset, ffo)
            }
            Err(e) => {
                issues.push(err(
                    IssueCode::PostambleInvalid,
                    ValidationLevel::Structure,
                    None,
                    Some(pa_offset),
                    format!("postamble invalid: {e}"),
                ));
                return None;
            }
        }
    } else {
        // Streaming mode: postamble at end of buffer
        if buf.len() < POSTAMBLE_SIZE {
            issues.push(err(
                IssueCode::BufferTooShort,
                ValidationLevel::Structure,
                None,
                Some(0),
                "buffer too short for postamble in streaming mode".to_string(),
            ));
            return None;
        }
        let pa_offset = buf.len() - POSTAMBLE_SIZE;
        match Postamble::read_from(&buf[pa_offset..]) {
            Ok(pa) => {
                let ffo = usize::try_from(pa.first_footer_offset).ok();
                if let Some(ffo_val) = ffo {
                    if ffo_val < PREAMBLE_SIZE || ffo_val > pa_offset {
                        issues.push(err(
                            IssueCode::FooterOffsetOutOfRange,
                            ValidationLevel::Structure,
                            None,
                            Some(pa_offset),
                            format!(
                                "first_footer_offset {} out of range [{PREAMBLE_SIZE}, {pa_offset}]",
                                ffo_val
                            ),
                        ));
                    }
                } else {
                    issues.push(err(
                        IssueCode::FooterOffsetOutOfRange,
                        ValidationLevel::Structure,
                        None,
                        Some(pa_offset),
                        format!(
                            "first_footer_offset {} does not fit in usize",
                            pa.first_footer_offset
                        ),
                    ));
                }
                (pa_offset, ffo)
            }
            Err(_) => {
                issues.push(err(
                    IssueCode::PostambleInvalid,
                    ValidationLevel::Structure,
                    None,
                    Some(pa_offset),
                    "missing or invalid postamble in streaming message".to_string(),
                ));
                return None;
            }
        }
    };

    // --- Frame walk ---
    let mut pos = PREAMBLE_SIZE;
    let mut current_phase = Phase::Headers;
    let mut meta_frames: Vec<(FrameType, &[u8])> = Vec::new();
    let mut data_objects: Vec<(&[u8], &[u8], usize)> = Vec::new();
    let mut observed_flags = MessageFlags::new(0);
    let mut pending_preceder = false;
    let mut obj_idx: usize = 0;
    let mut first_footer_pos: Option<usize> = None;

    while pos < msg_end {
        if pos + 2 > msg_end {
            if buf[pos..msg_end].iter().any(|&b| b != 0) {
                issues.push(warn(
                    IssueCode::NonZeroPadding,
                    ValidationLevel::Structure,
                    None,
                    Some(pos),
                    format!("unexpected non-zero padding bytes at offset {pos}"),
                ));
            }
            break;
        }
        // Check for alignment padding (only zeros allowed between frames)
        if &buf[pos..pos + 2] != FRAME_MAGIC {
            // Always advance at least to the next 8-byte boundary
            let next_aligned = ((pos + 8) & !7).min(msg_end);
            let advance_to = if next_aligned > pos {
                next_aligned
            } else {
                pos + 1
            };
            if buf[pos..advance_to.min(msg_end)].iter().all(|&b| b == 0) {
                pos = advance_to;
                continue;
            }
            issues.push(warn(
                IssueCode::NonZeroPadding,
                ValidationLevel::Structure,
                None,
                Some(pos),
                format!("unexpected non-zero padding bytes at offset {pos}"),
            ));
            pos = advance_to;
            continue;
        }

        if pos + FRAME_HEADER_SIZE > msg_end {
            issues.push(err(
                IssueCode::TruncatedFrameHeader,
                ValidationLevel::Structure,
                None,
                Some(pos),
                format!("truncated frame header at offset {pos}"),
            ));
            break;
        }

        let fh = match FrameHeader::read_from(&buf[pos..]) {
            Ok(fh) => fh,
            Err(e) => {
                issues.push(err(
                    IssueCode::InvalidFrameHeader,
                    ValidationLevel::Structure,
                    None,
                    Some(pos),
                    format!("invalid frame header at offset {pos}: {e}"),
                ));
                break;
            }
        };

        let frame_total = match usize::try_from(fh.total_length) {
            Ok(t) => t,
            Err(_) => {
                issues.push(err(
                    IssueCode::FrameLengthOverflow,
                    ValidationLevel::Structure,
                    None,
                    Some(pos),
                    format!(
                        "frame total_length {} overflows usize at offset {pos}",
                        fh.total_length
                    ),
                ));
                break;
            }
        };

        let min_size = if fh.frame_type == FrameType::DataObject {
            FRAME_HEADER_SIZE + DATA_OBJECT_FOOTER_SIZE
        } else {
            FRAME_HEADER_SIZE + FRAME_END.len()
        };
        if frame_total < min_size {
            issues.push(err(
                IssueCode::FrameTooSmall,
                ValidationLevel::Structure,
                None,
                Some(pos),
                format!(
                    "frame at offset {pos} total_length {} < minimum {min_size}",
                    frame_total
                ),
            ));
            break;
        }

        let frame_end = match pos.checked_add(frame_total) {
            Some(end) => end,
            None => {
                issues.push(err(
                    IssueCode::FrameLengthOverflow,
                    ValidationLevel::Structure,
                    None,
                    Some(pos),
                    format!("frame at offset {pos} total_length {frame_total} causes overflow"),
                ));
                break;
            }
        };
        // Frames must end before the postamble (msg_end is the byte after the last frame region)
        if frame_end > msg_end {
            issues.push(err(
                IssueCode::FrameExceedsMessage,
                ValidationLevel::Structure,
                None,
                Some(pos),
                format!(
                    "frame at offset {pos} extends into postamble (frame_end={frame_end}, postamble_start={msg_end})",
                ),
            ));
            break;
        }

        // Validate ENDF marker
        let endf_offset = pos + frame_total - FRAME_END.len();
        if &buf[endf_offset..endf_offset + FRAME_END.len()] != FRAME_END {
            issues.push(err(
                IssueCode::MissingEndMarker,
                ValidationLevel::Structure,
                None,
                Some(endf_offset),
                format!("missing ENDF marker at offset {endf_offset} for frame starting at {pos}"),
            ));
            break;
        }

        // Frame ordering
        let phase = frame_phase(fh.frame_type);
        if phase < current_phase {
            issues.push(err(
                IssueCode::FrameOrderViolation,
                ValidationLevel::Structure,
                None,
                Some(pos),
                format!(
                    "frame {:?} at offset {pos} appears after {:?} phase — wrong order",
                    fh.frame_type, current_phase
                ),
            ));
        }
        if phase == Phase::Footers && first_footer_pos.is_none() {
            first_footer_pos = Some(pos);
        }
        current_phase = current_phase.max(phase);

        // Preceder legality
        if pending_preceder && fh.frame_type != FrameType::DataObject {
            issues.push(err(
                IssueCode::PrecederNotFollowedByObject,
                ValidationLevel::Structure,
                None,
                Some(pos),
                format!(
                    "PrecederMetadata must be followed by DataObject, got {:?} at offset {pos}",
                    fh.frame_type
                ),
            ));
            pending_preceder = false;
        }

        // Track observed flags
        match fh.frame_type {
            FrameType::HeaderMetadata => observed_flags.set(MessageFlags::HEADER_METADATA),
            FrameType::FooterMetadata => observed_flags.set(MessageFlags::FOOTER_METADATA),
            FrameType::HeaderIndex => observed_flags.set(MessageFlags::HEADER_INDEX),
            FrameType::FooterIndex => observed_flags.set(MessageFlags::FOOTER_INDEX),
            FrameType::HeaderHash => observed_flags.set(MessageFlags::HEADER_HASHES),
            FrameType::FooterHash => observed_flags.set(MessageFlags::FOOTER_HASHES),
            FrameType::PrecederMetadata => {
                pending_preceder = true;
                observed_flags.set(MessageFlags::PRECEDER_METADATA);
            }
            FrameType::DataObject => {
                pending_preceder = false;
            }
        }

        // Extract payloads for Level 2/3 reuse
        match fh.frame_type {
            FrameType::DataObject => {
                let cbor_offset_pos = endf_offset - 8;
                if cbor_offset_pos < pos + FRAME_HEADER_SIZE {
                    issues.push(err(
                        IssueCode::DataObjectTooSmall,
                        ValidationLevel::Structure,
                        Some(obj_idx),
                        Some(pos),
                        format!("data object frame at offset {pos} too small for cbor_offset"),
                    ));
                } else {
                    let cbor_offset_raw = wire::read_u64_be(buf, cbor_offset_pos);
                    let cbor_offset = match usize::try_from(cbor_offset_raw) {
                        Ok(v) => v,
                        Err(_) => {
                            issues.push(err(
                                IssueCode::CborOffsetInvalid,
                                ValidationLevel::Structure,
                                Some(obj_idx),
                                Some(pos),
                                format!(
                                    "data object cbor_offset {} does not fit in usize at offset {pos}",
                                    cbor_offset_raw
                                ),
                            ));
                            obj_idx += 1;
                            let aligned = frame_end.saturating_add(7) & !7;
                            pos = if aligned <= msg_end {
                                aligned
                            } else {
                                frame_end
                            };
                            continue;
                        }
                    };
                    let abs_cbor_offset = match pos.checked_add(cbor_offset) {
                        Some(v) => v,
                        None => {
                            issues.push(err(
                                IssueCode::CborOffsetInvalid,
                                ValidationLevel::Structure,
                                Some(obj_idx),
                                Some(pos),
                                format!(
                                    "data object cbor_offset {} overflows at offset {pos}",
                                    cbor_offset
                                ),
                            ));
                            obj_idx += 1;
                            let aligned = frame_end.saturating_add(7) & !7;
                            pos = if aligned <= msg_end {
                                aligned
                            } else {
                                frame_end
                            };
                            continue;
                        }
                    };

                    if cbor_offset < FRAME_HEADER_SIZE || abs_cbor_offset > cbor_offset_pos {
                        issues.push(err(
                            IssueCode::CborOffsetInvalid,
                            ValidationLevel::Structure,
                            Some(obj_idx),
                            Some(pos),
                            format!(
                                "data object cbor_offset {} out of range at offset {pos}",
                                cbor_offset
                            ),
                        ));
                    } else {
                        let cbor_after = fh.flags & DataObjectFlags::CBOR_AFTER_PAYLOAD != 0;
                        let slices = if cbor_after {
                            let payload_start = pos + FRAME_HEADER_SIZE;
                            let cbor_start = abs_cbor_offset;
                            let cbor_end = cbor_offset_pos;
                            Some((&buf[cbor_start..cbor_end], &buf[payload_start..cbor_start]))
                        } else {
                            // CBOR-before: read through Cursor to find exact CBOR length
                            let cbor_start = abs_cbor_offset;
                            let region = &buf[cbor_start..cbor_offset_pos];
                            let mut cursor = std::io::Cursor::new(region);
                            match ciborium::from_reader::<ciborium::Value, _>(&mut cursor) {
                                Ok(_) => match usize::try_from(cursor.position()) {
                                    Ok(cbor_len) => {
                                        let payload_start = cbor_start + cbor_len;
                                        Some((
                                            &buf[cbor_start..cbor_start + cbor_len],
                                            &buf[payload_start..cbor_offset_pos],
                                        ))
                                    }
                                    Err(_) => {
                                        issues.push(err(
                                            IssueCode::CborBeforeBoundaryUnknown,
                                            ValidationLevel::Structure,
                                            Some(obj_idx),
                                            Some(pos),
                                            "CBOR-before descriptor length does not fit in usize"
                                                .to_string(),
                                        ));
                                        None
                                    }
                                },
                                Err(_) => {
                                    // CBOR parse fails — payload boundary unknown.
                                    // Skip this object; Level 2/3 can't work with
                                    // unreliable slices.
                                    issues.push(err(
                                        IssueCode::CborBeforeBoundaryUnknown,
                                        ValidationLevel::Structure,
                                        Some(obj_idx),
                                        Some(pos),
                                        "CBOR-before descriptor parse failed, cannot determine payload boundary".to_string(),
                                    ));
                                    None
                                }
                            }
                        };
                        if let Some((cbor_slice, payload_slice)) = slices {
                            data_objects.push((cbor_slice, payload_slice, pos));
                        }
                    }
                }
                obj_idx += 1;
            }
            _ => {
                let payload = &buf[pos + FRAME_HEADER_SIZE..endf_offset];
                meta_frames.push((fh.frame_type, payload));
            }
        }

        // Advance past frame (with alignment padding to 8-byte boundary)
        let aligned = frame_end.saturating_add(7) & !7;
        pos = if aligned <= msg_end {
            // Check alignment padding bytes are zero
            if buf[frame_end..aligned].iter().any(|&b| b != 0) {
                issues.push(warn(
                    IssueCode::NonZeroPadding,
                    ValidationLevel::Structure,
                    None,
                    Some(frame_end),
                    format!("non-zero alignment padding after frame at offset {frame_end}"),
                ));
            }
            aligned
        } else {
            frame_end
        };
    }

    // Dangling preceder
    if pending_preceder {
        issues.push(err(
            IssueCode::DanglingPreceder,
            ValidationLevel::Structure,
            None,
            None,
            "dangling PrecederMetadata: no DataObject frame followed".to_string(),
        ));
    }

    // Verify first_footer_offset against actual position
    if let Some(ffo) = declared_ffo {
        let expected_ffo = first_footer_pos.unwrap_or(msg_end);
        if ffo != expected_ffo {
            issues.push(warn(
                IssueCode::FooterOffsetMismatch,
                ValidationLevel::Structure,
                None,
                None,
                format!(
                    "first_footer_offset {} does not match actual first footer position {}",
                    ffo, expected_ffo
                ),
            ));
        }
    }

    // Flags consistency
    let declared = preamble.flags;
    let flag_checks: &[(u16, &str)] = &[
        (MessageFlags::HEADER_METADATA, "HEADER_METADATA"),
        (MessageFlags::FOOTER_METADATA, "FOOTER_METADATA"),
        (MessageFlags::HEADER_INDEX, "HEADER_INDEX"),
        (MessageFlags::FOOTER_INDEX, "FOOTER_INDEX"),
        (MessageFlags::HEADER_HASHES, "HEADER_HASHES"),
        (MessageFlags::FOOTER_HASHES, "FOOTER_HASHES"),
        (MessageFlags::PRECEDER_METADATA, "PRECEDER_METADATA"),
    ];
    for &(flag, name) in flag_checks {
        if declared.has(flag) != observed_flags.has(flag) {
            issues.push(warn(
                IssueCode::FlagMismatch,
                ValidationLevel::Structure,
                None,
                Some(10),
                format!("{name} flag mismatch: declared vs observed"),
            ));
        }
    }

    // Must have at least one metadata frame
    if !observed_flags.has(MessageFlags::HEADER_METADATA)
        && !observed_flags.has(MessageFlags::FOOTER_METADATA)
    {
        issues.push(err(
            IssueCode::NoMetadataFrame,
            ValidationLevel::Structure,
            None,
            None,
            "no metadata frame found (neither header nor footer)".to_string(),
        ));
    }

    Some(FrameWalkResult {
        meta_frames,
        data_objects,
    })
}
