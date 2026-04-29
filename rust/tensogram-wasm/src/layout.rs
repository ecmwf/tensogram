// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Wire-format layout helpers exposed to the TypeScript wrapper.
//!
//! These are thin bindings over public functions in the `tensogram`
//! core crate.  Together they give the TS wrapper enough primitives to
//! implement lazy per-object HTTP Range reads without re-implementing
//! any wire-format parsing:
//!
//! - [`read_preamble_info`], [`read_postamble_info`] let TS inspect
//!   one message header/footer without decoding frames.
//! - [`parse_header_chunk`], [`parse_footer_chunk`] accept a larger
//!   byte region (header 256 KB chunk, or footer suffix) and return
//!   the metadata + index frames found in it.
//! - [`read_data_object_frame_header`], [`read_data_object_frame_footer`],
//!   [`parse_descriptor_cbor`] let TS implement the CBOR-prefix-only
//!   descriptor optimisation for huge frames.
//! - [`decode_object_from_frame`], [`decode_range_from_frame`] decode
//!   a single frame once TS has fetched its bytes via Range.
//!
//! Every error is routed through [`crate::convert::js_err`] so the
//! TypeScript wrapper's error mapper sees a consistent shape.

use crate::DecodedMessage;
use crate::convert::{js_err, metadata_to_js, to_js};
use serde::Serialize;
use tensogram::{
    self as core, DecodeOptions,
    metadata::{cbor_to_global_metadata, cbor_to_index, cbor_to_object_descriptor},
    wire::{
        DATA_OBJECT_FOOTER_SIZE, DataObjectFlags, FRAME_COMMON_FOOTER_SIZE, FRAME_END,
        FRAME_HEADER_SIZE, FrameHeader, FrameType, MAGIC, MessageFlags, POSTAMBLE_SIZE,
        PREAMBLE_SIZE, Postamble, Preamble,
    },
};
use wasm_bindgen::prelude::*;

/// Set `(metadata, index)` on `out` with the metadata routed through
/// [`metadata_to_js`] so the wire-format `version` field gets
/// synthesised.  Without this the lazy-backend output would silently
/// return `metadata.version === undefined` while the eager-backend
/// [`crate::decode_metadata`] path returned `metadata.version ===
/// WIRE_VERSION` — a real cross-backend divergence the lazy
/// `messageMetadata` / `messageObject` tests rely on being absent.
fn set_metadata_and_index(
    out: &js_sys::Object,
    metadata: Option<core::GlobalMetadata>,
    index: Option<IndexFrameJs>,
) -> Result<(), JsError> {
    let meta_val = match metadata {
        Some(m) => metadata_to_js(&m)?,
        None => JsValue::NULL,
    };
    js_sys::Reflect::set(out, &JsValue::from_str("metadata"), &meta_val)
        .map_err(|_| JsError::new("internal: failed to set chunk.metadata"))?;
    let index_val = match index {
        Some(idx) => to_js(&idx)?,
        None => JsValue::NULL,
    };
    js_sys::Reflect::set(out, &JsValue::from_str("index"), &index_val)
        .map_err(|_| JsError::new("internal: failed to set chunk.index"))?;
    Ok(())
}

// ── Return shapes ────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct PreambleInfoJs {
    version: u16,
    flags: u16,
    total_length: u64,
    has_header_metadata: bool,
    has_header_index: bool,
    has_footer_metadata: bool,
    has_footer_index: bool,
    has_preceder_metadata: bool,
    hashes_present: bool,
}

#[derive(Serialize)]
struct PostambleInfoJs {
    first_footer_offset: u64,
    total_length: u64,
    end_magic_ok: bool,
}

#[derive(Serialize)]
struct IndexFrameJs {
    offsets: Vec<u64>,
    lengths: Vec<u64>,
}

#[derive(Serialize)]
struct FrameHeaderJs {
    frame_type: u16,
    version: u16,
    flags: u16,
    total_length: u64,
    is_data_object: bool,
    cbor_after_payload: bool,
}

#[derive(Serialize)]
struct DataObjectFooterJs {
    cbor_offset: u64,
    hash_hex: String,
    end_magic_ok: bool,
}

// ── Preamble / postamble ─────────────────────────────────────────────────────

/// Inspect a wire-format preamble.
///
/// @param bytes - At least 24 bytes; the first 24 bytes of a message.
#[wasm_bindgen]
pub fn read_preamble_info(bytes: &[u8]) -> Result<JsValue, JsError> {
    if bytes.len() < PREAMBLE_SIZE {
        return Err(JsError::new(&format!(
            "preamble buffer too short: {} < {PREAMBLE_SIZE}",
            bytes.len()
        )));
    }
    let pre = Preamble::read_from(&bytes[..PREAMBLE_SIZE]).map_err(js_err)?;
    let flags = pre.flags;
    to_js(&PreambleInfoJs {
        version: pre.version,
        flags: flags.bits(),
        total_length: pre.total_length,
        has_header_metadata: flags.has(MessageFlags::HEADER_METADATA),
        has_header_index: flags.has(MessageFlags::HEADER_INDEX),
        has_footer_metadata: flags.has(MessageFlags::FOOTER_METADATA),
        has_footer_index: flags.has(MessageFlags::FOOTER_INDEX),
        has_preceder_metadata: flags.has(MessageFlags::PRECEDER_METADATA),
        hashes_present: flags.has(MessageFlags::HASHES_PRESENT),
    })
}

/// Inspect a wire-format postamble.
///
/// @param bytes - At least 24 bytes; typically the last 24 bytes of a message.
#[wasm_bindgen]
pub fn read_postamble_info(bytes: &[u8]) -> Result<JsValue, JsError> {
    if bytes.len() < POSTAMBLE_SIZE {
        return Err(JsError::new(&format!(
            "postamble buffer too short: {} < {POSTAMBLE_SIZE}",
            bytes.len()
        )));
    }
    let pa = Postamble::read_from(&bytes[bytes.len() - POSTAMBLE_SIZE..]).map_err(js_err)?;
    let tail = &bytes[bytes.len() - core::wire::END_MAGIC.len()..];
    to_js(&PostambleInfoJs {
        first_footer_offset: pa.first_footer_offset,
        total_length: pa.total_length,
        end_magic_ok: tail == core::wire::END_MAGIC,
    })
}

// ── Header / footer chunk parsers ────────────────────────────────────────────

/// Parse header metadata + index from a chunk that starts at message
/// offset 0 (the preamble).  Stops at the first data-object or
/// preceder frame.  Returns nulls for frames that fall outside the
/// supplied chunk so TS can decide to widen its Range fetch.
#[wasm_bindgen]
pub fn parse_header_chunk(chunk: &[u8]) -> Result<JsValue, JsError> {
    if chunk.len() < PREAMBLE_SIZE {
        return Err(JsError::new(&format!(
            "header chunk too short: {} < {PREAMBLE_SIZE}",
            chunk.len()
        )));
    }
    if &chunk[..MAGIC.len()] != MAGIC {
        return Err(JsError::new("header chunk does not start with TENSOGRM"));
    }

    let mut metadata: Option<core::GlobalMetadata> = None;
    let mut index: Option<IndexFrameJs> = None;
    let mut body_start: Option<u64> = None;

    let mut pos = PREAMBLE_SIZE;
    while pos + FRAME_HEADER_SIZE <= chunk.len() {
        if &chunk[pos..pos + 2] != core::wire::FRAME_MAGIC {
            pos += 1;
            continue;
        }
        let fh = FrameHeader::read_from(&chunk[pos..]).map_err(js_err)?;
        let total = match usize::try_from(fh.total_length) {
            Ok(t) => t,
            Err(_) => return Err(JsError::new("frame total_length exceeds usize")),
        };
        if total < FRAME_HEADER_SIZE + FRAME_END.len() {
            return Err(JsError::new("frame total_length below minimum"));
        }
        let frame_end = match pos.checked_add(total) {
            Some(e) => e,
            None => return Err(JsError::new("frame end overflows")),
        };
        if frame_end > chunk.len() {
            break;
        }

        if matches!(
            fh.frame_type,
            FrameType::NTensorFrame | FrameType::PrecederMetadata
        ) {
            body_start = Some(pos as u64);
            break;
        }

        if &chunk[frame_end - FRAME_END.len()..frame_end] != FRAME_END {
            return Err(JsError::new("header frame missing ENDF trailer"));
        }

        let payload = &chunk[pos + FRAME_HEADER_SIZE..frame_end - FRAME_COMMON_FOOTER_SIZE];
        match fh.frame_type {
            FrameType::HeaderMetadata => {
                metadata = Some(cbor_to_global_metadata(payload).map_err(js_err)?);
            }
            FrameType::HeaderIndex => {
                let idx = cbor_to_index(payload).map_err(js_err)?;
                index = Some(IndexFrameJs {
                    offsets: idx.offsets,
                    lengths: idx.lengths,
                });
            }
            _ => {}
        }

        let aligned = (frame_end + 7) & !7;
        pos = aligned.min(chunk.len());
    }

    let out = js_sys::Object::new();
    set_metadata_and_index(&out, metadata, index)?;
    let body_start_val = match body_start {
        Some(b) => JsValue::from(b),
        None => JsValue::NULL,
    };
    js_sys::Reflect::set(&out, &JsValue::from_str("body_start"), &body_start_val)
        .map_err(|_| JsError::new("internal: failed to set header_chunk.body_start"))?;
    Ok(out.into())
}

/// Parse footer metadata + index from a chunk that covers the footer
/// region — i.e. `[first_footer_offset, message_end - POSTAMBLE_SIZE)`.
#[wasm_bindgen]
pub fn parse_footer_chunk(chunk: &[u8]) -> Result<JsValue, JsError> {
    let mut metadata: Option<core::GlobalMetadata> = None;
    let mut index: Option<IndexFrameJs> = None;

    let mut pos = 0usize;
    while pos + FRAME_HEADER_SIZE <= chunk.len() {
        if &chunk[pos..pos + 2] != core::wire::FRAME_MAGIC {
            pos += 1;
            continue;
        }
        let fh = FrameHeader::read_from(&chunk[pos..]).map_err(js_err)?;
        let total = match usize::try_from(fh.total_length) {
            Ok(t) => t,
            Err(_) => return Err(JsError::new("footer frame total_length exceeds usize")),
        };
        if total < FRAME_HEADER_SIZE + FRAME_END.len() {
            return Err(JsError::new("footer frame total_length below minimum"));
        }
        let frame_end = match pos.checked_add(total) {
            Some(e) if e <= chunk.len() => e,
            _ => break,
        };

        if &chunk[frame_end - FRAME_END.len()..frame_end] != FRAME_END {
            return Err(JsError::new("footer frame missing ENDF trailer"));
        }

        let payload = &chunk[pos + FRAME_HEADER_SIZE..frame_end - FRAME_COMMON_FOOTER_SIZE];
        match fh.frame_type {
            FrameType::FooterMetadata => {
                metadata = Some(cbor_to_global_metadata(payload).map_err(js_err)?);
            }
            FrameType::FooterIndex => {
                let idx = cbor_to_index(payload).map_err(js_err)?;
                index = Some(IndexFrameJs {
                    offsets: idx.offsets,
                    lengths: idx.lengths,
                });
            }
            _ => {}
        }

        let aligned = (frame_end + 7) & !7;
        pos = aligned.min(chunk.len());
    }

    let out = js_sys::Object::new();
    set_metadata_and_index(&out, metadata, index)?;
    Ok(out.into())
}

// ── Per-frame header + footer + descriptor CBOR ──────────────────────────────

/// Parse the 16-byte header of a data-object frame.
#[wasm_bindgen]
pub fn read_data_object_frame_header(bytes: &[u8]) -> Result<JsValue, JsError> {
    if bytes.len() < FRAME_HEADER_SIZE {
        return Err(JsError::new(&format!(
            "frame header buffer too short: {} < {FRAME_HEADER_SIZE}",
            bytes.len()
        )));
    }
    let fh = FrameHeader::read_from(&bytes[..FRAME_HEADER_SIZE]).map_err(js_err)?;
    to_js(&FrameHeaderJs {
        frame_type: fh.frame_type as u16,
        version: fh.version,
        flags: fh.flags,
        total_length: fh.total_length,
        is_data_object: fh.frame_type.is_data_object(),
        cbor_after_payload: fh.flags & DataObjectFlags::CBOR_AFTER_PAYLOAD != 0,
    })
}

/// Parse the 20-byte footer of a data-object frame (type `NTensorFrame`).
///
/// Expects the caller to supply the last 20 bytes of the frame.  Returns
/// the in-frame `cbor_offset` (where the descriptor CBOR lives) and the
/// 64-bit frame hash, plus a sanity flag on the `ENDF` trailer.
#[wasm_bindgen]
pub fn read_data_object_frame_footer(bytes: &[u8]) -> Result<JsValue, JsError> {
    if bytes.len() < DATA_OBJECT_FOOTER_SIZE {
        return Err(JsError::new(&format!(
            "data-object frame footer too short: {} < {DATA_OBJECT_FOOTER_SIZE}",
            bytes.len()
        )));
    }
    let footer = &bytes[bytes.len() - DATA_OBJECT_FOOTER_SIZE..];
    let cbor_offset = u64::from_be_bytes(
        footer[..8]
            .try_into()
            .map_err(|_| JsError::new("cbor_offset truncated"))?,
    );
    let hash = u64::from_be_bytes(
        footer[8..16]
            .try_into()
            .map_err(|_| JsError::new("hash truncated"))?,
    );
    let end = &footer[16..20];
    to_js(&DataObjectFooterJs {
        cbor_offset,
        hash_hex: format!("{hash:016x}"),
        end_magic_ok: end == FRAME_END,
    })
}

/// Decode a `DataObjectDescriptor` from its raw CBOR bytes.
#[wasm_bindgen]
pub fn parse_descriptor_cbor(cbor_bytes: &[u8]) -> Result<JsValue, JsError> {
    let desc = cbor_to_object_descriptor(cbor_bytes).map_err(js_err)?;
    to_js(&desc)
}

// ── Single-frame decode ──────────────────────────────────────────────────────

/// Decode a single data-object frame's full bytes to a `DecodedMessage`
/// that owns one decoded object.
///
/// @param verify_hash - Per-frame hash verification (default false).
///                      See `crate::decode` for the contract.
#[wasm_bindgen]
pub fn decode_object_from_frame(
    frame_bytes: &[u8],
    restore_non_finite: Option<bool>,
    verify_hash: Option<bool>,
) -> Result<DecodedMessage, JsError> {
    let options = DecodeOptions {
        restore_non_finite: restore_non_finite.unwrap_or(true),
        verify_hash: verify_hash.unwrap_or(false),
        ..Default::default()
    };
    let (desc, data) =
        core::decode::decode_object_from_frame(frame_bytes, &options).map_err(js_err)?;
    Ok(DecodedMessage::from_single_object(desc, data))
}

/// Extract ranges from a single data-object frame.
///
/// Mirrors [`crate::decode_range`] but takes one frame's bytes.
#[wasm_bindgen]
pub fn decode_range_from_frame(
    frame_bytes: &[u8],
    ranges: &js_sys::BigUint64Array,
) -> Result<JsValue, JsError> {
    let flat: Vec<u64> = ranges.to_vec();
    if !flat.len().is_multiple_of(2) {
        return Err(JsError::new(
            "ranges length must be a multiple of 2 (flat [offset, count] pairs)",
        ));
    }
    let range_pairs: Vec<(u64, u64)> = flat.chunks_exact(2).map(|w| (w[0], w[1])).collect();

    let options = DecodeOptions {
        ..Default::default()
    };
    let (descriptor, parts) =
        core::decode::decode_range_from_frame(frame_bytes, &range_pairs, &options)
            .map_err(js_err)?;

    let result = js_sys::Object::new();
    js_sys::Reflect::set(&result, &"descriptor".into(), &to_js(&descriptor)?)
        .map_err(|_| JsError::new("failed to set descriptor"))?;
    let parts_js = js_sys::Array::new_with_length(parts.len() as u32);
    for (i, bytes) in parts.iter().enumerate() {
        parts_js.set(i as u32, js_sys::Uint8Array::from(bytes.as_slice()).into());
    }
    js_sys::Reflect::set(&result, &"parts".into(), &parts_js)
        .map_err(|_| JsError::new("failed to set parts"))?;
    Ok(result.into())
}
