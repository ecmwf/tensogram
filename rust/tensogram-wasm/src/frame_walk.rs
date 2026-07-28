// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! WASM bindings for the structural frame walker.
//!
//! Mirrors the Rust core [`tensogram::frames`] /
//! [`tensogram::message_header`] (`rust/tensogram/src/frame_walk.rs`):
//! inspect one message's *shape* — which frames it holds and where —
//! without decoding payloads or CBOR.
//!
//! # One message per call
//!
//! Both entry points take the bytes of a **single** message, starting
//! at its `TENSOGRM` preamble magic.  Callers holding a multi-message
//! buffer must use [`crate::scan`] to obtain `[offset, length]` pairs
//! and slice before calling.
//!
//! # Payload ownership
//!
//! [`frames`] returns each frame's content as a JS-heap `Uint8Array`
//! **copy** (via `Uint8Array::from`), matching the mask accessors on
//! [`crate::DecodedMaskedMessage`].  A zero-copy view is not merely
//! undesirable here, it is unsound: `wasm_bindgen` materialises the
//! `&[u8]` argument in a temporary WASM allocation that is released
//! when this function returns, so a view into the walked payload would
//! dangle the moment the call completes.  Copies are also immune to
//! WASM linear-memory growth, so callers may retain them freely.
//!
//! # Eager materialisation
//!
//! The core walker is a lazy iterator; this binding materialises it
//! into a JS array because the frames-per-message count is small
//! (header frames + one per data object) and a lazily-pulled iterator
//! would have to keep the message bytes alive across the WASM
//! boundary.  A malformed frame chain therefore surfaces as a thrown
//! error rather than a truncated array — corruption is never silently
//! rendered as "fewer frames".

use crate::convert::{js_err, set_field, to_js};
use serde::Serialize;
use tensogram::{self as core, wire::FrameType};
use wasm_bindgen::prelude::*;

/// The canonical wire-format name of a frame type, as spelled in
/// `plans/WIRE_FORMAT.md` §6 and in the Rust `FrameType` enum.  Kept as
/// an explicit match so a new frame type cannot be added to the core
/// without this binding failing to compile.
fn frame_type_name(ft: FrameType) -> &'static str {
    match ft {
        FrameType::HeaderMetadata => "HeaderMetadata",
        FrameType::HeaderIndex => "HeaderIndex",
        FrameType::HeaderHash => "HeaderHash",
        FrameType::FooterHash => "FooterHash",
        FrameType::FooterIndex => "FooterIndex",
        FrameType::FooterMetadata => "FooterMetadata",
        FrameType::PrecederMetadata => "PrecederMetadata",
        FrameType::NTensorFrame => "NTensorFrame",
    }
}

/// Walk the frames of a single Tensogram message.
///
/// Thin binding over [`tensogram::frames`].  The preamble and
/// postamble are **not** frames and are not returned — use
/// [`message_header`] for the envelope.
///
/// @param message - Wire-format bytes of ONE message, starting at the
///   `TENSOGRM` preamble magic.  Use [`crate::scan`] to locate messages
///   in a multi-message buffer and slice before calling.
/// @returns An array of plain JS objects in wire order, one per frame:
///   `{ frame_type, frame_type_code, version, flags, offset, length,
///   payload, has_hash }`.
///   - `frame_type` — canonical name (`"HeaderMetadata"` …
///     `"NTensorFrame"`); `frame_type_code` — the wire number (1–9).
///   - `offset` — byte offset of the frame header, relative to the
///     start of `message`; `length` — whole-frame span through `ENDF`,
///     excluding trailing alignment padding.
///   - `payload` — the frame's content as a JS-heap `Uint8Array` copy:
///     the 16-byte frame header and the type-specific footer (20 B for
///     frame type 9, 12 B otherwise) are stripped.
///   - `has_hash` — the frame's `HASH_PRESENT` flag.
///
/// Throws when `message` does not start with a valid preamble, or when
/// the frame chain is malformed (truncated frame header, a declared
/// `total_length` that does not fit).
#[wasm_bindgen]
pub fn frames(message: &[u8]) -> Result<js_sys::Array, JsValue> {
    let out = js_sys::Array::new();
    for item in core::frames(message).map_err(js_err)? {
        let info = item.map_err(js_err)?;
        let obj = js_sys::Object::new();
        set_field(
            &obj,
            "frame_type",
            &JsValue::from_str(frame_type_name(info.frame_type)),
        )?;
        set_field(
            &obj,
            "frame_type_code",
            &JsValue::from(info.frame_type as u16),
        )?;
        set_field(&obj, "version", &JsValue::from(info.version))?;
        set_field(&obj, "flags", &JsValue::from(info.flags))?;
        // `offset` / `length` are `usize`, i.e. u32 on wasm32 — every
        // value is exactly representable as a JS number.
        set_field(&obj, "offset", &JsValue::from(info.offset as u32))?;
        set_field(&obj, "length", &JsValue::from(info.length as u32))?;
        set_field(
            &obj,
            "payload",
            &js_sys::Uint8Array::from(info.payload).into(),
        )?;
        set_field(&obj, "has_hash", &JsValue::from_bool(info.has_hash()))?;
        out.push(&obj);
    }
    Ok(out)
}

/// The message envelope, as returned by [`message_header`].
#[derive(Serialize)]
struct MessageHeaderJs {
    version: u16,
    total_length: u64,
    has_header_metadata: bool,
    has_footer_metadata: bool,
    has_header_index: bool,
    has_footer_index: bool,
    has_header_hashes: bool,
    has_footer_hashes: bool,
    has_preceder_metadata: bool,
    has_hashes_present: bool,
}

/// Read one message's envelope (preamble) as typed values, without
/// walking its frames.
///
/// Thin binding over [`tensogram::message_header`].  Tells a caller
/// whether a message is random-access (metadata / index / hashes in the
/// *header*) or streaming (in the *footer*) from 24 bytes alone.
///
/// @param message - Wire-format bytes of ONE message, starting at the
///   `TENSOGRM` preamble magic.
/// @returns `{ version, total_length, has_header_metadata,
///   has_footer_metadata, has_header_index, has_footer_index,
///   has_header_hashes, has_footer_hashes, has_preceder_metadata,
///   has_hashes_present }`.  `total_length` is `0` for a streaming
///   message whose length was never back-filled.
///
/// The predicates are exact for a buffered (random-access) message —
/// the encoder knows the whole message up front.  A streaming encoder
/// writes the preamble before any object, so its flags are advisory:
/// only `frame present ⇒ flag set` is guaranteed there.
///
/// Throws when `message` does not start with a valid preamble.
#[wasm_bindgen]
pub fn message_header(message: &[u8]) -> Result<JsValue, JsValue> {
    let h = core::message_header(message).map_err(js_err)?;
    to_js(&MessageHeaderJs {
        version: h.version,
        total_length: h.total_length,
        has_header_metadata: h.has_header_metadata(),
        has_footer_metadata: h.has_footer_metadata(),
        has_header_index: h.has_header_index(),
        has_footer_index: h.has_footer_index(),
        has_header_hashes: h.has_header_hashes(),
        has_footer_hashes: h.has_footer_hashes(),
        has_preceder_metadata: h.has_preceder_metadata(),
        has_hashes_present: h.has_hashes_present(),
    })
}
