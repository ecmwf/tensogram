// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! WASM bindings for the bidirectional remote-scan parsers.
//!
//! Four thin wrappers over [`tensogram::remote_scan_parse`] — the
//! TypeScript walker dispatches on the `{ kind: ..., ... }` JS shape
//! produced here, mirroring the Rust dispatch table in
//! `tensogram::remote::apply_round_outcomes`.
//!
//! Each export accepts a byte slice plus cursor / size scalars and
//! returns a serialised outcome object whose shape is pinned by the
//! `#[serde(tag = "kind", rename_all = "camelCase")]` attribute on
//! the underlying enum.  See the [`tensogram::remote_scan_parse`]
//! module docs for the full taxonomy.

use crate::convert::to_js;
use tensogram::{
    parse_backward_postamble, parse_forward_preamble,
    same_message_check as core_same_message_check, validate_backward_preamble,
};
use wasm_bindgen::prelude::*;

/// Parse a backward-postamble fetch and return its outcome.
///
/// @param pa_bytes - The 24-byte postamble at `[snap_prev - 24, snap_prev)`.
/// @param snap_next - Forward cursor (lower bound of the unknown gap).
/// @param snap_prev - Backward cursor (upper bound of the unknown gap).
/// @returns One of:
///   - `{ kind: "Format",                 reason: string }`
///   - `{ kind: "Streaming" }`
///   - `{ kind: "NeedPreambleValidation", msgStart: bigint, length: bigint }`
#[wasm_bindgen]
pub fn parse_backward_postamble_outcome(
    pa_bytes: &[u8],
    snap_next: u64,
    snap_prev: u64,
) -> Result<JsValue, JsError> {
    to_js(&parse_backward_postamble(pa_bytes, snap_next, snap_prev))
}

/// Validate a backward candidate preamble and return the commit decision.
///
/// @param preamble_bytes - The 24-byte preamble at `[msg_start, msg_start+24)`.
/// @param msg_start      - Candidate message-start offset from
///                          [`parse_backward_postamble_outcome`].
/// @param length         - Postamble's claimed `total_length`.
/// @returns One of:
///   - `{ kind: "Format", reason: string }`
///   - `{ kind: "Layout", offset: bigint, length: bigint }`
#[wasm_bindgen]
pub fn validate_backward_preamble_outcome(
    preamble_bytes: &[u8],
    msg_start: u64,
    length: u64,
) -> Result<JsValue, JsError> {
    to_js(&validate_backward_preamble(preamble_bytes, msg_start, length))
}

/// Parse a forward-preamble fetch and return its outcome.
///
/// @param preamble_bytes - The 24-byte preamble at `[pos, pos+24)`.
/// @param pos       - Absolute byte offset where the preamble begins.
/// @param file_size - Total file size.
/// @param bound     - Soft upper bound (`prev_scan_offset` in
///                    bidirectional mode, `file_size` in forward-only mode).
/// @returns One of:
///   - `{ kind: "Hit",          offset: bigint, length: bigint, msgEnd: bigint }`
///   - `{ kind: "ExceedsBound", offset: bigint, length: bigint, msgEnd: bigint }`
///   - `{ kind: "Streaming",    remaining: bigint }`
///   - `{ kind: "Terminate",    reason: string }`
#[wasm_bindgen]
pub fn parse_forward_preamble_outcome(
    preamble_bytes: &[u8],
    pos: u64,
    file_size: u64,
    bound: u64,
) -> Result<JsValue, JsError> {
    to_js(&parse_forward_preamble(preamble_bytes, pos, file_size, bound))
}

/// `true` iff a forward `Hit` and a backward-validated layout
/// describe the same message — used by the dispatch table to detect
/// the 1-message file and odd-count meet-in-the-middle cases.
#[wasm_bindgen]
pub fn same_message_check(
    fwd_offset: u64,
    fwd_length: u64,
    layout_offset: u64,
    layout_length: u64,
) -> bool {
    core_same_message_check(fwd_offset, fwd_length, layout_offset, layout_length)
}
