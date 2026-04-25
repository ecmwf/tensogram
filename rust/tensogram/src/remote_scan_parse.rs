// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Pure parsing primitives for the bidirectional remote-scan walker.
//!
//! Three small parsers — [`parse_backward_postamble`],
//! [`validate_backward_preamble`], [`parse_forward_preamble`] — plus
//! a same-message helper [`same_message_check`].  Each is a pure
//! function: takes byte slices and cursor values, returns a
//! discriminated outcome enum, mutates nothing.
//!
//! Used by the TypeScript bidirectional walker through four
//! `#[wasm_bindgen]` exports in `tensogram-wasm` that delegate to
//! these functions.  The Rust `remote` backend keeps a parallel
//! private set of richer parsers in `remote.rs` (carrying
//! `Preamble` and `CachedLayout` payloads required by the
//! state-mutating apply step) — both paths call the same
//! `Preamble::read_from` / `Postamble::read_from` core, so reason
//! strings stay structurally aligned.  The parity harness verifies
//! that the Rust and TypeScript walkers produce identical layouts
//! on every fixture, which would surface any drift between the two
//! parser surfaces.
//!
//! This module is cfg-independent: it does not depend on the
//! `remote` Cargo feature.  WASM builds use `default-features =
//! false` and need to call these parsers without pulling in
//! `object_store` and friends.
//!
//! ## Outcome enums and JS shape
//!
//! Each outcome enum is internally tagged via
//! `#[serde(tag = "kind", rename_all_fields = "camelCase")]`, which
//! camel-cases field names while preserving variant names in the
//! `"kind"` tag, so the `serde-wasm-bindgen` JS shape is stable
//! across language boundaries:
//!
//! ```text
//! { "kind": "Hit",                "offset": 0,  "length": 100, "msgEnd": 100 }
//! { "kind": "ExceedsBound",       "offset": 0,  "length": 100, "msgEnd": 100 }
//! { "kind": "Streaming",          "remaining": 1024 }
//! { "kind": "Terminate",          "reason": "bad-magic-fwd" }
//! { "kind": "Format",             "reason": "bad-end-magic-bwd" }
//! { "kind": "Streaming" }
//! { "kind": "NeedPreambleValidation", "msgStart": 8192, "length": 4096 }
//! { "kind": "Layout",             "offset": 8192, "length": 4096 }
//! ```
//!
//! Reason strings are stable identifiers — the parity harness pins
//! them, and the TypeScript walker emits them through `console.debug`
//! for symmetry with the Rust `tracing` events at
//! `target = "tensogram::remote_scan"`.

use serde::Serialize;

use crate::wire::{MAGIC, POSTAMBLE_SIZE, PREAMBLE_SIZE, Postamble, Preamble};

/// Forward fetch returned fewer than `PREAMBLE_SIZE` bytes.
pub const REASON_SHORT_FETCH_FWD: &str = "short-fetch-fwd";
/// Forward preamble does not start with the `TENSOGRM` magic.
pub const REASON_BAD_MAGIC_FWD: &str = "bad-magic-fwd";
/// Forward preamble parsed past the magic but failed wire-format checks.
pub const REASON_PREAMBLE_PARSE_ERROR_FWD: &str = "preamble-parse-error-fwd";
/// Forward `total_length` would push `pos + total_length` past `file_size`,
/// or the message length is smaller than the minimum.
pub const REASON_LENGTH_OUT_OF_RANGE_FWD: &str = "length-out-of-range-fwd";

/// Backward fetch returned fewer than `POSTAMBLE_SIZE` bytes.
pub const REASON_SHORT_FETCH_BWD: &str = "short-fetch-bwd";
/// Backward postamble does not end with the `END_MAGIC` trailer.
pub const REASON_BAD_END_MAGIC_BWD: &str = "bad-end-magic-bwd";
/// Backward postamble parsed past the trailer but failed wire-format checks.
pub const REASON_POSTAMBLE_PARSE_ERROR: &str = "postamble-parse-error";
/// Postamble's `total_length` is smaller than the minimum message size.
pub const REASON_LENGTH_BELOW_MINIMUM_BWD: &str = "length-below-minimum-bwd";
/// Postamble's `total_length` would underflow when subtracted from `prev`.
pub const REASON_BACKWARD_ARITH_UNDERFLOW: &str = "backward-arith-underflow";
/// Postamble's claimed `msg_start` lies inside the forward walker's
/// already-discovered prefix.
pub const REASON_BACKWARD_OVERLAPS_FORWARD: &str = "backward-overlaps-forward";

/// Backward candidate preamble does not start with the `TENSOGRM` magic.
pub const REASON_BAD_MAGIC_BWD: &str = "bad-magic-bwd";
/// Backward candidate preamble parsed past the magic but failed wire-format
/// checks.
pub const REASON_PREAMBLE_PARSE_ERROR_BWD: &str = "preamble-parse-error-bwd";
/// Backward candidate preamble carries `total_length == 0` (streaming) at a
/// non-tail position; backward yields with no recovery.
pub const REASON_STREAMING_PREAMBLE_NON_TAIL: &str = "streaming-preamble-non-tail";
/// Backward candidate preamble's `total_length` does not match the postamble's;
/// one of them is corrupt.
pub const REASON_PREAMBLE_POSTAMBLE_LENGTH_MISMATCH: &str = "preamble-postamble-length-mismatch";

/// Outcome of parsing a backward postamble fetch.
///
/// The caller decides what to do with this outcome — see the dispatch
/// table in `tensogram::remote::apply_round_outcomes` (Rust) or
/// `applyRoundOutcomes` (TypeScript).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all_fields = "camelCase")]
pub enum BackwardOutcome {
    /// Format error.  The caller disables the backward walker and
    /// discards any provisional suffix layouts (matching Rust's
    /// `disable_backward`); bidirectional is an optimisation, never
    /// a recovery mode.
    Format {
        /// Stable identifier for the format violation; one of the
        /// `REASON_*_BWD` constants.
        reason: &'static str,
    },
    /// Streaming-mode message at the postamble (`total_length == 0`).
    /// Caller disables the backward walker and discards provisional
    /// suffix layouts; forward continues.
    Streaming,
    /// Postamble parsed cleanly.  The caller must validate the
    /// candidate preamble at `msg_start` before committing the
    /// layout — wire-format integrity requires both ends to agree on
    /// `total_length`.
    NeedPreambleValidation {
        /// Candidate message-start offset, derived as `prev - total_length`.
        msg_start: u64,
        /// Postamble's `total_length` field.
        length: u64,
    },
}

/// Outcome of validating the candidate preamble at the offset
/// produced by [`parse_backward_postamble`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all_fields = "camelCase")]
pub enum BackwardCommit {
    /// Format error during preamble validation.  Caller disables
    /// the backward walker and discards any provisional suffix
    /// layouts (matching Rust's `disable_backward`).
    Format {
        /// Stable identifier; one of the `REASON_*_BWD` /
        /// `REASON_PREAMBLE_POSTAMBLE_LENGTH_MISMATCH` /
        /// `REASON_STREAMING_PREAMBLE_NON_TAIL` constants.
        reason: &'static str,
    },
    /// Preamble validated.  The caller's commit-decision step
    /// inspects this against the forward outcome to detect overlap,
    /// same-message meet, or a clean backward record.
    Layout {
        /// Confirmed message-start offset.
        offset: u64,
        /// Confirmed message length (bytes).
        length: u64,
    },
}

/// Outcome of parsing a forward preamble fetch.
///
/// Bidirectional callers pass `bound = prev_scan_offset` so the
/// forward walker cannot overrun into the suffix already claimed by
/// backward.  Forward-only callers pass `bound = file_size`, which
/// makes the [`ForwardOutcome::ExceedsBound`] branch unreachable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all_fields = "camelCase")]
pub enum ForwardOutcome {
    /// Forward parsed a message that fits entirely below `bound`.
    Hit {
        offset: u64,
        length: u64,
        msg_end: u64,
    },
    /// Forward parsed a message whose end exceeds `bound` while still
    /// fitting inside `file_size`.  Reachable only in bidirectional
    /// mode when `suffix_rev` already holds a backward-committed
    /// layout with a corrupt offset — forward's reading is canonical,
    /// so the caller disables backward (clearing `suffix_rev`) before
    /// committing the forward hop.
    ExceedsBound {
        offset: u64,
        length: u64,
        msg_end: u64,
    },
    /// Streaming-mode preamble (`total_length == 0`).  In forward-only
    /// mode this means a streaming tail; in bidirectional mode it
    /// indicates a non-tail streaming message inside the gap, which
    /// disables backward.
    Streaming { remaining: u64 },
    /// Forward terminates the walk.  Reasons cover short fetch,
    /// magic / parse / length errors at the current cursor.
    Terminate {
        /// Stable identifier; one of the `REASON_*_FWD` constants.
        reason: &'static str,
    },
}

/// Parse a backward postamble fetch.
///
/// `pa_bytes` is expected to be the 24-byte postamble at
/// `[snap_prev - POSTAMBLE_SIZE, snap_prev)`.  Implausible
/// `total_length` values are rejected by the arithmetic underflow
/// (`total > snap_prev`) and forward-overlap (`msg_start <
/// snap_next`) guards; the candidate-preamble validation step
/// catches anything that slips through, at the cost of a single
/// 24-byte GET.
///
/// Pure function — does not touch any backend state.
///
/// ```
/// use tensogram::remote_scan_parse::{BackwardOutcome, parse_backward_postamble};
/// // Buffer too short.
/// let outcome = parse_backward_postamble(&[0u8; 4], 0, 100);
/// assert!(matches!(outcome, BackwardOutcome::Format { reason: "short-fetch-bwd" }));
/// ```
pub fn parse_backward_postamble(
    pa_bytes: &[u8],
    snap_next: u64,
    snap_prev: u64,
) -> BackwardOutcome {
    let min_message_size = (PREAMBLE_SIZE + POSTAMBLE_SIZE) as u64;

    if pa_bytes.len() < POSTAMBLE_SIZE {
        return BackwardOutcome::Format {
            reason: REASON_SHORT_FETCH_BWD,
        };
    }
    let end_magic_start = POSTAMBLE_SIZE - crate::wire::END_MAGIC.len();
    if &pa_bytes[end_magic_start..POSTAMBLE_SIZE] != crate::wire::END_MAGIC {
        return BackwardOutcome::Format {
            reason: REASON_BAD_END_MAGIC_BWD,
        };
    }
    let postamble = match Postamble::read_from(&pa_bytes[..POSTAMBLE_SIZE]) {
        Ok(p) => p,
        Err(_) => {
            return BackwardOutcome::Format {
                reason: REASON_POSTAMBLE_PARSE_ERROR,
            };
        }
    };
    let total = postamble.total_length;
    if total == 0 {
        return BackwardOutcome::Streaming;
    }
    if total < min_message_size {
        return BackwardOutcome::Format {
            reason: REASON_LENGTH_BELOW_MINIMUM_BWD,
        };
    }
    let msg_start = match snap_prev.checked_sub(total) {
        Some(s) => s,
        None => {
            return BackwardOutcome::Format {
                reason: REASON_BACKWARD_ARITH_UNDERFLOW,
            };
        }
    };
    if msg_start < snap_next {
        return BackwardOutcome::Format {
            reason: REASON_BACKWARD_OVERLAPS_FORWARD,
        };
    }
    BackwardOutcome::NeedPreambleValidation {
        msg_start,
        length: total,
    }
}

/// Validate the backward candidate preamble against the postamble's
/// claimed `msg_start` and `length`.
///
/// Both ends of a well-formed message agree on `total_length`; this
/// function rejects mismatches and streaming preambles (which would
/// have indicated a streaming tail, not a backward-discoverable
/// message).
///
/// Pure function — does not touch any backend state.
///
/// ```
/// use tensogram::remote_scan_parse::{BackwardCommit, validate_backward_preamble};
/// // Buffer too short.
/// let outcome = validate_backward_preamble(&[0u8; 4], 0, 100);
/// assert!(matches!(outcome, BackwardCommit::Format { reason: "short-fetch-bwd" }));
/// ```
pub fn validate_backward_preamble(
    preamble_bytes: &[u8],
    msg_start: u64,
    length: u64,
) -> BackwardCommit {
    if preamble_bytes.len() < PREAMBLE_SIZE {
        return BackwardCommit::Format {
            reason: REASON_SHORT_FETCH_BWD,
        };
    }
    if &preamble_bytes[..MAGIC.len()] != MAGIC {
        return BackwardCommit::Format {
            reason: REASON_BAD_MAGIC_BWD,
        };
    }
    let preamble = match Preamble::read_from(preamble_bytes) {
        Ok(p) => p,
        Err(_) => {
            return BackwardCommit::Format {
                reason: REASON_PREAMBLE_PARSE_ERROR_BWD,
            };
        }
    };
    if preamble.total_length == 0 {
        return BackwardCommit::Format {
            reason: REASON_STREAMING_PREAMBLE_NON_TAIL,
        };
    }
    if preamble.total_length != length {
        return BackwardCommit::Format {
            reason: REASON_PREAMBLE_POSTAMBLE_LENGTH_MISMATCH,
        };
    }
    BackwardCommit::Layout {
        offset: msg_start,
        length,
    }
}

/// Parse a forward preamble fetch.
///
/// `pos` is the absolute byte offset where this preamble begins;
/// `file_size` caps the message-end against the known total file
/// length; `bound` is the soft cap (`prev_scan_offset` in
/// bidirectional mode, `file_size` in forward-only mode).
///
/// Pure function — does not touch any backend state.
///
/// ```
/// use tensogram::remote_scan_parse::{ForwardOutcome, parse_forward_preamble};
/// // Buffer too short.
/// let outcome = parse_forward_preamble(&[0u8; 4], 0, 1024, 1024);
/// assert!(matches!(outcome, ForwardOutcome::Terminate { reason: "short-fetch-fwd" }));
/// ```
pub fn parse_forward_preamble(
    preamble_bytes: &[u8],
    pos: u64,
    file_size: u64,
    bound: u64,
) -> ForwardOutcome {
    let min_message_size = (PREAMBLE_SIZE + POSTAMBLE_SIZE) as u64;

    if preamble_bytes.len() < PREAMBLE_SIZE {
        return ForwardOutcome::Terminate {
            reason: REASON_SHORT_FETCH_FWD,
        };
    }
    if &preamble_bytes[..MAGIC.len()] != MAGIC {
        return ForwardOutcome::Terminate {
            reason: REASON_BAD_MAGIC_FWD,
        };
    }
    let preamble = match Preamble::read_from(preamble_bytes) {
        Ok(p) => p,
        Err(_) => {
            return ForwardOutcome::Terminate {
                reason: REASON_PREAMBLE_PARSE_ERROR_FWD,
            };
        }
    };
    let msg_len = preamble.total_length;
    if msg_len == 0 {
        let remaining = file_size.saturating_sub(pos);
        return ForwardOutcome::Streaming { remaining };
    }
    let end = match pos.checked_add(msg_len) {
        Some(e) => e,
        None => {
            return ForwardOutcome::Terminate {
                reason: REASON_LENGTH_OUT_OF_RANGE_FWD,
            };
        }
    };
    if msg_len < min_message_size || end > file_size {
        return ForwardOutcome::Terminate {
            reason: REASON_LENGTH_OUT_OF_RANGE_FWD,
        };
    }
    if end > bound {
        return ForwardOutcome::ExceedsBound {
            offset: pos,
            length: msg_len,
            msg_end: end,
        };
    }
    ForwardOutcome::Hit {
        offset: pos,
        length: msg_len,
        msg_end: end,
    }
}

/// `true` iff a forward `Hit` and a backward-validated layout
/// describe exactly the same message.
///
/// Triggers on:
///
/// - 1-message files — forward and backward both identify the only
///   message.
/// - Odd-count crossovers — the meet-in-the-middle lands on a single
///   shared middle message.
///
/// In both cases the dispatch table commits the forward layout once
/// and yields the backward record silently (without clearing
/// `suffix_rev` from earlier rounds).
///
/// ```
/// use tensogram::remote_scan_parse::same_message_check;
/// assert!(same_message_check(0, 100, 0, 100));
/// assert!(!same_message_check(0, 100, 0, 200));
/// ```
pub fn same_message_check(
    fwd_offset: u64,
    fwd_length: u64,
    layout_offset: u64,
    layout_length: u64,
) -> bool {
    fwd_offset == layout_offset && fwd_length == layout_length
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{MessageFlags, WIRE_VERSION};

    fn make_postamble(total_length: u64) -> Vec<u8> {
        let pa = Postamble {
            first_footer_offset: 0,
            total_length,
        };
        let mut buf = Vec::with_capacity(POSTAMBLE_SIZE);
        pa.write_to(&mut buf);
        buf
    }

    fn make_preamble(total_length: u64) -> Vec<u8> {
        let pre = Preamble {
            version: WIRE_VERSION,
            flags: MessageFlags::new(0),
            reserved: 0,
            total_length,
        };
        let mut buf = Vec::with_capacity(PREAMBLE_SIZE);
        pre.write_to(&mut buf);
        buf
    }

    #[test]
    fn backward_short_fetch() {
        let outcome = parse_backward_postamble(&[0u8; 4], 0, 100);
        assert!(matches!(
            outcome,
            BackwardOutcome::Format {
                reason: REASON_SHORT_FETCH_BWD
            }
        ));
    }

    #[test]
    fn backward_bad_end_magic() {
        let buf = vec![0u8; POSTAMBLE_SIZE];
        let outcome = parse_backward_postamble(&buf, 0, POSTAMBLE_SIZE as u64);
        assert!(matches!(
            outcome,
            BackwardOutcome::Format {
                reason: REASON_BAD_END_MAGIC_BWD
            }
        ));
    }

    #[test]
    fn backward_streaming_postamble() {
        let buf = make_postamble(0);
        let outcome = parse_backward_postamble(&buf, 0, POSTAMBLE_SIZE as u64);
        assert_eq!(outcome, BackwardOutcome::Streaming);
    }

    #[test]
    fn backward_length_below_minimum() {
        let buf = make_postamble(1);
        let outcome = parse_backward_postamble(&buf, 0, 100);
        assert!(matches!(
            outcome,
            BackwardOutcome::Format {
                reason: REASON_LENGTH_BELOW_MINIMUM_BWD
            }
        ));
    }

    #[test]
    fn backward_arith_underflow_when_total_exceeds_snap_prev() {
        let buf = make_postamble(200);
        let outcome = parse_backward_postamble(&buf, 0, 100);
        assert!(matches!(
            outcome,
            BackwardOutcome::Format {
                reason: REASON_BACKWARD_ARITH_UNDERFLOW
            }
        ));
    }

    #[test]
    fn backward_overlap_when_msg_start_below_snap_next() {
        let buf = make_postamble(80);
        let outcome = parse_backward_postamble(&buf, 50, 100);
        assert!(matches!(
            outcome,
            BackwardOutcome::Format {
                reason: REASON_BACKWARD_OVERLAPS_FORWARD
            }
        ));
    }

    #[test]
    fn backward_need_preamble_validation_when_disjoint_from_forward() {
        let buf = make_postamble(80);
        let outcome = parse_backward_postamble(&buf, 10, 100);
        assert_eq!(
            outcome,
            BackwardOutcome::NeedPreambleValidation {
                msg_start: 20,
                length: 80,
            }
        );
    }

    #[test]
    fn backward_validate_short_fetch() {
        let outcome = validate_backward_preamble(&[0u8; 4], 0, 100);
        assert!(matches!(
            outcome,
            BackwardCommit::Format {
                reason: REASON_SHORT_FETCH_BWD
            }
        ));
    }

    #[test]
    fn backward_validate_bad_magic() {
        let mut buf = vec![0u8; PREAMBLE_SIZE];
        buf[..8].copy_from_slice(b"NOTMAGIC");
        let outcome = validate_backward_preamble(&buf, 0, 100);
        assert!(matches!(
            outcome,
            BackwardCommit::Format {
                reason: REASON_BAD_MAGIC_BWD
            }
        ));
    }

    #[test]
    fn backward_validate_streaming_at_non_tail() {
        let buf = make_preamble(0);
        let outcome = validate_backward_preamble(&buf, 0, 100);
        assert!(matches!(
            outcome,
            BackwardCommit::Format {
                reason: REASON_STREAMING_PREAMBLE_NON_TAIL
            }
        ));
    }

    #[test]
    fn backward_validate_length_mismatch() {
        let buf = make_preamble(99);
        let outcome = validate_backward_preamble(&buf, 0, 100);
        assert!(matches!(
            outcome,
            BackwardCommit::Format {
                reason: REASON_PREAMBLE_POSTAMBLE_LENGTH_MISMATCH
            }
        ));
    }

    #[test]
    fn backward_validate_layout() {
        let buf = make_preamble(100);
        let outcome = validate_backward_preamble(&buf, 50, 100);
        assert_eq!(
            outcome,
            BackwardCommit::Layout {
                offset: 50,
                length: 100,
            }
        );
    }

    #[test]
    fn forward_short_fetch() {
        let outcome = parse_forward_preamble(&[0u8; 4], 0, 1024, 1024);
        assert!(matches!(
            outcome,
            ForwardOutcome::Terminate {
                reason: REASON_SHORT_FETCH_FWD
            }
        ));
    }

    #[test]
    fn forward_bad_magic() {
        let mut buf = vec![0u8; PREAMBLE_SIZE];
        buf[..8].copy_from_slice(b"NOTMAGIC");
        let outcome = parse_forward_preamble(&buf, 0, 1024, 1024);
        assert!(matches!(
            outcome,
            ForwardOutcome::Terminate {
                reason: REASON_BAD_MAGIC_FWD
            }
        ));
    }

    #[test]
    fn forward_streaming_preamble() {
        let buf = make_preamble(0);
        let outcome = parse_forward_preamble(&buf, 0, 1024, 1024);
        assert_eq!(outcome, ForwardOutcome::Streaming { remaining: 1024 });
    }

    #[test]
    fn forward_length_below_minimum_message_size() {
        let buf = make_preamble(10);
        let outcome = parse_forward_preamble(&buf, 0, 1024, 1024);
        assert!(matches!(
            outcome,
            ForwardOutcome::Terminate {
                reason: REASON_LENGTH_OUT_OF_RANGE_FWD
            }
        ));
    }

    #[test]
    fn forward_length_exceeds_file() {
        let buf = make_preamble(2000);
        let outcome = parse_forward_preamble(&buf, 0, 1024, 1024);
        assert!(matches!(
            outcome,
            ForwardOutcome::Terminate {
                reason: REASON_LENGTH_OUT_OF_RANGE_FWD
            }
        ));
    }

    #[test]
    fn forward_hit() {
        let buf = make_preamble(100);
        let outcome = parse_forward_preamble(&buf, 0, 1024, 1024);
        assert_eq!(
            outcome,
            ForwardOutcome::Hit {
                offset: 0,
                length: 100,
                msg_end: 100,
            }
        );
    }

    #[test]
    fn forward_exceeds_bound_when_msg_end_above_bound_below_file_size() {
        let buf = make_preamble(100);
        let outcome = parse_forward_preamble(&buf, 0, 1024, 50);
        assert_eq!(
            outcome,
            ForwardOutcome::ExceedsBound {
                offset: 0,
                length: 100,
                msg_end: 100,
            }
        );
    }

    #[test]
    fn same_message_yes() {
        assert!(same_message_check(0, 100, 0, 100));
    }

    #[test]
    fn same_message_offset_diff() {
        assert!(!same_message_check(0, 100, 50, 100));
    }

    #[test]
    fn same_message_length_diff() {
        assert!(!same_message_check(0, 100, 0, 200));
    }
}
