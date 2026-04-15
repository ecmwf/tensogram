// (C) Copyright 2024- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Frame-by-frame streaming decoder for progressive chunk feeding.
//!
//! Accumulates bytes from a stream and decodes complete messages as
//! they arrive. Each decoded data object is emitted as a `DecodedFrame`
//! that the JS caller can pull via `next_frame()`.
//!
//! ```js
//! const decoder = new tensogram.StreamingDecoder();
//! const reader = response.body.getReader();
//! while (true) {
//!   const { done, value } = await reader.read();
//!   if (done) break;
//!   decoder.feed(value);
//!   let frame;
//!   while ((frame = decoder.next_frame())) {
//!     const data = frame.data_f32();
//!     renderToCanvas(frame.descriptor().shape, data);
//!     frame.free();
//!   }
//! }
//! decoder.free();
//! ```

use crate::convert::*;
use tensogram_core as core;
use wasm_bindgen::prelude::*;

/// Default maximum buffer size: 256 MiB.  Prevents unbounded memory
/// growth when the stream contains garbage or an incomplete message
/// header that never completes.
const DEFAULT_MAX_BUFFER: usize = 256 * 1024 * 1024;

/// Frame-by-frame streaming decoder.
///
/// Accumulates bytes from progressive feeding and emits decoded data
/// objects as complete messages arrive.
///
/// **Error visibility**: If a scanned message fails to decode (corrupt
/// payload), the error is captured in `last_error()` and the decoder
/// advances past the bad message.  Call `last_error()` after each
/// `feed()` to check for skipped messages.
///
/// **Memory limit**: The internal buffer is capped at 256 MiB by
/// default.  Call `set_max_buffer(n)` to change it.  Exceeding the
/// limit makes `feed()` return a `JsError`.
#[wasm_bindgen]
pub struct StreamingDecoder {
    buffer: Vec<u8>,
    /// Byte offset of the next unprocessed region.
    consumed: usize,
    /// Global metadata from the most recently decoded message.
    global_metadata: Option<core::GlobalMetadata>,
    /// Queue of decoded frames ready for JS to consume.
    ready_frames: std::collections::VecDeque<DecodedFrame>,
    /// Last decode error (if a scanned message failed to decode).
    last_decode_error: Option<String>,
    /// Count of messages that were scanned but failed to decode.
    skipped_messages: usize,
    /// Maximum buffer size in bytes.
    max_buffer: usize,
}

#[wasm_bindgen]
impl StreamingDecoder {
    /// Create a new streaming decoder.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            consumed: 0,
            global_metadata: None,
            ready_frames: std::collections::VecDeque::new(),
            last_decode_error: None,
            skipped_messages: 0,
            max_buffer: DEFAULT_MAX_BUFFER,
        }
    }

    /// Feed a chunk of bytes into the decoder.
    ///
    /// Internally scans for complete messages and decodes each one,
    /// emitting individual data objects as `DecodedFrame`s.
    ///
    /// Returns an error if the internal buffer exceeds `max_buffer` bytes.
    /// Check `last_error()` after feeding to detect skipped corrupt messages.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<(), JsError> {
        let new_size = (self.buffer.len() - self.consumed)
            .checked_add(chunk.len())
            .ok_or_else(|| JsError::new("buffer size overflow"))?;
        if new_size > self.max_buffer {
            return Err(JsError::new(&format!(
                "streaming buffer would grow to {} bytes (limit {})",
                new_size, self.max_buffer
            )));
        }
        // Compact before extending so the actual Vec length (and WASM memory)
        // stays close to the logical limit instead of growing by `consumed`.
        if self.consumed > 0 {
            self.buffer.drain(..self.consumed);
            self.consumed = 0;
        }
        self.buffer.extend_from_slice(chunk);
        self.last_decode_error = None; // clear previous error
        self.try_decode_messages();
        Ok(())
    }

    /// Pull the next decoded data object frame, or `undefined` if none ready.
    ///
    /// In JavaScript, `wasm-bindgen` maps Rust `None` to `undefined`.
    /// Use a truthiness check: `while ((frame = decoder.next_frame()))`.
    pub fn next_frame(&mut self) -> Option<DecodedFrame> {
        self.ready_frames.pop_front()
    }

    /// Whether global metadata has been received from at least one message.
    pub fn has_metadata(&self) -> bool {
        self.global_metadata.is_some()
    }

    /// Get the global metadata from the most recently decoded message.
    pub fn metadata(&self) -> Result<JsValue, JsError> {
        match &self.global_metadata {
            Some(meta) => to_js(meta),
            None => Ok(JsValue::NULL),
        }
    }

    /// Number of decoded frames ready to consume.
    pub fn pending_count(&self) -> usize {
        self.ready_frames.len()
    }

    /// Total bytes buffered but not yet decoded.
    pub fn buffered_bytes(&self) -> usize {
        self.buffer.len() - self.consumed
    }

    /// Error message from the last skipped (corrupt) message, or null.
    ///
    /// Cleared on each `feed()` call.  If non-null, at least one message
    /// found by the scanner failed to decode and was skipped.
    pub fn last_error(&self) -> Option<String> {
        self.last_decode_error.clone()
    }

    /// Total number of messages that were skipped due to decode errors
    /// since the decoder was created or last reset.
    pub fn skipped_count(&self) -> usize {
        self.skipped_messages
    }

    /// Set the maximum internal buffer size in bytes (default: 256 MiB).
    ///
    /// The limit applies to the total *unprocessed* bytes (already-buffered
    /// bytes plus the next incoming chunk).  If adding a new chunk would
    /// exceed this limit, `feed()` returns an error and the chunk is not
    /// buffered.  Reducing the limit below the current buffer size takes
    /// effect on the next `feed()` call.
    pub fn set_max_buffer(&mut self, max_bytes: usize) {
        self.max_buffer = max_bytes;
    }

    /// Reset the decoder, clearing all buffered data and pending frames.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.consumed = 0;
        self.global_metadata = None;
        self.ready_frames.clear();
        self.last_decode_error = None;
        self.skipped_messages = 0;
    }
}

impl StreamingDecoder {
    fn try_decode_messages(&mut self) {
        debug_assert!(
            self.consumed <= self.buffer.len(),
            "consumed ({}) > buffer.len() ({})",
            self.consumed,
            self.buffer.len()
        );

        let remaining = &self.buffer[self.consumed..];
        if remaining.is_empty() {
            return;
        }

        // Scan once for ALL complete messages in the remaining buffer.
        // This is O(n) instead of re-scanning after each decoded message.
        let positions = core::scan(remaining);

        let options = core::DecodeOptions {
            verify_hash: false,
            ..Default::default()
        };

        // Track the furthest byte position we've successfully processed.
        // `msg_end` values are relative to `remaining` (= buffer[consumed..]),
        // so we record the max end seen and advance `consumed` by that amount
        // once at the end.
        let mut furthest = 0usize;

        for (msg_start, msg_len) in positions {
            let msg_end = msg_start + msg_len;

            if msg_end > remaining.len() {
                break; // Incomplete trailing message — wait for more data
            }

            let msg_bytes = &remaining[msg_start..msg_end];

            match core::decode(msg_bytes, &options) {
                Ok((metadata, objects)) => {
                    let base_entries = &metadata.base;

                    for (i, (descriptor, data)) in objects.into_iter().enumerate() {
                        let base_entry = base_entries.get(i).cloned();
                        self.ready_frames.push_back(DecodedFrame {
                            descriptor,
                            data,
                            base_entry,
                        });
                    }

                    self.global_metadata = Some(metadata);
                }
                Err(e) => {
                    // Record the error so JS callers can inspect it.
                    // We still advance past the bad message to avoid an
                    // infinite re-scan loop.
                    self.last_decode_error = Some(e.to_string());
                    self.skipped_messages += 1;
                }
            }

            furthest = msg_end;
        }

        self.consumed += furthest;
    }
}

// ── DecodedFrame ─────────────────────────────────────────────────────────────

/// A single decoded data object from the streaming decoder.
///
/// Owns the decoded payload data.  Use `data_f32()` etc. for zero-copy
/// TypedArray views.  Call `.free()` when done to release WASM memory.
#[wasm_bindgen]
pub struct DecodedFrame {
    descriptor: core::DataObjectDescriptor,
    data: Vec<u8>,
    base_entry: Option<std::collections::BTreeMap<String, ciborium::Value>>,
}

#[wasm_bindgen]
impl DecodedFrame {
    /// Object descriptor (shape, dtype, encoding, etc.) as a JS object.
    pub fn descriptor(&self) -> Result<JsValue, JsError> {
        to_js(&self.descriptor)
    }

    /// Per-object metadata entry from the base array (if available).
    pub fn base_entry(&self) -> Result<JsValue, JsError> {
        match &self.base_entry {
            Some(entry) => to_js(entry),
            None => Ok(JsValue::NULL),
        }
    }

    /// Zero-copy Float32Array view into decoded payload.
    ///
    /// **Warning**: invalidated if WASM memory grows.
    pub fn data_f32(&self) -> Result<js_sys::Float32Array, JsError> {
        view_as_f32(&self.data)
    }

    /// Zero-copy Float64Array view.
    pub fn data_f64(&self) -> Result<js_sys::Float64Array, JsError> {
        view_as_f64(&self.data)
    }

    /// Zero-copy Int32Array view.
    pub fn data_i32(&self) -> Result<js_sys::Int32Array, JsError> {
        view_as_i32(&self.data)
    }

    /// Zero-copy Uint8Array view.
    pub fn data_u8(&self) -> Result<js_sys::Uint8Array, JsError> {
        Ok(view_as_u8(&self.data))
    }

    /// Payload byte length.
    pub fn byte_length(&self) -> usize {
        self.data.len()
    }
}
