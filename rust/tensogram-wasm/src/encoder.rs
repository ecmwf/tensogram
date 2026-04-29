// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Frame-at-a-time streaming encoder for JavaScript callers.
//!
//! Two operating modes, both driven by the Rust-core
//! `StreamingEncoder<W: Write>` generic:
//!
//! - **Buffered (default).**  The sink is an in-memory `Vec<u8>`.  Every
//!   `write_object` / `write_preceder` appends to the buffer; `finish()`
//!   returns the complete wire-format message as a `Uint8Array`.
//!   Matches the Python `StreamingEncoder` model.
//!
//! - **Streaming.**  The sink is a [`JsCallbackWriter`] that forwards
//!   every chunk of bytes the core encoder emits to a caller-provided
//!   `(chunk: Uint8Array) => void` JS callback.  No full-message
//!   buffering — the callback is invoked during construction (preamble +
//!   header frames), during each `write_object` (one data-object frame's
//!   bytes), and during `finish()` (footer frames + postamble).
//!   `finish()` still returns a `Uint8Array`, but in streaming mode it
//!   is empty because every byte has already gone through the callback.
//!
//! The TypeScript wrapper selects the mode via
//! `StreamingEncoderOptions.onBytes`; the mode is fixed at construction
//! time and cannot be switched.
//!
//! ```js
//! // Buffered (default):
//! const enc = new StreamingEncoder({ version: 3 });
//! enc.writeObject(descriptor, new Float32Array([1, 2, 3]));
//! const bytes = enc.finish();          // full wire-format message
//!
//! // Streaming:
//! const chunks = [];
//! const enc = new StreamingEncoder({ version: 3 }, {
//!   onBytes: (chunk) => chunks.push(new Uint8Array(chunk)),
//! });
//! enc.writeObject(descriptor, new Float32Array([1, 2, 3]));
//! enc.finish();                        // callback has received every chunk
//! const bytes = Uint8Array.from(chunks.flatMap((c) => Array.from(c)));
//! ```

use crate::convert::*;
use std::collections::BTreeMap;
use std::io::Write;
use tensogram::{self as core, TensogramError};
use wasm_bindgen::prelude::*;

// ── Sinks ───────────────────────────────────────────────────────────────────

type BufferedEncoder = core::streaming::StreamingEncoder<Vec<u8>>;
type StreamingCoreEncoder = core::streaming::StreamingEncoder<JsCallbackWriter>;

/// Internal sink selection — the mode is fixed at construction.
enum Inner {
    Buffered(BufferedEncoder),
    Streaming(StreamingCoreEncoder),
}

impl Inner {
    fn write_preceder(
        &mut self,
        map: BTreeMap<String, ciborium::Value>,
    ) -> Result<(), TensogramError> {
        match self {
            Inner::Buffered(e) => e.write_preceder(map),
            Inner::Streaming(e) => e.write_preceder(map),
        }
    }

    fn write_object(
        &mut self,
        desc: &core::DataObjectDescriptor,
        bytes: &[u8],
    ) -> Result<(), TensogramError> {
        match self {
            Inner::Buffered(e) => e.write_object(desc, bytes),
            Inner::Streaming(e) => e.write_object(desc, bytes),
        }
    }

    fn write_object_pre_encoded(
        &mut self,
        desc: &core::DataObjectDescriptor,
        bytes: &[u8],
    ) -> Result<(), TensogramError> {
        match self {
            Inner::Buffered(e) => e.write_object_pre_encoded(desc, bytes),
            Inner::Streaming(e) => e.write_object_pre_encoded(desc, bytes),
        }
    }

    fn object_count(&self) -> usize {
        match self {
            Inner::Buffered(e) => e.object_count(),
            Inner::Streaming(e) => e.object_count(),
        }
    }

    fn bytes_written(&self) -> u64 {
        match self {
            Inner::Buffered(e) => e.bytes_written(),
            Inner::Streaming(e) => e.bytes_written(),
        }
    }
}

/// `std::io::Write` sink that forwards every chunk of bytes to a
/// synchronous JavaScript callback.
///
/// The callback must be synchronous — any `Promise` it returns is
/// silently discarded because the Rust `Write::write` contract is
/// synchronous.  The TS wrapper documents this contract; callers that
/// need async work (e.g. `fetch` upload) should either buffer
/// internally first or use the buffered mode with a single `fetch`
/// call.
///
/// Errors thrown by the callback surface as
/// `std::io::Error::other(...)`, which the core encoder then wraps as
/// `TensogramError::Io` — the TypeScript wrapper routes this to
/// `IoError` via the standard `mapTensogramError` path.
struct JsCallbackWriter {
    callback: js_sys::Function,
}

impl JsCallbackWriter {
    fn new(callback: js_sys::Function) -> Self {
        Self { callback }
    }
}

impl Write for JsCallbackWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // `Uint8Array::from(&[u8])` copies into JS-heap memory — the
        // callback receives a fresh, JS-owned view each time, so the
        // caller can hold on to it past the Rust side of this call.
        let chunk = js_sys::Uint8Array::from(buf);
        let this = JsValue::NULL;
        match self.callback.call1(&this, &chunk) {
            Ok(_) => Ok(buf.len()),
            Err(js_err) => {
                let message = js_err.as_string().unwrap_or_else(|| format!("{js_err:?}"));
                Err(std::io::Error::other(format!(
                    "streaming sink callback failed: {message}"
                )))
            }
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

// ── Exported class ──────────────────────────────────────────────────────────

/// Streaming encoder with a selectable sink: in-memory buffer (default)
/// or caller-supplied JS callback.
///
/// Lifecycle:
/// 1. `new(meta, hash?, on_bytes?)` — writes preamble + header metadata
///    frame.  In buffered mode these bytes accumulate internally; in
///    streaming mode they flow to `on_bytes` immediately.
/// 2. Zero or more `write_preceder(meta)` / `write_object(desc, data)` /
///    `write_object_pre_encoded(desc, data)` calls.
/// 3. `finish()` writes the footer + postamble.  In buffered mode
///    returns the complete `Uint8Array`; in streaming mode returns an
///    empty `Uint8Array` (the callback has seen every byte).
///
/// After `finish()` the encoder is closed — every further method call
/// throws "already finished".  Callers must still invoke the
/// wasm-bindgen `free()` method when done with the handle.
#[wasm_bindgen]
pub struct StreamingEncoder {
    /// `None` once `finish()` has been called — every mutator checks for
    /// this and raises a clean "already finished" error.
    inner: Option<Inner>,
}

#[wasm_bindgen]
impl StreamingEncoder {
    /// Begin a new streaming message.
    ///
    /// @param metadata_js - GlobalMetadata (JS object).  Must contain
    ///   `version`; `base` may carry per-object entries; `_reserved_` is
    ///   rejected (library-managed).
    /// @param hash - When `true` (default), xxh3 hashes are computed
    ///   per object and stored in the footer hash frame.  When `false`,
    ///   hashing is disabled entirely.
    /// @param on_bytes - Optional synchronous callback invoked with
    ///   each chunk of wire-format bytes the encoder produces.  When
    ///   present, no internal buffering is performed and `finish()`
    ///   returns an empty `Uint8Array`.  When absent, the encoder
    ///   buffers to an internal `Vec<u8>` and `finish()` returns the
    ///   complete message.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        metadata_js: JsValue,
        hash: Option<bool>,
        on_bytes: Option<js_sys::Function>,
        allow_nan: Option<bool>,
        allow_inf: Option<bool>,
        nan_mask_method: Option<String>,
        pos_inf_mask_method: Option<String>,
        neg_inf_mask_method: Option<String>,
        small_mask_threshold_bytes: Option<usize>,
    ) -> Result<StreamingEncoder, JsValue> {
        let metadata = metadata_from_js(&metadata_js)?;
        let options = build_encode_options_full(
            hash,
            allow_nan,
            allow_inf,
            nan_mask_method.as_deref(),
            pos_inf_mask_method.as_deref(),
            neg_inf_mask_method.as_deref(),
            small_mask_threshold_bytes,
        )?;
        let inner = match on_bytes {
            Some(cb) => {
                let sink = JsCallbackWriter::new(cb);
                Inner::Streaming(
                    StreamingCoreEncoder::new(sink, &metadata, &options).map_err(js_err)?,
                )
            }
            None => Inner::Buffered(
                BufferedEncoder::new(Vec::new(), &metadata, &options).map_err(js_err)?,
            ),
        };
        Ok(StreamingEncoder { inner: Some(inner) })
    }

    /// Write a PrecederMetadata frame for the next data object.
    ///
    /// The provided object is merged into a GlobalMetadata with a
    /// single-entry `base` array.  Must be followed by exactly one
    /// `write_object` / `write_object_pre_encoded` call before another
    /// `write_preceder` or `finish`.
    pub fn write_preceder(&mut self, metadata_js: JsValue) -> Result<(), JsValue> {
        let inner = self.inner.as_mut().ok_or_else(already_finished)?;
        let map: BTreeMap<String, ciborium::Value> =
            serde_wasm_bindgen::from_value(metadata_js).map_err(js_err_display)?;
        inner.write_preceder(map).map_err(js_err)
    }

    /// Encode and write a single data object.
    ///
    /// @param descriptor_js - `DataObjectDescriptor` as a plain JS object.
    /// @param data - Raw native-endian payload as any TypedArray.
    pub fn write_object(&mut self, descriptor_js: JsValue, data: JsValue) -> Result<(), JsValue> {
        self.write_with(descriptor_js, data, |inner, desc, bytes| {
            inner.write_object(desc, bytes)
        })
    }

    /// Write a pre-encoded data object directly (no pipeline).
    ///
    /// `data` must already be encoded according to the descriptor's
    /// `encoding` / `filter` / `compression`.  The library does not run
    /// the pipeline — it validates descriptor structure and the szip
    /// block offsets (if any) and writes bytes verbatim.  The hash is
    /// recomputed from these bytes if the encoder was constructed with
    /// `hash: true`.
    pub fn write_object_pre_encoded(
        &mut self,
        descriptor_js: JsValue,
        data: JsValue,
    ) -> Result<(), JsValue> {
        self.write_with(descriptor_js, data, |inner, desc, bytes| {
            inner.write_object_pre_encoded(desc, bytes)
        })
    }

    /// Number of data objects written so far.  Zero after `new()`,
    /// increments on every successful `write_object` /
    /// `write_object_pre_encoded`.
    pub fn object_count(&self) -> Result<usize, JsValue> {
        Ok(self
            .inner
            .as_ref()
            .ok_or_else(already_finished)?
            .object_count())
    }

    /// Total bytes produced so far (preamble + header frames + all
    /// completed data-object frames).  In buffered mode this equals
    /// the length of the internal buffer; in streaming mode it equals
    /// the sum of byte-lengths passed to the callback.
    ///
    /// Returned as `f64` because JS numbers are the lingua-franca for
    /// sizes on the wire boundary.  `Number.MAX_SAFE_INTEGER` ≈ 9 PiB,
    /// which is well beyond any realistic Tensogram message.
    pub fn bytes_written(&self) -> Result<f64, JsValue> {
        Ok(self
            .inner
            .as_ref()
            .ok_or_else(already_finished)?
            .bytes_written() as f64)
    }

    /// Finalise the encoder, writing footer frames + postamble.
    ///
    /// In buffered mode returns the complete wire-format `Uint8Array`.
    /// In streaming mode the footer bytes flow through the callback
    /// and the return value is an empty `Uint8Array` (zero-length
    /// marker, not a failure — every byte has already been delivered).
    ///
    /// After this call the encoder is closed — any further method call
    /// throws "already finished".  Callers must still invoke the
    /// wasm-bindgen `free()` method when done with the handle.
    pub fn finish(&mut self) -> Result<js_sys::Uint8Array, JsValue> {
        let inner = self.inner.take().ok_or_else(already_finished)?;
        match inner {
            Inner::Buffered(e) => {
                let buf = e.finish().map_err(js_err)?;
                Ok(js_sys::Uint8Array::from(buf.as_slice()))
            }
            Inner::Streaming(e) => {
                // The core returns the sink (JsCallbackWriter); we
                // discard it — every byte has already gone to the JS
                // callback.  A zero-length `Uint8Array` keeps the
                // return type stable across both modes.
                let _sink = e.finish().map_err(js_err)?;
                Ok(js_sys::Uint8Array::new_with_length(0))
            }
        }
    }
}

impl StreamingEncoder {
    /// Shared dispatch for `write_object` / `write_object_pre_encoded`:
    /// deserialise the descriptor, extract raw bytes from any
    /// TypedArray, then hand off to the supplied core-level writer.
    fn write_with(
        &mut self,
        descriptor_js: JsValue,
        data: JsValue,
        core_fn: impl FnOnce(
            &mut Inner,
            &core::DataObjectDescriptor,
            &[u8],
        ) -> Result<(), TensogramError>,
    ) -> Result<(), JsValue> {
        let inner = self.inner.as_mut().ok_or_else(already_finished)?;
        let desc: core::DataObjectDescriptor =
            serde_wasm_bindgen::from_value(descriptor_js).map_err(js_err_display)?;
        let bytes = typed_array_or_u8_to_bytes(&data)
            .ok_or_else(|| JsValue::from(js_sys::Error::new("data must be a TypedArray or Uint8Array")))?;
        core_fn(inner, &desc, &bytes).map_err(js_err)
    }
}

fn already_finished() -> JsValue {
    js_sys::Error::new("StreamingEncoder already finished").into()
}
