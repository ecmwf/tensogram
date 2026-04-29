// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::collections::BTreeMap;

use crate::dtype::Dtype;
use crate::error::{Result, TensogramError};
use crate::framing::{self, EncodedObject};
use crate::hash::HashAlgorithm;
use crate::metadata::RESERVED_KEY;
use crate::substitute_and_mask::{self, MaskSet};
use crate::types::{DataObjectDescriptor, GlobalMetadata, MaskDescriptor, MasksMetadata};
pub use tensogram_encodings::bitmask::MaskMethod;
#[cfg(feature = "blosc2")]
use tensogram_encodings::pipeline::Blosc2Codec;
#[cfg(feature = "sz3")]
use tensogram_encodings::pipeline::Sz3ErrorBound;
#[cfg(feature = "zfp")]
use tensogram_encodings::pipeline::ZfpMode;
use tensogram_encodings::pipeline::{
    self, ByteOrder, CompressionType, EncodingType, FilterType, PipelineConfig,
};
use tensogram_encodings::simple_packing::{self, SimplePackingParams};

/// Options for encoding.
#[derive(Debug, Clone)]
pub struct EncodeOptions {
    /// Hash algorithm to use for payload integrity. None = no hashing.
    pub hash_algorithm: Option<HashAlgorithm>,
    /// Reserved for future buffered-mode preceder support.
    ///
    /// Currently, setting this to `true` in buffered mode (`encode()`)
    /// returns an error — use [`StreamingEncoder::write_preceder`](crate::streaming::StreamingEncoder::write_preceder) instead.
    /// The streaming encoder ignores this field; it emits preceders only
    /// when `write_preceder()` is called explicitly.
    pub emit_preceders: bool,
    /// Which backend to use for szip / zstd when both FFI and pure-Rust
    /// implementations are compiled in.
    ///
    /// Defaults to `Ffi` on native (faster, battle-tested) and `Pure` on
    /// `wasm32` (FFI cannot exist).  Override with
    /// `TENSOGRAM_COMPRESSION_BACKEND=pure` env variable, or set this
    /// field explicitly.
    pub compression_backend: pipeline::CompressionBackend,
    /// Thread budget for the multi-threaded coding pipeline.
    ///
    /// - `0` (default) — sequential (current behaviour).  Can be
    ///   overridden at runtime via `TENSOGRAM_THREADS=N`.
    /// - `1` — explicit single-threaded execution (bypasses env).
    /// - `N ≥ 2` — scoped pool of `N` workers.  Output bytes are
    ///   byte-identical to the sequential path regardless of `N`.
    ///
    /// When more than one data object is being encoded the budget is
    /// spent axis-B-first (intra-codec parallelism) — this codebase
    /// tends to have a small number of very large messages.  See the
    /// [multi-threaded pipeline guide](../../docs/src/guide/multi-threaded-pipeline.md)
    /// for the full policy.
    ///
    /// Ignored with a one-time `tracing::warn!` when the `threads`
    /// cargo feature is disabled.
    pub threads: u32,
    /// Minimum total payload bytes below which the parallel path is
    /// skipped even when `threads > 0`.
    ///
    /// `None` uses [`crate::DEFAULT_PARALLEL_THRESHOLD_BYTES`] (64 KiB).
    /// Set to `Some(0)` to force the parallel path for testing; set to
    /// `Some(usize::MAX)` to force sequential.
    pub parallel_threshold_bytes: Option<usize>,
    /// When `true`, NaN values in float / complex payloads are
    /// substituted with `0.0` and recorded in a bitmask companion
    /// section of the data-object frame (wire type 9
    /// `NTensorFrame`, see `plans/WIRE_FORMAT.md` §6.5).  When
    /// `false` (the default) any NaN in the input is a hard encode
    /// error.
    pub allow_nan: bool,
    /// When `true`, `+Inf` AND `-Inf` values are substituted with
    /// `0.0` and recorded in per-sign bitmasks (see `allow_nan`).
    /// The flag is a single switch for both signs on purpose — callers
    /// who want only one sign must pre-process their data.  When
    /// `false` (the default), any `±Inf` in the input is a hard encode
    /// error.
    pub allow_inf: bool,
    /// Compression method for the NaN mask (see
    /// [`tensogram_encodings::bitmask::MaskMethod`]).  Default
    /// [`MaskMethod::Roaring`].  Only consulted when `allow_nan` is
    /// `true` AND the input actually contained at least one NaN
    /// element.
    pub nan_mask_method: MaskMethod,
    /// Compression method for the `+Inf` mask.  Default
    /// [`MaskMethod::Roaring`].
    pub pos_inf_mask_method: MaskMethod,
    /// Compression method for the `-Inf` mask.  Default
    /// [`MaskMethod::Roaring`].
    pub neg_inf_mask_method: MaskMethod,
    /// Uncompressed byte-count threshold below which mask blobs are
    /// written with [`MaskMethod::None`] (raw packed bytes) regardless
    /// of the requested method.  Default `128`.  Single threshold
    /// across all three masks.  Set to `0` to disable the auto-fallback
    /// and always use the requested method.
    pub small_mask_threshold_bytes: usize,
    /// Emit a `HeaderHash` frame aggregating per-object hashes.
    ///
    /// Buffered mode default: `true`.  Streaming mode: forced to
    /// `false` — header frames are written before any data object,
    /// so the hashes aren't known yet.  Setting this to `true` while
    /// building a `StreamingEncoder` is a construction-time
    /// `EncodingError`.
    ///
    /// Ignored when `hash_algorithm` is `None`: if hashing is
    /// disabled there is nothing to aggregate.
    pub create_header_hashes: bool,
    /// Emit a `FooterHash` frame aggregating per-object hashes.
    ///
    /// Buffered mode default: `false` (the `HeaderHash` frame is
    /// the canonical place for the aggregate in buffered mode).
    /// Streaming mode default: `true` (the only place where the
    /// aggregate can live, since streamed data objects precede the
    /// footer).
    ///
    /// Ignored when `hash_algorithm` is `None`.
    pub create_footer_hashes: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            hash_algorithm: Some(HashAlgorithm::Xxh3),
            emit_preceders: false,
            compression_backend: pipeline::CompressionBackend::default(),
            threads: 0,
            parallel_threshold_bytes: None,
            allow_nan: false,
            allow_inf: false,
            nan_mask_method: MaskMethod::default(),
            pos_inf_mask_method: MaskMethod::default(),
            neg_inf_mask_method: MaskMethod::default(),
            small_mask_threshold_bytes: 128,
            // Buffered defaults: header-only aggregate.
            create_header_hashes: true,
            create_footer_hashes: false,
        }
    }
}

pub(crate) fn validate_object(desc: &DataObjectDescriptor, data_len: usize) -> Result<()> {
    if desc.obj_type.is_empty() {
        return Err(TensogramError::Metadata(
            "obj_type must not be empty".to_string(),
        ));
    }
    if desc.ndim as usize != desc.shape.len() {
        return Err(TensogramError::Metadata(format!(
            "ndim {} does not match shape.len() {}",
            desc.ndim,
            desc.shape.len()
        )));
    }
    if desc.strides.len() != desc.shape.len() {
        return Err(TensogramError::Metadata(format!(
            "strides.len() {} does not match shape.len() {}",
            desc.strides.len(),
            desc.shape.len()
        )));
    }
    if desc.encoding == "none" {
        let product = desc
            .shape
            .iter()
            .try_fold(1u64, |acc, &x| acc.checked_mul(x))
            .ok_or_else(|| TensogramError::Metadata("shape product overflow".to_string()))?;
        if desc.dtype.byte_width() > 0 {
            let expected_bytes = product
                .checked_mul(desc.dtype.byte_width() as u64)
                .ok_or_else(|| TensogramError::Metadata("shape product overflow".to_string()))?;
            if expected_bytes != data_len as u64 {
                return Err(TensogramError::Metadata(format!(
                    "data_len {data_len} does not match expected {expected_bytes} bytes from shape and dtype"
                )));
            }
        } else if desc.dtype == crate::Dtype::Bitmask {
            // Bitmask: expected data length is ceil(shape_product / 8)
            let expected_bytes = product.div_ceil(8);
            if expected_bytes != data_len as u64 {
                return Err(TensogramError::Metadata(format!(
                    "data_len {data_len} does not match expected {expected_bytes} bytes for bitmask (ceil({product}/8))"
                )));
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum EncodeMode {
    Raw,
    PreEncoded,
}

/// Encode a single object: run the pipeline (or validate pre-encoded
/// bytes), compute its hash, and return the `EncodedObject`.
///
/// `intra_codec_threads` is passed through to [`PipelineConfig`] and
/// honoured by axis-B-capable codecs (blosc2, zstd, simple_packing,
/// shuffle).  Pure functional — no shared state, safe to call from
/// multiple rayon workers in parallel.
fn encode_one_object(
    desc: &DataObjectDescriptor,
    data: &[u8],
    mode: EncodeMode,
    options: &EncodeOptions,
    intra_codec_threads: u32,
) -> Result<EncodedObject> {
    validate_object(desc, data.len())?;

    // Pre-pipeline substitute-and-mask stage (Raw mode only).  Two
    // flavours gated by the same call:
    //
    // - `allow_nan == false && allow_inf == false` — behaves like the
    //   0.17 default finite check: first NaN / Inf errors out.
    // - Either flag true — substitute non-finite values with the
    //   dtype-specific zero and collect per-kind bitmasks for the
    //   frame-writing stage.  See `plans/WIRE_FORMAT.md` §6.5 and
    //   `docs/src/guide/nan-inf-handling.md` for the user-facing
    //   semantics.
    //
    // Pre-encoded bytes are opaque: skip the stage.
    let (pipeline_input, mask_set) = if matches!(mode, EncodeMode::Raw) {
        let parallel = crate::parallel::should_parallelise(
            intra_codec_threads,
            data.len(),
            options.parallel_threshold_bytes,
        );
        let (cow, masks) = substitute_and_mask::substitute_and_mask(
            data,
            desc.dtype,
            desc.byte_order,
            options.allow_nan,
            options.allow_inf,
            parallel,
        )?;
        (cow, masks)
    } else {
        (std::borrow::Cow::Borrowed(data), MaskSet::empty(0))
    };

    let num_elements = desc.num_elements()?;
    let dtype = desc.dtype;

    // Build the descriptor we will emit on the wire.  For Raw mode this
    // includes the auto-compute step: when `encoding=simple_packing` and
    // the user left out `sp_reference_value` / `sp_binary_scale_factor`
    // we derive them from the data here so the final descriptor carries
    // the full explicit 4-key set.  `PreEncoded` skips this — the bytes
    // are opaque, the user must have supplied complete params.
    //
    // Use the ORIGINAL `data` (pre-substitute), not `pipeline_input`:
    // when `allow_nan` / `allow_inf` are on, NaN / Inf get replaced
    // with `0.0` and any auto-compute over the substituted bytes
    // would derive a `sp_reference_value` distorted by those zeros
    // (silent precision loss).  Using the original data makes
    // `simple_packing::compute_params` surface non-finite inputs as a
    // clear `PackingError`, telling the user to either supply explicit
    // params or pre-substitute their data.
    let mut final_desc = desc.clone();
    if matches!(mode, EncodeMode::Raw) {
        resolve_simple_packing_params(&mut final_desc, data)?;
    }

    let mut config = build_pipeline_config_with_backend(
        &final_desc,
        num_elements,
        dtype,
        options.compression_backend,
        intra_codec_threads,
    )?;

    // When xxh3 hashing is requested and we are running the pipeline
    // (Raw mode), ask the pipeline to compute it inline — this avoids
    // a second walk over the encoded payload.  The pipeline's inline
    // path is xxh3-specific; other `HashAlgorithm` variants and
    // `PreEncoded` mode fall back to `compute_hash` further down.
    //
    // The match on `options.hash_algorithm` is exhaustive so that
    // adding a new `HashAlgorithm` variant becomes a compile error
    // here, forcing the maintainer to either wire a new inline path
    // for that algorithm or to route it explicitly through the
    // post-hoc `compute_hash` fallback below.
    let inline_hash_requested = matches!(mode, EncodeMode::Raw)
        && match options.hash_algorithm {
            Some(HashAlgorithm::Xxh3) => true,
            None => false,
        };
    config.compute_hash = inline_hash_requested;

    let (encoded_payload, inline_hash) = match mode {
        EncodeMode::Raw => {
            // Run the full encoding pipeline on the (possibly
            // substituted) payload.  When substitution occurred the
            // `pipeline_input` is `Cow::Owned`; otherwise it's the
            // caller's bytes borrowed zero-cost.
            let result = pipeline::encode_pipeline(pipeline_input.as_ref(), &config)
                .map_err(|e| TensogramError::Encoding(e.to_string()))?;

            // Store szip block offsets if produced
            if let Some(offsets) = &result.block_offsets {
                final_desc.params.insert(
                    "szip_block_offsets".to_string(),
                    ciborium::Value::Array(
                        offsets
                            .iter()
                            .map(|&o| ciborium::Value::Integer(o.into()))
                            .collect(),
                    ),
                );
            }

            (result.encoded_bytes, result.hash)
        }
        EncodeMode::PreEncoded => {
            // Caller's bytes are already encoded — use them directly.
            // `build_pipeline_config_with_backend` was called above purely
            // for defence-in-depth validation of the declared
            // encoding/compression params.
            validate_no_szip_offsets_for_non_szip(desc)?;
            if desc.compression == "szip" && desc.params.contains_key("szip_block_offsets") {
                validate_szip_block_offsets(&desc.params, data.len())?;
            }
            (data.to_vec(), None)
        }
    };

    // ── Compose the payload region: [encoded_payload][mask_nan][mask_inf+][mask_inf-] ──
    //
    // When no masks were collected (the common case), the region is
    // just `encoded_payload` and the descriptor gets `masks = None`,
    // making the frame byte-identical to the legacy `NTensorFrame`
    // payload layout except for the frame-type number.
    //
    // When masks ARE present, each kind is compressed via the
    // user-specified `MaskMethod` (with auto-fallback to `None` for
    // tiny masks), appended to the payload region in the canonical
    // order `nan`, `inf+`, `inf-` (matching the CBOR key sort order),
    // and each kind's CBOR descriptor records its byte offset and
    // length relative to the region start.  See
    // `plans/WIRE_FORMAT.md` §6.5.
    let (payload_region, masks_metadata) = compose_payload_region(
        encoded_payload,
        mask_set,
        &options.nan_mask_method,
        &options.pos_inf_mask_method,
        &options.neg_inf_mask_method,
        options.small_mask_threshold_bytes,
    )?;
    if let Some(m) = masks_metadata {
        final_desc.masks = Some(m);
    }
    let encoded_payload = payload_region;

    // v3: the per-object hash is no longer written into the CBOR
    // descriptor.  It lives in the inline hash slot of the frame's
    // footer (see `plans/WIRE_FORMAT.md` §2.4), populated by
    // `encode_data_object_frame` at frame-build time.  The
    // aggregate HashFrame reads those slots back in
    // `framing::build_hash_frame_cbor` — no second pass here.
    //
    // `inline_hash` from the pipeline is redundant with the
    // inline slot (same digest, different storage) and is
    // intentionally unused on this path; keeping it in the return
    // tuple preserves the pipeline's hash-while-encoding
    // invariant for callers that want a digest without going
    // through the frame layer.
    let _ = (inline_hash, options);

    Ok(EncodedObject {
        descriptor: final_desc,
        encoded_payload,
    })
}

fn encode_inner(
    global_metadata: &GlobalMetadata,
    descriptors: &[(&DataObjectDescriptor, &[u8])],
    options: &EncodeOptions,
    mode: EncodeMode,
) -> Result<Vec<u8>> {
    // Buffered encode does not support emit_preceders — use StreamingEncoder
    // with write_preceder() instead.
    if options.emit_preceders {
        return Err(TensogramError::Encoding(
            "emit_preceders is not supported in buffered mode; use StreamingEncoder::write_preceder() instead".to_string(),
        ));
    }

    // ── Thread-budget dispatch (axis-B-first policy) ────────────────────
    //
    // Resolve the effective thread budget (explicit option > env var),
    // decide if the workload is large enough to parallelise, and pick
    // axis A (par_iter across objects) vs axis B (sequential, codec
    // uses the budget internally).
    let budget = crate::parallel::resolve_budget(options.threads);
    let total_bytes: usize = descriptors.iter().map(|(_, d)| d.len()).sum();
    let parallel =
        crate::parallel::should_parallelise(budget, total_bytes, options.parallel_threshold_bytes);

    let any_axis_b = descriptors
        .iter()
        .any(|(d, _)| crate::parallel::is_axis_b_friendly(&d.encoding, &d.filter, &d.compression));
    let use_axis_a = parallel && crate::parallel::use_axis_a(descriptors.len(), budget, any_axis_b);

    // Axis B gets the full budget; axis A keeps codecs sequential so
    // that the product of axis A and axis B threads never exceeds the
    // caller's ask.
    let intra_codec_threads = if parallel && !use_axis_a { budget } else { 0 };

    let encode_one = |(desc, data): &(&DataObjectDescriptor, &[u8])| {
        encode_one_object(desc, data, mode, options, intra_codec_threads)
    };

    let encoded_objects: Vec<EncodedObject> = if use_axis_a {
        // Axis A: par_iter across objects.  Requires the `threads`
        // feature; when it's off, the caller's budget silently falls
        // back to sequential (with a one-time warning from `with_pool`).
        #[cfg(feature = "threads")]
        {
            use rayon::prelude::*;
            crate::parallel::with_pool(budget, || {
                descriptors
                    .par_iter()
                    .map(&encode_one)
                    .collect::<Result<Vec<_>>>()
            })?
        }
        #[cfg(not(feature = "threads"))]
        {
            descriptors.iter().map(encode_one).collect::<Result<_>>()?
        }
    } else {
        // Axis B (or purely sequential): iterate objects in order.
        // Install the pool when there's an intra-codec budget so that
        // parallel primitives inside codec implementations (e.g.
        // `simple_packing` chunked par_iter) actually use it.
        crate::parallel::run_maybe_pooled(budget, parallel, intra_codec_threads, || {
            descriptors.iter().map(encode_one).collect::<Result<_>>()
        })?
    };

    // Validate that the caller hasn't written to _reserved_ at any level.
    validate_no_client_reserved(global_metadata)?;

    // Validate base/descriptor count: base may be shorter (auto-extended) or
    // equal, but having MORE base entries than descriptors is an error —
    // the extra entries would be silently discarded.
    if global_metadata.base.len() > descriptors.len() {
        return Err(TensogramError::Metadata(format!(
            "metadata base has {} entries but only {} descriptors provided; \
             extra base entries would be discarded",
            global_metadata.base.len(),
            descriptors.len()
        )));
    }

    // Populate per-object base entries with _reserved_.tensor (ndim/shape/strides/dtype).
    // Pre-existing application keys (e.g. "mars") are preserved.
    let mut enriched_meta = global_metadata.clone();
    populate_base_entries(&mut enriched_meta.base, &encoded_objects);
    populate_reserved_provenance(&mut enriched_meta.reserved);

    // Derive the aggregate HashFrame policy from the buffered-mode
    // options.  Streaming uses a different path (force-false
    // `create_header_hashes`); here in `encode()` we honor the
    // caller's choice as-is.
    let hash_policy = framing::HashFramePolicy {
        header: options.create_header_hashes,
        footer: options.create_footer_hashes,
    };
    framing::encode_message(
        &enriched_meta,
        &encoded_objects,
        options.hash_algorithm,
        hash_policy,
    )
}

/// Encode a complete Tensogram message.
///
/// `global_metadata` is the message-level metadata (version, MARS keys, etc.).
/// `descriptors` is a list of (DataObjectDescriptor, raw_data) pairs.
/// Returns the complete wire-format message.
#[tracing::instrument(skip(global_metadata, descriptors, options), fields(objects = descriptors.len()))]
pub fn encode(
    global_metadata: &GlobalMetadata,
    descriptors: &[(&DataObjectDescriptor, &[u8])],
    options: &EncodeOptions,
) -> Result<Vec<u8>> {
    encode_inner(global_metadata, descriptors, options, EncodeMode::Raw)
}

/// Encode a pre-encoded Tensogram message where callers supply already-encoded bytes.
///
/// Use this when the payload bytes have already been encoded/compressed by an external
/// pipeline. The library will:
/// - Validate object descriptors (shape, dtype, etc.)
/// - Validate encoding/compression params via `build_pipeline_config()` (defense-in-depth)
/// - Use the caller's bytes directly as the encoded payload (no pipeline call)
/// - Compute a fresh xxh3 hash over the caller's bytes (overwrites any caller-supplied hash)
/// - Preserve caller-supplied `szip_block_offsets` in descriptor params
///
/// Callers must NOT set:
/// - `emit_preceders = true` — use `StreamingEncoder::write_preceder()` for streaming
///   preceder support.
///
/// Unlike `encode()`, this path does NOT run the finite-value check — the caller's
/// bytes are assumed to be already well-formed for the declared encoding and are
/// written as-is.  If the pre-encoded bytes decode to NaN / Inf, that round-trips
/// through the wire unchanged.
#[tracing::instrument(name = "encode_pre_encoded", skip_all, fields(num_objects = descriptors.len()))]
pub fn encode_pre_encoded(
    global_metadata: &GlobalMetadata,
    descriptors: &[(&DataObjectDescriptor, &[u8])],
    options: &EncodeOptions,
) -> Result<Vec<u8>> {
    encode_inner(
        global_metadata,
        descriptors,
        options,
        EncodeMode::PreEncoded,
    )
}

/// Validate that the caller hasn't written to `_reserved_` at any level.
///
/// The `_reserved_` namespace is library-managed.  Client code must not
/// set it in the message-level metadata or in any `base[i]` entry.
fn validate_no_client_reserved(meta: &GlobalMetadata) -> Result<()> {
    if !meta.reserved.is_empty() {
        return Err(TensogramError::Metadata(format!(
            "client code must not write to '{RESERVED_KEY}' at message level; \
             this field is populated by the library"
        )));
    }
    for (i, entry) in meta.base.iter().enumerate() {
        if entry.contains_key(RESERVED_KEY) {
            return Err(TensogramError::Metadata(format!(
                "client code must not write to '{RESERVED_KEY}' in base[{i}]; \
                 this field is populated by the library"
            )));
        }
    }
    Ok(())
}

/// Populate per-object base entries with tensor metadata under `_reserved_.tensor`.
///
/// Resizes `base` to match the object count, then inserts a `_reserved_`
/// map containing `tensor: {ndim, shape, strides, dtype}` into each entry.
/// Pre-existing application keys (e.g. `"mars"`) are preserved.
pub(crate) fn populate_base_entries(
    base: &mut Vec<BTreeMap<String, ciborium::Value>>,
    encoded_objects: &[crate::framing::EncodedObject],
) {
    use ciborium::Value;

    // Ensure base has exactly one entry per object.
    base.resize_with(encoded_objects.len(), BTreeMap::new);

    for (entry, obj) in base.iter_mut().zip(encoded_objects.iter()) {
        let desc = &obj.descriptor;

        let tensor_map = Value::Map(vec![
            (
                Value::Text("ndim".to_string()),
                Value::Integer(desc.ndim.into()),
            ),
            (
                Value::Text("shape".to_string()),
                Value::Array(
                    desc.shape
                        .iter()
                        .map(|&d| Value::Integer(d.into()))
                        .collect(),
                ),
            ),
            (
                Value::Text("strides".to_string()),
                Value::Array(
                    desc.strides
                        .iter()
                        .map(|&s| Value::Integer(s.into()))
                        .collect(),
                ),
            ),
            (
                Value::Text("dtype".to_string()),
                Value::Text(desc.dtype.to_string()),
            ),
        ]);

        let reserved_map = Value::Map(vec![(Value::Text("tensor".to_string()), tensor_map)]);

        entry.insert(RESERVED_KEY.to_string(), reserved_map);
    }
}

/// Populate the `reserved` section with provenance fields as specified in
/// `WIRE_FORMAT.md`:
///
/// - `encoder.name` — `"tensogram"`
/// - `encoder.version` — library version at encode time
/// - `time` — UTC ISO 8601 timestamp
/// - `uuid` — RFC 4122 v4 UUID
///
/// Pre-existing keys in `reserved` are preserved; only these four are
/// set (or overwritten).
pub(crate) fn populate_reserved_provenance(reserved: &mut BTreeMap<String, ciborium::Value>) {
    use ciborium::Value;
    #[cfg(not(target_arch = "wasm32"))]
    use std::time::SystemTime;

    // encoder.name + encoder.version
    let version_str = env!("CARGO_PKG_VERSION");
    let encoder_map = Value::Map(vec![
        (
            Value::Text("name".to_string()),
            Value::Text("tensogram".to_string()),
        ),
        (
            Value::Text("version".to_string()),
            Value::Text(version_str.to_string()),
        ),
    ]);
    reserved.insert("encoder".to_string(), encoder_map);

    // time — ISO 8601 UTC
    // On wasm32-unknown-unknown, SystemTime::now() panics. Skip the `time`
    // field entirely rather than encoding a misleading epoch-0 timestamp.
    // Callers can set a timestamp via `_extra_` if needed.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let secs = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Simple UTC format: YYYY-MM-DDThh:mm:ssZ
        // We compute from epoch seconds to avoid adding a datetime crate.
        let days = secs / 86400;
        let day_secs = secs % 86400;
        let hours = day_secs / 3600;
        let minutes = (day_secs % 3600) / 60;
        let seconds = day_secs % 60;
        // Civil date from days since 1970-01-01 (Howard Hinnant algorithm)
        let (y, m, d) = civil_from_days(days as i64);
        let timestamp = format!("{y:04}-{m:02}-{d:02}T{hours:02}:{minutes:02}:{seconds:02}Z");
        reserved.insert("time".to_string(), Value::Text(timestamp));
    }

    // uuid — RFC 4122 v4
    let id = uuid::Uuid::new_v4();
    reserved.insert("uuid".to_string(), Value::Text(id.to_string()));
}

/// Convert days since 1970-01-01 to (year, month, day).
/// Howard Hinnant's algorithm (public domain).
#[cfg(not(target_arch = "wasm32"))]
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    // doe (day of era) is guaranteed in [0, 146096] by the era computation,
    // so the u32 cast cannot truncate.
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

pub(crate) fn build_pipeline_config(
    desc: &DataObjectDescriptor,
    num_values: usize,
    dtype: Dtype,
) -> Result<PipelineConfig> {
    build_pipeline_config_with_backend(
        desc,
        num_values,
        dtype,
        pipeline::CompressionBackend::default(),
        0,
    )
}

/// Resolve the encoding type from a descriptor.
fn resolve_encoding(desc: &DataObjectDescriptor, dtype: Dtype) -> Result<EncodingType> {
    match desc.encoding.as_str() {
        "none" => Ok(EncodingType::None),
        "simple_packing" => {
            // Strict float64 check.  A `byte_width() != 8` test would
            // also let `Int64` / `Uint64` / `Complex64` through, and
            // the pipeline would then re-interpret their bytes as
            // f64 and produce silently-wrong output.
            if dtype != Dtype::Float64 {
                return Err(TensogramError::Encoding(format!(
                    "simple_packing only supports float64 dtype; got {dtype:?}"
                )));
            }
            let params = extract_simple_packing_params(&desc.params)?;
            Ok(EncodingType::SimplePacking(params))
        }
        other => Err(TensogramError::Encoding(format!(
            "unknown encoding: {other}"
        ))),
    }
}

/// Resolve the filter type from a descriptor.
fn resolve_filter(desc: &DataObjectDescriptor) -> Result<FilterType> {
    match desc.filter.as_str() {
        "none" => Ok(FilterType::None),
        "shuffle" => {
            let element_size = usize::try_from(get_u64_param(
                &desc.params,
                "shuffle_element_size",
            )?)
            .map_err(|_| {
                TensogramError::Metadata("shuffle_element_size out of usize range".to_string())
            })?;
            Ok(FilterType::Shuffle { element_size })
        }
        other => Err(TensogramError::Encoding(format!("unknown filter: {other}"))),
    }
}

/// Resolve the compression type from a descriptor, using the resolved
/// encoding and filter for any codec that depends on them (szip bits_per_sample,
/// blosc2 typesize).
fn resolve_compression(
    desc: &DataObjectDescriptor,
    dtype: Dtype,
    encoding: &EncodingType,
    filter: &FilterType,
) -> Result<CompressionType> {
    match desc.compression.as_str() {
        "none" => Ok(CompressionType::None),
        #[cfg(any(feature = "szip", feature = "szip-pure"))]
        "szip" => {
            let rsi = u32::try_from(get_u64_param(&desc.params, "szip_rsi")?)
                .map_err(|_| TensogramError::Metadata("szip_rsi out of u32 range".to_string()))?;
            let block_size = u32::try_from(get_u64_param(&desc.params, "szip_block_size")?)
                .map_err(|_| {
                    TensogramError::Metadata("szip_block_size out of u32 range".to_string())
                })?;
            let flags = u32::try_from(get_u64_param(&desc.params, "szip_flags")?)
                .map_err(|_| TensogramError::Metadata("szip_flags out of u32 range".to_string()))?;
            let bits_per_sample = match (encoding, filter) {
                (EncodingType::SimplePacking(params), _) => params.bits_per_value,
                (EncodingType::None, FilterType::Shuffle { .. }) => 8,
                (EncodingType::None, FilterType::None) => (dtype.byte_width() * 8) as u32,
            };
            Ok(CompressionType::Szip {
                rsi,
                block_size,
                flags,
                bits_per_sample,
            })
        }
        #[cfg(any(feature = "zstd", feature = "zstd-pure"))]
        "zstd" => {
            let level_i64 = get_i64_param(&desc.params, "zstd_level").unwrap_or(3);
            let level = i32::try_from(level_i64).map_err(|_| {
                TensogramError::Metadata(format!("zstd_level value {level_i64} out of i32 range"))
            })?;
            Ok(CompressionType::Zstd { level })
        }
        #[cfg(feature = "lz4")]
        "lz4" => Ok(CompressionType::Lz4),
        #[cfg(feature = "blosc2")]
        "blosc2" => {
            let codec_str = match desc.params.get("blosc2_codec") {
                Some(ciborium::Value::Text(s)) => s.as_str(),
                _ => "lz4",
            };
            let codec = match codec_str {
                "blosclz" => Blosc2Codec::Blosclz,
                "lz4" => Blosc2Codec::Lz4,
                "lz4hc" => Blosc2Codec::Lz4hc,
                "zlib" => Blosc2Codec::Zlib,
                "zstd" => Blosc2Codec::Zstd,
                other => {
                    return Err(TensogramError::Encoding(format!(
                        "unknown blosc2 codec: {other}"
                    )));
                }
            };
            let clevel_i64 = get_i64_param(&desc.params, "blosc2_clevel").unwrap_or(5);
            let clevel = i32::try_from(clevel_i64).map_err(|_| {
                TensogramError::Metadata(format!(
                    "blosc2_clevel value {clevel_i64} out of i32 range"
                ))
            })?;
            let typesize = match (encoding, filter) {
                (EncodingType::SimplePacking(params), _) => {
                    (params.bits_per_value as usize).div_ceil(8)
                }
                (EncodingType::None, FilterType::Shuffle { .. }) => 1,
                (EncodingType::None, FilterType::None) => dtype.byte_width(),
            };
            Ok(CompressionType::Blosc2 {
                codec,
                clevel,
                typesize,
            })
        }
        #[cfg(feature = "zfp")]
        "zfp" => {
            let mode_str = match desc.params.get("zfp_mode") {
                Some(ciborium::Value::Text(s)) => s.clone(),
                _ => {
                    return Err(TensogramError::Metadata(
                        "missing required parameter: zfp_mode".to_string(),
                    ));
                }
            };
            let mode = match mode_str.as_str() {
                "fixed_rate" => {
                    let rate = get_f64_param(&desc.params, "zfp_rate")?;
                    ZfpMode::FixedRate { rate }
                }
                "fixed_precision" => {
                    let precision = u32::try_from(get_u64_param(&desc.params, "zfp_precision")?)
                        .map_err(|_| {
                            TensogramError::Metadata("zfp_precision out of u32 range".to_string())
                        })?;
                    ZfpMode::FixedPrecision { precision }
                }
                "fixed_accuracy" => {
                    let tolerance = get_f64_param(&desc.params, "zfp_tolerance")?;
                    ZfpMode::FixedAccuracy { tolerance }
                }
                other => {
                    return Err(TensogramError::Encoding(format!(
                        "unknown zfp_mode: {other}"
                    )));
                }
            };
            Ok(CompressionType::Zfp { mode })
        }
        #[cfg(feature = "sz3")]
        "sz3" => {
            let mode_str = match desc.params.get("sz3_error_bound_mode") {
                Some(ciborium::Value::Text(s)) => s.clone(),
                _ => {
                    return Err(TensogramError::Metadata(
                        "missing required parameter: sz3_error_bound_mode".to_string(),
                    ));
                }
            };
            let bound_val = get_f64_param(&desc.params, "sz3_error_bound")?;
            let error_bound = match mode_str.as_str() {
                "abs" => Sz3ErrorBound::Absolute(bound_val),
                "rel" => Sz3ErrorBound::Relative(bound_val),
                "psnr" => Sz3ErrorBound::Psnr(bound_val),
                other => {
                    return Err(TensogramError::Encoding(format!(
                        "unknown sz3_error_bound_mode: {other}"
                    )));
                }
            };
            Ok(CompressionType::Sz3 { error_bound })
        }
        "rle" => {
            // Bitmask-only codec — see `plans/WIRE_FORMAT.md` §8.
            if dtype != Dtype::Bitmask {
                return Err(TensogramError::Encoding(format!(
                    "compression \"rle\" only supports dtype=bitmask, got dtype={:?}",
                    dtype
                )));
            }
            Ok(CompressionType::Rle)
        }
        "roaring" => {
            // Bitmask-only codec — see `plans/WIRE_FORMAT.md` §8.
            if dtype != Dtype::Bitmask {
                return Err(TensogramError::Encoding(format!(
                    "compression \"roaring\" only supports dtype=bitmask, got dtype={:?}",
                    dtype
                )));
            }
            Ok(CompressionType::Roaring)
        }
        other => Err(TensogramError::Encoding(format!(
            "unknown compression: {other}"
        ))),
    }
}

/// Build a pipeline config with an explicit compression backend override
/// and an intra-codec thread budget.
///
/// `intra_codec_threads == 0` preserves the pre-threads behaviour and is
/// what direct pipeline callers (benchmarks, external code) should use.
pub(crate) fn build_pipeline_config_with_backend(
    desc: &DataObjectDescriptor,
    num_values: usize,
    dtype: Dtype,
    compression_backend: pipeline::CompressionBackend,
    intra_codec_threads: u32,
) -> Result<PipelineConfig> {
    let encoding = resolve_encoding(desc, dtype)?;
    let filter = resolve_filter(desc)?;
    let compression = resolve_compression(desc, dtype, &encoding, &filter)?;

    Ok(PipelineConfig {
        encoding,
        filter,
        compression,
        num_values,
        byte_order: desc.byte_order,
        dtype_byte_width: dtype.byte_width(),
        swap_unit_size: dtype.swap_unit_size(),
        compression_backend,
        intra_codec_threads,
        // `compute_hash` is not carried in the descriptor — the caller
        // (encode_one_object / streaming) flips it on when a hash is
        // requested.  Default off so direct pipeline callers pay nothing
        // for hashing unless they opt in.
        compute_hash: false,
    })
}

fn extract_simple_packing_params(
    params: &BTreeMap<String, ciborium::Value>,
) -> Result<SimplePackingParams> {
    let reference_value = get_f64_param(params, "sp_reference_value")?;
    if reference_value.is_nan() || reference_value.is_infinite() {
        return Err(TensogramError::Metadata(format!(
            "sp_reference_value must be finite, got {reference_value}"
        )));
    }
    Ok(SimplePackingParams {
        reference_value,
        binary_scale_factor: i32::try_from(get_i64_param(params, "sp_binary_scale_factor")?)
            .map_err(|_| {
                TensogramError::Metadata("sp_binary_scale_factor out of i32 range".to_string())
            })?,
        decimal_scale_factor: i32::try_from(get_i64_param(params, "sp_decimal_scale_factor")?)
            .map_err(|_| {
                TensogramError::Metadata("sp_decimal_scale_factor out of i32 range".to_string())
            })?,
        bits_per_value: u32::try_from(get_u64_param(params, "sp_bits_per_value")?).map_err(
            |_| TensogramError::Metadata("sp_bits_per_value out of u32 range".to_string()),
        )?,
    })
}

/// Auto-compute the reference / binary-scale params for a
/// `simple_packing` descriptor when they are absent.
///
/// When a caller writes (in any language):
///
/// ```text
/// desc = { encoding: "simple_packing", sp_bits_per_value: 16, ... }
/// ```
///
/// the four-key explicit form
/// (`sp_reference_value` + `sp_binary_scale_factor` + the two knob
/// keys) is derived from the input data on the fly.  The descriptor
/// is then stamped with all four so that the wire-format representation
/// stays self-describing.
///
/// Contract:
/// * No-op when `encoding != "simple_packing"`.
/// * No-op when both `sp_reference_value` and `sp_binary_scale_factor`
///   are already present — explicit user values win and are not
///   recomputed.  This supports advanced workflows that pin the
///   reference value across many objects (e.g. for time-series delta
///   encoding downstream).
/// * `sp_bits_per_value` is required on both the auto-compute and
///   explicit-params paths — error otherwise.
/// * `sp_decimal_scale_factor` defaults to `0` when absent.
/// * The data bytes are interpreted as float64 in the descriptor's
///   declared byte order — simple_packing is strictly `Dtype::Float64`
///   (other 8-byte dtypes such as `Int64`, `Uint64`, `Complex64` are
///   rejected up-front to avoid silent reinterpretation).  The pipeline
///   builder re-checks the same constraint for callers that bypass
///   this resolver.
pub(crate) fn resolve_simple_packing_params(
    desc: &mut DataObjectDescriptor,
    data_bytes: &[u8],
) -> Result<()> {
    if desc.encoding != "simple_packing" {
        return Ok(());
    }

    // simple_packing only supports float64.  Other 8-byte dtypes
    // (Int64 / Uint64 / Complex64) would pass a byte-width check but
    // re-interpreting their bytes as f64 produces garbage params.
    // Tighten to exact equality with `Dtype::Float64`.
    if desc.dtype != Dtype::Float64 {
        return Err(TensogramError::Encoding(format!(
            "simple_packing only supports float64 dtype; got {:?}",
            desc.dtype
        )));
    }

    // sp_bits_per_value is required regardless of which path we take —
    // the explicit 4-key form needs it for the bit-packing layout, and
    // the auto-compute form needs it for `compute_params`.  Check it
    // here so the error is consistent and points at the canonical
    // missing key, rather than failing later in the pipeline-config
    // builder with a less specific message.
    if !desc.params.contains_key("sp_bits_per_value") {
        return Err(TensogramError::Metadata(
            "simple_packing requires sp_bits_per_value (the encoder can \
             auto-compute sp_reference_value + sp_binary_scale_factor from \
             the data, but the bit-width and decimal scale are the user \
             knobs).  Provide at least sp_bits_per_value, or the full \
             explicit 4-key set."
                .to_string(),
        ));
    }

    // The two derived keys are an all-or-nothing pair.  Providing only
    // one would either silently get overwritten by auto-compute (if we
    // ran it) or produce a meaningless mix of user-supplied + derived
    // values.  Detecting this here gives the user a clear error
    // before any encoding work happens.
    let has_ref = desc.params.contains_key("sp_reference_value");
    let has_bsf = desc.params.contains_key("sp_binary_scale_factor");
    if has_ref ^ has_bsf {
        let (set, missing) = if has_ref {
            ("sp_reference_value", "sp_binary_scale_factor")
        } else {
            ("sp_binary_scale_factor", "sp_reference_value")
        };
        return Err(TensogramError::Metadata(format!(
            "simple_packing: descriptor sets {set} but not {missing}.  \
             Provide both for explicit-params encoding, or neither to \
             let the encoder auto-compute them from the data."
        )));
    }

    // Explicit computed params present — trust them, skip the
    // auto-compute work entirely.  We still default the
    // sp_decimal_scale_factor knob when absent so the pipeline's
    // extract_simple_packing_params doesn't fault on a missing key.
    if has_ref && has_bsf {
        desc.params
            .entry("sp_decimal_scale_factor".to_string())
            .or_insert(ciborium::Value::Integer(0i64.into()));
        return Ok(());
    }

    let bits_per_value = u32::try_from(get_u64_param(&desc.params, "sp_bits_per_value")?)
        .map_err(|_| TensogramError::Metadata("sp_bits_per_value out of u32 range".to_string()))?;
    let decimal_scale_factor = if desc.params.contains_key("sp_decimal_scale_factor") {
        i32::try_from(get_i64_param(&desc.params, "sp_decimal_scale_factor")?).map_err(|_| {
            TensogramError::Metadata("sp_decimal_scale_factor out of i32 range".to_string())
        })?
    } else {
        0
    };

    let values = bytes_as_f64_vec(data_bytes, desc.byte_order)?;
    let params = simple_packing::compute_params(&values, bits_per_value, decimal_scale_factor)
        .map_err(|e| TensogramError::Encoding(e.to_string()))?;

    desc.params.insert(
        "sp_reference_value".to_string(),
        ciborium::Value::Float(params.reference_value),
    );
    desc.params.insert(
        "sp_binary_scale_factor".to_string(),
        ciborium::Value::Integer(i64::from(params.binary_scale_factor).into()),
    );
    desc.params.insert(
        "sp_decimal_scale_factor".to_string(),
        ciborium::Value::Integer(i64::from(params.decimal_scale_factor).into()),
    );
    desc.params.insert(
        "sp_bits_per_value".to_string(),
        ciborium::Value::Integer(i64::from(params.bits_per_value).into()),
    );
    Ok(())
}

/// Reinterpret raw bytes as float64 honouring the descriptor's
/// byte order.  Used by the simple_packing auto-compute path.
///
/// Uses fallible `try_reserve_exact` rather than `collect()` so that
/// allocation failure on very large inputs surfaces as a structured
/// `TensogramError` instead of aborting the process — matching the
/// pattern in `tensogram_encodings::pipeline::bytes_to_f64`.
fn bytes_as_f64_vec(bytes: &[u8], byte_order: ByteOrder) -> Result<Vec<f64>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(TensogramError::Metadata(format!(
            "simple_packing: input byte length {} is not a multiple of 8 (float64)",
            bytes.len()
        )));
    }
    let n = bytes.len() / 8;
    let mut out: Vec<f64> = Vec::new();
    out.try_reserve_exact(n).map_err(|e| {
        TensogramError::Encoding(format!(
            "simple_packing: failed to reserve {} bytes for byte-to-f64 \
             conversion: {e}",
            n.saturating_mul(std::mem::size_of::<f64>()),
        ))
    })?;
    for chunk in bytes.chunks_exact(8) {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(chunk);
        out.push(match byte_order {
            ByteOrder::Big => f64::from_be_bytes(buf),
            ByteOrder::Little => f64::from_le_bytes(buf),
        });
    }
    Ok(out)
}

pub(crate) fn get_f64_param(params: &BTreeMap<String, ciborium::Value>, key: &str) -> Result<f64> {
    match params.get(key) {
        Some(ciborium::Value::Float(f)) => Ok(*f),
        Some(ciborium::Value::Integer(i)) => {
            // i128 → f64 may lose precision for very large integers (> 2^53),
            // but this is acceptable for a float accessor on an integer value.
            let n: i128 = (*i).into();
            Ok(n as f64)
        }
        Some(other) => Err(TensogramError::Metadata(format!(
            "expected number for {key}, got {other:?}"
        ))),
        None => Err(TensogramError::Metadata(format!(
            "missing required parameter: {key}"
        ))),
    }
}

pub(crate) fn get_i64_param(params: &BTreeMap<String, ciborium::Value>, key: &str) -> Result<i64> {
    match params.get(key) {
        Some(ciborium::Value::Integer(i)) => {
            let n: i128 = (*i).into();
            i64::try_from(n).map_err(|_| {
                TensogramError::Metadata(format!("integer value {n} out of i64 range for {key}"))
            })
        }
        Some(other) => Err(TensogramError::Metadata(format!(
            "expected integer for {key}, got {other:?}"
        ))),
        None => Err(TensogramError::Metadata(format!(
            "missing required parameter: {key}"
        ))),
    }
}

pub(crate) fn get_u64_param(params: &BTreeMap<String, ciborium::Value>, key: &str) -> Result<u64> {
    match params.get(key) {
        Some(ciborium::Value::Integer(i)) => {
            let n: i128 = (*i).into();
            u64::try_from(n).map_err(|_| {
                TensogramError::Metadata(format!("integer value {n} out of u64 range for {key}"))
            })
        }
        Some(other) => Err(TensogramError::Metadata(format!(
            "expected integer for {key}, got {other:?}"
        ))),
        None => Err(TensogramError::Metadata(format!(
            "missing required parameter: {key}"
        ))),
    }
}

pub(crate) fn validate_szip_block_offsets(
    params: &BTreeMap<String, ciborium::Value>,
    encoded_bytes_len: usize,
) -> Result<()> {
    let value = params.get("szip_block_offsets").ok_or_else(|| {
        TensogramError::Metadata(
            "missing required parameter: szip_block_offsets for szip compression".to_string(),
        )
    })?;

    let offsets = match value {
        ciborium::Value::Array(arr) => arr,
        other => {
            return Err(TensogramError::Metadata(format!(
                "szip_block_offsets must be an array, got {other:?}"
            )));
        }
    };

    if offsets.is_empty() {
        return Err(TensogramError::Metadata(
            "szip_block_offsets must not be empty; first offset must be 0".to_string(),
        ));
    }

    let bit_bound = encoded_bytes_len.checked_mul(8).ok_or_else(|| {
        TensogramError::Metadata(format!(
            "encoded byte length {encoded_bytes_len} overflows bit-bound calculation"
        ))
    })?;
    let bit_bound_u64 = u64::try_from(bit_bound).map_err(|_| {
        TensogramError::Metadata(format!(
            "bit-bound {bit_bound} derived from {encoded_bytes_len} bytes does not fit in u64"
        ))
    })?;

    let mut parsed_offsets = Vec::with_capacity(offsets.len());
    for (idx, item) in offsets.iter().enumerate() {
        let offset = match item {
            ciborium::Value::Integer(i) => {
                let n: i128 = (*i).into();
                u64::try_from(n).map_err(|_| {
                    TensogramError::Metadata(format!(
                        "szip_block_offsets[{idx}] = {n} out of u64 range"
                    ))
                })?
            }
            other => {
                return Err(TensogramError::Metadata(format!(
                    "szip_block_offsets[{idx}] must be an integer, got {other:?}"
                )));
            }
        };

        if offset > bit_bound_u64 {
            return Err(TensogramError::Metadata(format!(
                "szip_block_offsets[{idx}] = {offset} exceeds bit bound {bit_bound_u64} (encoded_bytes_len = {encoded_bytes_len} bytes, {bit_bound_u64} bits)"
            )));
        }

        if idx == 0 {
            if offset != 0 {
                return Err(TensogramError::Metadata(format!(
                    "szip_block_offsets[0] must be 0, got {offset}"
                )));
            }
        } else {
            let prev = parsed_offsets[idx - 1];
            if offset <= prev {
                return Err(TensogramError::Metadata(format!(
                    "szip_block_offsets must be strictly increasing: szip_block_offsets[{}] = {}, szip_block_offsets[{idx}] = {offset}",
                    idx - 1,
                    prev
                )));
            }
        }

        parsed_offsets.push(offset);
    }

    Ok(())
}

pub(crate) fn validate_no_szip_offsets_for_non_szip(desc: &DataObjectDescriptor) -> Result<()> {
    if desc.compression != "szip" && desc.params.contains_key("szip_block_offsets") {
        return Err(TensogramError::Metadata(format!(
            "szip_block_offsets provided but compression is '{}', not 'szip'",
            desc.compression
        )));
    }
    Ok(())
}

/// Compose the payload region for a data-object frame.
///
/// The layout emitted is
/// `[encoded_payload][mask_nan][mask_inf+][mask_inf-]`
/// where each mask section is present iff the corresponding
/// [`MaskSet`] field is `Some`.  The returned [`MasksMetadata`]
/// records each section's byte offset (relative to the start of the
/// region) and length.
///
/// When [`MaskSet::is_empty`], the returned region is the caller's
/// `encoded_payload` unchanged and the metadata is `None` — the
/// resulting frame is byte-identical to the legacy `NTensorFrame`
/// payload layout except for the frame-type number.
///
/// # Small-mask fallback
///
/// When a mask's uncompressed bit-packed byte count is
/// `≤ small_threshold` (default 128, configurable and set to `0` to
/// disable), the method is forced to [`MaskMethod::None`] regardless
/// of the caller's requested method.  The resulting
/// [`MaskDescriptor::method`] reflects what was actually written,
/// not the caller's request.
///
/// Takes the per-kind methods + threshold directly rather than an
/// [`EncodeOptions`] reference so `StreamingEncoder` can call it
/// from its field snapshot without borrowing a synthesised options
/// struct.
pub(crate) fn compose_payload_region(
    mut encoded_payload: Vec<u8>,
    masks: MaskSet,
    nan_method: &MaskMethod,
    pos_inf_method: &MaskMethod,
    neg_inf_method: &MaskMethod,
    small_threshold: usize,
) -> Result<(Vec<u8>, Option<MasksMetadata>)> {
    if masks.is_empty() {
        return Ok((encoded_payload, None));
    }

    let mut metadata = MasksMetadata::default();
    let mut region_cursor = encoded_payload.len() as u64;

    // Append each present mask to the payload region and record its
    // descriptor.  Canonical order matches the CBOR key sort —
    // nan < inf+ < inf- — so the mask region stays byte-stable
    // across identical inputs.
    let mut append_one =
        |bits_opt: Option<&Vec<bool>>, method: &MaskMethod| -> Result<Option<MaskDescriptor>> {
            let Some(bits) = bits_opt else {
                return Ok(None);
            };
            let (blob, used_method) = encode_one_mask(bits, method.clone(), small_threshold)?;
            let desc = MaskDescriptor {
                method: used_method.name().to_string(),
                offset: region_cursor,
                length: blob.len() as u64,
                params: mask_params_cbor(&used_method),
            };
            region_cursor += blob.len() as u64;
            encoded_payload.extend_from_slice(&blob);
            Ok(Some(desc))
        };
    metadata.nan = append_one(masks.nan.as_ref(), nan_method)?;
    metadata.pos_inf = append_one(masks.pos_inf.as_ref(), pos_inf_method)?;
    metadata.neg_inf = append_one(masks.neg_inf.as_ref(), neg_inf_method)?;

    Ok((encoded_payload, Some(metadata)))
}

/// Compress one mask using the caller's chosen method, with auto-
/// fallback to [`MaskMethod::None`] for small masks.  Returns the
/// serialised blob AND the method actually used (may differ from the
/// requested method due to the small-mask fallback — see
/// [`compose_payload_region`]).
fn encode_one_mask(
    bits: &[bool],
    requested: MaskMethod,
    small_threshold: usize,
) -> Result<(Vec<u8>, MaskMethod)> {
    use tensogram_encodings::bitmask;

    // Small-mask fallback: compare the raw bit-packed byte count
    // against the threshold.  When `small_threshold == 0` the
    // fallback is disabled and we always honour the requested method.
    let uncompressed_bytes = bits.len().div_ceil(8);
    let method = if small_threshold > 0 && uncompressed_bytes <= small_threshold {
        MaskMethod::None
    } else {
        requested
    };

    let blob = match &method {
        MaskMethod::None => bitmask::codecs::encode_none(bits)
            .map_err(|e| TensogramError::Encoding(format!("bitmask pack: {e}")))?,
        MaskMethod::Rle => bitmask::rle::encode(bits),
        MaskMethod::Roaring => bitmask::roaring::encode(bits)
            .map_err(|e| TensogramError::Encoding(format!("roaring mask encode: {e}")))?,
        MaskMethod::Lz4 => bitmask::codecs::encode_lz4(bits)
            .map_err(|e| TensogramError::Encoding(format!("lz4 mask encode: {e}")))?,
        MaskMethod::Zstd { level } => bitmask::codecs::encode_zstd(bits, *level)
            .map_err(|e| TensogramError::Encoding(format!("zstd mask encode: {e}")))?,
        #[cfg(feature = "blosc2")]
        MaskMethod::Blosc2 { codec, level } => bitmask::codecs::encode_blosc2(bits, *codec, *level)
            .map_err(|e| TensogramError::Encoding(format!("blosc2 mask encode: {e}")))?,
    };

    Ok((blob, method))
}

/// Build the `params` sub-map for a [`MaskDescriptor`] per
/// `plans/WIRE_FORMAT.md` §6.5.1.  Empty for the parameter-less
/// methods; populated for `zstd` / `blosc2`.
fn mask_params_cbor(method: &MaskMethod) -> BTreeMap<String, ciborium::Value> {
    let mut params = BTreeMap::new();
    match method {
        MaskMethod::None | MaskMethod::Rle | MaskMethod::Roaring | MaskMethod::Lz4 => {}
        MaskMethod::Zstd { level } => {
            if let Some(l) = level {
                params.insert(
                    "level".to_string(),
                    ciborium::Value::Integer((*l as i64).into()),
                );
            }
        }
        #[cfg(feature = "blosc2")]
        MaskMethod::Blosc2 { codec, level } => {
            let codec_str = match codec {
                Blosc2Codec::Blosclz => "blosclz",
                Blosc2Codec::Lz4 => "lz4",
                Blosc2Codec::Lz4hc => "lz4hc",
                Blosc2Codec::Zlib => "zlib",
                Blosc2Codec::Zstd => "zstd",
            };
            params.insert(
                "codec".to_string(),
                ciborium::Value::Text(codec_str.to_string()),
            );
            params.insert(
                "level".to_string(),
                ciborium::Value::Integer((*level as i64).into()),
            );
        }
    }
    params
}

// ── Edge case tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{DecodeOptions, decode};
    use crate::types::{ByteOrder, GlobalMetadata};
    use std::collections::BTreeMap;

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
            byte_order: ByteOrder::native(),
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            masks: None,
        }
    }

    // ── Category 1: base array mismatches ────────────────────────────────

    #[test]
    fn test_base_more_entries_than_descriptors_rejected() {
        // base has 5 entries but only 2 descriptors — should error.
        let meta = GlobalMetadata {
            base: vec![
                BTreeMap::new(),
                BTreeMap::new(),
                BTreeMap::new(),
                BTreeMap::new(),
                BTreeMap::new(),
            ],
            ..Default::default()
        };
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        let result = encode(
            &meta,
            &[(&desc, data.as_slice()), (&desc, data.as_slice())],
            &options,
        );
        assert!(
            result.is_err(),
            "5 base entries with 2 descriptors should fail"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("5") && err.contains("2"),
            "error should mention counts: {err}"
        );
    }

    #[test]
    fn test_base_fewer_entries_than_descriptors_auto_extended() {
        // base has 0 entries but 3 descriptors — auto-extends, _reserved_ inserted.
        let meta = GlobalMetadata {
            base: vec![],
            ..Default::default()
        };
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        let msg = encode(
            &meta,
            &[
                (&desc, data.as_slice()),
                (&desc, data.as_slice()),
                (&desc, data.as_slice()),
            ],
            &options,
        )
        .unwrap();

        let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();
        assert_eq!(decoded.base.len(), 3);
        // Each entry should have _reserved_ with tensor info
        for entry in &decoded.base {
            assert!(
                entry.contains_key("_reserved_"),
                "auto-extended base entry should have _reserved_"
            );
        }
    }

    #[test]
    fn test_base_entry_with_top_level_key_names_no_collision() {
        // base[0] contains a key named "version" — no collision with top-level version.
        let mut entry = BTreeMap::new();
        entry.insert(
            "version".to_string(),
            ciborium::Value::Text("my-version".to_string()),
        );
        entry.insert(
            "base".to_string(),
            ciborium::Value::Text("not-the-real-base".to_string()),
        );
        let meta = GlobalMetadata {
            base: vec![entry],
            ..Default::default()
        };
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        let msg = encode(&meta, &[(&desc, data.as_slice())], &options).unwrap();
        let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();

        // `version` and `base` are just free-form keys inside a
        // per-object `base[0]` entry — they have no special meaning
        // there.  The wire-format version lives in the preamble
        // (see `plans/WIRE_FORMAT.md` §3), not in CBOR metadata.
        assert_eq!(
            decoded.base[0].get("version"),
            Some(&ciborium::Value::Text("my-version".to_string()))
        );
        assert_eq!(
            decoded.base[0].get("base"),
            Some(&ciborium::Value::Text("not-the-real-base".to_string()))
        );
    }

    #[test]
    fn test_base_entry_with_deeply_nested_reserved_allowed() {
        // Only top-level _reserved_ in base[i] should be rejected.
        // Deeply nested _reserved_ (like {"foo": {"_reserved_": ...}}) is fine.
        let nested = ciborium::Value::Map(vec![(
            ciborium::Value::Text("_reserved_".to_string()),
            ciborium::Value::Text("nested-is-ok".to_string()),
        )]);
        let mut entry = BTreeMap::new();
        entry.insert("foo".to_string(), nested);
        let meta = GlobalMetadata {
            base: vec![entry],
            ..Default::default()
        };
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        // Should succeed — only top-level _reserved_ is rejected
        let msg = encode(&meta, &[(&desc, data.as_slice())], &options).unwrap();
        let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();
        // The nested _reserved_ should survive
        let foo = decoded.base[0].get("foo").unwrap();
        if let ciborium::Value::Map(pairs) = foo {
            assert_eq!(pairs.len(), 1);
        } else {
            panic!("expected map for foo");
        }
    }

    // ── Category 2: _reserved_ edge cases ────────────────────────────────

    #[test]
    fn test_reserved_rejected_at_message_level() {
        let mut reserved = BTreeMap::new();
        reserved.insert(
            "rogue".to_string(),
            ciborium::Value::Text("bad".to_string()),
        );
        let meta = GlobalMetadata {
            reserved,
            ..Default::default()
        };
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let result = encode(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("_reserved_") && err.contains("message level"),
            "error: {err}"
        );
    }

    #[test]
    fn test_reserved_rejected_in_base_entry() {
        let mut entry = BTreeMap::new();
        entry.insert("_reserved_".to_string(), ciborium::Value::Map(vec![]));
        let meta = GlobalMetadata {
            base: vec![entry],
            ..Default::default()
        };
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let result = encode(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("_reserved_") && err.contains("base[0]"),
            "error: {err}"
        );
    }

    #[test]
    fn test_reserved_tensor_has_four_keys_after_encode() {
        let meta = GlobalMetadata::default();
        let desc = make_descriptor(vec![3, 4]);
        let data = vec![0u8; 3 * 4 * 4]; // 3*4 float32
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        let msg = encode(&meta, &[(&desc, data.as_slice())], &options).unwrap();
        let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();

        let reserved = decoded.base[0]
            .get("_reserved_")
            .expect("_reserved_ missing");
        if let ciborium::Value::Map(pairs) = reserved {
            // Should have "tensor" key
            let tensor_entry = pairs
                .iter()
                .find(|(k, _)| *k == ciborium::Value::Text("tensor".to_string()));
            assert!(tensor_entry.is_some(), "missing tensor key in _reserved_");
            if let Some((_, ciborium::Value::Map(tensor_pairs))) = tensor_entry {
                let keys: Vec<String> = tensor_pairs
                    .iter()
                    .filter_map(|(k, _)| {
                        if let ciborium::Value::Text(s) = k {
                            Some(s.clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                assert_eq!(keys.len(), 4, "tensor should have 4 keys, got: {keys:?}");
                assert!(keys.contains(&"ndim".to_string()));
                assert!(keys.contains(&"shape".to_string()));
                assert!(keys.contains(&"strides".to_string()));
                assert!(keys.contains(&"dtype".to_string()));
            } else {
                panic!("tensor is not a map");
            }
        } else {
            panic!("_reserved_ is not a map");
        }
    }

    #[test]
    fn test_reserved_tensor_scalar_ndim_zero() {
        // Scalar: ndim=0, shape=[], strides=[]
        let desc = DataObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 0,
            shape: vec![],
            strides: vec![],
            dtype: Dtype::Float32,
            byte_order: ByteOrder::native(),
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            masks: None,
        };
        let data = vec![0u8; 4]; // 1 float32
        let meta = GlobalMetadata::default();
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        let msg = encode(&meta, &[(&desc, data.as_slice())], &options).unwrap();
        let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();

        let reserved = decoded.base[0]
            .get("_reserved_")
            .expect("_reserved_ missing");
        if let ciborium::Value::Map(pairs) = reserved {
            let tensor_entry = pairs
                .iter()
                .find(|(k, _)| *k == ciborium::Value::Text("tensor".to_string()));
            if let Some((_, ciborium::Value::Map(tensor_pairs))) = tensor_entry {
                // ndim should be 0
                let ndim = tensor_pairs
                    .iter()
                    .find(|(k, _)| *k == ciborium::Value::Text("ndim".to_string()));
                assert!(
                    matches!(ndim, Some((_, ciborium::Value::Integer(i))) if i128::from(*i) == 0),
                    "ndim should be 0 for scalar"
                );
                // shape should be empty array
                let shape = tensor_pairs
                    .iter()
                    .find(|(k, _)| *k == ciborium::Value::Text("shape".to_string()));
                assert!(
                    matches!(shape, Some((_, ciborium::Value::Array(a))) if a.is_empty()),
                    "shape should be [] for scalar"
                );
            } else {
                panic!("tensor missing or not a map");
            }
        } else {
            panic!("_reserved_ is not a map");
        }
    }

    // ── Category 3: _extra_ edge cases ───────────────────────────────────

    #[test]
    fn test_extra_with_keys_colliding_with_base_entry_keys() {
        // _extra_ has key "mars", base[0] also has key "mars" — different scopes, both survive
        let mut entry = BTreeMap::new();
        entry.insert(
            "mars".to_string(),
            ciborium::Value::Text("base-mars".to_string()),
        );
        let mut extra = BTreeMap::new();
        extra.insert(
            "mars".to_string(),
            ciborium::Value::Text("extra-mars".to_string()),
        );
        let meta = GlobalMetadata {
            base: vec![entry],
            extra,
            ..Default::default()
        };
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        let msg = encode(&meta, &[(&desc, data.as_slice())], &options).unwrap();
        let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();

        assert_eq!(
            decoded.base[0].get("mars"),
            Some(&ciborium::Value::Text("base-mars".to_string()))
        );
        assert_eq!(
            decoded.extra.get("mars"),
            Some(&ciborium::Value::Text("extra-mars".to_string()))
        );
    }

    #[test]
    fn test_empty_extra_omitted_from_cbor() {
        let meta = GlobalMetadata {
            extra: BTreeMap::new(),
            ..Default::default()
        };
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        let msg = encode(&meta, &[(&desc, data.as_slice())], &options).unwrap();
        let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();
        assert!(decoded.extra.is_empty());
    }

    #[test]
    fn test_extra_with_nested_maps_round_trips() {
        let nested = ciborium::Value::Map(vec![
            (
                ciborium::Value::Text("key1".to_string()),
                ciborium::Value::Integer(42.into()),
            ),
            (
                ciborium::Value::Text("key2".to_string()),
                ciborium::Value::Map(vec![(
                    ciborium::Value::Text("deep".to_string()),
                    ciborium::Value::Bool(true),
                )]),
            ),
        ]);
        let mut extra = BTreeMap::new();
        extra.insert("nested".to_string(), nested.clone());
        let meta = GlobalMetadata {
            extra,
            ..Default::default()
        };
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        let msg = encode(&meta, &[(&desc, data.as_slice())], &options).unwrap();
        let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();
        // Nested maps should round-trip
        assert!(decoded.extra.contains_key("nested"));
    }

    // ── Category 4: Serde deserialization ────────────────────────────────

    #[test]
    fn test_legacy_top_level_keys_routed_to_extra() {
        // Simulate a legacy v2-style message carrying `common` / `payload`
        // and a stray `version` top-level key.  Under the free-form rule
        // (see `plans/WIRE_FORMAT.md` §6.1), these unknown keys must flow
        // into `_extra_` rather than being silently dropped — the wire
        // version lives exclusively in the preamble (see [`crate::wire`]).
        use ciborium::Value;
        let cbor = Value::Map(vec![
            (Value::Text("common".to_string()), Value::Map(vec![])),
            (Value::Text("payload".to_string()), Value::Array(vec![])),
            (Value::Text("version".to_string()), Value::Integer(3.into())),
        ]);
        let mut bytes = Vec::new();
        ciborium::into_writer(&cbor, &mut bytes).unwrap();

        let decoded: GlobalMetadata = crate::metadata::cbor_to_global_metadata(&bytes).unwrap();
        assert!(decoded.base.is_empty());
        assert!(decoded.reserved.is_empty());
        assert!(decoded.extra.contains_key("common"));
        assert!(decoded.extra.contains_key("payload"));
        assert_eq!(
            decoded.extra.get("version"),
            Some(&Value::Integer(3.into()))
        );
    }

    #[test]
    fn test_old_reserved_key_name_routed_to_extra() {
        // "reserved" (unescaped, old v1 name) is NOT the library-managed
        // namespace — only the exact key `_reserved_` is.  Under the
        // free-form rule, `reserved` is just another top-level key and
        // flows into `_extra_` on decode.
        use ciborium::Value;
        let cbor = Value::Map(vec![(
            Value::Text("reserved".to_string()),
            Value::Map(vec![(
                Value::Text("rogue".to_string()),
                Value::Text("value".to_string()),
            )]),
        )]);
        let mut bytes = Vec::new();
        ciborium::into_writer(&cbor, &mut bytes).unwrap();

        let decoded: GlobalMetadata = crate::metadata::cbor_to_global_metadata(&bytes).unwrap();
        assert!(
            decoded.reserved.is_empty(),
            "legacy 'reserved' must NOT bleed into library-managed `_reserved_`"
        );
        assert!(
            decoded.extra.contains_key("reserved"),
            "legacy 'reserved' key must land in `_extra_`"
        );
    }

    // ── Category 4b: validate_no_client_reserved — multi-entry base ────

    #[test]
    fn test_reserved_rejected_in_second_base_entry_only() {
        // base[0] is clean, base[1] has _reserved_ → should fail, mentioning base[1]
        let mut entry0 = BTreeMap::new();
        entry0.insert("clean".to_string(), ciborium::Value::Text("ok".to_string()));
        let mut entry1 = BTreeMap::new();
        entry1.insert("_reserved_".to_string(), ciborium::Value::Map(vec![]));
        let meta = GlobalMetadata {
            base: vec![entry0, entry1],
            ..Default::default()
        };
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let result = encode(
            &meta,
            &[(&desc, data.as_slice()), (&desc, data.as_slice())],
            &EncodeOptions::default(),
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("base[1]"),
            "error should mention base[1]: {err}"
        );
    }

    #[test]
    fn test_reserved_accepted_when_all_base_entries_clean() {
        // Multiple base entries, none have _reserved_ → should succeed
        let mut e0 = BTreeMap::new();
        e0.insert(
            "key0".to_string(),
            ciborium::Value::Text("val0".to_string()),
        );
        let mut e1 = BTreeMap::new();
        e1.insert(
            "key1".to_string(),
            ciborium::Value::Text("val1".to_string()),
        );
        let meta = GlobalMetadata {
            base: vec![e0, e1],
            ..Default::default()
        };
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        let msg = encode(
            &meta,
            &[(&desc, data.as_slice()), (&desc, data.as_slice())],
            &options,
        )
        .unwrap();
        let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();
        assert_eq!(decoded.base.len(), 2);
        assert!(decoded.base[0].contains_key("key0"));
        assert!(decoded.base[1].contains_key("key1"));
    }

    // ── Category 5: populate_base_entries — all dtypes ───────────────────

    #[test]
    fn test_reserved_tensor_dtype_strings_for_all_dtypes() {
        // Verify that _reserved_.tensor.dtype string is correct for every Dtype variant
        let dtypes_and_expected = [
            (Dtype::Float16, "float16"),
            (Dtype::Bfloat16, "bfloat16"),
            (Dtype::Float32, "float32"),
            (Dtype::Float64, "float64"),
            (Dtype::Complex64, "complex64"),
            (Dtype::Complex128, "complex128"),
            (Dtype::Int8, "int8"),
            (Dtype::Int16, "int16"),
            (Dtype::Int32, "int32"),
            (Dtype::Int64, "int64"),
            (Dtype::Uint8, "uint8"),
            (Dtype::Uint16, "uint16"),
            (Dtype::Uint32, "uint32"),
            (Dtype::Uint64, "uint64"),
        ];

        for (dtype, expected_str) in dtypes_and_expected {
            let byte_width = dtype.byte_width();
            let num_elements: u64 = 4;
            let data_len = num_elements as usize * byte_width;

            let desc = DataObjectDescriptor {
                obj_type: "ntensor".to_string(),
                ndim: 1,
                shape: vec![num_elements],
                strides: vec![1],
                dtype,
                byte_order: ByteOrder::native(),
                encoding: "none".to_string(),
                filter: "none".to_string(),
                compression: "none".to_string(),
                params: BTreeMap::new(),
                masks: None,
            };
            let data = vec![0u8; data_len];
            let meta = GlobalMetadata::default();
            let options = EncodeOptions {
                hash_algorithm: None,
                ..Default::default()
            };
            let msg = encode(&meta, &[(&desc, data.as_slice())], &options).unwrap();
            let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();

            let reserved = decoded.base[0]
                .get("_reserved_")
                .unwrap_or_else(|| panic!("_reserved_ missing for dtype {dtype}"));
            if let ciborium::Value::Map(pairs) = reserved {
                let tensor_entry = pairs
                    .iter()
                    .find(|(k, _)| *k == ciborium::Value::Text("tensor".to_string()));
                if let Some((_, ciborium::Value::Map(tensor_pairs))) = tensor_entry {
                    let dtype_val = tensor_pairs
                        .iter()
                        .find(|(k, _)| *k == ciborium::Value::Text("dtype".to_string()));
                    assert!(
                        matches!(
                            dtype_val,
                            Some((_, ciborium::Value::Text(s))) if s == expected_str
                        ),
                        "dtype for {dtype} should be '{expected_str}', got: {dtype_val:?}"
                    );
                } else {
                    panic!("tensor missing or not a map for dtype {dtype}");
                }
            } else {
                panic!("_reserved_ is not a map for dtype {dtype}");
            }
        }
    }

    // ── Category 6: GlobalMetadata serde with all fields ─────────────────

    #[test]
    fn test_global_metadata_serde_all_fields_populated() {
        // base + reserved + extra all populated — verify CBOR round-trip
        use ciborium::Value;

        let mut base_entry = BTreeMap::new();
        base_entry.insert("key".to_string(), Value::Text("base_val".to_string()));
        let mut reserved = BTreeMap::new();
        reserved.insert("encoder".to_string(), Value::Text("test".to_string()));
        let mut extra = BTreeMap::new();
        extra.insert("custom".to_string(), Value::Integer(42.into()));

        let meta = GlobalMetadata {
            base: vec![base_entry],
            reserved,
            extra,
        };

        // Serialize to CBOR and back
        let cbor_bytes = crate::metadata::global_metadata_to_cbor(&meta).unwrap();
        let decoded: GlobalMetadata =
            crate::metadata::cbor_to_global_metadata(&cbor_bytes).unwrap();
        assert_eq!(decoded.base.len(), 1);
        assert_eq!(
            decoded.base[0].get("key"),
            Some(&Value::Text("base_val".to_string()))
        );
        assert!(decoded.reserved.contains_key("encoder"));
        assert_eq!(
            decoded.extra.get("custom"),
            Some(&Value::Integer(42.into()))
        );
    }

    // ── Category 7: populate_reserved_provenance ─────────────────────────

    #[test]
    fn test_provenance_fields_present_after_encode() {
        let meta = GlobalMetadata::default();
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };
        let msg = encode(&meta, &[(&desc, data.as_slice())], &options).unwrap();
        let (decoded, _) = decode(&msg, &DecodeOptions::default()).unwrap();

        // Message-level reserved should have encoder, time, uuid
        assert!(decoded.reserved.contains_key("encoder"));
        assert!(decoded.reserved.contains_key("time"));
        assert!(decoded.reserved.contains_key("uuid"));

        // encoder should contain name and version
        if let ciborium::Value::Map(pairs) = decoded.reserved.get("encoder").unwrap() {
            let has_name = pairs
                .iter()
                .any(|(k, _)| *k == ciborium::Value::Text("name".to_string()));
            let has_version = pairs
                .iter()
                .any(|(k, _)| *k == ciborium::Value::Text("version".to_string()));
            assert!(has_name, "encoder map should have 'name' key");
            assert!(has_version, "encoder map should have 'version' key");
        } else {
            panic!("encoder should be a map");
        }

        // uuid should be a valid UUID string (36 chars with hyphens)
        if let ciborium::Value::Text(uuid_str) = decoded.reserved.get("uuid").unwrap() {
            assert_eq!(uuid_str.len(), 36, "UUID should be 36 chars: {uuid_str}");
            assert_eq!(
                uuid_str.chars().filter(|c| *c == '-').count(),
                4,
                "UUID should have 4 hyphens: {uuid_str}"
            );
        } else {
            panic!("uuid should be a text");
        }

        // time should be an ISO 8601 timestamp ending with Z
        if let ciborium::Value::Text(time_str) = decoded.reserved.get("time").unwrap() {
            assert!(
                time_str.ends_with('Z'),
                "time should end with Z: {time_str}"
            );
            assert!(
                time_str.contains('T'),
                "time should contain T separator: {time_str}"
            );
        } else {
            panic!("time should be a text");
        }
    }

    #[test]
    fn test_both_reserved_and_reserved_underscore_only_new_captured() {
        // Both "reserved" and "_reserved_" present — only "_reserved_" should be captured.
        use ciborium::Value;
        let cbor = Value::Map(vec![
            (
                Value::Text("_reserved_".to_string()),
                Value::Map(vec![(
                    Value::Text("encoder".to_string()),
                    Value::Text("tensogram".to_string()),
                )]),
            ),
            (
                Value::Text("reserved".to_string()),
                Value::Map(vec![(
                    Value::Text("old".to_string()),
                    Value::Text("ignored".to_string()),
                )]),
            ),
            (Value::Text("version".to_string()), Value::Integer(3.into())),
        ]);
        let mut bytes = Vec::new();
        ciborium::into_writer(&cbor, &mut bytes).unwrap();

        let decoded: GlobalMetadata = crate::metadata::cbor_to_global_metadata(&bytes).unwrap();
        assert!(decoded.reserved.contains_key("encoder"));
        assert!(!decoded.reserved.contains_key("old"));
    }

    // ── Category 8: encode_pre_encoded smoke tests ───────────────────────

    /// Roundtrip: encode raw bytes via encode(), then re-encode the exact same
    /// payload bytes via encode_pre_encoded(). Both decoded payloads must be
    /// byte-identical. We compare payload bytes, NOT raw wire messages (provenance
    /// UUIDs make raw message equality impossible).
    #[test]
    fn test_encode_pre_encoded_roundtrip_simple_packing() {
        // Use encoding="none" (raw float32) for maximum portability — no feature flags needed.
        let desc = make_descriptor(vec![4]);
        let raw_data: Vec<u8> = vec![0u8; 4 * 4]; // 4 float32 values, all-zero

        let meta = GlobalMetadata::default();
        let options = EncodeOptions::default();

        // Step 1: encode normally
        let msg1 = encode(&meta, &[(&desc, raw_data.as_slice())], &options).unwrap();

        // Step 2: decode to get the encoded payload bytes + descriptor
        let (_, objects1) = decode(&msg1, &DecodeOptions::default()).unwrap();
        let (decoded_desc1, decoded_payload1) = &objects1[0];

        // Step 3: re-encode the same bytes via encode_pre_encoded
        let msg2 = encode_pre_encoded(
            &meta,
            &[(&decoded_desc1.clone(), decoded_payload1.as_slice())],
            &options,
        )
        .unwrap();

        // Step 4: decode the second message
        let (_, objects2) = decode(&msg2, &DecodeOptions::default()).unwrap();
        let (_, decoded_payload2) = &objects2[0];

        // Payloads must be identical — same bytes, same encoding
        // (raw wire messages differ due to non-deterministic provenance UUIDs)
        assert_eq!(
            decoded_payload1, decoded_payload2,
            "decoded payloads should be equal after encode/re-encode roundtrip"
        );
    }

    /// emit_preceders=true must be rejected by encode_pre_encoded (buffered mode).
    #[test]
    fn test_encode_pre_encoded_rejects_emit_preceders() {
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        let meta = GlobalMetadata::default();
        let options = EncodeOptions {
            emit_preceders: true,
            ..Default::default()
        };
        let result = encode_pre_encoded(&meta, &[(&desc, data.as_slice())], &options);
        assert!(
            result.is_err(),
            "encode_pre_encoded with emit_preceders=true should fail"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("emit_preceders"),
            "error should mention emit_preceders: {err}"
        );
    }

    /// `encode_pre_encoded` populates each data-object frame's
    /// inline hash slot with the xxh3-64 of the frame body when
    /// `EncodeOptions.hash_algorithm` is `Some(Xxh3)` — same
    /// contract as `encode`.  v3 equivalent of the pre-v3
    /// "library overwrites caller-supplied descriptor hash" test
    /// (caller-supplied hashes are structurally impossible in v3
    /// because `DataObjectDescriptor.hash` is gone).
    #[test]
    fn test_encode_pre_encoded_populates_inline_hash_slot() {
        use crate::framing::{decode_message, scan};
        use crate::hash::verify_frame_hash;
        use crate::wire::{FrameHeader, MessageFlags, Preamble};

        let desc = make_descriptor(vec![2]);
        let data = vec![0xABu8; 8];
        let meta = GlobalMetadata::default();
        let options = EncodeOptions::default();

        let msg = encode_pre_encoded(&meta, &[(&desc, data.as_slice())], &options).unwrap();

        // Preamble HASHES_PRESENT must be set.
        let preamble = Preamble::read_from(&msg).unwrap();
        assert!(preamble.flags.has(MessageFlags::HASHES_PRESENT));

        // Every data-object frame's inline slot verifies.
        let messages = scan(&msg);
        assert_eq!(messages.len(), 1);
        let (offset, len) = messages[0];
        let only_msg = &msg[offset..offset + len];
        let decoded = decode_message(only_msg).unwrap();
        for (_, _, _, frame_offset) in &decoded.objects {
            let frame = &only_msg[*frame_offset..];
            let fh = FrameHeader::read_from(frame).unwrap();
            let frame_bytes = &frame[..fh.total_length as usize];
            verify_frame_hash(frame_bytes, fh.frame_type)
                .expect("inline hash slot must verify against body");
        }
    }

    #[test]
    fn test_validate_szip_block_offsets_happy_path() {
        let mut params = BTreeMap::new();
        params.insert(
            "szip_block_offsets".to_string(),
            ciborium::Value::Array(vec![0u64, 100, 200].into_iter().map(|n| n.into()).collect()),
        );

        assert!(validate_szip_block_offsets(&params, 100).is_ok());
    }

    #[test]
    fn test_validate_szip_block_offsets_missing_key() {
        let params = BTreeMap::new();

        let err = validate_szip_block_offsets(&params, 100)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("missing") && err.contains("szip_block_offsets"),
            "error: {err}"
        );
    }

    #[test]
    fn test_validate_szip_block_offsets_not_array() {
        let mut params = BTreeMap::new();
        params.insert(
            "szip_block_offsets".to_string(),
            ciborium::Value::Integer(0.into()),
        );

        let err = validate_szip_block_offsets(&params, 100)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("array") && err.contains("szip_block_offsets"),
            "error: {err}"
        );
    }

    #[test]
    fn test_validate_szip_block_offsets_non_integer_element() {
        let mut params = BTreeMap::new();
        params.insert(
            "szip_block_offsets".to_string(),
            ciborium::Value::Array(vec![
                ciborium::Value::Integer(0.into()),
                ciborium::Value::Text("x".to_string()),
            ]),
        );

        let err = validate_szip_block_offsets(&params, 100)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("integer") && err.contains("szip_block_offsets"),
            "error: {err}"
        );
    }

    #[test]
    fn test_validate_szip_block_offsets_nonzero_first() {
        let mut params = BTreeMap::new();
        params.insert(
            "szip_block_offsets".to_string(),
            ciborium::Value::Array(vec![5u64, 100, 200].into_iter().map(|n| n.into()).collect()),
        );

        let err = validate_szip_block_offsets(&params, 100)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("must be 0") && err.contains("got 5"),
            "error: {err}"
        );
    }

    #[test]
    fn test_validate_szip_block_offsets_non_monotonic() {
        let mut params = BTreeMap::new();
        params.insert(
            "szip_block_offsets".to_string(),
            ciborium::Value::Array(vec![0u64, 100, 50].into_iter().map(|n| n.into()).collect()),
        );

        let err = validate_szip_block_offsets(&params, 100)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("increasing") || err.contains("monotonic"),
            "error: {err}"
        );
    }

    #[test]
    fn test_validate_szip_block_offsets_offset_beyond_bound() {
        let mut params = BTreeMap::new();
        params.insert(
            "szip_block_offsets".to_string(),
            ciborium::Value::Array(vec![0u64, 100, 801].into_iter().map(|n| n.into()).collect()),
        );

        let err = validate_szip_block_offsets(&params, 100)
            .unwrap_err()
            .to_string();
        assert!(err.contains("800") && err.contains("801"), "error: {err}");
    }

    #[test]
    fn test_validate_no_szip_offsets_for_non_szip_rejects() {
        let mut params = BTreeMap::new();
        params.insert(
            "szip_block_offsets".to_string(),
            ciborium::Value::Array(vec![0u64, 1].into_iter().map(|n| n.into()).collect()),
        );
        let desc = DataObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 1,
            shape: vec![2],
            strides: vec![1],
            dtype: Dtype::Float32,
            byte_order: ByteOrder::native(),
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "zstd".to_string(),
            params,
            masks: None,
        };

        let err = validate_no_szip_offsets_for_non_szip(&desc)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("szip_block_offsets") && err.contains("zstd"),
            "error: {err}"
        );
    }

    #[test]
    fn test_validate_no_szip_offsets_for_non_szip_allows_szip() {
        let mut params = BTreeMap::new();
        params.insert(
            "szip_block_offsets".to_string(),
            ciborium::Value::Array(vec![0u64, 1].into_iter().map(|n| n.into()).collect()),
        );
        let desc = DataObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim: 1,
            shape: vec![2],
            strides: vec![1],
            dtype: Dtype::Float32,
            byte_order: ByteOrder::native(),
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "szip".to_string(),
            params,
            masks: None,
        };

        assert!(validate_no_szip_offsets_for_non_szip(&desc).is_ok());
    }
}
