// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::collections::BTreeMap;
use std::io::Write;

use crate::encode::{
    EncodeOptions, MaskMethod, build_pipeline_config, populate_base_entries,
    populate_reserved_provenance, validate_no_szip_offsets_for_non_szip, validate_object,
    validate_szip_block_offsets,
};
use crate::error::{Result, TensogramError};
use crate::framing::EncodedObject;
use crate::hash::HashAlgorithm;
use crate::metadata::{self, RESERVED_KEY};
use crate::substitute_and_mask;
use crate::types::{DataObjectDescriptor, GlobalMetadata, HashDescriptor, HashFrame, IndexFrame};
use crate::wire::{
    FRAME_END, FRAME_HEADER_SIZE, FrameHeader, FrameType, MessageFlags, POSTAMBLE_SIZE,
    PREAMBLE_SIZE, Postamble, Preamble,
};
use tensogram_encodings::pipeline;

/// A streaming encoder that writes Tensogram frames progressively to a sink.
///
/// Unlike [`crate::encode::encode`], which builds the entire message in memory,
/// `StreamingEncoder` writes each data object frame immediately. This allows
/// encoding to a socket or pipe without buffering the full message.
///
/// The trade-off is that header-based index and hash frames are not possible;
/// instead, these are written as footer frames when [`finish`](StreamingEncoder::finish)
/// is called.
///
/// # Example
/// ```no_run
/// use std::io::BufWriter;
/// use std::fs::File;
/// use tensogram::streaming::StreamingEncoder;
/// use tensogram::{GlobalMetadata, EncodeOptions};
///
/// let file = BufWriter::new(File::create("output.tgm").unwrap());
/// let meta = GlobalMetadata::default();
/// let mut enc = StreamingEncoder::new(file, &meta, &EncodeOptions::default()).unwrap();
/// // enc.write_object(&desc, &data).unwrap();
/// // enc.finish().unwrap();
/// ```
pub struct StreamingEncoder<W: Write> {
    writer: W,
    /// Byte offsets of each data object frame from message start.
    object_offsets: Vec<u64>,
    /// Total byte length of each data object frame, excluding alignment padding.
    object_lengths: Vec<u64>,
    /// Per-object hash entries: (hash_type, hash_value).
    hash_entries: Vec<Option<(String, String)>>,
    /// Descriptors of completed objects (payloads not retained) — used to
    /// populate per-object payload entries in the footer metadata frame.
    completed_objects: Vec<EncodedObject>,
    /// Total bytes written so far.
    bytes_written: u64,
    /// Hash algorithm to use for payload integrity.
    hash_algorithm: Option<HashAlgorithm>,
    /// Original global metadata — re-used to build the footer metadata frame.
    global_meta: GlobalMetadata,
    /// True when a PrecederMetadata frame has been written but the
    /// corresponding DataObject has not yet been written.
    pending_preceder: bool,
    /// Per-object preceder payloads — stored so the footer metadata can
    /// include all per-object metadata (for decoders that skip preceders).
    preceder_payloads: Vec<Option<BTreeMap<String, ciborium::Value>>>,
    /// Intra-codec thread budget resolved from `EncodeOptions.threads`
    /// at construction time.  Passed through to every `write_object`
    /// pipeline call; axis A is not applicable in streaming mode
    /// because each `write_object` is a separate caller-paced event.
    intra_codec_threads: u32,
    /// Snapshot of the parallel-threshold option for the same reason.
    parallel_threshold_bytes: Option<usize>,
    /// Snapshot of `EncodeOptions.allow_nan` / `allow_inf` — determines
    /// whether `write_object` substitutes non-finite values and emits
    /// a mask companion section (see `plans/BITMASK_FRAME.md`).
    allow_nan: bool,
    allow_inf: bool,
    /// Per-kind mask compression method snapshots.
    nan_mask_method: MaskMethod,
    pos_inf_mask_method: MaskMethod,
    neg_inf_mask_method: MaskMethod,
    /// Snapshot of the small-mask fallback threshold.
    small_mask_threshold_bytes: usize,
}

impl<W: Write> StreamingEncoder<W> {
    /// Begin a new streaming message.
    ///
    /// Writes the preamble (with `total_length = 0` for streaming mode)
    /// and a header metadata frame containing the global metadata.
    pub fn new(
        mut writer: W,
        global_meta: &GlobalMetadata,
        options: &EncodeOptions,
    ) -> Result<Self> {
        let meta_cbor = metadata::global_metadata_to_cbor(global_meta)?;

        // Streaming preamble: total_length=0 signals unknown length at write time.
        // Always set PRECEDER_METADATA in streaming mode — the flag is advisory
        // and decoders handle the absence of actual preceder frames gracefully.
        let mut flags = MessageFlags::default();
        flags.set(MessageFlags::HEADER_METADATA);
        flags.set(MessageFlags::FOOTER_METADATA);
        flags.set(MessageFlags::FOOTER_INDEX);
        flags.set(MessageFlags::PRECEDER_METADATA);
        if options.hash_algorithm.is_some() {
            flags.set(MessageFlags::FOOTER_HASHES);
        }

        let preamble = Preamble {
            version: 3,
            flags,
            reserved: 0,
            total_length: 0,
        };
        let preamble_bytes = preamble_to_bytes(&preamble);
        writer.write_all(&preamble_bytes)?;
        let mut bytes_written = PREAMBLE_SIZE as u64;

        // Write header metadata frame
        let frame_bytes = build_frame(FrameType::HeaderMetadata, 1, 0, &meta_cbor);
        writer.write_all(&frame_bytes)?;
        bytes_written += frame_bytes.len() as u64;

        write_padding(&mut writer, &mut bytes_written)?;

        // Snapshot the thread budget now so that mid-message changes to
        // TENSOGRAM_THREADS don't leak in between write_object calls —
        // one message is deterministic.
        let intra_codec_threads = crate::parallel::resolve_budget(options.threads);

        Ok(Self {
            writer,
            object_offsets: Vec::new(),
            object_lengths: Vec::new(),
            hash_entries: Vec::new(),
            completed_objects: Vec::new(),
            bytes_written,
            hash_algorithm: options.hash_algorithm,
            global_meta: global_meta.clone(),
            pending_preceder: false,
            preceder_payloads: Vec::new(),
            intra_codec_threads,
            parallel_threshold_bytes: options.parallel_threshold_bytes,
            allow_nan: options.allow_nan,
            allow_inf: options.allow_inf,
            nan_mask_method: options.nan_mask_method.clone(),
            pos_inf_mask_method: options.pos_inf_mask_method.clone(),
            neg_inf_mask_method: options.neg_inf_mask_method.clone(),
            small_mask_threshold_bytes: options.small_mask_threshold_bytes,
        })
    }

    /// Write a PrecederMetadata frame for the next data object.
    ///
    /// The `metadata` map becomes `base[0]` in a `GlobalMetadata` CBOR
    /// wrapper.  Must be followed by exactly one
    /// [`write_object`](Self::write_object) or
    /// [`write_object_pre_encoded`](Self::write_object_pre_encoded) call
    /// before another `write_preceder` or [`finish`](Self::finish).
    pub fn write_preceder(&mut self, metadata: BTreeMap<String, ciborium::Value>) -> Result<()> {
        if self.pending_preceder {
            return Err(TensogramError::Framing(
                "write_preceder called twice without an intervening write_object/write_object_pre_encoded".to_string(),
            ));
        }

        // Reject _reserved_ in preceder metadata — this namespace is library-managed
        // and would collide with the encoder's auto-populated _reserved_.tensor.
        if metadata.contains_key(RESERVED_KEY) {
            return Err(TensogramError::Metadata(format!(
                "client code must not write '{RESERVED_KEY}' in preceder metadata; \
                     this field is populated by the library"
            )));
        }

        let preceder_meta = GlobalMetadata {
            version: self.global_meta.version,
            base: vec![metadata.clone()],
            ..Default::default()
        };
        let cbor = crate::metadata::global_metadata_to_cbor(&preceder_meta)?;
        let frame_bytes = build_frame(FrameType::PrecederMetadata, 1, 0, &cbor);
        self.writer.write_all(&frame_bytes)?;
        self.bytes_written += frame_bytes.len() as u64;

        write_padding(&mut self.writer, &mut self.bytes_written)?;

        self.pending_preceder = true;
        // Store for inclusion in footer metadata
        self.preceder_payloads.push(Some(metadata));
        Ok(())
    }

    /// Encode and write a single data object frame.
    ///
    /// The descriptor's encoding/filter/compression pipeline is applied,
    /// the payload is hashed (if configured), and the frame is written
    /// immediately — no buffering.
    ///
    /// When `EncodeOptions.threads > 0` was passed to
    /// [`StreamingEncoder::new`], the pipeline call may use up to that
    /// many threads internally (axis B).  Axis A is not available in
    /// streaming mode — each `write_object` is a caller-paced event
    /// with no cross-object parallelism opportunity.
    pub fn write_object(&mut self, desc: &DataObjectDescriptor, data: &[u8]) -> Result<()> {
        validate_object(desc, data.len())?;

        let shape_product = desc
            .shape
            .iter()
            .try_fold(1u64, |acc, &x| acc.checked_mul(x))
            .ok_or_else(|| TensogramError::Metadata("shape product overflow".to_string()))?;
        let num_elements = usize::try_from(shape_product)
            .map_err(|_| TensogramError::Metadata("element count overflows usize".to_string()))?;

        // Honour the intra-codec thread budget captured at construction.
        // Small-message threshold: if the payload is below the threshold,
        // skip the pool (the overhead would outweigh any codec win).
        let parallel = crate::parallel::should_parallelise(
            self.intra_codec_threads,
            data.len(),
            self.parallel_threshold_bytes,
        );

        // Pre-pipeline substitute-and-mask stage — see
        // [`crate::encode::encode_one_object`] for semantics.
        // `write_object_pre_encoded` treats its input as opaque and
        // bypasses the stage, matching buffered `encode_pre_encoded`.
        let (pipeline_input, mask_set) = substitute_and_mask::substitute_and_mask(
            data,
            desc.dtype,
            desc.byte_order,
            self.allow_nan,
            self.allow_inf,
            parallel,
        )?;
        let intra = if parallel {
            self.intra_codec_threads
        } else {
            0
        };

        let config = crate::encode::build_pipeline_config_with_backend(
            desc,
            num_elements,
            desc.dtype,
            tensogram_encodings::pipeline::CompressionBackend::default(),
            intra,
        )?;

        let result =
            crate::parallel::run_maybe_pooled(self.intra_codec_threads, parallel, intra, || {
                pipeline::encode_pipeline(pipeline_input.as_ref(), &config)
            })
            .map_err(|e| TensogramError::Encoding(e.to_string()))?;

        // Build final descriptor with computed fields
        let mut final_desc = desc.clone();

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

        // Compose the payload region: [encoded_payload][masks...] — see
        // `plans/BITMASK_FRAME.md` §3.2.  When no masks were produced
        // (the common case) this is a zero-cost passthrough.
        let (payload_region, masks_metadata) = crate::encode::compose_payload_region(
            result.encoded_bytes,
            mask_set,
            &self.nan_mask_method,
            &self.pos_inf_mask_method,
            &self.neg_inf_mask_method,
            self.small_mask_threshold_bytes,
        )?;
        if let Some(m) = masks_metadata {
            final_desc.masks = Some(m);
        }

        self.write_object_inner(final_desc, &payload_region)
    }

    /// Write a pre-encoded data object frame directly.
    ///
    /// Unlike [`write_object`](Self::write_object), this method does **not**
    /// run the encoding pipeline — `pre_encoded_bytes` are written to the
    /// stream as-is.  The descriptor must accurately describe the encoding
    /// that was already applied (encoding, filter, compression, params) so
    /// that decoders can reconstruct the original payload.
    ///
    /// This method participates in the same preceder consumption logic as
    /// [`write_object`](Self::write_object) and can be freely intermixed
    /// with it.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor is invalid or the frame cannot be
    /// written to the underlying writer.
    #[tracing::instrument(skip(self, descriptor, pre_encoded_bytes))]
    pub fn write_object_pre_encoded(
        &mut self,
        descriptor: &DataObjectDescriptor,
        pre_encoded_bytes: &[u8],
    ) -> Result<()> {
        validate_object(descriptor, pre_encoded_bytes.len())?;

        let shape_product = descriptor
            .shape
            .iter()
            .try_fold(1u64, |acc, &x| acc.checked_mul(x))
            .ok_or_else(|| TensogramError::Metadata("shape product overflow".to_string()))?;
        let num_elements = usize::try_from(shape_product)
            .map_err(|_| TensogramError::Metadata("element count overflows usize".to_string()))?;

        // Validate descriptor pipeline configuration without encoding.
        build_pipeline_config(descriptor, num_elements, descriptor.dtype)?;

        // Validate szip metadata — same checks as buffered encode_pre_encoded.
        validate_no_szip_offsets_for_non_szip(descriptor)?;
        if descriptor.compression == "szip" && descriptor.params.contains_key("szip_block_offsets")
        {
            validate_szip_block_offsets(&descriptor.params, pre_encoded_bytes.len())?;
        }

        self.write_object_inner(descriptor.clone(), pre_encoded_bytes)
    }

    /// Shared inner implementation for both [`write_object`](Self::write_object) and
    /// [`write_object_pre_encoded`](Self::write_object_pre_encoded).
    ///
    /// Writes the data object frame directly to the sink and — when a hash
    /// algorithm is configured — computes the xxh3-64 digest inline with
    /// the payload write.  The payload bytes are therefore walked exactly
    /// once: no intermediate frame buffer is built in memory.
    ///
    /// Updates all bookkeeping and consumes any pending preceder.
    fn write_object_inner(
        &mut self,
        mut final_desc: DataObjectDescriptor,
        encoded_bytes: &[u8],
    ) -> Result<()> {
        let start_offset = self.bytes_written;

        let frame_len = write_data_object_frame_hashed(
            &mut self.writer,
            &mut final_desc,
            encoded_bytes,
            self.hash_algorithm,
        )?;
        self.bytes_written += frame_len;

        // The helper writes the real hash back into `final_desc.hash`;
        // mirror it into `hash_entries` for footer-frame bookkeeping.
        let hash_entry = final_desc
            .hash
            .as_ref()
            .map(|h| (h.hash_type.clone(), h.value.clone()));

        self.object_offsets.push(start_offset);
        self.object_lengths.push(frame_len);
        self.hash_entries.push(hash_entry);
        // Retain only the descriptor for footer metadata population.
        // The encoded payload has already been written to the stream;
        // keeping it in memory would negate streaming's memory benefits.
        self.completed_objects.push(EncodedObject {
            descriptor: final_desc,
            encoded_payload: Vec::new(),
        });

        // Consume pending preceder — if no preceder was written for this
        // object, record None so preceder_payloads stays aligned with objects.
        if self.pending_preceder {
            self.pending_preceder = false;
        } else {
            self.preceder_payloads.push(None);
        }

        // Align to 8 bytes
        write_padding(&mut self.writer, &mut self.bytes_written)?;

        Ok(())
    }

    /// Finalize the streaming message.
    ///
    /// Writes footer frames (payload metadata + hash + index) and the postamble.
    /// Consumes the encoder and returns the underlying writer.
    pub fn finish(mut self) -> Result<W> {
        if self.pending_preceder {
            return Err(TensogramError::Framing(
                "dangling PrecederMetadata: finish called without a following write_object/write_object_pre_encoded"
                    .to_string(),
            ));
        }

        let footer_start = self.bytes_written;

        // Footer metadata frame: updated global metadata with per-object payload entries.
        // The header metadata was written without knowing the objects; here we write
        // a footer metadata frame that supersedes it with payload populated.
        // Preceder payloads are merged in so the footer is complete even for
        // decoders that skip PrecederMetadata frames.
        {
            let mut enriched_meta = self.global_meta.clone();
            populate_base_entries(&mut enriched_meta.base, &self.completed_objects);
            populate_reserved_provenance(&mut enriched_meta.reserved);

            // Merge preceder payloads into footer metadata base entries
            // (preceder wins).  preceder_payloads is aligned 1:1 with
            // completed_objects by write_preceder/write_object bookkeeping,
            // so the lengths must match.
            if self.preceder_payloads.len() != self.completed_objects.len() {
                return Err(TensogramError::Framing(format!(
                    "internal: preceder_payloads ({}) out of sync with completed_objects ({})",
                    self.preceder_payloads.len(),
                    self.completed_objects.len()
                )));
            }
            for (i, prec) in self.preceder_payloads.iter().enumerate() {
                if let Some(prec_map) = prec
                    && i < enriched_meta.base.len()
                {
                    for (k, v) in prec_map {
                        enriched_meta.base[i].insert(k.clone(), v.clone());
                    }
                }
            }
            let meta_cbor = metadata::global_metadata_to_cbor(&enriched_meta)?;
            let frame_bytes = build_frame(FrameType::FooterMetadata, 1, 0, &meta_cbor);
            self.writer.write_all(&frame_bytes)?;
            self.bytes_written += frame_bytes.len() as u64;
            write_padding(&mut self.writer, &mut self.bytes_written)?;
        }

        // Footer hash frame (if any objects had hashes)
        let has_hashes = self.hash_entries.iter().any(|e| e.is_some());
        if has_hashes {
            let hash_type = self
                .hash_algorithm
                .map(|a| a.as_str().to_string())
                .unwrap_or_default();
            let hashes: Vec<String> = self
                .hash_entries
                .iter()
                .map(|e| e.as_ref().map(|(_, v)| v.clone()).unwrap_or_default())
                .collect();
            let hash_frame = HashFrame {
                object_count: self.object_offsets.len() as u64,
                hash_type,
                hashes,
            };
            let hash_cbor = metadata::hash_frame_to_cbor(&hash_frame)?;
            let frame_bytes = build_frame(FrameType::FooterHash, 1, 0, &hash_cbor);
            self.writer.write_all(&frame_bytes)?;
            self.bytes_written += frame_bytes.len() as u64;

            write_padding(&mut self.writer, &mut self.bytes_written)?;
        }

        // Footer index frame
        let index = IndexFrame {
            object_count: self.object_offsets.len() as u64,
            offsets: self.object_offsets,
            lengths: self.object_lengths,
        };
        let index_cbor = metadata::index_to_cbor(&index)?;
        let frame_bytes = build_frame(FrameType::FooterIndex, 1, 0, &index_cbor);
        self.writer.write_all(&frame_bytes)?;
        self.bytes_written += frame_bytes.len() as u64;

        write_padding(&mut self.writer, &mut self.bytes_written)?;

        // Postamble.
        //
        // Streaming mode writes `total_length = 0` by default — the
        // writer is typed `W: Write` and we cannot seek back to
        // rewrite the preamble.  Callers with a seekable sink can
        // opt into back-filling via `finish_with_backfill` (below),
        // which populates both the preamble's and postamble's
        // `total_length` fields.
        let postamble = Postamble {
            first_footer_offset: footer_start,
            total_length: 0,
        };
        let mut postamble_bytes = Vec::with_capacity(POSTAMBLE_SIZE);
        postamble.write_to(&mut postamble_bytes);
        self.writer.write_all(&postamble_bytes)?;
        self.bytes_written += postamble_bytes.len() as u64;

        self.writer.flush()?;

        Ok(self.writer)
    }

    /// Returns the number of data objects written so far.
    pub fn object_count(&self) -> usize {
        self.object_offsets.len()
    }

    /// Returns the total bytes written so far.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
}

// ── Seekable-sink specialisation ─────────────────────────────────────────────

impl<W: Write + std::io::Seek> StreamingEncoder<W> {
    /// Finalize the streaming message and back-fill the `total_length`
    /// field in both the preamble and postamble (v3 §7 and §9.4).
    ///
    /// Equivalent to [`finish`] but at the end seeks back to the
    /// preamble offset (0) and the postamble's `total_length` slot
    /// (`end_pos - 16`), writing the real message length to both.
    /// Readers can then backward-scan from EOF using the mirrored
    /// length without fallback to forward scanning.
    ///
    /// Use this when the writer backs a seekable sink (file, cursor
    /// over a buffer) — it's a no-op semantic change over
    /// `finish()` but enables O(1) backward scan on the produced
    /// file.
    ///
    /// # Errors
    ///
    /// Returns any I/O error from the underlying writer's `seek`,
    /// `write_all`, or `flush`.
    ///
    /// [`finish`]: StreamingEncoder::finish
    pub fn finish_with_backfill(self) -> Result<W> {
        use std::io::SeekFrom;
        // `finish()` consumes self and returns the inner writer;
        // we then seek back to patch both length slots.
        let mut writer = self.finish()?;

        // Determine the current file position, which equals the full
        // message length after finish() (finish() consumed self, so
        // we re-derive via `stream_position`).
        let end_pos = writer.stream_position()?;
        let total_length = end_pos;

        // Back-fill the preamble's total_length (bytes 16..24).
        writer.seek(SeekFrom::Start(16))?;
        writer.write_all(&total_length.to_be_bytes())?;

        // Back-fill the postamble's total_length (second u64 field,
        // located at `end - 16 .. end - 8`; the trailing 8 bytes are
        // the END_MAGIC).
        writer.seek(SeekFrom::Start(end_pos - 16))?;
        writer.write_all(&total_length.to_be_bytes())?;

        // Seek back to EOF so the returned writer matches
        // `finish()`'s post-condition (cursor at the very end).
        writer.seek(SeekFrom::Start(end_pos))?;
        writer.flush()?;
        Ok(writer)
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn preamble_to_bytes(preamble: &Preamble) -> Vec<u8> {
    let mut out = Vec::with_capacity(PREAMBLE_SIZE);
    preamble.write_to(&mut out);
    out
}

fn build_frame(frame_type: FrameType, version: u16, flags: u16, payload: &[u8]) -> Vec<u8> {
    let total_length = (FRAME_HEADER_SIZE + payload.len() + FRAME_END.len()) as u64;
    let fh = FrameHeader {
        frame_type,
        version,
        flags,
        total_length,
    };
    let mut out = Vec::with_capacity(total_length as usize);
    fh.write_to(&mut out);
    out.extend_from_slice(payload);
    out.extend_from_slice(FRAME_END);
    out
}

const ZERO_PAD: [u8; 7] = [0; 7];

fn write_padding(writer: &mut impl Write, bytes_written: &mut u64) -> std::io::Result<()> {
    let pad = (8 - (*bytes_written as usize % 8)) % 8;
    if pad > 0 {
        writer.write_all(&ZERO_PAD[..pad])?;
        *bytes_written += pad as u64;
    }
    Ok(())
}

/// Write a data object frame directly to `writer` while optionally hashing
/// the payload in a single pass.
///
/// This is the streaming equivalent of
/// [`crate::framing::encode_data_object_frame`], but instead of allocating a
/// `Vec<u8>` for the whole frame it streams pieces straight to the sink.
/// When `hash_algorithm` is `Some(_)`, the payload is fed to an
/// `Xxh3Default` hasher in 64 KiB chunks as it is written, so the payload
/// is walked exactly once end-to-end.
///
/// Frame layout (always `CBOR_AFTER_PAYLOAD`):
///
/// ```text
/// [FrameHeader 16B][Payload][CBOR descriptor][cbor_offset 8B][ENDF 4B]
/// ```
///
/// # CBOR length invariant
///
/// The CBOR descriptor embeds the hash value, but the frame header's
/// `total_length` field must be written *before* the hash is known.  We
/// rely on each [`HashAlgorithm`] variant producing a hex string of a
/// fixed advertised length (see [`HashAlgorithm::hex_digest_len`]) so
/// that the CBOR byte count is the same whether we serialise a
/// placeholder or the real digest.  The length is verified twice:
///
/// 1. Once **before writing any bytes**, by serialising with two
///    distinct placeholder digests and confirming they produce the same
///    CBOR length.  If this fails we return a
///    [`TensogramError::Framing`] without touching the sink, so the
///    caller's stream is untouched.
/// 2. Once **after** the real CBOR is built, as a debug-only sanity
///    check (the first check makes this redundant, but it catches
///    divergences in `object_descriptor_to_cbor` itself).
///
/// The descriptor's `hash` field is cleared on entry so any
/// caller-supplied hash does not leak into the size estimate, and on exit
/// carries the real `HashDescriptor` produced by the inline hash (or
/// `None` when `hash_algorithm == None`).
///
/// # Returns
///
/// Number of bytes written to `writer`, not including any trailing 8-byte
/// alignment padding (the caller's responsibility).
///
/// # Errors
///
/// * [`TensogramError::Framing`] if the configured [`HashAlgorithm`]
///   produces a CBOR representation whose length depends on the digest
///   value, or if the frame's `total_length` would overflow `u64`.  In
///   both cases no bytes are written to `writer`.
/// * [`TensogramError::Metadata`] if CBOR serialisation fails.
/// * [`TensogramError::Io`] on any `writer` failure — partial writes may
///   already be on the sink.
fn write_data_object_frame_hashed<W: Write>(
    writer: &mut W,
    descriptor: &mut DataObjectDescriptor,
    payload: &[u8],
    hash_algorithm: Option<HashAlgorithm>,
) -> Result<u64> {
    use crate::wire::{DATA_OBJECT_FOOTER_SIZE, DataObjectFlags, FRAME_END};

    // ── CBOR size estimate ────────────────────────────────────────────
    //
    // Build a placeholder descriptor that carries the exact same
    // `hash_type` label the final descriptor will carry, plus a hex
    // value of `alg.hex_digest_len()` zeros.
    let cbor_len_estimate = match hash_algorithm {
        None => {
            descriptor.hash = None;
            metadata::object_descriptor_to_cbor(descriptor)?.len()
        }
        Some(alg) => {
            let placeholder_len = alg.hex_digest_len();
            descriptor.hash = Some(HashDescriptor {
                hash_type: alg.as_str().to_string(),
                value: "0".repeat(placeholder_len),
            });
            let len_zeros = metadata::object_descriptor_to_cbor(descriptor)?.len();

            // Verify the invariant BEFORE writing any bytes: a second
            // placeholder with different digit values must serialise to
            // the same length, otherwise the CBOR length depends on the
            // hash value and we would corrupt `total_length`.
            descriptor.hash = Some(HashDescriptor {
                hash_type: alg.as_str().to_string(),
                value: "f".repeat(placeholder_len),
            });
            let len_ones = metadata::object_descriptor_to_cbor(descriptor)?.len();
            if len_zeros != len_ones {
                return Err(TensogramError::Framing(format!(
                    "streaming encoder requires a hash algorithm with a \
                     value-independent CBOR encoding length; {} produced \
                     {len_zeros} bytes for an all-zero digest and \
                     {len_ones} bytes for an all-'f' digest of the same \
                     length ({placeholder_len} chars).  No frame bytes \
                     have been written.  Use the buffered encode() API \
                     for this hash algorithm, or extend \
                     write_data_object_frame_hashed to handle variable-\
                     length digests.",
                    alg.as_str()
                )));
            }
            len_zeros
        }
    };

    let payload_len = payload.len();

    // Compute `total_length` in u64 with checked arithmetic so an
    // overflow (only reachable on pathological 32-bit inputs) becomes
    // a clean `TensogramError::Framing` before any bytes are written.
    let total_length = (FRAME_HEADER_SIZE as u64)
        .checked_add(cbor_len_estimate as u64)
        .and_then(|n| n.checked_add(payload_len as u64))
        .and_then(|n| n.checked_add(DATA_OBJECT_FOOTER_SIZE as u64))
        .ok_or_else(|| {
            TensogramError::Framing(format!(
                "data object frame total_length overflows u64 \
                 (payload {payload_len} bytes, CBOR {cbor_len_estimate} bytes, \
                 framing {} bytes)",
                FRAME_HEADER_SIZE + DATA_OBJECT_FOOTER_SIZE
            ))
        })?;

    // ── 1) Frame header ──────────────────────────────────────────────
    //
    // 0.17+ emits `NTensorMaskedFrame` (type 9) for every new data-object
    // frame.  Without a `masks` sub-map in the CBOR descriptor the on-wire
    // layout matches pre-0.17 type 4 byte-for-byte except for the type
    // number.
    let mut header_bytes = Vec::with_capacity(FRAME_HEADER_SIZE);
    FrameHeader {
        frame_type: FrameType::NTensorMaskedFrame,
        version: 1,
        flags: DataObjectFlags::CBOR_AFTER_PAYLOAD,
        total_length,
    }
    .write_to(&mut header_bytes);
    writer.write_all(&header_bytes)?;

    // ── 2) Payload — single walk, hashing inline in 64 KiB chunks ────
    //
    // Inline streaming hashing is xxh3-specific today: the digest
    // format (`format_xxh3_digest`) and hasher type (`Xxh3Default`)
    // are both Xxh3-bound.  The inner match on `HashAlgorithm` is
    // exhaustive, so adding a new variant becomes a compile error
    // here — forcing the maintainer to either wire up a new hasher
    // and digest formatter, or to reject the new algorithm with a
    // clean `TensogramError::Framing` (and route it through the
    // buffered `encode()` path instead).
    //
    // We couple the hasher with its algorithm tag in a single
    // `Option<(hasher, alg)>` so the digest-install match in step 3
    // cannot get out of sync with the construction.
    const CHUNK: usize = 64 * 1024;
    let mut inline_hasher: Option<(xxhash_rust::xxh3::Xxh3Default, HashAlgorithm)> = hash_algorithm
        .map(|alg| match alg {
            HashAlgorithm::Xxh3 => (xxhash_rust::xxh3::Xxh3Default::new(), alg),
        });
    let mut offset = 0;
    while offset < payload_len {
        let end = (offset + CHUNK).min(payload_len);
        let chunk = &payload[offset..end];
        if let Some((h, _)) = &mut inline_hasher {
            h.update(chunk);
        }
        writer.write_all(chunk)?;
        offset = end;
    }

    // ── 3) Install the real hash digest on the descriptor ────────────
    //
    // The match on `alg` is exhaustive over `HashAlgorithm`; a new
    // variant forces a compile error until its hex-digest formatter
    // is wired in (or the construction in step 2 is extended to
    // reject it).
    descriptor.hash = inline_hasher.map(|(h, alg)| match alg {
        HashAlgorithm::Xxh3 => HashDescriptor {
            hash_type: alg.as_str().to_string(),
            value: crate::hash::format_xxh3_digest(h.digest()),
        },
    });

    // ── 4) CBOR descriptor — re-serialised with the real hash value ──
    //
    // Length must match the placeholder estimate byte-for-byte, which
    // the pre-write check above has already guaranteed for this
    // algorithm.  The `debug_assert_eq!` is a dev-time tripwire for
    // bugs inside `object_descriptor_to_cbor` itself (non-canonical
    // serialisation etc.), not for the hash-length invariant.
    let cbor_bytes = metadata::object_descriptor_to_cbor(descriptor)?;
    debug_assert_eq!(
        cbor_bytes.len(),
        cbor_len_estimate,
        "write_data_object_frame_hashed: final CBOR length \
         ({}) differs from pre-write estimate ({cbor_len_estimate}) — \
         this indicates a non-deterministic CBOR serialiser, not a \
         hash-length problem (that would have been caught earlier)",
        cbor_bytes.len(),
    );
    writer.write_all(&cbor_bytes)?;

    // ── 5) cbor_offset — frame-relative byte offset of CBOR start ────
    let cbor_offset = (FRAME_HEADER_SIZE + payload_len) as u64;
    writer.write_all(&cbor_offset.to_be_bytes())?;

    // ── 6) ENDF terminator ───────────────────────────────────────────
    writer.write_all(FRAME_END)?;

    Ok(total_length)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dtype;
    use crate::decode::{DecodeOptions, decode};
    use crate::encode::{EncodeOptions, encode};
    use crate::types::{ByteOrder, DataObjectDescriptor};
    use std::collections::BTreeMap;

    fn make_descriptor(shape: Vec<u64>) -> DataObjectDescriptor {
        let ndim = shape.len() as u64;
        let mut strides = vec![0u64; shape.len()];
        if !shape.is_empty() {
            strides[shape.len() - 1] = 1;
            for i in (0..shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        DataObjectDescriptor {
            obj_type: "ntensor".to_string(),
            ndim,
            shape,
            strides,
            dtype: Dtype::Float32,
            byte_order: ByteOrder::native(),
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            masks: None,
            hash: None,
        }
    }

    #[test]
    fn streaming_single_object_round_trip() {
        let meta = GlobalMetadata::default();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 4 * 4];

        // Streaming encode
        let buf = Vec::new();
        let mut enc = StreamingEncoder::new(buf, &meta, &EncodeOptions::default()).unwrap();
        enc.write_object(&desc, &data).unwrap();
        let result = enc.finish().unwrap();

        // Decode should succeed
        let (decoded_meta, objects) = decode(&result, &DecodeOptions::default()).unwrap();
        assert_eq!(decoded_meta.version, 3);
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].1, data);
    }

    #[test]
    fn streaming_multi_object_round_trip() {
        let meta = GlobalMetadata::default();
        let desc1 = make_descriptor(vec![4]);
        let desc2 = make_descriptor(vec![8]);
        let data1 = vec![1u8; 4 * 4];
        let data2 = vec![2u8; 8 * 4];

        let buf = Vec::new();
        let mut enc = StreamingEncoder::new(buf, &meta, &EncodeOptions::default()).unwrap();
        enc.write_object(&desc1, &data1).unwrap();
        enc.write_object(&desc2, &data2).unwrap();
        assert_eq!(enc.object_count(), 2);
        let result = enc.finish().unwrap();

        let (_, objects) = decode(&result, &DecodeOptions::default()).unwrap();
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].1, data1);
        assert_eq!(objects[1].1, data2);
    }

    #[test]
    fn streaming_matches_buffered_single_object() {
        let meta = GlobalMetadata::default();
        let desc = make_descriptor(vec![4]);
        let data = vec![42u8; 4 * 4];
        let options = EncodeOptions {
            hash_algorithm: Some(HashAlgorithm::Xxh3),
            ..Default::default()
        };

        // Buffered encode
        let buffered = encode(&meta, &[(&desc, &data)], &options).unwrap();
        let (buf_meta, buf_objects) = decode(
            &buffered,
            &DecodeOptions {
                verify_hash: true,
                ..Default::default()
            },
        )
        .unwrap();

        // Streaming encode
        let buf = Vec::new();
        let mut enc = StreamingEncoder::new(buf, &meta, &options).unwrap();
        enc.write_object(&desc, &data).unwrap();
        let streamed = enc.finish().unwrap();
        let (str_meta, str_objects) = decode(
            &streamed,
            &DecodeOptions {
                verify_hash: true,
                ..Default::default()
            },
        )
        .unwrap();

        // Data must match (wire bytes may differ due to header vs footer layout)
        assert_eq!(buf_meta.version, str_meta.version);
        assert_eq!(buf_objects.len(), str_objects.len());
        assert_eq!(buf_objects[0].0.shape, str_objects[0].0.shape);
        assert_eq!(buf_objects[0].0.dtype, str_objects[0].0.dtype);
        assert_eq!(buf_objects[0].1, str_objects[0].1);
        // Hash values must match
        assert_eq!(
            buf_objects[0].0.hash.as_ref().unwrap().value,
            str_objects[0].0.hash.as_ref().unwrap().value
        );
    }

    #[test]
    fn streaming_hash_verification() {
        let meta = GlobalMetadata::default();
        let desc = make_descriptor(vec![4]);
        let data = vec![42u8; 4 * 4];
        let options = EncodeOptions {
            hash_algorithm: Some(HashAlgorithm::Xxh3),
            ..Default::default()
        };

        let buf = Vec::new();
        let mut enc = StreamingEncoder::new(buf, &meta, &options).unwrap();
        enc.write_object(&desc, &data).unwrap();
        let result = enc.finish().unwrap();

        // Verify hash passes
        let verify_opts = DecodeOptions {
            verify_hash: true,
            ..Default::default()
        };
        let (_, objects) = decode(&result, &verify_opts).unwrap();
        assert!(objects[0].0.hash.is_some());
    }

    #[test]
    fn streaming_no_objects() {
        let meta = GlobalMetadata::default();
        let options = EncodeOptions {
            hash_algorithm: None,
            ..Default::default()
        };

        let buf = Vec::new();
        let enc = StreamingEncoder::new(buf, &meta, &options).unwrap();
        assert_eq!(enc.object_count(), 0);
        let result = enc.finish().unwrap();

        let (decoded_meta, objects) = decode(&result, &DecodeOptions::default()).unwrap();
        assert_eq!(decoded_meta.version, 3);
        assert_eq!(objects.len(), 0);
    }

    /// Threads budget on `StreamingEncoder` must not change the encoded
    /// payload for transparent pipelines.  This locks in the pass-3
    /// consistency: axis-B dispatch inside `write_object` is opt-in and
    /// transparent-codec output is byte-identical across thread counts.
    #[test]
    fn streaming_threads_byte_identical_transparent() {
        let meta = GlobalMetadata::default();
        // One large object — 200 KiB — above the 64 KiB default threshold.
        let desc = make_descriptor(vec![50_000]);
        let data: Vec<u8> = (0..50_000)
            .flat_map(|i| (250.0f32 + (i as f32).sin() * 30.0).to_ne_bytes())
            .collect();

        let mk = |threads: u32| -> Vec<u8> {
            let buf = Vec::new();
            let opts = EncodeOptions {
                threads,
                parallel_threshold_bytes: Some(0), // force parallel
                ..Default::default()
            };
            let mut enc = StreamingEncoder::new(buf, &meta, &opts).unwrap();
            enc.write_object(&desc, &data).unwrap();
            enc.finish().unwrap()
        };

        // Compare encoded payload bytes (ignore provenance).
        let payloads = |buf: &[u8]| -> Vec<Vec<u8>> {
            crate::framing::decode_message(buf)
                .unwrap()
                .objects
                .iter()
                .map(|(_, p, _, _)| p.to_vec())
                .collect()
        };

        let baseline = mk(0);
        let payloads_baseline = payloads(&baseline);

        for t in [1u32, 2, 4, 8] {
            let got = mk(t);
            assert_eq!(
                payloads_baseline,
                payloads(&got),
                "streaming threads={t} payload must match sequential"
            );
        }
    }

    #[test]
    fn streaming_with_metadata() {
        let mut extra = BTreeMap::new();
        extra.insert(
            "centre".to_string(),
            ciborium::Value::Text("ecmwf".to_string()),
        );
        let meta = GlobalMetadata {
            version: 3,
            extra,
            ..Default::default()
        };

        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 4 * 4];

        let buf = Vec::new();
        let mut enc = StreamingEncoder::new(buf, &meta, &EncodeOptions::default()).unwrap();
        enc.write_object(&desc, &data).unwrap();
        let result = enc.finish().unwrap();

        let (decoded_meta, _) = decode(&result, &DecodeOptions::default()).unwrap();
        assert_eq!(
            decoded_meta.extra.get("centre"),
            Some(&ciborium::Value::Text("ecmwf".to_string()))
        );
    }

    // ── PrecederMetadata tests ───────────────────────────────────────────

    #[test]
    fn streaming_preceder_round_trip() {
        let meta = GlobalMetadata::default();
        let desc = make_descriptor(vec![4]);
        let data = vec![42u8; 4 * 4];

        let mut prec = BTreeMap::new();
        prec.insert(
            "mars".to_string(),
            ciborium::Value::Map(vec![(
                ciborium::Value::Text("param".to_string()),
                ciborium::Value::Text("2t".to_string()),
            )]),
        );

        let buf = Vec::new();
        let mut enc = StreamingEncoder::new(buf, &meta, &EncodeOptions::default()).unwrap();
        enc.write_preceder(prec).unwrap();
        enc.write_object(&desc, &data).unwrap();
        let result = enc.finish().unwrap();

        let (decoded_meta, objects) = decode(&result, &DecodeOptions::default()).unwrap();
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].1, data);

        // Preceder mars keys should be in base[0]
        let mars = decoded_meta.base[0].get("mars");
        assert!(mars.is_some(), "mars key should be in base[0]");
    }

    #[test]
    fn streaming_preceder_wins_over_footer() {
        // Pre-populate global_meta.base[0] with a value — the preceder
        // should override it after decode.
        let mut footer_base = BTreeMap::new();
        footer_base.insert(
            "source".to_string(),
            ciborium::Value::Text("footer".to_string()),
        );
        let meta = GlobalMetadata {
            version: 3,
            base: vec![footer_base],
            ..Default::default()
        };

        let mut prec = BTreeMap::new();
        prec.insert(
            "source".to_string(),
            ciborium::Value::Text("preceder".to_string()),
        );

        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 4 * 4];

        let buf = Vec::new();
        let mut enc = StreamingEncoder::new(buf, &meta, &EncodeOptions::default()).unwrap();
        enc.write_preceder(prec).unwrap();
        enc.write_object(&desc, &data).unwrap();
        let result = enc.finish().unwrap();

        let (decoded_meta, _) = decode(&result, &DecodeOptions::default()).unwrap();
        let source = decoded_meta.base[0].get("source").and_then(|v| match v {
            ciborium::Value::Text(s) => Some(s.as_str()),
            _ => None,
        });
        assert_eq!(source, Some("preceder"), "preceder should win over footer");
    }

    #[test]
    fn streaming_consecutive_preceder_error() {
        let meta = GlobalMetadata::default();
        let buf = Vec::new();
        let mut enc = StreamingEncoder::new(buf, &meta, &EncodeOptions::default()).unwrap();

        enc.write_preceder(BTreeMap::new()).unwrap();
        let result = enc.write_preceder(BTreeMap::new());
        assert!(
            result.is_err(),
            "two write_preceder calls without intervening write_object should fail"
        );
    }

    #[test]
    fn streaming_dangling_preceder_error() {
        let meta = GlobalMetadata::default();
        let buf = Vec::new();
        let mut enc = StreamingEncoder::new(buf, &meta, &EncodeOptions::default()).unwrap();

        enc.write_preceder(BTreeMap::new()).unwrap();
        let result = enc.finish();
        assert!(
            result.is_err(),
            "finish with a dangling preceder should fail"
        );
    }

    #[test]
    fn streaming_mixed_objects_with_and_without_preceders() {
        let meta = GlobalMetadata::default();
        let desc0 = make_descriptor(vec![4]);
        let desc1 = make_descriptor(vec![8]);
        let data0 = vec![1u8; 4 * 4];
        let data1 = vec![2u8; 8 * 4];

        let mut prec = BTreeMap::new();
        prec.insert(
            "note".to_string(),
            ciborium::Value::Text("only for obj 0".to_string()),
        );

        let buf = Vec::new();
        let mut enc = StreamingEncoder::new(buf, &meta, &EncodeOptions::default()).unwrap();
        // Object 0: with preceder
        enc.write_preceder(prec).unwrap();
        enc.write_object(&desc0, &data0).unwrap();
        // Object 1: without preceder
        enc.write_object(&desc1, &data1).unwrap();
        let result = enc.finish().unwrap();

        let (decoded_meta, objects) = decode(&result, &DecodeOptions::default()).unwrap();
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].1, data0);
        assert_eq!(objects[1].1, data1);

        // base[0] should have preceder entry
        assert!(decoded_meta.base[0].contains_key("note"));
        // base[1] should NOT have it
        assert!(!decoded_meta.base[1].contains_key("note"));
    }

    #[test]
    fn streaming_preceder_metadata_preservation() {
        // Verify application metadata from preceder survives the full
        // encode → footer-merge → decode → preceder-merge path.
        let meta = GlobalMetadata::default();
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 2 * 4];

        let mut prec = BTreeMap::new();
        prec.insert("units".to_string(), ciborium::Value::Text("K".to_string()));
        prec.insert(
            "mars".to_string(),
            ciborium::Value::Map(vec![
                (
                    ciborium::Value::Text("param".to_string()),
                    ciborium::Value::Text("2t".to_string()),
                ),
                (
                    ciborium::Value::Text("levtype".to_string()),
                    ciborium::Value::Text("sfc".to_string()),
                ),
            ]),
        );

        let buf = Vec::new();
        let mut enc = StreamingEncoder::new(buf, &meta, &EncodeOptions::default()).unwrap();
        enc.write_preceder(prec).unwrap();
        enc.write_object(&desc, &data).unwrap();
        let result = enc.finish().unwrap();

        let (decoded_meta, _) = decode(&result, &DecodeOptions::default()).unwrap();
        let p = &decoded_meta.base[0];
        assert_eq!(
            p.get("units"),
            Some(&ciborium::Value::Text("K".to_string()))
        );
        assert!(p.contains_key("mars"));
        // Structural keys (ndim, shape) should be under _reserved_.tensor
        assert!(p.contains_key("_reserved_"));
    }

    // ── Edge case: preceder with _reserved_ rejected ─────────────────────

    #[test]
    fn streaming_preceder_with_reserved_rejected() {
        let meta = GlobalMetadata::default();
        let buf = Vec::new();
        let mut enc = StreamingEncoder::new(buf, &meta, &EncodeOptions::default()).unwrap();

        let mut prec = BTreeMap::new();
        prec.insert("_reserved_".to_string(), ciborium::Value::Map(vec![]));

        let result = enc.write_preceder(prec);
        assert!(result.is_err(), "_reserved_ in preceder should be rejected");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("_reserved_"),
            "error should mention _reserved_: {err}"
        );
    }

    #[test]
    fn streaming_preceder_reserved_stripped_on_decode() {
        // If a non-standard producer includes _reserved_ in a preceder,
        // the decoder strips it rather than failing, and the encoder's
        // _reserved_.tensor is preserved.

        // Build a raw message with a preceder that contains _reserved_.
        // We bypass the encoder's validation by constructing frames manually.
        let mut prec_entry = BTreeMap::new();
        prec_entry.insert(
            "mars".to_string(),
            ciborium::Value::Map(vec![(
                ciborium::Value::Text("param".to_string()),
                ciborium::Value::Text("2t".to_string()),
            )]),
        );
        prec_entry.insert(
            "_reserved_".to_string(),
            ciborium::Value::Map(vec![(
                ciborium::Value::Text("rogue".to_string()),
                ciborium::Value::Text("bad".to_string()),
            )]),
        );

        // Encode normally to get a valid message first, then decode
        // and verify _reserved_ from preceder doesn't clobber.
        // We test via the framing level directly.
        let preceder_meta = GlobalMetadata {
            version: 3,
            base: vec![prec_entry],
            ..Default::default()
        };
        let preceder_cbor = crate::metadata::global_metadata_to_cbor(&preceder_meta).unwrap();

        // Build a raw message with preceder + data object
        let desc_for_frame = make_descriptor(vec![4]);
        let payload = vec![0u8; 4 * 4];
        let frame =
            crate::framing::encode_data_object_frame(&desc_for_frame, &payload, false).unwrap();

        // Footer metadata with _reserved_.tensor
        let mut footer_base = BTreeMap::new();
        let tensor_map = ciborium::Value::Map(vec![
            (
                ciborium::Value::Text("ndim".to_string()),
                ciborium::Value::Integer(1.into()),
            ),
            (
                ciborium::Value::Text("shape".to_string()),
                ciborium::Value::Array(vec![ciborium::Value::Integer(4.into())]),
            ),
            (
                ciborium::Value::Text("strides".to_string()),
                ciborium::Value::Array(vec![ciborium::Value::Integer(1.into())]),
            ),
            (
                ciborium::Value::Text("dtype".to_string()),
                ciborium::Value::Text("float32".to_string()),
            ),
        ]);
        footer_base.insert(
            "_reserved_".to_string(),
            ciborium::Value::Map(vec![(
                ciborium::Value::Text("tensor".to_string()),
                tensor_map,
            )]),
        );
        let footer_meta = GlobalMetadata {
            version: 3,
            base: vec![footer_base],
            ..Default::default()
        };
        let footer_cbor = crate::metadata::global_metadata_to_cbor(&footer_meta).unwrap();

        // Assemble raw message
        use crate::wire::*;
        let header_meta_cbor =
            crate::metadata::global_metadata_to_cbor(&GlobalMetadata::default()).unwrap();

        let mut out = Vec::new();
        out.extend_from_slice(&[0u8; PREAMBLE_SIZE]);

        // Header metadata
        let total_length = (FRAME_HEADER_SIZE + header_meta_cbor.len() + FRAME_END.len()) as u64;
        let fh = FrameHeader {
            frame_type: FrameType::HeaderMetadata,
            version: 1,
            flags: 0,
            total_length,
        };
        fh.write_to(&mut out);
        out.extend_from_slice(&header_meta_cbor);
        out.extend_from_slice(FRAME_END);
        let pad = (8 - (out.len() % 8)) % 8;
        out.extend(std::iter::repeat_n(0u8, pad));

        // Preceder metadata
        let total_length = (FRAME_HEADER_SIZE + preceder_cbor.len() + FRAME_END.len()) as u64;
        let fh = FrameHeader {
            frame_type: FrameType::PrecederMetadata,
            version: 1,
            flags: 0,
            total_length,
        };
        fh.write_to(&mut out);
        out.extend_from_slice(&preceder_cbor);
        out.extend_from_slice(FRAME_END);
        let pad = (8 - (out.len() % 8)) % 8;
        out.extend(std::iter::repeat_n(0u8, pad));

        // Data object
        out.extend_from_slice(&frame);
        let pad = (8 - (out.len() % 8)) % 8;
        out.extend(std::iter::repeat_n(0u8, pad));

        // Footer metadata
        let total_length = (FRAME_HEADER_SIZE + footer_cbor.len() + FRAME_END.len()) as u64;
        let fh = FrameHeader {
            frame_type: FrameType::FooterMetadata,
            version: 1,
            flags: 0,
            total_length,
        };
        fh.write_to(&mut out);
        out.extend_from_slice(&footer_cbor);
        out.extend_from_slice(FRAME_END);
        let pad = (8 - (out.len() % 8)) % 8;
        out.extend(std::iter::repeat_n(0u8, pad));

        // Postamble (patched with total_length after preamble is
        // finalised).
        let postamble_offset = out.len();
        let postamble = Postamble {
            first_footer_offset: postamble_offset as u64,
            total_length: 0,
        };
        postamble.write_to(&mut out);

        // Patch preamble
        let total_length = out.len() as u64;
        let mut flags = MessageFlags::default();
        flags.set(MessageFlags::HEADER_METADATA);
        flags.set(MessageFlags::FOOTER_METADATA);
        flags.set(MessageFlags::PRECEDER_METADATA);
        let preamble = Preamble {
            version: 3,
            flags,
            reserved: 0,
            total_length,
        };
        let mut preamble_bytes = Vec::new();
        preamble.write_to(&mut preamble_bytes);
        out[0..PREAMBLE_SIZE].copy_from_slice(&preamble_bytes);
        // Patch postamble total_length
        out[postamble_offset + 8..postamble_offset + 16]
            .copy_from_slice(&total_length.to_be_bytes());

        // Decode
        let decoded = crate::framing::decode_message(&out).unwrap();

        // The preceder's _reserved_ should have been stripped by the decoder.
        // The footer's _reserved_.tensor should be preserved.
        let base0 = &decoded.global_metadata.base[0];
        assert!(
            base0.contains_key("mars"),
            "mars from preceder should survive"
        );
        // _reserved_ should come from footer, not preceder
        let reserved = base0.get("_reserved_");
        assert!(
            reserved.is_some(),
            "_reserved_ from footer should be present"
        );
        if let Some(ciborium::Value::Map(pairs)) = reserved {
            let has_tensor = pairs
                .iter()
                .any(|(k, _)| *k == ciborium::Value::Text("tensor".to_string()));
            assert!(has_tensor, "tensor key from footer should be preserved");
            let has_rogue = pairs
                .iter()
                .any(|(k, _)| *k == ciborium::Value::Text("rogue".to_string()));
            assert!(
                !has_rogue,
                "rogue key from preceder's _reserved_ should have been stripped"
            );
        }
    }

    // ── write_object_pre_encoded tests ───────────────────────────────────

    #[test]
    fn test_streaming_mixed_mode_pre_encoded() {
        // write_object (raw), write_object_pre_encoded, write_object (raw) — decode all 3.
        let meta = GlobalMetadata::default();

        let desc0 = make_descriptor(vec![4]);
        let desc2 = make_descriptor(vec![6]);
        // Pre-encoded object: encoding="none" so pre-encoded bytes == raw bytes.
        let desc1 = make_descriptor(vec![5]);

        let data0 = vec![1u8; 4 * 4];
        let pre_encoded1 = vec![2u8; 5 * 4]; // treated as already-encoded
        let data2 = vec![3u8; 6 * 4];

        let buf = Vec::new();
        let mut enc = StreamingEncoder::new(buf, &meta, &EncodeOptions::default()).unwrap();
        enc.write_object(&desc0, &data0).unwrap();
        enc.write_object_pre_encoded(&desc1, &pre_encoded1).unwrap();
        enc.write_object(&desc2, &data2).unwrap();
        assert_eq!(enc.object_count(), 3);
        let result = enc.finish().unwrap();

        let (_, objects) = decode(&result, &DecodeOptions::default()).unwrap();
        assert_eq!(objects.len(), 3);
        // Don't compare raw message bytes (provenance is non-deterministic).
        // Compare decoded payloads.
        assert_eq!(objects[0].1, data0, "object 0 payload mismatch");
        assert_eq!(objects[1].1, pre_encoded1, "object 1 payload mismatch");
        assert_eq!(objects[2].1, data2, "object 2 payload mismatch");
    }

    #[test]
    fn test_streaming_preceder_then_pre_encoded() {
        // write_preceder followed by write_object_pre_encoded — preceder metadata
        // should appear in base[0] after decode.
        let meta = GlobalMetadata::default();
        let desc = make_descriptor(vec![4]);
        let pre_encoded = vec![42u8; 4 * 4];

        let mut prec = BTreeMap::new();
        prec.insert(
            "mars".to_string(),
            ciborium::Value::Map(vec![(
                ciborium::Value::Text("param".to_string()),
                ciborium::Value::Text("2t".to_string()),
            )]),
        );

        let buf = Vec::new();
        let mut enc = StreamingEncoder::new(buf, &meta, &EncodeOptions::default()).unwrap();
        enc.write_preceder(prec).unwrap();
        enc.write_object_pre_encoded(&desc, &pre_encoded).unwrap();
        let result = enc.finish().unwrap();

        let (decoded_meta, objects) = decode(&result, &DecodeOptions::default()).unwrap();
        assert_eq!(objects.len(), 1);
        // Payload must round-trip correctly.
        assert_eq!(objects[0].1, pre_encoded, "pre-encoded payload mismatch");
        // Preceder mars key should be in base[0].
        let mars = decoded_meta.base[0].get("mars");
        assert!(
            mars.is_some(),
            "mars key from preceder should be in base[0]"
        );
    }

    #[test]
    fn streaming_finish_preserves_preceder_does_not_clobber_reserved_tensor() {
        // Verify that preceder metadata does NOT clobber the encoder's
        // _reserved_.tensor in the footer metadata.
        let meta = GlobalMetadata::default();
        let desc = make_descriptor(vec![4]);
        let data = vec![42u8; 4 * 4];

        let mut prec = BTreeMap::new();
        prec.insert("units".to_string(), ciborium::Value::Text("K".to_string()));

        let buf = Vec::new();
        let mut enc = StreamingEncoder::new(buf, &meta, &EncodeOptions::default()).unwrap();
        enc.write_preceder(prec).unwrap();
        enc.write_object(&desc, &data).unwrap();
        let result = enc.finish().unwrap();

        let (decoded_meta, _) = decode(&result, &DecodeOptions::default()).unwrap();
        let base0 = &decoded_meta.base[0];

        // preceder key should be present
        assert!(base0.contains_key("units"));

        // _reserved_.tensor should also be present
        let reserved = base0.get("_reserved_").expect("_reserved_ missing");
        if let ciborium::Value::Map(pairs) = reserved {
            let has_tensor = pairs
                .iter()
                .any(|(k, _)| *k == ciborium::Value::Text("tensor".to_string()));
            assert!(
                has_tensor,
                "_reserved_.tensor should be present after preceder merge"
            );
        } else {
            panic!("_reserved_ should be a map");
        }
    }
}
