// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Asynchronous streaming encoder — sibling of
//! [`crate::streaming::StreamingEncoder`] that writes to any
//! `tokio::io::AsyncWrite + Unpin` sink.
//!
//! Driver: the HPC producer scenario in `plans/PLAN_CPP_ASYNC.md` §1 —
//! a producer job emits forecast steps as they are produced, writing to
//! a shared filesystem or object-store sink without blocking the
//! caller's compute thread.
//!
//! ## Wire-format compatibility
//!
//! The async encoder produces **byte-identical output** to the sync
//! [`crate::streaming::StreamingEncoder`] for the same logical sequence
//! of writes.  Frame layout, hash semantics, and footer-frame ordering
//! all share helpers in [`crate::streaming`].  The only difference is
//! the I/O surface (`AsyncWrite` vs `Write`).
//!
//! ## Threading model
//!
//! Like the sync encoder, the async encoder is single-task-owned.  The
//! `AsyncWrite` trait is naturally serial; concurrent
//! [`AsyncStreamingEncoder::write_object`] calls against the same
//! encoder are not permitted.  Hashing remains in the calling task
//! and never crosses thread boundaries — the transparent-codec
//! byte-identical-across-threads invariant is preserved.

use std::collections::BTreeMap;

use tokio::io::{AsyncSeekExt, AsyncWrite, AsyncWriteExt};

use crate::encode::{
    EncodeOptions, MaskMethod, build_pipeline_config, populate_base_entries,
    populate_reserved_provenance, validate_no_szip_offsets_for_non_szip, validate_object,
    validate_szip_block_offsets,
};
use crate::error::{Result, TensogramError};
use crate::framing::{EncodedObject, encode_data_object_frame};
use crate::metadata::{self, RESERVED_KEY};
use crate::streaming::{build_frame, build_preamble_and_header_bytes, padding_for};
use crate::substitute_and_mask;
use crate::types::{DataObjectDescriptor, GlobalMetadata, HashFrame, IndexFrame};
use crate::wire::{FrameType, POSTAMBLE_SIZE, Postamble};
use tensogram_encodings::pipeline;

/// Asynchronous streaming encoder writing Tensogram frames progressively
/// to any [`AsyncWrite`] sink.
///
/// See the module docs for the full design rationale.
///
/// # Example
///
/// ```no_run
/// # async fn run() -> tensogram::Result<()> {
/// use tokio::fs::File;
/// use tensogram::streaming_async::AsyncStreamingEncoder;
/// use tensogram::{GlobalMetadata, EncodeOptions};
///
/// let file = File::create("output.tgm").await?;
/// let meta = GlobalMetadata::default();
/// let mut enc = AsyncStreamingEncoder::new(file, &meta, &EncodeOptions::default()).await?;
/// // enc.write_object(&desc, &data).await?;
/// // enc.finish().await?;
/// # Ok(()) }
/// ```
pub struct AsyncStreamingEncoder<W: AsyncWrite + Unpin> {
    writer: W,
    object_offsets: Vec<u64>,
    object_lengths: Vec<u64>,
    hash_entries: Vec<Option<(String, String)>>,
    completed_objects: Vec<EncodedObject>,
    bytes_written: u64,
    hashing: bool,
    emit_footer_hash_frame: bool,
    global_meta: GlobalMetadata,
    pending_preceder: bool,
    preceder_payloads: Vec<Option<BTreeMap<String, ciborium::Value>>>,
    intra_codec_threads: u32,
    parallel_threshold_bytes: Option<usize>,
    allow_nan: bool,
    allow_inf: bool,
    nan_mask_method: MaskMethod,
    pos_inf_mask_method: MaskMethod,
    neg_inf_mask_method: MaskMethod,
    small_mask_threshold_bytes: usize,
}

impl<W: AsyncWrite + Unpin> AsyncStreamingEncoder<W> {
    /// Begin a new streaming message.
    ///
    /// Writes the preamble (with `total_length = 0` for streaming mode)
    /// and a `HeaderMetadata` frame to the underlying sink.  Mirrors
    /// [`crate::streaming::StreamingEncoder::new`] — the byte-level
    /// output is identical.
    pub async fn new(
        mut writer: W,
        global_meta: &GlobalMetadata,
        options: &EncodeOptions,
    ) -> Result<Self> {
        let resolved = options.aggregate_hash.resolved_streaming()?;
        let emit_footer_hash = options.hashing && resolved.emits_footer();

        let (preamble_bytes, header_frame_bytes) =
            build_preamble_and_header_bytes(global_meta, options.hashing, emit_footer_hash)?;

        let mut bytes_written = 0u64;
        writer.write_all(&preamble_bytes).await?;
        bytes_written += preamble_bytes.len() as u64;
        writer.write_all(&header_frame_bytes).await?;
        bytes_written += header_frame_bytes.len() as u64;
        write_padding(&mut writer, &mut bytes_written).await?;

        let intra_codec_threads = crate::parallel::resolve_budget(options.threads)?;

        Ok(Self {
            writer,
            object_offsets: Vec::new(),
            object_lengths: Vec::new(),
            hash_entries: Vec::new(),
            completed_objects: Vec::new(),
            bytes_written,
            hashing: options.hashing,
            emit_footer_hash_frame: emit_footer_hash,
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

    /// Write a `PrecederMetadata` frame for the next data object.
    ///
    /// Mirror of [`crate::streaming::StreamingEncoder::write_preceder`].
    pub async fn write_preceder(
        &mut self,
        metadata: BTreeMap<String, ciborium::Value>,
    ) -> Result<()> {
        if self.pending_preceder {
            return Err(TensogramError::Framing(
                "write_preceder called twice without an intervening write_object/write_object_pre_encoded".to_string(),
            ));
        }
        if metadata.contains_key(RESERVED_KEY) {
            return Err(TensogramError::Metadata(format!(
                "client code must not write '{RESERVED_KEY}' in preceder metadata; \
                     this field is populated by the library"
            )));
        }

        let preceder_meta = GlobalMetadata {
            base: vec![metadata.clone()],
            ..Default::default()
        };
        let cbor = metadata::global_metadata_to_cbor(&preceder_meta)?;
        let frame_bytes = build_frame(FrameType::PrecederMetadata, 1, 0, &cbor, self.hashing);
        self.writer.write_all(&frame_bytes).await?;
        self.bytes_written += frame_bytes.len() as u64;

        write_padding(&mut self.writer, &mut self.bytes_written).await?;

        self.pending_preceder = true;
        self.preceder_payloads.push(Some(metadata));
        Ok(())
    }

    /// Encode and write a single data object frame.
    ///
    /// Builds the frame in memory using the same helpers as the sync
    /// encoder, then issues a single async `write_all`.  The produced
    /// bytes are identical to those of
    /// [`crate::streaming::StreamingEncoder::write_object`] for the
    /// same inputs.
    ///
    /// Buffering one frame in memory is acceptable for the v1 producer
    /// scenario where individual frames are bounded; true chunked
    /// AsyncWrite streaming is a follow-up optimisation.
    pub async fn write_object(&mut self, desc: &DataObjectDescriptor, data: &[u8]) -> Result<()> {
        validate_object(desc, data.len())?;
        let num_elements = desc.num_elements()?;

        let parallel = crate::parallel::should_parallelise(
            self.intra_codec_threads,
            data.len(),
            self.parallel_threshold_bytes,
        );

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

        let mut final_desc = desc.clone();
        crate::encode::resolve_simple_packing_params(&mut final_desc, data)?;

        let config = crate::encode::build_pipeline_config_with_backend(
            &final_desc,
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

        self.write_object_inner(final_desc, &payload_region).await
    }

    /// Write a pre-encoded data object frame.
    ///
    /// Mirror of
    /// [`crate::streaming::StreamingEncoder::write_object_pre_encoded`].
    pub async fn write_object_pre_encoded(
        &mut self,
        descriptor: &DataObjectDescriptor,
        pre_encoded_bytes: &[u8],
    ) -> Result<()> {
        validate_object(descriptor, pre_encoded_bytes.len())?;
        let num_elements = descriptor.num_elements()?;
        build_pipeline_config(descriptor, num_elements, descriptor.dtype)?;
        validate_no_szip_offsets_for_non_szip(descriptor)?;
        if descriptor.compression == "szip" && descriptor.params.contains_key("szip_block_offsets")
        {
            validate_szip_block_offsets(&descriptor.params, pre_encoded_bytes.len())?;
        }
        self.write_object_inner(descriptor.clone(), pre_encoded_bytes)
            .await
    }

    async fn write_object_inner(
        &mut self,
        final_desc: DataObjectDescriptor,
        encoded_bytes: &[u8],
    ) -> Result<()> {
        let start_offset = self.bytes_written;

        // Build the full data-object frame in memory using the shared
        // helper, then async-write the buffer in one shot.  Bytes are
        // identical to the sync encoder's chunk-streamed path.
        // `cbor_before = false` matches the sync streaming layout
        // (CBOR_AFTER_PAYLOAD).
        let frame_bytes =
            encode_data_object_frame(&final_desc, encoded_bytes, false, self.hashing)?;
        let frame_len = frame_bytes.len() as u64;

        // Extract the inline hash slot (8 bytes at offset frame_len-12)
        // when hashing is on, so we can populate the FooterHash frame
        // later without re-hashing the body.
        let inline_digest = if self.hashing {
            let end = frame_bytes.len();
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&frame_bytes[end - 12..end - 4]);
            Some(u64::from_be_bytes(buf))
        } else {
            None
        };

        self.writer.write_all(&frame_bytes).await?;
        self.bytes_written += frame_len;

        let hash_entry = inline_digest.map(|d| {
            (
                crate::hash::HASH_ALGORITHM_NAME.to_string(),
                crate::hash::format_xxh3_digest(d),
            )
        });

        self.object_offsets.push(start_offset);
        self.object_lengths.push(frame_len);
        self.hash_entries.push(hash_entry);
        self.completed_objects.push(EncodedObject {
            descriptor: final_desc,
            encoded_payload: Vec::new(),
        });

        if self.pending_preceder {
            self.pending_preceder = false;
        } else {
            self.preceder_payloads.push(None);
        }

        write_padding(&mut self.writer, &mut self.bytes_written).await?;
        Ok(())
    }

    /// Finalise the streaming message.  Writes footer frames + postamble.
    pub async fn finish(mut self) -> Result<W> {
        self.write_footer_frames_and_postamble().await?;
        self.writer.flush().await?;
        Ok(self.writer)
    }

    async fn write_footer_frames_and_postamble(&mut self) -> Result<()> {
        if self.pending_preceder {
            return Err(TensogramError::Framing(
                "dangling PrecederMetadata: finish called without a following write_object/write_object_pre_encoded"
                    .to_string(),
            ));
        }

        let footer_start = self.bytes_written;

        // Footer metadata frame (preceder payloads merged in).
        {
            let mut enriched_meta = self.global_meta.clone();
            populate_base_entries(&mut enriched_meta.base, &self.completed_objects);
            populate_reserved_provenance(&mut enriched_meta.reserved);
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
            let frame_bytes =
                build_frame(FrameType::FooterMetadata, 1, 0, &meta_cbor, self.hashing);
            self.writer.write_all(&frame_bytes).await?;
            self.bytes_written += frame_bytes.len() as u64;
            write_padding(&mut self.writer, &mut self.bytes_written).await?;
        }

        // Footer hash frame (if opted in).
        if self.emit_footer_hash_frame && self.hash_entries.iter().any(|e| e.is_some()) {
            let algorithm = if self.hashing {
                crate::hash::HASH_ALGORITHM_NAME.to_string()
            } else {
                String::new()
            };
            let hashes: Vec<String> = self
                .hash_entries
                .iter()
                .map(|e| e.as_ref().map(|(_, v)| v.clone()).unwrap_or_default())
                .collect();
            let hash_frame = HashFrame { algorithm, hashes };
            let hash_cbor = metadata::hash_frame_to_cbor(&hash_frame)?;
            let frame_bytes = build_frame(FrameType::FooterHash, 1, 0, &hash_cbor, self.hashing);
            self.writer.write_all(&frame_bytes).await?;
            self.bytes_written += frame_bytes.len() as u64;
            write_padding(&mut self.writer, &mut self.bytes_written).await?;
        }

        // Footer index frame.
        let index = IndexFrame {
            offsets: std::mem::take(&mut self.object_offsets),
            lengths: std::mem::take(&mut self.object_lengths),
        };
        let index_cbor = metadata::index_to_cbor(&index)?;
        let frame_bytes = build_frame(FrameType::FooterIndex, 1, 0, &index_cbor, self.hashing);
        self.writer.write_all(&frame_bytes).await?;
        self.bytes_written += frame_bytes.len() as u64;
        write_padding(&mut self.writer, &mut self.bytes_written).await?;

        // Postamble (streaming mode: total_length = 0).
        let postamble = Postamble {
            first_footer_offset: footer_start,
            total_length: 0,
        };
        let mut postamble_bytes = Vec::with_capacity(POSTAMBLE_SIZE);
        postamble.write_to(&mut postamble_bytes);
        self.writer.write_all(&postamble_bytes).await?;
        self.bytes_written += postamble_bytes.len() as u64;

        Ok(())
    }

    /// Number of data objects written so far.
    pub fn object_count(&self) -> usize {
        self.object_offsets.len()
    }

    /// Total bytes written so far.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
}

// ── Seekable-sink specialisation ─────────────────────────────────────────────

impl<W: AsyncWrite + tokio::io::AsyncSeek + Unpin> AsyncStreamingEncoder<W> {
    /// Finalise and back-fill the `total_length` field in both the
    /// preamble and postamble.
    ///
    /// Mirror of
    /// [`crate::streaming::StreamingEncoder::finish_with_backfill`]
    /// for seekable async sinks (e.g. `tokio::fs::File`).  Object-store
    /// multipart sinks cannot seek; for those use [`Self::finish`]
    /// instead and accept `total_length = 0` (forward-scan only).
    pub async fn finish_with_backfill(mut self) -> Result<W> {
        use std::io::SeekFrom;
        self.write_footer_frames_and_postamble().await?;

        let end_pos = self.writer.stream_position().await?;
        let total_length = end_pos;

        self.writer.seek(SeekFrom::Start(16)).await?;
        self.writer.write_all(&total_length.to_be_bytes()).await?;

        self.writer.seek(SeekFrom::Start(end_pos - 16)).await?;
        self.writer.write_all(&total_length.to_be_bytes()).await?;

        self.writer.seek(SeekFrom::Start(end_pos)).await?;
        self.writer.flush().await?;
        Ok(self.writer)
    }
}

// ── Async padding helper ────────────────────────────────────────────────────

const ZERO_PAD: [u8; 7] = [0; 7];

async fn write_padding<W: AsyncWrite + Unpin>(
    writer: &mut W,
    bytes_written: &mut u64,
) -> std::io::Result<()> {
    let pad = padding_for(*bytes_written);
    if pad > 0 {
        writer.write_all(&ZERO_PAD[..pad]).await?;
        *bytes_written += pad as u64;
    }
    Ok(())
}
