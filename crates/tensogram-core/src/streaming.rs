// (C) Copyright 2024- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::collections::BTreeMap;
use std::io::Write;

use crate::encode::{
    build_pipeline_config, populate_base_entries, populate_reserved_provenance,
    validate_no_szip_offsets_for_non_szip, validate_object, validate_szip_block_offsets,
    EncodeOptions,
};
use crate::error::{Result, TensogramError};
use crate::framing::EncodedObject;
use crate::hash::{compute_hash, HashAlgorithm};
use crate::metadata::{self, RESERVED_KEY};
use crate::types::{DataObjectDescriptor, GlobalMetadata, HashDescriptor, HashFrame, IndexFrame};
use crate::wire::{
    FrameHeader, FrameType, MessageFlags, Postamble, Preamble, FRAME_END, FRAME_HEADER_SIZE,
    PREAMBLE_SIZE,
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
/// use tensogram_core::streaming::StreamingEncoder;
/// use tensogram_core::{GlobalMetadata, EncodeOptions};
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
            version: 2,
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
    pub fn write_object(&mut self, desc: &DataObjectDescriptor, data: &[u8]) -> Result<()> {
        validate_object(desc, data.len())?;

        let shape_product = desc
            .shape
            .iter()
            .try_fold(1u64, |acc, &x| acc.checked_mul(x))
            .ok_or_else(|| TensogramError::Metadata("shape product overflow".to_string()))?;
        let num_elements = usize::try_from(shape_product)
            .map_err(|_| TensogramError::Metadata("element count overflows usize".to_string()))?;

        let config = build_pipeline_config(desc, num_elements, desc.dtype)?;
        let result = pipeline::encode_pipeline(data, &config)
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

        self.write_object_inner(final_desc, &result.encoded_bytes)
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
    /// Computes the hash, builds the data object frame, updates all bookkeeping,
    /// consumes any pending preceder, and writes the frame to the stream.
    fn write_object_inner(
        &mut self,
        mut final_desc: DataObjectDescriptor,
        encoded_bytes: &[u8],
    ) -> Result<()> {
        // Compute hash
        let hash_entry = if let Some(algorithm) = self.hash_algorithm {
            let hash_value = compute_hash(encoded_bytes, algorithm);
            let hash_type = algorithm.as_str().to_string();
            let entry = Some((hash_type.clone(), hash_value.clone()));
            final_desc.hash = Some(HashDescriptor {
                hash_type,
                value: hash_value,
            });
            entry
        } else {
            None
        };

        // Build the data object frame bytes
        let frame_bytes =
            crate::framing::encode_data_object_frame(&final_desc, encoded_bytes, false)?;

        self.object_offsets.push(self.bytes_written);
        self.object_lengths.push(frame_bytes.len() as u64);
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

        // Write frame
        self.writer.write_all(&frame_bytes)?;
        self.bytes_written += frame_bytes.len() as u64;

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
                if let Some(prec_map) = prec {
                    if i < enriched_meta.base.len() {
                        for (k, v) in prec_map {
                            enriched_meta.base[i].insert(k.clone(), v.clone());
                        }
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

        // Postamble
        let postamble = Postamble {
            first_footer_offset: footer_start,
        };
        let mut postamble_bytes = Vec::with_capacity(16);
        postamble.write_to(&mut postamble_bytes);
        self.writer.write_all(&postamble_bytes)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{decode, DecodeOptions};
    use crate::encode::{encode, EncodeOptions};
    use crate::types::{ByteOrder, DataObjectDescriptor};
    use crate::Dtype;
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
        assert_eq!(decoded_meta.version, 2);
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
            compression_backend: Default::default(),
            hash_algorithm: Some(HashAlgorithm::Xxh3),
            emit_preceders: false,
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
            compression_backend: Default::default(),
            hash_algorithm: Some(HashAlgorithm::Xxh3),
            emit_preceders: false,
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
            compression_backend: Default::default(),
            hash_algorithm: None,
            emit_preceders: false,
        };

        let buf = Vec::new();
        let enc = StreamingEncoder::new(buf, &meta, &options).unwrap();
        assert_eq!(enc.object_count(), 0);
        let result = enc.finish().unwrap();

        let (decoded_meta, objects) = decode(&result, &DecodeOptions::default()).unwrap();
        assert_eq!(decoded_meta.version, 2);
        assert_eq!(objects.len(), 0);
    }

    #[test]
    fn streaming_with_metadata() {
        let mut extra = BTreeMap::new();
        extra.insert(
            "centre".to_string(),
            ciborium::Value::Text("ecmwf".to_string()),
        );
        let meta = GlobalMetadata {
            version: 2,
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
            version: 2,
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
            version: 2,
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
            version: 2,
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

        // Postamble
        let postamble_offset = out.len();
        let postamble = Postamble {
            first_footer_offset: postamble_offset as u64,
        };
        postamble.write_to(&mut out);

        // Patch preamble
        let total_length = out.len() as u64;
        let mut flags = MessageFlags::default();
        flags.set(MessageFlags::HEADER_METADATA);
        flags.set(MessageFlags::FOOTER_METADATA);
        flags.set(MessageFlags::PRECEDER_METADATA);
        let preamble = Preamble {
            version: 2,
            flags,
            reserved: 0,
            total_length,
        };
        let mut preamble_bytes = Vec::new();
        preamble.write_to(&mut preamble_bytes);
        out[0..PREAMBLE_SIZE].copy_from_slice(&preamble_bytes);

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
