use std::collections::BTreeMap;
use std::io::Write;

use crate::encode::{
    build_pipeline_config, populate_payload_entries, validate_object, EncodeOptions,
};
use crate::error::{Result, TensogramError};
use crate::framing::EncodedObject;
use crate::hash::{compute_hash, HashAlgorithm};
use crate::metadata;
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
    /// Encoded payload length of each data object frame.
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
    /// The `metadata` map becomes `payload[0]` in a `GlobalMetadata` CBOR
    /// with `common` empty.  Must be followed by exactly one
    /// [`write_object`](Self::write_object) call before another
    /// `write_preceder` or [`finish`](Self::finish).
    pub fn write_preceder(&mut self, metadata: BTreeMap<String, ciborium::Value>) -> Result<()> {
        if self.pending_preceder {
            return Err(TensogramError::Framing(
                "write_preceder called twice without an intervening write_object".to_string(),
            ));
        }

        let preceder_meta = GlobalMetadata {
            version: self.global_meta.version,
            payload: vec![metadata.clone()],
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

        let num_elements = usize::try_from(desc.shape.iter().product::<u64>())
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

        // Compute hash
        let hash_entry = if let Some(algorithm) = self.hash_algorithm {
            let hash_value = compute_hash(&result.encoded_bytes, algorithm);
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
            crate::framing::encode_data_object_frame(&final_desc, &result.encoded_bytes, false)?;

        // Record offset before writing
        self.object_offsets.push(self.bytes_written);
        self.object_lengths.push(result.encoded_bytes.len() as u64);
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
                "dangling PrecederMetadata: finish called without a following write_object"
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
            populate_payload_entries(&mut enriched_meta.payload, &self.completed_objects);

            // Merge preceder payloads into footer metadata (preceder wins)
            for (i, prec) in self.preceder_payloads.iter().enumerate() {
                if let Some(prec_map) = prec {
                    if i < enriched_meta.payload.len() {
                        for (k, v) in prec_map {
                            enriched_meta.payload[i].insert(k.clone(), v.clone());
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
            byte_order: ByteOrder::Big,
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
            hash_algorithm: Some(HashAlgorithm::Xxh3),
            emit_preceders: false,
        };

        // Buffered encode
        let buffered = encode(&meta, &[(&desc, &data)], &options).unwrap();
        let (buf_meta, buf_objects) =
            decode(&buffered, &DecodeOptions { verify_hash: true }).unwrap();

        // Streaming encode
        let buf = Vec::new();
        let mut enc = StreamingEncoder::new(buf, &meta, &options).unwrap();
        enc.write_object(&desc, &data).unwrap();
        let streamed = enc.finish().unwrap();
        let (str_meta, str_objects) =
            decode(&streamed, &DecodeOptions { verify_hash: true }).unwrap();

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
            emit_preceders: false,
        };

        let buf = Vec::new();
        let mut enc = StreamingEncoder::new(buf, &meta, &options).unwrap();
        enc.write_object(&desc, &data).unwrap();
        let result = enc.finish().unwrap();

        // Verify hash passes
        let verify_opts = DecodeOptions { verify_hash: true };
        let (_, objects) = decode(&result, &verify_opts).unwrap();
        assert!(objects[0].0.hash.is_some());
    }

    #[test]
    fn streaming_no_objects() {
        let meta = GlobalMetadata::default();
        let options = EncodeOptions {
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
        let mut common = BTreeMap::new();
        common.insert(
            "centre".to_string(),
            ciborium::Value::Text("ecmwf".to_string()),
        );
        let meta = GlobalMetadata {
            version: 2,
            common,
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
            decoded_meta.common.get("centre"),
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

        // Preceder mars keys should be in payload[0]
        let mars = decoded_meta.payload[0].get("mars");
        assert!(mars.is_some(), "mars key should be in payload[0]");
    }

    #[test]
    fn streaming_preceder_wins_over_footer() {
        // Pre-populate global_meta.payload[0] with a value — the preceder
        // should override it after decode.
        let mut footer_payload = BTreeMap::new();
        footer_payload.insert(
            "source".to_string(),
            ciborium::Value::Text("footer".to_string()),
        );
        let meta = GlobalMetadata {
            version: 2,
            payload: vec![footer_payload],
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
        let source = decoded_meta.payload[0].get("source").and_then(|v| match v {
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

        // Payload[0] should have preceder entry
        assert!(decoded_meta.payload[0].contains_key("note"));
        // Payload[1] should NOT have it
        assert!(!decoded_meta.payload[1].contains_key("note"));
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
        let p = &decoded_meta.payload[0];
        assert_eq!(
            p.get("units"),
            Some(&ciborium::Value::Text("K".to_string()))
        );
        assert!(p.contains_key("mars"));
        // Structural keys (ndim, shape) should also be present from footer
        assert!(p.contains_key("ndim"));
        assert!(p.contains_key("shape"));
    }
}
