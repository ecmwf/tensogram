use std::io::Write;

use crate::encode::{build_pipeline_config, validate_object, EncodeOptions};
use crate::error::{Result, TensogramError};
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
    /// Total bytes written so far.
    bytes_written: u64,
    /// Hash algorithm to use for payload integrity.
    hash_algorithm: Option<HashAlgorithm>,
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

        // Streaming preamble: total_length=0 signals unknown length at write time
        let mut flags = MessageFlags::default();
        flags.set(MessageFlags::HEADER_METADATA);
        flags.set(MessageFlags::FOOTER_INDEX);
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
            bytes_written,
            hash_algorithm: options.hash_algorithm,
        })
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

        // Write frame
        self.writer.write_all(&frame_bytes)?;
        self.bytes_written += frame_bytes.len() as u64;

        // Align to 8 bytes
        write_padding(&mut self.writer, &mut self.bytes_written)?;

        Ok(())
    }

    /// Finalize the streaming message.
    ///
    /// Writes footer frames (hash + index) and the postamble.
    /// Consumes the encoder and returns the underlying writer.
    pub fn finish(mut self) -> Result<W> {
        let footer_start = self.bytes_written;

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
}
