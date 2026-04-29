// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

//! Iterator types for lazy traversal of messages and objects.
//!
//! # Quick start
//!
//! ```ignore
//! // Iterate over messages in a byte buffer (zero-copy)
//! for msg in tensogram::iter::messages(&buf) {
//!     let (meta, objs) = tensogram::decode(msg, &Default::default())?;
//! }
//!
//! // Iterate over objects in a single message
//! for result in tensogram::iter::objects(&msg_bytes, Default::default())? {
//!     let (descriptor, data) = result?;
//! }
//!
//! // File-based lazy iteration
//! let mut file = TensogramFile::open("data.tgm")?;
//! for raw in file.iter()? {
//!     let raw = raw?;
//! }
//! ```

use std::path::PathBuf;

use crate::decode::DecodeOptions;
use crate::encode::build_pipeline_config;
use crate::error::Result;
use crate::framing;
use crate::types::DataObjectDescriptor;

/// Create a zero-copy iterator over messages in a byte buffer.
///
/// Calls [`framing::scan`] once to locate all message boundaries, then yields
/// `&[u8]` slices pointing into the original buffer on each `next()` call.
/// Garbage between valid messages is silently skipped.
pub fn messages(buf: &[u8]) -> MessageIter<'_> {
    let offsets = framing::scan(buf);
    MessageIter {
        buf,
        offsets,
        pos: 0,
    }
}

/// Create an iterator that decodes each object in a message on demand.
///
/// Parses the frame header and metadata once, then decodes objects lazily via
/// the full pipeline (encoding + filter + decompression).
pub fn objects(buf: &[u8], options: DecodeOptions) -> Result<ObjectIter> {
    let msg = framing::decode_message(buf)?;
    let object_data: Vec<(DataObjectDescriptor, Vec<u8>, Vec<u8>)> = msg
        .objects
        .into_iter()
        .map(|(desc, payload, mask_region, _)| (desc, payload.to_vec(), mask_region.to_vec()))
        .collect();
    Ok(ObjectIter {
        objects: object_data,
        index: 0,
        options,
    })
}

/// Return an iterator over the [`DataObjectDescriptor`]s in a message without
/// decoding any payload data.
pub fn objects_metadata(buf: &[u8]) -> Result<impl Iterator<Item = DataObjectDescriptor> + use<>> {
    let msg = framing::decode_message(buf)?;
    Ok(msg
        .objects
        .into_iter()
        .map(|(desc, _, _, _)| desc)
        .collect::<Vec<_>>()
        .into_iter())
}

// ── MessageIter ──────────────────────────────────────────────────────────────

/// Zero-copy iterator over messages in a byte buffer.
///
/// Yields `&[u8]` slices pointing into the original buffer.
/// Implements [`ExactSizeIterator`] because all boundaries are known after the
/// initial scan.
pub struct MessageIter<'a> {
    buf: &'a [u8],
    offsets: Vec<(usize, usize)>,
    pos: usize,
}

impl<'a> Iterator for MessageIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.offsets.len() {
            return None;
        }
        let (offset, length) = self.offsets[self.pos];
        self.pos += 1;
        Some(&self.buf[offset..offset + length])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.offsets.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for MessageIter<'_> {}

// ── ObjectIter ───────────────────────────────────────────────────────────────

/// Iterator over the decoded objects (tensors) in a single message.
///
/// Decodes each object through the full pipeline on demand.
/// Yields `Result<(DataObjectDescriptor, Vec<u8>)>`.
/// Implements [`ExactSizeIterator`].
pub struct ObjectIter {
    objects: Vec<(DataObjectDescriptor, Vec<u8>, Vec<u8>)>,
    index: usize,
    options: DecodeOptions,
}

impl Iterator for ObjectIter {
    type Item = Result<(DataObjectDescriptor, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.objects.len() {
            return None;
        }
        let i = self.index;
        self.index += 1;
        let (ref desc, ref payload_bytes, ref mask_region) = self.objects[i];

        // v3: hash verification moved to frame-level (see the inline
        // slot in `plans/WIRE_FORMAT.md` §2.4).  `options.verify_hash`
        // is retained on the public API for source compatibility
        // but a full-iter caller that wants integrity checks should
        // go through `validate --checksum`.
        let _ = (self.options.verify_hash, desc, payload_bytes);

        let num_elements = match desc.num_elements() {
            Ok(n) => n,
            Err(e) => return Some(Err(e)),
        };

        let config = match build_pipeline_config(desc, num_elements, desc.dtype) {
            Ok(c) => c,
            Err(e) => return Some(Err(e)),
        };

        let mut decoded = match tensogram_encodings::pipeline::decode_pipeline(
            payload_bytes,
            &config,
            self.options.native_byte_order,
        ) {
            Ok(d) => d,
            Err(e) => return Some(Err(crate::error::TensogramError::Encoding(e.to_string()))),
        };

        if self.options.restore_non_finite {
            let output_byte_order = if self.options.native_byte_order {
                tensogram_encodings::ByteOrder::native()
            } else {
                desc.byte_order
            };
            if let Err(e) = crate::restore::restore_non_finite_into(
                &mut decoded,
                desc,
                mask_region,
                output_byte_order,
            ) {
                return Some(Err(e));
            }
        }

        Some(Ok((desc.clone(), decoded)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.objects.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ObjectIter {}

// ── FileMessageIter ──────────────────────────────────────────────────────────

/// Lazy iterator over messages stored in a file.
///
/// Holds a persistent file handle and seeks to each message offset on demand,
/// avoiding both full-file reads and repeated open/close syscalls.
/// Constructed via [`TensogramFile::iter`].
///
/// [`TensogramFile::iter`]: crate::file::TensogramFile::iter
pub struct FileMessageIter {
    file: std::fs::File,
    offsets: Vec<(usize, usize)>,
    pos: usize,
}

impl FileMessageIter {
    pub(crate) fn new(path: PathBuf, offsets: Vec<(usize, usize)>) -> Result<Self> {
        let file = std::fs::File::open(&path)?;
        Ok(Self {
            file,
            offsets,
            pos: 0,
        })
    }
}

impl Iterator for FileMessageIter {
    type Item = Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        use std::io::{Read, Seek, SeekFrom};

        if self.pos >= self.offsets.len() {
            return None;
        }
        let (offset, length) = self.offsets[self.pos];
        self.pos += 1;

        let result = (|| {
            self.file.seek(SeekFrom::Start(offset as u64))?;
            let mut buf = vec![0u8; length];
            self.file.read_exact(&mut buf)?;
            Ok(buf)
        })();
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.offsets.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for FileMessageIter {}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::DecodeOptions;
    use crate::dtype::Dtype;
    use crate::encode::{EncodeOptions, encode};
    use crate::types::{ByteOrder, DataObjectDescriptor, GlobalMetadata};
    use std::collections::BTreeMap;

    fn make_global_meta() -> GlobalMetadata {
        GlobalMetadata {
            extra: BTreeMap::new(),
            ..Default::default()
        }
    }

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
            byte_order: ByteOrder::Little,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            masks: None,
        }
    }

    fn encode_msg(shape: Vec<u64>, fill: u8) -> Vec<u8> {
        let n: usize = shape.iter().product::<u64>() as usize * 4;
        let data = vec![fill; n];
        let meta = make_global_meta();
        let desc = make_descriptor(shape);
        encode(
            &meta,
            &[(&desc, &data)],
            &EncodeOptions {
                hash_algorithm: None,
                ..Default::default()
            },
        )
        .unwrap()
    }

    // ── MessageIter ──

    #[test]
    fn test_message_iter_empty_buffer() {
        let buf = vec![];
        let mut it = messages(&buf);
        assert_eq!(it.len(), 0);
        assert!(it.next().is_none());
    }

    #[test]
    fn test_message_iter_single_message() {
        let msg = encode_msg(vec![4], 1);
        let collected: Vec<&[u8]> = messages(&msg).collect();
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0], msg.as_slice());
    }

    #[test]
    fn test_message_iter_multiple_messages() {
        let msg0 = encode_msg(vec![4], 0);
        let msg1 = encode_msg(vec![4], 1);
        let msg2 = encode_msg(vec![4], 2);
        let mut buf = msg0.clone();
        buf.extend_from_slice(&msg1);
        buf.extend_from_slice(&msg2);

        let collected: Vec<&[u8]> = messages(&buf).collect();
        assert_eq!(collected.len(), 3);
    }

    #[test]
    fn test_message_iter_with_garbage() {
        let msg0 = encode_msg(vec![4], 0);
        let msg1 = encode_msg(vec![4], 1);
        let mut buf = vec![0xDE, 0xAD, 0xBE, 0xEF];
        buf.extend_from_slice(&msg0);
        buf.extend_from_slice(&[0xFF, 0xFF]);
        buf.extend_from_slice(&msg1);
        let collected: Vec<&[u8]> = messages(&buf).collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn test_message_iter_yields_decodable_slices() {
        let msg0 = encode_msg(vec![3], 0xAB);
        let msg1 = encode_msg(vec![5], 0xCD);
        let mut buf = msg0;
        buf.extend_from_slice(&msg1);

        for (i, slice) in messages(&buf).enumerate() {
            let (_meta, objs) = crate::decode::decode(slice, &DecodeOptions::default()).unwrap();
            let expected_shape = if i == 0 { vec![3u64] } else { vec![5u64] };
            assert_eq!(objs[0].0.shape, expected_shape);
        }
    }

    // ── ObjectIter ──

    #[test]
    fn test_object_iter_zero_objects() {
        let meta = make_global_meta();
        let msg = encode(
            &meta,
            &[],
            &EncodeOptions {
                hash_algorithm: None,
                ..Default::default()
            },
        )
        .unwrap();
        let mut it = objects(&msg, DecodeOptions::default()).unwrap();
        assert_eq!(it.len(), 0);
        assert!(it.next().is_none());
    }

    #[test]
    fn test_object_iter_single_object() {
        let msg = encode_msg(vec![4], 42);
        let collected: Vec<_> = objects(&msg, DecodeOptions::default()).unwrap().collect();
        assert_eq!(collected.len(), 1);
        let (desc, data) = collected[0].as_ref().unwrap();
        assert_eq!(desc.shape, vec![4u64]);
        assert_eq!(data.len(), 16);
        assert_eq!(data, &vec![42u8; 16]);
    }

    #[test]
    fn test_object_iter_multi_object() {
        let shape = vec![4u64];
        let data0 = vec![0u8; 16];
        let data1 = vec![1u8; 16];
        let meta = make_global_meta();
        let desc0 = make_descriptor(shape.clone());
        let desc1 = make_descriptor(shape.clone());

        let msg = encode(
            &meta,
            &[(&desc0, data0.as_slice()), (&desc1, data1.as_slice())],
            &EncodeOptions {
                hash_algorithm: None,
                ..Default::default()
            },
        )
        .unwrap();
        let mut it = objects(&msg, DecodeOptions::default()).unwrap();
        assert_eq!(it.len(), 2);
        let (d0_desc, d0) = it.next().unwrap().unwrap();
        assert_eq!(d0_desc.shape, shape);
        assert_eq!(d0, data0);
        let (d1_desc, d1) = it.next().unwrap().unwrap();
        assert_eq!(d1_desc.shape, shape);
        assert_eq!(d1, data1);
        assert_eq!(it.len(), 0);
        assert!(it.next().is_none());
    }

    #[test]
    fn test_objects_metadata_only() {
        let msg = encode_msg(vec![3, 4], 7);
        let descs: Vec<DataObjectDescriptor> = objects_metadata(&msg).unwrap().collect();
        assert_eq!(descs.len(), 1);
        assert_eq!(descs[0].shape, vec![3u64, 4u64]);
        assert_eq!(descs[0].dtype, Dtype::Float32);
    }

    // ── FileMessageIter ──

    #[test]
    fn test_file_iter_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.tgm");
        std::fs::write(&path, []).unwrap();
        let it = FileMessageIter::new(path, vec![]).unwrap();
        assert_eq!(it.len(), 0);
        assert_eq!(it.collect::<Vec<_>>().len(), 0);
    }

    #[test]
    fn test_file_iter_three_messages() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("three.tgm");

        let msg0 = encode_msg(vec![4], 0);
        let msg1 = encode_msg(vec![4], 1);
        let msg2 = encode_msg(vec![4], 2);
        let mut content = msg0.clone();
        content.extend_from_slice(&msg1);
        content.extend_from_slice(&msg2);
        std::fs::write(&path, &content).unwrap();

        let offsets = framing::scan(&content);
        let it = FileMessageIter::new(path, offsets).unwrap();
        assert_eq!(it.len(), 3);
        let collected: Vec<_> = it.collect();
        assert_eq!(collected[0].as_ref().unwrap(), &msg0);
        assert_eq!(collected[1].as_ref().unwrap(), &msg1);
        assert_eq!(collected[2].as_ref().unwrap(), &msg2);
    }

    #[test]
    fn test_file_iter_each_decodable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("decode.tgm");

        let msgs: Vec<Vec<u8>> = (0u8..3).map(|fill| encode_msg(vec![2], fill)).collect();
        let content: Vec<u8> = msgs.iter().flatten().copied().collect();
        std::fs::write(&path, &content).unwrap();

        let offsets = framing::scan(&content);
        for raw in FileMessageIter::new(path, offsets).unwrap() {
            let raw = raw.unwrap();
            let (_meta, _) = crate::decode::decode(&raw, &DecodeOptions::default()).unwrap();
        }
    }
}
