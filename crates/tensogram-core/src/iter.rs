//! Iterator types for lazy traversal of messages and objects.
//!
//! # Quick start
//!
//! ```ignore
//! // Iterate over messages in a byte buffer (zero-copy)
//! for msg in tensogram_core::iter::messages(&buf) {
//!     let (meta, _) = tensogram_core::decode(msg, &Default::default())?;
//! }
//!
//! // Iterate over objects in a single message
//! for result in tensogram_core::iter::objects(&msg_bytes, Default::default())? {
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

use crate::decode::{decode_metadata, decode_object, DecodeOptions};
use crate::error::Result;
use crate::framing;
use crate::types::{Metadata, ObjectDescriptor};

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
/// Parses the metadata header once, then decodes objects lazily via the full
/// pipeline (encoding + filter + decompression).
pub fn objects(buf: &[u8], options: DecodeOptions) -> Result<ObjectIter> {
    let metadata = decode_metadata(buf)?;
    Ok(ObjectIter {
        buf: buf.to_vec(),
        metadata,
        index: 0,
        options,
    })
}

/// Return an iterator over the [`ObjectDescriptor`]s in a message without
/// decoding any payload data.
pub fn objects_metadata(buf: &[u8]) -> Result<impl Iterator<Item = ObjectDescriptor>> {
    let metadata = decode_metadata(buf)?;
    Ok(metadata.objects.into_iter())
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
/// Yields `Result<(ObjectDescriptor, Vec<u8>)>`.
/// Implements [`ExactSizeIterator`].
pub struct ObjectIter {
    buf: Vec<u8>,
    metadata: Metadata,
    index: usize,
    options: DecodeOptions,
}

impl Iterator for ObjectIter {
    type Item = Result<(ObjectDescriptor, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.metadata.objects.len() {
            return None;
        }
        let i = self.index;
        self.index += 1;
        let descriptor = self.metadata.objects[i].clone();
        Some(decode_object(&self.buf, i, &self.options).map(|(_, data)| (descriptor, data)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.metadata.objects.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ObjectIter {}

// ── FileMessageIter ──────────────────────────────────────────────────────────

/// Lazy iterator over messages stored in a file.
///
/// Reads each message from disk on demand using seek + read, avoiding loading
/// the entire file into memory. Constructed via [`TensogramFile::iter`].
///
/// [`TensogramFile::iter`]: crate::file::TensogramFile::iter
pub struct FileMessageIter {
    path: PathBuf,
    offsets: Vec<(usize, usize)>,
    pos: usize,
}

impl FileMessageIter {
    pub(crate) fn new(path: PathBuf, offsets: Vec<(usize, usize)>) -> Self {
        Self {
            path,
            offsets,
            pos: 0,
        }
    }
}

impl Iterator for FileMessageIter {
    type Item = Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.offsets.len() {
            return None;
        }
        let (offset, length) = self.offsets[self.pos];
        self.pos += 1;
        Some(read_bytes_at(&self.path, offset, length))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.offsets.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for FileMessageIter {}

fn read_bytes_at(path: &std::path::Path, offset: usize, length: usize) -> Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path)?;
    file.seek(SeekFrom::Start(offset as u64))?;
    let mut buf = vec![0u8; length];
    file.read_exact(&mut buf)?;
    Ok(buf)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::decode::DecodeOptions;
    use crate::dtype::Dtype;
    use crate::encode::{encode, EncodeOptions};
    use crate::types::{ByteOrder, ObjectDescriptor, PayloadDescriptor};

    fn make_meta_1obj(shape: Vec<u64>) -> crate::types::Metadata {
        let strides = {
            let mut s = vec![1u64; shape.len()];
            for i in (0..shape.len().saturating_sub(1)).rev() {
                s[i] = s[i + 1] * shape[i + 1];
            }
            s
        };
        crate::types::Metadata {
            version: 1,
            objects: vec![ObjectDescriptor {
                obj_type: "ntensor".to_string(),
                ndim: shape.len() as u64,
                shape: shape.clone(),
                strides,
                dtype: Dtype::Float32,
                extra: BTreeMap::new(),
            }],
            payload: vec![PayloadDescriptor {
                byte_order: ByteOrder::Little,
                encoding: "none".to_string(),
                filter: "none".to_string(),
                compression: "none".to_string(),
                params: BTreeMap::new(),
                hash: None,
            }],
            extra: BTreeMap::new(),
        }
    }

    fn encode_msg(shape: Vec<u64>, fill: u8) -> Vec<u8> {
        let n: usize = shape.iter().product::<u64>() as usize * 4; // float32 = 4 bytes
        let data = vec![fill; n];
        let meta = make_meta_1obj(shape);
        encode(
            &meta,
            &[&data],
            &EncodeOptions {
                hash_algorithm: None,
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
        assert_eq!(collected[0], msg0.as_slice());
        assert_eq!(collected[1], msg1.as_slice());
        assert_eq!(collected[2], msg2.as_slice());
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
    fn test_message_iter_partial_message_skipped() {
        let msg = encode_msg(vec![4], 0);
        let truncated = &msg[..msg.len() / 2];
        let collected: Vec<&[u8]> = messages(truncated).collect();
        assert_eq!(collected.len(), 0);
    }

    #[test]
    fn test_message_iter_exact_size_decreases() {
        let msg0 = encode_msg(vec![4], 0);
        let msg1 = encode_msg(vec![4], 1);
        let mut buf = msg0;
        buf.extend_from_slice(&msg1);

        let mut it = messages(&buf);
        assert_eq!(it.len(), 2);
        it.next();
        assert_eq!(it.len(), 1);
        it.next();
        assert_eq!(it.len(), 0);
        assert!(it.next().is_none());
    }

    #[test]
    fn test_message_iter_yields_decodable_slices() {
        let msg0 = encode_msg(vec![3], 0xAB);
        let msg1 = encode_msg(vec![5], 0xCD);
        let mut buf = msg0;
        buf.extend_from_slice(&msg1);

        for (i, slice) in messages(&buf).enumerate() {
            let (meta, _) = crate::decode::decode(slice, &DecodeOptions::default()).unwrap();
            assert_eq!(meta.version, 1);
            let expected_shape = if i == 0 { vec![3u64] } else { vec![5u64] };
            assert_eq!(meta.objects[0].shape, expected_shape);
        }
    }

    // ── ObjectIter ──

    #[test]
    fn test_object_iter_zero_objects() {
        let meta = crate::types::Metadata {
            version: 1,
            objects: vec![],
            payload: vec![],
            extra: BTreeMap::new(),
        };
        let msg = encode(
            &meta,
            &[],
            &EncodeOptions {
                hash_algorithm: None,
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
        assert_eq!(data.len(), 16); // 4 × float32
        assert_eq!(data, &vec![42u8; 16]);
    }

    #[test]
    fn test_object_iter_multi_object() {
        let shape = vec![4u64];
        let strides = vec![1u64];
        let data0 = vec![0u8; 16];
        let data1 = vec![1u8; 16];
        let meta = crate::types::Metadata {
            version: 1,
            objects: vec![
                ObjectDescriptor {
                    obj_type: "ntensor".to_string(),
                    ndim: 1,
                    shape: shape.clone(),
                    strides: strides.clone(),
                    dtype: Dtype::Float32,
                    extra: BTreeMap::new(),
                },
                ObjectDescriptor {
                    obj_type: "ntensor".to_string(),
                    ndim: 1,
                    shape: shape.clone(),
                    strides: strides.clone(),
                    dtype: Dtype::Float32,
                    extra: BTreeMap::new(),
                },
            ],
            payload: vec![
                PayloadDescriptor {
                    byte_order: ByteOrder::Little,
                    encoding: "none".to_string(),
                    filter: "none".to_string(),
                    compression: "none".to_string(),
                    params: BTreeMap::new(),
                    hash: None,
                },
                PayloadDescriptor {
                    byte_order: ByteOrder::Little,
                    encoding: "none".to_string(),
                    filter: "none".to_string(),
                    compression: "none".to_string(),
                    params: BTreeMap::new(),
                    hash: None,
                },
            ],
            extra: BTreeMap::new(),
        };
        let msg = encode(
            &meta,
            &[&data0, &data1],
            &EncodeOptions {
                hash_algorithm: None,
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
        let descs: Vec<ObjectDescriptor> = objects_metadata(&msg).unwrap().collect();
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
        let it = FileMessageIter::new(path, vec![]);
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
        let it = FileMessageIter::new(path, offsets);
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
        for raw in FileMessageIter::new(path, offsets) {
            let raw = raw.unwrap();
            let (meta, _) = crate::decode::decode(&raw, &DecodeOptions::default()).unwrap();
            assert_eq!(meta.version, 1);
        }
    }
}
