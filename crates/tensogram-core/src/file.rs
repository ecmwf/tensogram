use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::decode::{self, DecodeOptions};
use crate::encode::{self, EncodeOptions};
use crate::error::{Result, TensogramError};
use crate::framing;
use crate::types::Metadata;

/// A handle for reading/writing Tensogram message files.
pub struct TensogramFile {
    path: PathBuf,
    /// Cached message offsets (offset, length) from scan.
    message_offsets: Option<Vec<(usize, usize)>>,
}

impl TensogramFile {
    /// Open an existing file for reading.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(TensogramError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("file not found: {}", path.display()),
            )));
        }
        Ok(TensogramFile {
            path,
            message_offsets: None,
        })
    }

    /// Create a new file for writing (truncates if exists).
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        // Create parent directories if needed
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        // Create/truncate the file
        fs::File::create(&path)?;
        Ok(TensogramFile {
            path,
            message_offsets: None,
        })
    }

    /// Scan the file for message boundaries. Caches results.
    fn ensure_scanned(&mut self) -> Result<()> {
        if self.message_offsets.is_some() {
            return Ok(());
        }
        let data = fs::read(&self.path)?;
        let offsets = framing::scan(&data);
        self.message_offsets = Some(offsets);
        Ok(())
    }

    /// Count messages without decoding them.
    pub fn message_count(&mut self) -> Result<usize> {
        self.ensure_scanned()?;
        Ok(self
            .message_offsets
            .as_ref()
            .expect("message_offsets guaranteed after ensure_scanned() succeeds")
            .len())
    }

    /// Append a message to the file.
    pub fn append(
        &mut self,
        metadata: &Metadata,
        data_objects: &[&[u8]],
        options: &EncodeOptions,
    ) -> Result<()> {
        let msg = encode::encode(metadata, data_objects, options)?;
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        file.write_all(&msg)?;
        // Invalidate cached scan
        self.message_offsets = None;
        Ok(())
    }

    /// Read raw message bytes at a specific index.
    ///
    /// Uses `seek()` + `read_exact()` to read only the requested message
    /// without loading the entire file into memory.
    pub fn read_message(&mut self, index: usize) -> Result<Vec<u8>> {
        self.ensure_scanned()?;
        let offsets = self
            .message_offsets
            .as_ref()
            .expect("message_offsets guaranteed after ensure_scanned() succeeds");
        if index >= offsets.len() {
            return Err(TensogramError::Framing(format!(
                "message index {} out of range (count={})",
                index,
                offsets.len()
            )));
        }
        let (offset, length) = offsets[index];
        let mut file = fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(offset as u64))?;
        let mut buf = vec![0u8; length];
        file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Iterate over messages. Each item is the raw message bytes.
    #[deprecated(note = "Use message_count() + read_message(index) for lazy access")]
    pub fn messages(&mut self) -> Result<Vec<Vec<u8>>> {
        self.ensure_scanned()?;
        let count = self
            .message_offsets
            .as_ref()
            .expect("message_offsets guaranteed after ensure_scanned() succeeds")
            .len();
        let mut msgs = Vec::with_capacity(count);
        for i in 0..count {
            msgs.push(self.read_message(i)?);
        }
        Ok(msgs)
    }

    /// Decode a specific message by index.
    pub fn decode_message(
        &mut self,
        index: usize,
        options: &DecodeOptions,
    ) -> Result<(Metadata, Vec<Vec<u8>>)> {
        let msg = self.read_message(index)?;
        decode::decode(&msg, options)
    }

    /// Return a lazy iterator over the messages in this file.
    ///
    /// Scans the file once to locate message boundaries, then yields raw
    /// message bytes on demand via seek-based reads. The iterator does not
    /// borrow `self` — it owns a cloned path and offset list.
    pub fn iter(&mut self) -> Result<crate::iter::FileMessageIter> {
        self.ensure_scanned()?;
        let offsets = self
            .message_offsets
            .as_ref()
            .expect("message_offsets guaranteed after ensure_scanned() succeeds")
            .clone();
        Ok(crate::iter::FileMessageIter::new(
            self.path.clone(),
            offsets,
        ))
    }

    /// Get the file path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;
    use crate::types::{ByteOrder, ObjectDescriptor, PayloadDescriptor};
    use std::collections::BTreeMap;

    fn make_metadata(shape: Vec<u64>) -> Metadata {
        let strides = if shape.is_empty() {
            vec![]
        } else {
            let mut s = vec![1u64; shape.len()];
            for i in (0..shape.len() - 1).rev() {
                s[i] = s[i + 1] * shape[i + 1];
            }
            s
        };

        Metadata {
            version: 1,
            objects: vec![ObjectDescriptor {
                obj_type: "ntensor".to_string(),
                ndim: shape.len() as u64,
                shape,
                strides,
                dtype: Dtype::Float32,
                extra: BTreeMap::new(),
            }],
            payload: vec![PayloadDescriptor {
                byte_order: ByteOrder::Big,
                encoding: "none".to_string(),
                filter: "none".to_string(),
                compression: "none".to_string(),
                params: BTreeMap::new(),
                hash: None,
            }],
            extra: BTreeMap::new(),
        }
    }

    #[test]
    fn test_file_create_append_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.tgm");

        let mut file = TensogramFile::create(&path).unwrap();
        let metadata = make_metadata(vec![4]);
        let data = vec![0u8; 16]; // 4 float32 = 16 bytes
        file.append(&metadata, &[&data], &EncodeOptions::default())
            .unwrap();
        file.append(&metadata, &[&data], &EncodeOptions::default())
            .unwrap();

        assert_eq!(file.message_count().unwrap(), 2);

        #[allow(deprecated)]
        let msgs = file.messages().unwrap();
        assert_eq!(msgs.len(), 2);

        let (meta, objects) = file.decode_message(0, &DecodeOptions::default()).unwrap();
        assert_eq!(meta.version, 1);
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0], data);
    }

    #[test]
    fn test_file_lazy_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lazy.tgm");

        let mut file = TensogramFile::create(&path).unwrap();
        let metadata = make_metadata(vec![4]);

        let data0 = vec![0u8; 16];
        let data1 = vec![1u8; 16];
        let data2 = vec![2u8; 16];

        file.append(&metadata, &[&data0], &EncodeOptions::default())
            .unwrap();
        file.append(&metadata, &[&data1], &EncodeOptions::default())
            .unwrap();
        file.append(&metadata, &[&data2], &EncodeOptions::default())
            .unwrap();

        assert_eq!(file.message_count().unwrap(), 3);

        let (_, obj1) = file.decode_message(1, &DecodeOptions::default()).unwrap();
        assert_eq!(obj1[0], data1);

        let (_, obj0) = file.decode_message(0, &DecodeOptions::default()).unwrap();
        assert_eq!(obj0[0], data0);

        let (_, obj2) = file.decode_message(2, &DecodeOptions::default()).unwrap();
        assert_eq!(obj2[0], data2);
    }

    #[test]
    fn test_file_read_message_by_index() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test2.tgm");

        let mut file = TensogramFile::create(&path).unwrap();
        let data1 = vec![1u8; 16];
        let data2 = vec![2u8; 16];

        let metadata = make_metadata(vec![4]);
        file.append(&metadata, &[&data1], &EncodeOptions::default())
            .unwrap();
        file.append(&metadata, &[&data2], &EncodeOptions::default())
            .unwrap();

        let (_, obj1) = file.decode_message(0, &DecodeOptions::default()).unwrap();
        let (_, obj2) = file.decode_message(1, &DecodeOptions::default()).unwrap();
        assert_eq!(obj1[0], data1);
        assert_eq!(obj2[0], data2);
    }

    #[test]
    fn test_file_iter_via_tensogram_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("iter.tgm");

        let mut file = TensogramFile::create(&path).unwrap();
        let data0 = vec![0u8; 16];
        let data1 = vec![1u8; 16];
        let data2 = vec![2u8; 16];
        let metadata = make_metadata(vec![4]);
        file.append(&metadata, &[&data0], &EncodeOptions::default())
            .unwrap();
        file.append(&metadata, &[&data1], &EncodeOptions::default())
            .unwrap();
        file.append(&metadata, &[&data2], &EncodeOptions::default())
            .unwrap();

        let raw_messages: Vec<Vec<u8>> = file.iter().unwrap().map(|r| r.unwrap()).collect();
        assert_eq!(raw_messages.len(), 3);

        // Verify each message decodes to the correct data
        for (i, raw) in raw_messages.iter().enumerate() {
            let (_, objects) = crate::decode::decode(raw, &DecodeOptions::default()).unwrap();
            assert_eq!(objects[0], vec![i as u8; 16]);
        }
    }
}
