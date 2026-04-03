use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::decode::{self, DecodeOptions};
use crate::encode::{self, EncodeOptions};
use crate::error::{Result, TensogramError};
use crate::framing;
use crate::types::{DataObjectDescriptor, DecodedObject, GlobalMetadata};

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
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
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
            .ok_or_else(|| TensogramError::Framing("scan result missing".to_string()))?
            .len())
    }

    /// Append a message to the file.
    pub fn append(
        &mut self,
        global_metadata: &GlobalMetadata,
        descriptors: &[(&DataObjectDescriptor, &[u8])],
        options: &EncodeOptions,
    ) -> Result<()> {
        let msg = encode::encode(global_metadata, descriptors, options)?;
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        file.write_all(&msg)?;
        self.message_offsets = None;
        Ok(())
    }

    /// Read raw message bytes at a specific index.
    pub fn read_message(&mut self, index: usize) -> Result<Vec<u8>> {
        self.ensure_scanned()?;
        let offsets = self
            .message_offsets
            .as_ref()
            .ok_or_else(|| TensogramError::Framing("scan result missing".to_string()))?;
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
            .ok_or_else(|| TensogramError::Framing("scan result missing".to_string()))?
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
    ) -> Result<(GlobalMetadata, Vec<DecodedObject>)> {
        let msg = self.read_message(index)?;
        decode::decode(&msg, options)
    }

    /// Return a lazy iterator over the messages in this file.
    pub fn iter(&mut self) -> Result<crate::iter::FileMessageIter> {
        self.ensure_scanned()?;
        let offsets = self
            .message_offsets
            .as_ref()
            .ok_or_else(|| TensogramError::Framing("scan result missing".to_string()))?
            .clone();
        crate::iter::FileMessageIter::new(self.path.clone(), offsets)
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
    use crate::types::ByteOrder;
    use std::collections::BTreeMap;

    fn make_global_meta() -> GlobalMetadata {
        GlobalMetadata {
            version: 2,
            extra: BTreeMap::new(),
        }
    }

    fn make_descriptor(shape: Vec<u64>) -> DataObjectDescriptor {
        let strides = if shape.is_empty() {
            vec![]
        } else {
            let mut s = vec![1u64; shape.len()];
            for i in (0..shape.len() - 1).rev() {
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
            byte_order: ByteOrder::Big,
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        }
    }

    #[test]
    fn test_file_create_append_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.tgm");

        let mut file = TensogramFile::create(&path).unwrap();
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();

        assert_eq!(file.message_count().unwrap(), 2);

        #[allow(deprecated)]
        let msgs = file.messages().unwrap();
        assert_eq!(msgs.len(), 2);

        let (decoded_meta, objects) = file.decode_message(0, &DecodeOptions::default()).unwrap();
        assert_eq!(decoded_meta.version, 2);
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].1, data);
    }

    #[test]
    fn test_file_lazy_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lazy.tgm");

        let mut file = TensogramFile::create(&path).unwrap();
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);

        let data0 = vec![0u8; 16];
        let data1 = vec![1u8; 16];
        let data2 = vec![2u8; 16];

        file.append(
            &meta,
            &[(&desc, data0.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();
        file.append(
            &meta,
            &[(&desc, data1.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();
        file.append(
            &meta,
            &[(&desc, data2.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();

        assert_eq!(file.message_count().unwrap(), 3);

        let (_, obj1) = file.decode_message(1, &DecodeOptions::default()).unwrap();
        assert_eq!(obj1[0].1, data1);

        let (_, obj0) = file.decode_message(0, &DecodeOptions::default()).unwrap();
        assert_eq!(obj0[0].1, data0);

        let (_, obj2) = file.decode_message(2, &DecodeOptions::default()).unwrap();
        assert_eq!(obj2[0].1, data2);
    }

    #[test]
    fn test_file_iter_via_tensogram_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("iter.tgm");

        let mut file = TensogramFile::create(&path).unwrap();
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);

        let data0 = vec![0u8; 16];
        let data1 = vec![1u8; 16];
        let data2 = vec![2u8; 16];

        file.append(
            &meta,
            &[(&desc, data0.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();
        file.append(
            &meta,
            &[(&desc, data1.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();
        file.append(
            &meta,
            &[(&desc, data2.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();

        let raw_messages: Vec<Vec<u8>> = file.iter().unwrap().map(|r| r.unwrap()).collect();
        assert_eq!(raw_messages.len(), 3);

        for (i, raw) in raw_messages.iter().enumerate() {
            let (_, objects) = crate::decode::decode(raw, &DecodeOptions::default()).unwrap();
            assert_eq!(objects[0].1, vec![i as u8; 16]);
        }
    }
}
