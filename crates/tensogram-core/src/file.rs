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
    /// Memory-mapped file buffer (available with `mmap` feature).
    #[cfg(feature = "mmap")]
    mmap: Option<memmap2::Mmap>,
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
            #[cfg(feature = "mmap")]
            mmap: None,
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
            #[cfg(feature = "mmap")]
            mmap: None,
        })
    }

    /// Open a file with memory-mapped I/O for zero-copy reads.
    ///
    /// The file is mapped into memory and `scan()` runs directly on the
    /// mapped buffer. `read_message()` returns a copy from the mapped region.
    #[cfg(feature = "mmap")]
    pub fn open_mmap(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = fs::File::open(&path)?;
        // SAFETY: the file is opened read-only and we hold the path for
        // the lifetime of TensogramFile. Concurrent writes by other
        // processes would violate the contract, but that is the standard
        // mmap caveat documented by memmap2.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let offsets = framing::scan(&mmap);
        Ok(TensogramFile {
            path,
            message_offsets: Some(offsets),
            mmap: Some(mmap),
        })
    }

    /// Scan the file for message boundaries using streaming I/O.
    ///
    /// Reads only preamble-sized chunks and seeks, avoiding loading the
    /// entire file into memory. Caches results for subsequent calls.
    fn ensure_scanned(&mut self) -> Result<()> {
        if self.message_offsets.is_some() {
            return Ok(());
        }
        let mut file = fs::File::open(&self.path)?;
        let offsets = framing::scan_file(&mut file)?;
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

        // Use mmap buffer if available, otherwise seek+read
        #[cfg(feature = "mmap")]
        if let Some(ref mmap) = self.mmap {
            return Ok(mmap[offset..offset + length].to_vec());
        }

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

    // ── Async API (requires `async` feature) ─────────────────────────

    /// Open a file and scan it asynchronously.
    ///
    /// The scan is CPU-bound, so it runs on a blocking thread via
    /// `tokio::task::spawn_blocking`.
    #[cfg(feature = "async")]
    pub async fn open_async(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(TensogramError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("file not found: {}", path.display()),
            )));
        }
        let p = path.clone();
        let offsets = tokio::task::spawn_blocking(move || {
            let mut file = fs::File::open(&p)?;
            framing::scan_file(&mut file)
        })
        .await
        .map_err(|e| TensogramError::Io(std::io::Error::other(e)))??;

        Ok(TensogramFile {
            path,
            message_offsets: Some(offsets),
            #[cfg(feature = "mmap")]
            mmap: None,
        })
    }

    /// Read raw message bytes at a specific index, asynchronously.
    #[cfg(feature = "async")]
    pub async fn read_message_async(&mut self, index: usize) -> Result<Vec<u8>> {
        // Ensure scanned (sync — already cached after open_async)
        if self.message_offsets.is_none() {
            let p = self.path.clone();
            let offsets = tokio::task::spawn_blocking(move || {
                let mut file = fs::File::open(&p)?;
                framing::scan_file(&mut file)
            })
            .await
            .map_err(|e| TensogramError::Io(std::io::Error::other(e)))??;
            self.message_offsets = Some(offsets);
        }

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

        let p = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let mut file = fs::File::open(&p)?;
            file.seek(SeekFrom::Start(offset as u64))?;
            let mut buf = vec![0u8; length];
            file.read_exact(&mut buf)?;
            Ok(buf)
        })
        .await
        .map_err(|e| TensogramError::Io(std::io::Error::other(e)))?
    }

    /// Decode a specific message asynchronously.
    ///
    /// Both I/O and CPU-intensive decode run on blocking threads
    /// because the encoding pipeline may call FFI (libaec, zfp, blosc2).
    #[cfg(feature = "async")]
    pub async fn decode_message_async(
        &mut self,
        index: usize,
        options: &DecodeOptions,
    ) -> Result<(GlobalMetadata, Vec<DecodedObject>)> {
        let msg = self.read_message_async(index).await?;
        let opts = options.clone();
        tokio::task::spawn_blocking(move || decode::decode(&msg, &opts))
            .await
            .map_err(|e| TensogramError::Io(std::io::Error::other(e)))?
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
            ..Default::default()
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

    // ── Phase 4: mmap tests ────────────────────────────────────────────

    #[cfg(feature = "mmap")]
    #[test]
    fn test_mmap_open_and_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mmap.tgm");

        // Write two messages normally
        let mut file = TensogramFile::create(&path).unwrap();
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data0 = vec![10u8; 16];
        let data1 = vec![20u8; 16];
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

        // Reopen with mmap
        let mut mmap_file = TensogramFile::open_mmap(&path).unwrap();
        assert_eq!(mmap_file.message_count().unwrap(), 2);

        let (decoded_meta, objects) = mmap_file
            .decode_message(0, &DecodeOptions::default())
            .unwrap();
        assert_eq!(decoded_meta.version, 2);
        assert_eq!(objects[0].1, data0);

        let (_, objects1) = mmap_file
            .decode_message(1, &DecodeOptions::default())
            .unwrap();
        assert_eq!(objects1[0].1, data1);
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn test_mmap_matches_regular_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mmap_vs_regular.tgm");

        let mut file = TensogramFile::create(&path).unwrap();
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![42u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();

        // Read with regular open
        let mut regular = TensogramFile::open(&path).unwrap();
        let regular_msg = regular.read_message(0).unwrap();

        // Read with mmap
        let mut mmap = TensogramFile::open_mmap(&path).unwrap();
        let mmap_msg = mmap.read_message(0).unwrap();

        assert_eq!(regular_msg, mmap_msg);
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

    // ── Phase 5: async tests ─────────────────────────────────────────

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_async_open_and_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("async.tgm");

        // Write messages with sync API
        let mut file = TensogramFile::create(&path).unwrap();
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data0 = vec![10u8; 16];
        let data1 = vec![20u8; 16];
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

        // Read with async API
        let mut async_file = TensogramFile::open_async(&path).await.unwrap();
        assert_eq!(async_file.message_count().unwrap(), 2);

        let msg0 = async_file.read_message_async(0).await.unwrap();
        let (meta0, objects0) = crate::decode::decode(&msg0, &DecodeOptions::default()).unwrap();
        assert_eq!(meta0.version, 2);
        assert_eq!(objects0[0].1, data0);

        let (_, objects1) = async_file
            .decode_message_async(1, &DecodeOptions::default())
            .await
            .unwrap();
        assert_eq!(objects1[0].1, data1);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_async_matches_sync() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("async_vs_sync.tgm");

        let mut file = TensogramFile::create(&path).unwrap();
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![42u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )
        .unwrap();

        // Sync read
        let mut sync_file = TensogramFile::open(&path).unwrap();
        let sync_msg = sync_file.read_message(0).unwrap();

        // Async read
        let mut async_file = TensogramFile::open_async(&path).await.unwrap();
        let async_msg = async_file.read_message_async(0).await.unwrap();

        assert_eq!(sync_msg, async_msg);
    }
}
