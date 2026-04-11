use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::decode::{self, DecodeOptions};
use crate::encode::{self, EncodeOptions};
use crate::error::{Result, TensogramError};
use crate::framing;
use crate::types::{DataObjectDescriptor, DecodedObject, GlobalMetadata};

// ── Backend enum ─────────────────────────────────────────────────────────────

enum Backend {
    Local {
        path: PathBuf,
        #[cfg(feature = "mmap")]
        mmap: Option<memmap2::Mmap>,
    },
    #[cfg(feature = "remote")]
    Remote(crate::remote::RemoteBackend),
}

impl Backend {
    fn source_string(&self) -> String {
        match self {
            Backend::Local { path, .. } => path.display().to_string(),
            #[cfg(feature = "remote")]
            Backend::Remote(r) => r.source_url().to_string(),
        }
    }
}

/// A handle for reading/writing Tensogram message files.
///
/// Supports local files, memory-mapped files (`mmap` feature), and
/// remote object stores (`remote` feature) via a unified API.
pub struct TensogramFile {
    backend: Backend,
    message_offsets: Option<Vec<(usize, usize)>>,
}

impl TensogramFile {
    #[tracing::instrument(skip(path), fields(path = %path.as_ref().display()))]
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(TensogramError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("file not found: {}", path.display()),
            )));
        }
        Ok(TensogramFile {
            backend: Backend::Local {
                path,
                #[cfg(feature = "mmap")]
                mmap: None,
            },
            message_offsets: None,
        })
    }

    /// Open a local file or remote URL.
    ///
    /// Auto-detects remote URLs (s3://, s3a://, gs://, az://, azure://, http://, https://)
    /// when the `remote` feature is enabled; otherwise treats the source
    /// as a local path.
    pub fn open_source(source: impl AsRef<str>) -> Result<Self> {
        let source = source.as_ref();

        #[cfg(feature = "remote")]
        if crate::remote::is_remote_url(source) {
            return Self::open_remote(source, &std::collections::BTreeMap::new());
        }

        Self::open(source)
    }

    /// Open a remote file with explicit storage options (credentials, region, etc.).
    #[cfg(feature = "remote")]
    pub fn open_remote(
        source: &str,
        storage_options: &std::collections::BTreeMap<String, String>,
    ) -> Result<Self> {
        let remote = crate::remote::RemoteBackend::open(source, storage_options)?;
        Ok(TensogramFile {
            backend: Backend::Remote(remote),
            message_offsets: None,
        })
    }

    /// Create a new local file for writing (truncates if exists).
    #[tracing::instrument(skip(path), fields(path = %path.as_ref().display()))]
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                TensogramError::Io(std::io::Error::new(
                    e.kind(),
                    format!("cannot create parent directory for {}: {e}", path.display()),
                ))
            })?;
        }
        fs::File::create(&path).map_err(|e| {
            TensogramError::Io(std::io::Error::new(
                e.kind(),
                format!("cannot create {}: {e}", path.display()),
            ))
        })?;
        Ok(TensogramFile {
            backend: Backend::Local {
                path,
                #[cfg(feature = "mmap")]
                mmap: None,
            },
            message_offsets: None,
        })
    }

    /// Open a file with memory-mapped I/O for zero-copy reads.
    ///
    /// The file is mapped into memory and `scan()` runs directly on the
    /// mapped buffer. `read_message()` returns a copy from the mapped region.
    #[cfg(feature = "mmap")]
    pub fn open_mmap(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = fs::File::open(&path).map_err(|e| {
            TensogramError::Io(std::io::Error::new(
                e.kind(),
                format!("{}: {e}", path.display()),
            ))
        })?;
        // SAFETY: the file is opened read-only and we hold the path for
        // the lifetime of TensogramFile. Concurrent writes by other
        // processes would violate the contract, but that is the standard
        // mmap caveat documented by memmap2.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let offsets = framing::scan(&mmap);
        Ok(TensogramFile {
            backend: Backend::Local {
                path,
                mmap: Some(mmap),
            },
            message_offsets: Some(offsets),
        })
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    fn local_path(&self) -> Result<&PathBuf> {
        match &self.backend {
            Backend::Local { path, .. } => Ok(path),
            #[cfg(feature = "remote")]
            Backend::Remote(_) => Err(TensogramError::Remote(
                "operation not supported on remote files".to_string(),
            )),
        }
    }

    fn ensure_scanned(&mut self) -> Result<()> {
        if self.message_offsets.is_some() {
            return Ok(());
        }
        let path = self.local_path()?.clone();
        let mut file = fs::File::open(&path).map_err(|e| {
            TensogramError::Io(std::io::Error::new(
                e.kind(),
                format!("{}: {e}", path.display()),
            ))
        })?;
        let offsets = framing::scan_file(&mut file)?;
        self.message_offsets = Some(offsets);
        Ok(())
    }

    fn checked_offsets(&self, index: usize) -> Result<(usize, usize)> {
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
        Ok(offsets[index])
    }

    // ── Public API ───────────────────────────────────────────────────────

    pub fn message_count(&mut self) -> Result<usize> {
        #[cfg(feature = "remote")]
        if let Backend::Remote(remote) = &mut self.backend {
            return remote.message_count();
        }
        self.ensure_scanned()?;
        Ok(self
            .message_offsets
            .as_ref()
            .ok_or_else(|| TensogramError::Framing("scan result missing".to_string()))?
            .len())
    }

    pub fn append(
        &mut self,
        global_metadata: &GlobalMetadata,
        descriptors: &[(&DataObjectDescriptor, &[u8])],
        options: &EncodeOptions,
    ) -> Result<()> {
        let path = self.local_path()?.clone();
        let msg = encode::encode(global_metadata, descriptors, options)?;
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| {
                TensogramError::Io(std::io::Error::new(
                    e.kind(),
                    format!("{}: {e}", path.display()),
                ))
            })?;
        file.write_all(&msg)?;
        self.message_offsets = None;
        Ok(())
    }

    pub fn read_message(&mut self, index: usize) -> Result<Vec<u8>> {
        #[cfg(feature = "remote")]
        if let Backend::Remote(remote) = &mut self.backend {
            return remote.read_message(index);
        }

        self.ensure_scanned()?;

        let (offset, length) = self.checked_offsets(index)?;

        #[cfg(feature = "mmap")]
        if let Backend::Local {
            mmap: Some(ref mmap),
            ..
        } = self.backend
        {
            return Ok(mmap[offset..offset + length].to_vec());
        }

        match &self.backend {
            Backend::Local { path, .. } => {
                let mut file = fs::File::open(path).map_err(|e| {
                    TensogramError::Io(std::io::Error::new(
                        e.kind(),
                        format!("{}: {e}", path.display()),
                    ))
                })?;
                file.seek(SeekFrom::Start(offset as u64))?;
                let mut buf = vec![0u8; length];
                file.read_exact(&mut buf)?;
                Ok(buf)
            }
            #[cfg(feature = "remote")]
            Backend::Remote(_) => Err(TensogramError::Remote(
                "unreachable: remote handled above".to_string(),
            )),
        }
    }

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

    pub fn decode_message(
        &mut self,
        index: usize,
        options: &DecodeOptions,
    ) -> Result<(GlobalMetadata, Vec<DecodedObject>)> {
        let msg = self.read_message(index)?;
        decode::decode(&msg, options)
    }

    pub fn iter(&mut self) -> Result<crate::iter::FileMessageIter> {
        self.ensure_scanned()?;
        let path = self.local_path()?.clone();
        let offsets = self
            .message_offsets
            .as_ref()
            .ok_or_else(|| TensogramError::Framing("scan result missing".to_string()))?
            .clone();
        crate::iter::FileMessageIter::new(path, offsets)
    }

    pub fn path(&self) -> Option<&Path> {
        match &self.backend {
            Backend::Local { path, .. } => Some(path),
            #[cfg(feature = "remote")]
            Backend::Remote(_) => None,
        }
    }

    pub fn source(&self) -> String {
        self.backend.source_string()
    }

    pub fn invalidate_offsets(&mut self) {
        match &self.backend {
            Backend::Local { .. } => {
                self.message_offsets = None;
            }
            #[cfg(feature = "remote")]
            Backend::Remote(_) => {}
        }
    }

    // ── Object-level access (efficient for remote) ───────────────────────

    pub fn decode_metadata(&mut self, msg_idx: usize) -> Result<GlobalMetadata> {
        match &mut self.backend {
            #[cfg(feature = "remote")]
            Backend::Remote(remote) => remote.read_metadata(msg_idx),
            _ => {
                let msg = self.read_message(msg_idx)?;
                decode::decode_metadata(&msg)
            }
        }
    }

    pub fn decode_descriptors(
        &mut self,
        msg_idx: usize,
    ) -> Result<(GlobalMetadata, Vec<DataObjectDescriptor>)> {
        match &mut self.backend {
            #[cfg(feature = "remote")]
            Backend::Remote(remote) => remote.read_descriptors(msg_idx),
            _ => {
                let msg = self.read_message(msg_idx)?;
                decode::decode_descriptors(&msg)
            }
        }
    }

    pub fn decode_object(
        &mut self,
        msg_idx: usize,
        obj_idx: usize,
        options: &DecodeOptions,
    ) -> Result<(GlobalMetadata, DataObjectDescriptor, Vec<u8>)> {
        match &mut self.backend {
            #[cfg(feature = "remote")]
            Backend::Remote(remote) => remote.read_object(msg_idx, obj_idx, options),
            _ => {
                let msg = self.read_message(msg_idx)?;
                decode::decode_object(&msg, obj_idx, options)
            }
        }
    }

    pub fn decode_range(
        &mut self,
        msg_idx: usize,
        obj_idx: usize,
        ranges: &[(u64, u64)],
        options: &DecodeOptions,
    ) -> Result<Vec<Vec<u8>>> {
        match &mut self.backend {
            #[cfg(feature = "remote")]
            Backend::Remote(remote) => remote.read_range(msg_idx, obj_idx, ranges, options),
            _ => {
                let msg = self.read_message(msg_idx)?;
                decode::decode_range(&msg, obj_idx, ranges, options)
            }
        }
    }

    pub fn is_remote(&self) -> bool {
        #[cfg(feature = "remote")]
        if matches!(self.backend, Backend::Remote(_)) {
            return true;
        }
        false
    }

    // ── Async API (requires `async` feature) ─────────────────────────────

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
            let mut file = fs::File::open(&p).map_err(|e| {
                TensogramError::Io(std::io::Error::new(
                    e.kind(),
                    format!("{}: {e}", p.display()),
                ))
            })?;
            framing::scan_file(&mut file)
        })
        .await
        .map_err(|e| TensogramError::Io(std::io::Error::other(e)))??;

        Ok(TensogramFile {
            backend: Backend::Local {
                path,
                #[cfg(feature = "mmap")]
                mmap: None,
            },
            message_offsets: Some(offsets),
        })
    }

    #[cfg(feature = "async")]
    pub async fn read_message_async(&mut self, index: usize) -> Result<Vec<u8>> {
        #[cfg(feature = "remote")]
        if let Backend::Remote(remote) = &mut self.backend {
            return remote.read_message_async(index).await;
        }

        if self.message_offsets.is_none() {
            let p = self.local_path()?.clone();
            let offsets = tokio::task::spawn_blocking(move || {
                let mut file = fs::File::open(&p)?;
                framing::scan_file(&mut file)
            })
            .await
            .map_err(|e| TensogramError::Io(std::io::Error::other(e)))??;
            self.message_offsets = Some(offsets);
        }

        let (offset, length) = self.checked_offsets(index)?;

        let p = self.local_path()?.clone();
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

    #[cfg(all(feature = "remote", feature = "async"))]
    pub async fn open_source_async(source: impl AsRef<str>) -> Result<Self> {
        let source = source.as_ref();

        if crate::remote::is_remote_url(source) {
            return Self::open_remote_async(source, &std::collections::BTreeMap::new()).await;
        }

        Self::open_async(source).await
    }

    #[cfg(all(feature = "remote", feature = "async"))]
    pub async fn open_remote_async(
        source: &str,
        storage_options: &std::collections::BTreeMap<String, String>,
    ) -> Result<Self> {
        let remote = crate::remote::RemoteBackend::open_async(source, storage_options).await?;
        Ok(TensogramFile {
            backend: Backend::Remote(remote),
            message_offsets: None,
        })
    }

    #[cfg(feature = "async")]
    pub async fn decode_metadata_async(&mut self, msg_idx: usize) -> Result<GlobalMetadata> {
        #[cfg(feature = "remote")]
        if let Backend::Remote(remote) = &mut self.backend {
            return remote.read_metadata_async(msg_idx).await;
        }

        let msg = self.read_message_async(msg_idx).await?;
        tokio::task::spawn_blocking(move || decode::decode_metadata(&msg))
            .await
            .map_err(|e| TensogramError::Io(std::io::Error::other(e)))?
    }

    #[cfg(feature = "async")]
    pub async fn decode_descriptors_async(
        &mut self,
        msg_idx: usize,
    ) -> Result<(GlobalMetadata, Vec<DataObjectDescriptor>)> {
        #[cfg(feature = "remote")]
        if let Backend::Remote(remote) = &mut self.backend {
            return remote.read_descriptors_async(msg_idx).await;
        }

        let msg = self.read_message_async(msg_idx).await?;
        tokio::task::spawn_blocking(move || decode::decode_descriptors(&msg))
            .await
            .map_err(|e| TensogramError::Io(std::io::Error::other(e)))?
    }

    #[cfg(feature = "async")]
    pub async fn decode_object_async(
        &mut self,
        msg_idx: usize,
        obj_idx: usize,
        options: &DecodeOptions,
    ) -> Result<(GlobalMetadata, DataObjectDescriptor, Vec<u8>)> {
        #[cfg(feature = "remote")]
        if let Backend::Remote(remote) = &mut self.backend {
            return remote.read_object_async(msg_idx, obj_idx, options).await;
        }

        let msg = self.read_message_async(msg_idx).await?;
        let opts = options.clone();
        tokio::task::spawn_blocking(move || decode::decode_object(&msg, obj_idx, &opts))
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
            byte_order: ByteOrder::native(),
            encoding: "none".to_string(),
            filter: "none".to_string(),
            compression: "none".to_string(),
            params: BTreeMap::new(),
            hash: None,
        }
    }

    #[test]
    fn test_file_create_append_read() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("test.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        assert_eq!(file.message_count()?, 2);

        #[allow(deprecated)]
        let msgs = file.messages()?;
        assert_eq!(msgs.len(), 2);

        let (decoded_meta, objects) = file.decode_message(0, &DecodeOptions::default())?;
        assert_eq!(decoded_meta.version, 2);
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].1, data);
        Ok(())
    }

    #[test]
    fn test_file_lazy_read() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("lazy.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);

        let data0 = vec![0u8; 16];
        let data1 = vec![1u8; 16];
        let data2 = vec![2u8; 16];

        file.append(
            &meta,
            &[(&desc, data0.as_slice())],
            &EncodeOptions::default(),
        )?;
        file.append(
            &meta,
            &[(&desc, data1.as_slice())],
            &EncodeOptions::default(),
        )?;
        file.append(
            &meta,
            &[(&desc, data2.as_slice())],
            &EncodeOptions::default(),
        )?;

        assert_eq!(file.message_count()?, 3);

        let (_, obj1) = file.decode_message(1, &DecodeOptions::default())?;
        assert_eq!(obj1[0].1, data1);

        let (_, obj0) = file.decode_message(0, &DecodeOptions::default())?;
        assert_eq!(obj0[0].1, data0);

        let (_, obj2) = file.decode_message(2, &DecodeOptions::default())?;
        assert_eq!(obj2[0].1, data2);
        Ok(())
    }

    #[test]
    fn test_file_decode_metadata() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("meta.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        let decoded_meta = file.decode_metadata(0)?;
        assert_eq!(decoded_meta.version, 2);
        Ok(())
    }

    #[test]
    fn test_file_decode_descriptors() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("descs.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        let (decoded_meta, descriptors) = file.decode_descriptors(0)?;
        assert_eq!(decoded_meta.version, 2);
        assert_eq!(descriptors.len(), 1);
        assert_eq!(descriptors[0].shape, vec![4]);
        Ok(())
    }

    #[test]
    fn test_file_decode_object() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("obj.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![42u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        let (decoded_meta, decoded_desc, decoded_data) =
            file.decode_object(0, 0, &DecodeOptions::default())?;
        assert_eq!(decoded_meta.version, 2);
        assert_eq!(decoded_desc.shape, vec![4]);
        assert_eq!(decoded_data, data);
        Ok(())
    }

    // ── mmap tests ───────────────────────────────────────────────────────

    #[cfg(feature = "mmap")]
    #[test]
    fn test_mmap_open_and_read() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("mmap.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data0 = vec![10u8; 16];
        let data1 = vec![20u8; 16];
        file.append(
            &meta,
            &[(&desc, data0.as_slice())],
            &EncodeOptions::default(),
        )?;
        file.append(
            &meta,
            &[(&desc, data1.as_slice())],
            &EncodeOptions::default(),
        )?;

        let mut mmap_file = TensogramFile::open_mmap(&path)?;
        assert_eq!(mmap_file.message_count()?, 2);

        let (decoded_meta, objects) = mmap_file.decode_message(0, &DecodeOptions::default())?;
        assert_eq!(decoded_meta.version, 2);
        assert_eq!(objects[0].1, data0);

        let (_, objects1) = mmap_file.decode_message(1, &DecodeOptions::default())?;
        assert_eq!(objects1[0].1, data1);
        Ok(())
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn test_mmap_matches_regular_open() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("mmap_vs_regular.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![42u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        let mut regular = TensogramFile::open(&path)?;
        let regular_msg = regular.read_message(0)?;

        let mut mmap = TensogramFile::open_mmap(&path)?;
        let mmap_msg = mmap.read_message(0)?;

        assert_eq!(regular_msg, mmap_msg);
        Ok(())
    }

    #[test]
    fn test_file_iter_via_tensogram_file() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("iter.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);

        let data0 = vec![0u8; 16];
        let data1 = vec![1u8; 16];
        let data2 = vec![2u8; 16];

        file.append(
            &meta,
            &[(&desc, data0.as_slice())],
            &EncodeOptions::default(),
        )?;
        file.append(
            &meta,
            &[(&desc, data1.as_slice())],
            &EncodeOptions::default(),
        )?;
        file.append(
            &meta,
            &[(&desc, data2.as_slice())],
            &EncodeOptions::default(),
        )?;

        let raw_messages: Vec<Vec<u8>> =
            file.iter()?.collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(raw_messages.len(), 3);

        for (i, raw) in raw_messages.iter().enumerate() {
            let (_, objects) = crate::decode::decode(raw, &DecodeOptions::default())?;
            assert_eq!(objects[0].1, vec![i as u8; 16]);
        }
        Ok(())
    }

    // ── async tests ──────────────────────────────────────────────────────

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_async_open_and_read() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("async.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data0 = vec![10u8; 16];
        let data1 = vec![20u8; 16];
        file.append(
            &meta,
            &[(&desc, data0.as_slice())],
            &EncodeOptions::default(),
        )?;
        file.append(
            &meta,
            &[(&desc, data1.as_slice())],
            &EncodeOptions::default(),
        )?;

        let mut async_file = TensogramFile::open_async(&path).await?;
        assert_eq!(async_file.message_count()?, 2);

        let msg0 = async_file.read_message_async(0).await?;
        let (meta0, objects0) = crate::decode::decode(&msg0, &DecodeOptions::default())?;
        assert_eq!(meta0.version, 2);
        assert_eq!(objects0[0].1, data0);

        let (_, objects1) = async_file
            .decode_message_async(1, &DecodeOptions::default())
            .await?;
        assert_eq!(objects1[0].1, data1);
        Ok(())
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_async_matches_sync() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("async_vs_sync.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![42u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        let mut sync_file = TensogramFile::open(&path)?;
        let sync_msg = sync_file.read_message(0)?;

        let mut async_file = TensogramFile::open_async(&path).await?;
        let async_msg = async_file.read_message_async(0).await?;

        assert_eq!(sync_msg, async_msg);
        Ok(())
    }
}
