// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

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
    message_offsets: OnceLock<Vec<(usize, usize)>>,
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
            message_offsets: OnceLock::new(),
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
            message_offsets: OnceLock::new(),
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
            message_offsets: OnceLock::new(),
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
            message_offsets: OnceLock::from(offsets),
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

    fn ensure_scanned(&self) -> Result<()> {
        if self.message_offsets.get().is_none() {
            let path = self.local_path()?.clone();
            let mut file = fs::File::open(&path).map_err(|e| {
                TensogramError::Io(std::io::Error::new(
                    e.kind(),
                    format!("{}: {e}", path.display()),
                ))
            })?;
            let offsets = framing::scan_file(&mut file)?;
            let _ = self.message_offsets.set(offsets);
        }
        Ok(())
    }

    fn checked_offsets(&self, index: usize) -> Result<(usize, usize)> {
        let offsets = self
            .message_offsets
            .get()
            .ok_or_else(|| TensogramError::Framing("internal invariant violated: message_offsets was not populated by ensure_scanned() before use (this is a library bug — please report)".to_string()))?;
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

    pub fn message_count(&self) -> Result<usize> {
        #[cfg(feature = "remote")]
        if let Backend::Remote(remote) = &self.backend {
            return remote.message_count();
        }
        self.ensure_scanned()?;
        Ok(self
            .message_offsets
            .get()
            .ok_or_else(|| TensogramError::Framing("internal invariant violated: message_offsets was not populated by ensure_scanned() before use (this is a library bug — please report)".to_string()))?
            .len())
    }

    pub fn append(
        &mut self,
        global_metadata: &GlobalMetadata,
        descriptors: &[(&DataObjectDescriptor, &[u8])],
        options: &EncodeOptions,
    ) -> Result<()> {
        #[cfg(feature = "mmap")]
        if let Backend::Local { mmap: Some(_), .. } = &self.backend {
            return Err(TensogramError::Io(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "cannot append to a memory-mapped file (open without mmap to append)",
            )));
        }
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
        self.message_offsets = OnceLock::new();
        Ok(())
    }

    pub fn read_message(&self, index: usize) -> Result<Vec<u8>> {
        #[cfg(feature = "remote")]
        if let Backend::Remote(remote) = &self.backend {
            return remote.read_message(index);
        }

        self.ensure_scanned()?;

        let (offset, length) = self.checked_offsets(index)?;

        #[cfg(feature = "mmap")]
        if let Backend::Local {
            mmap: Some(mmap), ..
        } = &self.backend
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
    pub fn messages(&self) -> Result<Vec<Vec<u8>>> {
        self.ensure_scanned()?;
        let count = self
            .message_offsets
            .get()
            .ok_or_else(|| TensogramError::Framing("internal invariant violated: message_offsets was not populated by ensure_scanned() before use (this is a library bug — please report)".to_string()))?
            .len();
        let mut msgs = Vec::with_capacity(count);
        for i in 0..count {
            msgs.push(self.read_message(i)?);
        }
        Ok(msgs)
    }

    pub fn decode_message(
        &self,
        index: usize,
        options: &DecodeOptions,
    ) -> Result<(GlobalMetadata, Vec<DecodedObject>)> {
        let msg = self.read_message(index)?;
        decode::decode(&msg, options)
    }

    pub fn iter(&self) -> Result<crate::iter::FileMessageIter> {
        self.ensure_scanned()?;
        let path = self.local_path()?.clone();
        let offsets = self
            .message_offsets
            .get()
            .ok_or_else(|| TensogramError::Framing("internal invariant violated: message_offsets was not populated by ensure_scanned() before use (this is a library bug — please report)".to_string()))?
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
                self.message_offsets = OnceLock::new();
            }
            #[cfg(feature = "remote")]
            Backend::Remote(_) => {}
        }
    }

    // ── Object-level access (efficient for remote) ───────────────────────

    pub fn decode_metadata(&self, msg_idx: usize) -> Result<GlobalMetadata> {
        match &self.backend {
            #[cfg(feature = "remote")]
            Backend::Remote(remote) => remote.read_metadata(msg_idx),
            _ => {
                let msg = self.read_message(msg_idx)?;
                decode::decode_metadata(&msg)
            }
        }
    }

    pub fn decode_descriptors(
        &self,
        msg_idx: usize,
    ) -> Result<(GlobalMetadata, Vec<DataObjectDescriptor>)> {
        match &self.backend {
            #[cfg(feature = "remote")]
            Backend::Remote(remote) => remote.read_descriptors(msg_idx),
            _ => {
                let msg = self.read_message(msg_idx)?;
                decode::decode_descriptors(&msg)
            }
        }
    }

    pub fn decode_object(
        &self,
        msg_idx: usize,
        obj_idx: usize,
        options: &DecodeOptions,
    ) -> Result<(GlobalMetadata, DataObjectDescriptor, Vec<u8>)> {
        match &self.backend {
            #[cfg(feature = "remote")]
            Backend::Remote(remote) => remote.read_object(msg_idx, obj_idx, options),
            _ => {
                let msg = self.read_message(msg_idx)?;
                decode::decode_object(&msg, obj_idx, options)
            }
        }
    }

    #[cfg(feature = "remote")]
    pub fn decode_range_batch(
        &self,
        msg_indices: &[usize],
        obj_idx: usize,
        ranges: &[(u64, u64)],
        options: &DecodeOptions,
    ) -> Result<Vec<(DataObjectDescriptor, Vec<Vec<u8>>)>> {
        match &self.backend {
            Backend::Remote(remote) => {
                remote.read_range_batch(msg_indices, obj_idx, ranges, options)
            }
            _ => Err(TensogramError::Io(std::io::Error::other(
                "batch range decode requires a remote backend",
            ))),
        }
    }

    #[cfg(feature = "remote")]
    pub fn decode_object_batch(
        &self,
        msg_indices: &[usize],
        obj_idx: usize,
        options: &DecodeOptions,
    ) -> Result<Vec<(GlobalMetadata, DataObjectDescriptor, Vec<u8>)>> {
        match &self.backend {
            Backend::Remote(remote) => remote.read_object_batch(msg_indices, obj_idx, options),
            _ => Err(TensogramError::Io(std::io::Error::other(
                "batch object decode requires a remote backend",
            ))),
        }
    }

    pub fn decode_range(
        &self,
        msg_idx: usize,
        obj_idx: usize,
        ranges: &[(u64, u64)],
        options: &DecodeOptions,
    ) -> Result<(DataObjectDescriptor, Vec<Vec<u8>>)> {
        match &self.backend {
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
            message_offsets: OnceLock::from(offsets),
        })
    }

    #[cfg(feature = "async")]
    pub async fn message_count_async(&self) -> Result<usize> {
        #[cfg(feature = "remote")]
        if let Backend::Remote(remote) = &self.backend {
            return remote.message_count_async().await;
        }

        if self.message_offsets.get().is_none() {
            let p = self.local_path()?.clone();
            let offsets = tokio::task::spawn_blocking(move || {
                let mut file = fs::File::open(&p)?;
                framing::scan_file(&mut file)
            })
            .await
            .map_err(|e| TensogramError::Io(std::io::Error::other(e)))??;
            let _ = self.message_offsets.set(offsets);
        }

        Ok(self
            .message_offsets
            .get()
            .ok_or_else(|| TensogramError::Framing("internal invariant violated: message_offsets was not populated by ensure_scanned() before use (this is a library bug — please report)".to_string()))?
            .len())
    }

    #[cfg(feature = "async")]
    pub async fn read_message_async(&self, index: usize) -> Result<Vec<u8>> {
        #[cfg(feature = "remote")]
        if let Backend::Remote(remote) = &self.backend {
            return remote.read_message_async(index).await;
        }

        if self.message_offsets.get().is_none() {
            let p = self.local_path()?.clone();
            let offsets = tokio::task::spawn_blocking(move || {
                let mut file = fs::File::open(&p)?;
                framing::scan_file(&mut file)
            })
            .await
            .map_err(|e| TensogramError::Io(std::io::Error::other(e)))??;
            let _ = self.message_offsets.set(offsets);
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
        &self,
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
            message_offsets: OnceLock::new(),
        })
    }

    #[cfg(feature = "async")]
    pub async fn decode_metadata_async(&self, msg_idx: usize) -> Result<GlobalMetadata> {
        #[cfg(feature = "remote")]
        if let Backend::Remote(remote) = &self.backend {
            return remote.read_metadata_async(msg_idx).await;
        }

        let msg = self.read_message_async(msg_idx).await?;
        tokio::task::spawn_blocking(move || decode::decode_metadata(&msg))
            .await
            .map_err(|e| TensogramError::Io(std::io::Error::other(e)))?
    }

    #[cfg(feature = "async")]
    pub async fn decode_descriptors_async(
        &self,
        msg_idx: usize,
    ) -> Result<(GlobalMetadata, Vec<DataObjectDescriptor>)> {
        #[cfg(feature = "remote")]
        if let Backend::Remote(remote) = &self.backend {
            return remote.read_descriptors_async(msg_idx).await;
        }

        let msg = self.read_message_async(msg_idx).await?;
        tokio::task::spawn_blocking(move || decode::decode_descriptors(&msg))
            .await
            .map_err(|e| TensogramError::Io(std::io::Error::other(e)))?
    }

    #[cfg(feature = "async")]
    pub async fn decode_object_async(
        &self,
        msg_idx: usize,
        obj_idx: usize,
        options: &DecodeOptions,
    ) -> Result<(GlobalMetadata, DataObjectDescriptor, Vec<u8>)> {
        #[cfg(feature = "remote")]
        if let Backend::Remote(remote) = &self.backend {
            return remote.read_object_async(msg_idx, obj_idx, options).await;
        }

        let msg = self.read_message_async(msg_idx).await?;
        let opts = options.clone();
        tokio::task::spawn_blocking(move || decode::decode_object(&msg, obj_idx, &opts))
            .await
            .map_err(|e| TensogramError::Io(std::io::Error::other(e)))?
    }

    #[cfg(feature = "async")]
    pub async fn decode_range_async(
        &self,
        msg_idx: usize,
        obj_idx: usize,
        ranges: &[(u64, u64)],
        options: &DecodeOptions,
    ) -> Result<(DataObjectDescriptor, Vec<Vec<u8>>)> {
        #[cfg(feature = "remote")]
        if let Backend::Remote(remote) = &self.backend {
            return remote
                .read_range_async(msg_idx, obj_idx, ranges, options)
                .await;
        }

        let msg = self.read_message_async(msg_idx).await?;
        let ranges = ranges.to_vec();
        let opts = options.clone();
        tokio::task::spawn_blocking(move || decode::decode_range(&msg, obj_idx, &ranges, &opts))
            .await
            .map_err(|e| TensogramError::Io(std::io::Error::other(e)))?
    }

    #[cfg(all(feature = "remote", feature = "async"))]
    pub async fn prefetch_layouts_async(&self, msg_indices: &[usize]) -> Result<()> {
        if let Backend::Remote(remote) = &self.backend {
            return remote.ensure_all_layouts_batch_async(msg_indices).await;
        }
        Ok(())
    }

    #[cfg(all(feature = "remote", feature = "async"))]
    pub async fn decode_object_batch_async(
        &self,
        msg_indices: &[usize],
        obj_idx: usize,
        options: &DecodeOptions,
    ) -> Result<Vec<(GlobalMetadata, DataObjectDescriptor, Vec<u8>)>> {
        if let Backend::Remote(remote) = &self.backend {
            return remote
                .read_object_batch_async(msg_indices, obj_idx, options)
                .await;
        }
        Err(TensogramError::Io(std::io::Error::other(
            "batch object decode requires a remote backend",
        )))
    }

    #[cfg(all(feature = "remote", feature = "async"))]
    pub async fn decode_range_batch_async(
        &self,
        msg_indices: &[usize],
        obj_idx: usize,
        ranges: &[(u64, u64)],
        options: &DecodeOptions,
    ) -> Result<Vec<(DataObjectDescriptor, Vec<Vec<u8>>)>> {
        if let Backend::Remote(remote) = &self.backend {
            return remote
                .read_range_batch_async(msg_indices, obj_idx, ranges, options)
                .await;
        }
        Err(TensogramError::Io(std::io::Error::other(
            "batch range decode requires a remote backend",
        )))
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
            masks: None,
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

        let (_decoded_meta, objects) = file.decode_message(0, &DecodeOptions::default())?;
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

        let _decoded_meta = file.decode_metadata(0)?;
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

        let (_decoded_meta, descriptors) = file.decode_descriptors(0)?;
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

        let (_decoded_meta, decoded_desc, decoded_data) =
            file.decode_object(0, 0, &DecodeOptions::default())?;
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

        let mmap_file = TensogramFile::open_mmap(&path)?;
        assert_eq!(mmap_file.message_count()?, 2);

        let (_decoded_meta, objects) = mmap_file.decode_message(0, &DecodeOptions::default())?;
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

        let regular = TensogramFile::open(&path)?;
        let regular_msg = regular.read_message(0)?;

        let mmap = TensogramFile::open_mmap(&path)?;
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

        let async_file = TensogramFile::open_async(&path).await?;
        assert_eq!(async_file.message_count()?, 2);

        let msg0 = async_file.read_message_async(0).await?;
        let (_meta0, objects0) = crate::decode::decode(&msg0, &DecodeOptions::default())?;
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

        let sync_file = TensogramFile::open(&path)?;
        let sync_msg = sync_file.read_message(0)?;

        let async_file = TensogramFile::open_async(&path).await?;
        let async_msg = async_file.read_message_async(0).await?;

        assert_eq!(sync_msg, async_msg);
        Ok(())
    }

    // ── decode_range on local files ──────────────────────────────────────

    #[test]
    fn test_local_decode_range_valid() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("range.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![10]);
        let data: Vec<u8> = (0..40).collect(); // 10 float32s = 40 bytes
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        // Decode elements 2..5 (3 elements)
        let (ret_desc, parts) = file.decode_range(0, 0, &[(2, 3)], &DecodeOptions::default())?;
        assert_eq!(ret_desc.shape, vec![10]);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].len(), 3 * 4); // 3 float32s = 12 bytes
        Ok(())
    }

    #[test]
    fn test_local_decode_range_multiple_ranges()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("range_multi.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![16]);
        let data: Vec<u8> = (0..64).collect(); // 16 float32s = 64 bytes
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        let ranges = vec![(0u64, 4u64), (8u64, 4u64)];
        let (_, parts) = file.decode_range(0, 0, &ranges, &DecodeOptions::default())?;
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].len(), 4 * 4);
        assert_eq!(parts[1].len(), 4 * 4);
        Ok(())
    }

    #[test]
    fn test_local_decode_range_invalid_object_index()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("range_bad_obj.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        let result = file.decode_range(0, 5, &[(0, 1)], &DecodeOptions::default());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("out of range"),
            "expected 'out of range', got: {msg}"
        );
        Ok(())
    }

    #[test]
    fn test_local_decode_range_invalid_message_index()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("range_bad_msg.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        let result = file.decode_range(5, 0, &[(0, 1)], &DecodeOptions::default());
        assert!(result.is_err());
        Ok(())
    }

    // ── mmap append → Unsupported error ──────────────────────────────────

    #[cfg(feature = "mmap")]
    #[test]
    fn test_mmap_append_returns_unsupported() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("mmap_append.tgm");

        // Create and populate a file first
        {
            let mut file = TensogramFile::create(&path)?;
            let meta = make_global_meta();
            let desc = make_descriptor(vec![4]);
            let data = vec![0u8; 16];
            file.append(
                &meta,
                &[(&desc, data.as_slice())],
                &EncodeOptions::default(),
            )?;
        }

        // Open with mmap and try to append
        let mut mmap_file = TensogramFile::open_mmap(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        let result = mmap_file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        );
        match result {
            Ok(_) => panic!("expected error for append on mmap file"),
            Err(e) => {
                let err_msg = e.to_string();
                assert!(
                    err_msg.contains("memory-mapped") || err_msg.contains("mmap"),
                    "expected mmap-related error, got: {err_msg}"
                );
            }
        }
        Ok(())
    }

    // ── Async error cases ────────────────────────────────────────────────

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_async_open_nonexistent_file()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let result = TensogramFile::open_async("/tmp/nonexistent_tensogram_file_12345.tgm").await;
        match result {
            Ok(_) => panic!("expected error for nonexistent file"),
            Err(e) => {
                let err_msg = e.to_string();
                assert!(
                    err_msg.contains("not found") || err_msg.contains("NotFound"),
                    "expected not-found error, got: {err_msg}"
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_async_message_index_out_of_range()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("async_oor.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        let async_file = TensogramFile::open_async(&path).await?;
        let result = async_file.read_message_async(5).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("out of range"),
            "expected 'out of range', got: {err_msg}"
        );
        Ok(())
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_async_decode_metadata_out_of_range()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("async_meta_oor.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        let async_file = TensogramFile::open_async(&path).await?;
        let result = async_file.decode_metadata_async(10).await;
        assert!(result.is_err());
        Ok(())
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_async_decode_descriptors() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("async_descs.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        let async_file = TensogramFile::open_async(&path).await?;
        let (_decoded_meta, descriptors) = async_file.decode_descriptors_async(0).await?;
        assert_eq!(descriptors.len(), 1);
        assert_eq!(descriptors[0].shape, vec![4]);
        Ok(())
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_async_decode_object() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("async_obj.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![42u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        let async_file = TensogramFile::open_async(&path).await?;
        let (_decoded_meta, decoded_desc, decoded_data) = async_file
            .decode_object_async(0, 0, &DecodeOptions::default())
            .await?;
        assert_eq!(decoded_desc.shape, vec![4]);
        assert_eq!(decoded_data, data);
        Ok(())
    }

    // ── open_source with local path ──────────────────────────────────────

    #[test]
    fn test_open_source_local_path() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("local_source.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        let path_str = path.to_str().unwrap();
        let opened = TensogramFile::open_source(path_str)?;
        assert!(!opened.is_remote());
        assert_eq!(opened.message_count()?, 1);
        Ok(())
    }

    // ── path() and source() for local files ──────────────────────────────

    #[test]
    fn test_path_returns_some_for_local() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("path_test.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        let opened = TensogramFile::open(&path)?;
        assert!(opened.path().is_some());
        assert_eq!(opened.path().unwrap(), path.as_path());
        Ok(())
    }

    #[test]
    fn test_source_returns_path_string_for_local()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("source_test.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        let opened = TensogramFile::open(&path)?;
        let source = opened.source();
        assert!(
            source.contains("source_test.tgm"),
            "source() should contain the filename, got: {source}"
        );
        Ok(())
    }

    // ── open nonexistent file (sync) ─────────────────────────────────────

    #[test]
    fn test_open_nonexistent_file() {
        let result = TensogramFile::open("/tmp/nonexistent_tensogram_9999.tgm");
        match result {
            Ok(_) => panic!("expected error for nonexistent file"),
            Err(e) => {
                let err_msg = e.to_string();
                assert!(
                    err_msg.contains("not found"),
                    "expected 'not found', got: {err_msg}"
                );
            }
        }
    }

    // ── Coverage closers ─────────────────────────────────────────────────

    #[test]
    fn test_create_in_nested_path_creates_parent()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        // Exercises the `fs::create_dir_all(parent)` branch in `create()`.
        let dir = tempfile::tempdir()?;
        let deep_path = dir.path().join("a").join("b").join("c").join("deep.tgm");
        let mut file = TensogramFile::create(&deep_path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;
        assert_eq!(file.message_count()?, 1);
        assert!(deep_path.exists());
        Ok(())
    }

    /// Create a temp directory, drop its write bit, attempt to create a
    /// `.tgm` file inside, and assert that a typed `Io` error comes back.
    /// Restores the mode at the end so the tempdir cleans up correctly.
    ///
    /// Unix-only because Windows's permission model does not honour
    /// `chmod` in the same way. Skips gracefully if we detect that the
    /// current user bypasses directory-mode permission checks (running
    /// as root in a container, CAP_DAC_OVERRIDE set, etc.) — in those
    /// environments the test's premise cannot be deterministically met.
    #[cfg(unix)]
    #[test]
    fn test_create_in_nonwritable_location_returns_io_error()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir()?;
        let dir_path = dir.path().to_path_buf();

        // Drop all write bits (mode 0o555 = r-xr-xr-x).
        let original = std::fs::metadata(&dir_path)?.permissions();
        let mut readonly = original.clone();
        readonly.set_mode(0o555);
        std::fs::set_permissions(&dir_path, readonly)?;

        // Probe: try to write to the chmod'd dir via std::fs. If the
        // probe succeeds, we're in an environment where dir-mode checks
        // are bypassed (root, CAP_DAC_OVERRIDE, etc.) — skip gracefully.
        let probe_path = dir_path.join(".perm_probe");
        let probe_result = std::fs::File::create(&probe_path);
        if probe_result.is_ok() {
            // Clean up and skip.
            let _ = std::fs::remove_file(&probe_path);
            std::fs::set_permissions(&dir_path, original)?;
            return Ok(());
        }
        drop(probe_result);

        let target = dir_path.join("nope.tgm");
        let result = TensogramFile::create(&target);

        // Always restore permissions so tempdir can clean up.
        std::fs::set_permissions(&dir_path, original)?;

        // TensogramFile doesn't implement Debug, so use match instead of expect_err.
        let msg = match result {
            Ok(_) => panic!("expected Io error creating in non-writable dir, got Ok"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("cannot create")
                || msg.contains("permission")
                || msg.contains("read-only"),
            "expected permission-related error, got: {msg}"
        );
        Ok(())
    }

    #[test]
    fn test_read_message_from_deleted_file_errors()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        // TensogramFile is path-backed: each read reopens the file. If the
        // underlying path disappears out from under an open handle, the
        // next read_message call must surface a typed I/O error rather
        // than panic.
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("deleted.tgm");

        // Phase 1 — scope a write handle and let it flush on drop so the
        // file exists on disk before we delete it.
        {
            let mut writer = TensogramFile::create(&path)?;
            let meta = make_global_meta();
            let desc = make_descriptor(vec![2]);
            writer.append(
                &meta,
                &[(&desc, vec![0u8; 8].as_slice())],
                &EncodeOptions::default(),
            )?;
        }

        // Phase 2 — open a read handle, cache the message offsets, then
        // delete the underlying file. The handle is still alive; the next
        // disk-backed operation should fail.
        let reader = TensogramFile::open(&path)?;
        assert_eq!(reader.message_count()?, 1);

        std::fs::remove_file(&path)?;

        // read_message must return an Io error because the path is gone.
        let read_result = reader.read_message(0);
        assert!(
            read_result.is_err(),
            "expected read_message to fail after underlying file was deleted, got Ok"
        );
        let err_msg = read_result.unwrap_err().to_string();
        assert!(
            err_msg.contains("not found")
                || err_msg.contains("No such")
                || err_msg.contains("cannot"),
            "expected I/O error mentioning missing file, got: {err_msg}"
        );

        // Reopening the deleted path must also fail.
        assert!(TensogramFile::open(&path).is_err());
        Ok(())
    }

    /// Appending a message with zero descriptors produces a valid
    /// header-only message (preamble + metadata + postamble). Exercises
    /// the edge case in `encode_inner` where the data-objects loop runs
    /// zero times.
    #[test]
    fn test_append_empty_message() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("empty_append.tgm");
        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        file.append(&meta, &[], &EncodeOptions::default())?;
        assert_eq!(file.message_count()?, 1);
        // The one message has zero objects.
        let (_decoded_meta, objects) = file.decode_message(0, &DecodeOptions::default())?;
        assert_eq!(objects.len(), 0);
        Ok(())
    }

    #[test]
    fn test_file_iter_after_modification() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("modified.tgm");
        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![2]);
        let data = vec![0u8; 8];
        for _ in 0..3 {
            file.append(
                &meta,
                &[(&desc, data.as_slice())],
                &EncodeOptions::default(),
            )?;
        }
        assert_eq!(file.message_count()?, 3);
        drop(file);

        // Reopen and verify count persists
        let reopened = TensogramFile::open(&path)?;
        assert_eq!(reopened.message_count()?, 3);
        Ok(())
    }

    #[test]
    fn test_decode_message_out_of_range_clearly_errors()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("oor.tgm");
        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![2]);
        file.append(
            &meta,
            &[(&desc, vec![0u8; 8].as_slice())],
            &EncodeOptions::default(),
        )?;
        let result = file.decode_message(99, &DecodeOptions::default());
        assert!(result.is_err());
        Ok(())
    }

    // ── invalidate_offsets ────────────────────────────────────────────────

    #[test]
    fn test_invalidate_offsets_forces_rescan() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("invalidate.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        assert_eq!(file.message_count()?, 1);

        // Append another message
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;
        // After append, offsets are invalidated internally; verify rescan
        assert_eq!(file.message_count()?, 2);
        Ok(())
    }

    // ── open_source dispatches to local backend ──────────────────────────

    #[test]
    fn test_open_source_local_path_source_and_path_accessors()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("open_src_acc.tgm");

        // Create a file with one message
        {
            let mut file = TensogramFile::create(&path)?;
            let meta = make_global_meta();
            let desc = make_descriptor(vec![4]);
            let data = vec![7u8; 16];
            file.append(
                &meta,
                &[(&desc, data.as_slice())],
                &EncodeOptions::default(),
            )?;
        }

        let path_str = path.to_str().unwrap();
        let opened = TensogramFile::open_source(path_str)?;

        // source() should include the filename
        assert!(
            opened.source().contains("open_src_acc.tgm"),
            "source() = {}",
            opened.source()
        );
        // path() should return the exact path for local files
        assert_eq!(opened.path().unwrap(), path.as_path());
        // is_remote() is false for local files
        assert!(!opened.is_remote());
        // Can read the message back
        let count = opened.message_count()?;
        assert_eq!(count, 1);
        Ok(())
    }

    // ── open nonexistent via open_source ──────────────────────────────────

    #[test]
    fn test_open_source_nonexistent() {
        let result = TensogramFile::open_source("/tmp/no_such_file_tensogram_abc.tgm");
        match result {
            Ok(_) => panic!("expected error for nonexistent file"),
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("not found"),
                    "expected 'not found', got: {msg}"
                );
            }
        }
    }

    // ── message index out of range (sync) ────────────────────────────────

    #[test]
    fn test_read_message_index_out_of_range() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("oor.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        // Index 0 works
        assert!(file.read_message(0).is_ok());

        // Index 1 is out of range
        let result = file.read_message(1);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("out of range"),
            "expected 'out of range', got: {msg}"
        );

        // Large index
        let result = file.read_message(999);
        assert!(result.is_err());
        Ok(())
    }

    // ── decode_range valid extraction ─────────────────────────────────────

    #[test]
    fn test_local_decode_range_full_tensor() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("range_full.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![8]);
        let data: Vec<u8> = (0..32).collect(); // 8 float32s = 32 bytes
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        // Decode the entire range
        let (ret_desc, parts) = file.decode_range(0, 0, &[(0, 8)], &DecodeOptions::default())?;
        assert_eq!(ret_desc.shape, vec![8]);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].len(), 32); // all 8 float32s
        assert_eq!(parts[0], data);
        Ok(())
    }

    // ── decode_object on local file ──────────────────────────────────────

    #[test]
    fn test_local_decode_object_valid() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("dec_obj.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![99u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        let (_decoded_meta, decoded_desc, decoded_data) =
            file.decode_object(0, 0, &DecodeOptions::default())?;
        assert_eq!(decoded_desc.shape, vec![4]);
        assert_eq!(decoded_data, data);
        Ok(())
    }

    #[test]
    fn test_local_decode_object_out_of_range() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("dec_obj_oor.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        // Object index 5 doesn't exist (only object 0)
        let result = file.decode_object(0, 5, &DecodeOptions::default());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("out of range"),
            "expected 'out of range', got: {msg}"
        );
        Ok(())
    }

    // ── Backend::Local — is_remote() returns false ────────────────────────

    #[test]
    fn test_is_remote_false_for_local() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("not_remote.tgm");
        let file = TensogramFile::create(&path)?;
        assert!(!file.is_remote());
        Ok(())
    }

    #[test]
    fn test_is_remote_false_for_opened_local() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("not_remote2.tgm");
        {
            let mut f = TensogramFile::create(&path)?;
            let meta = make_global_meta();
            let desc = make_descriptor(vec![2]);
            f.append(&meta, &[(&desc, &[0u8; 8])], &EncodeOptions::default())?;
        }
        let opened = TensogramFile::open(&path)?;
        assert!(!opened.is_remote());
        Ok(())
    }

    // ── decode_descriptors via file ──────────────────────────────────────

    #[test]
    fn test_file_decode_descriptors_msg_out_of_range()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("desc_oor.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        // Message index 10 is out of range
        let result = file.decode_descriptors(10);
        assert!(result.is_err());
        Ok(())
    }

    // ── decode_metadata via file with out-of-range index ─────────────────

    #[test]
    fn test_file_decode_metadata_out_of_range()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("meta_oor.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        let result = file.decode_metadata(99);
        assert!(result.is_err());
        Ok(())
    }

    // ── invalidate_offsets explicit ───────────────────────────────────────

    #[test]
    fn test_invalidate_offsets_explicit() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("inv_explicit.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;
        assert_eq!(file.message_count()?, 1);

        // Manually invalidate and verify rescan works
        file.invalidate_offsets();
        assert_eq!(file.message_count()?, 1);
        Ok(())
    }

    // ── open file then re-open and read ──────────────────────────────────

    #[test]
    fn test_open_existing_file_and_read() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("reopen.tgm");

        // Create and write
        {
            let mut file = TensogramFile::create(&path)?;
            let meta = make_global_meta();
            let desc = make_descriptor(vec![4]);
            let data = vec![55u8; 16];
            file.append(
                &meta,
                &[(&desc, data.as_slice())],
                &EncodeOptions::default(),
            )?;
        }

        // Re-open with open()
        let file = TensogramFile::open(&path)?;
        assert_eq!(file.message_count()?, 1);
        let (_meta, objs) = file.decode_message(0, &DecodeOptions::default())?;
        assert_eq!(objs[0].1, vec![55u8; 16]);
        Ok(())
    }

    // ── decode_range out-of-bounds element range ─────────────────────────

    #[test]
    fn test_local_decode_range_out_of_bounds_elements()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("range_oob.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]); // 4 float32s
        let data = vec![0u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        // Request range that exceeds the tensor: offset=2, count=10 but only 4 elements
        let result = file.decode_range(0, 0, &[(2, 10)], &DecodeOptions::default());
        assert!(result.is_err(), "expected error for out-of-bounds range");
        Ok(())
    }

    // ── path() returns None for remote-like scenarios ────────────────────
    // (We can only test the local branch here since remote requires feature)

    #[test]
    fn test_path_some_for_created_file() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("path_created.tgm");
        let file = TensogramFile::create(&path)?;
        assert!(file.path().is_some());
        assert_eq!(file.path().unwrap(), path.as_path());
        Ok(())
    }

    // ── async decode_range ───────────────────────────────────────────────

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_async_decode_range() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("async_range.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![10]);
        let data: Vec<u8> = (0..40).collect(); // 10 float32s
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        let async_file = TensogramFile::open_async(&path).await?;
        let (ret_desc, parts) = async_file
            .decode_range_async(0, 0, &[(0, 5)], &DecodeOptions::default())
            .await?;
        assert_eq!(ret_desc.shape, vec![10]);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].len(), 5 * 4); // 5 float32s = 20 bytes
        Ok(())
    }

    // ── async decode_object ──────────────────────────────────────────────

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_async_decode_object_out_of_range()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("async_obj_oor.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        let async_file = TensogramFile::open_async(&path).await?;
        let result = async_file
            .decode_object_async(0, 99, &DecodeOptions::default())
            .await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("out of range"),
            "expected 'out of range', got: {msg}"
        );
        Ok(())
    }

    // ── async message_count_async ────────────────────────────────────────

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_async_message_count() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("async_count.tgm");

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

        let async_file = TensogramFile::open_async(&path).await?;
        let count = async_file.message_count_async().await?;
        assert_eq!(count, 2);
        Ok(())
    }

    // ── async decode_descriptors ─────────────────────────────────────────

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_async_decode_descriptors_out_of_range()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("async_descs_oor.tgm");

        let mut file = TensogramFile::create(&path)?;
        let meta = make_global_meta();
        let desc = make_descriptor(vec![4]);
        let data = vec![0u8; 16];
        file.append(
            &meta,
            &[(&desc, data.as_slice())],
            &EncodeOptions::default(),
        )?;

        let async_file = TensogramFile::open_async(&path).await?;
        let result = async_file.decode_descriptors_async(10).await;
        assert!(result.is_err());
        Ok(())
    }
}
