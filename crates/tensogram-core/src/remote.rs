//! Remote object store backend for `TensogramFile`.
//!
//! Enables reading `.tgm` files from S3, GCS, Azure, and HTTP(S) via
//! the `object_store` crate. Feature-gated behind `remote`.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use object_store::path::Path as ObjectPath;
use object_store::{GetOptions, GetRange, ObjectStore, ObjectStoreExt};
use url::Url;

use crate::decode::DecodeOptions;
use crate::error::{Result, TensogramError};
use crate::framing;
use crate::metadata;
use crate::types::{DataObjectDescriptor, GlobalMetadata, IndexFrame};
use crate::wire::{
    FrameHeader, FrameType, MessageFlags, Preamble, FRAME_END, FRAME_HEADER_SIZE, MAGIC,
    POSTAMBLE_SIZE, PREAMBLE_SIZE,
};

// ── URL scheme detection ─────────────────────────────────────────────────────

const REMOTE_SCHEMES: &[&str] = &["s3", "s3a", "gs", "az", "azure", "http", "https"];

/// Run an async operation synchronously, safe from nested-runtime panics.
///
/// Spawns a dedicated thread with a fresh single-threaded tokio runtime.
/// This allows `TensogramFile` sync methods to work regardless of whether
/// the caller is already inside an async context (e.g. Python event loop,
/// `#[tokio::test]`).
fn block_on_thread<T, F, Fut>(store: Arc<dyn ObjectStore>, f: F) -> Result<T>
where
    T: Send,
    F: FnOnce(Arc<dyn ObjectStore>) -> Fut + Send,
    Fut: std::future::Future<Output = std::result::Result<T, object_store::Error>>,
{
    std::thread::scope(|s| {
        s.spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| TensogramError::Remote(format!("tokio runtime: {e}")))?;
            rt.block_on(f(store))
                .map_err(|e| TensogramError::Remote(e.to_string()))
        })
        .join()
        .unwrap_or_else(|_| Err(TensogramError::Remote("I/O thread panicked".to_string())))
    })
}

pub fn is_remote_url(source: &str) -> bool {
    match source.find("://") {
        Some(pos) => {
            let scheme = &source[..pos];
            REMOTE_SCHEMES
                .iter()
                .any(|s| s.eq_ignore_ascii_case(scheme))
        }
        None => false,
    }
}

// ── Cached per-message layout ────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct MessageLayout {
    offset: u64,
    length: u64,
    preamble: Preamble,
    index: Option<IndexFrame>,
    global_metadata: Option<GlobalMetadata>,
}

// ── Remote backend ───────────────────────────────────────────────────────────

pub(crate) struct RemoteBackend {
    source_url: String,
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    file_size: u64,
    layouts: Vec<MessageLayout>,
}

impl std::fmt::Debug for RemoteBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteBackend")
            .field("source", &self.source_url)
            .field("file_size", &self.file_size)
            .field("messages", &self.layouts.len())
            .finish()
    }
}

impl RemoteBackend {
    pub(crate) fn source_url(&self) -> &str {
        &self.source_url
    }

    pub(crate) fn open(source: &str, storage_options: &BTreeMap<String, String>) -> Result<Self> {
        let url = Url::parse(source)
            .map_err(|e| TensogramError::Remote(format!("invalid URL '{source}': {e}")))?;

        let mut opts: Vec<(String, String)> = storage_options
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        if url.scheme() == "http" && !opts.iter().any(|(k, _)| k == "allow_http") {
            opts.push(("allow_http".to_string(), "true".to_string()));
        }
        let (store, path) = object_store::parse_url_opts(&url, opts)
            .map_err(|e| TensogramError::Remote(format!("cannot open '{source}': {e}")))?;

        let store: Arc<dyn ObjectStore> = Arc::from(store);

        let meta = block_on_thread(store.clone(), |s| {
            let path = path.clone();
            async move { s.head(&path).await }
        })?;

        let file_size = meta.size as u64;
        if file_size < (PREAMBLE_SIZE + POSTAMBLE_SIZE) as u64 {
            return Err(TensogramError::Remote(format!(
                "remote file too small ({file_size} bytes)"
            )));
        }

        let mut backend = RemoteBackend {
            source_url: source.to_string(),
            store,
            path,
            file_size,
            layouts: Vec::new(),
        };
        backend.scan_messages()?;
        Ok(backend)
    }

    // ── Range reads ──────────────────────────────────────────────────────

    fn get_range(&self, range: Range<u64>) -> Result<Bytes> {
        let path = self.path.clone();
        block_on_thread(self.store.clone(), move |s| async move {
            s.get_range(&path, range).await
        })
    }

    #[allow(dead_code)]
    fn get_suffix(&self, nbytes: u64) -> Result<Bytes> {
        let path = self.path.clone();
        block_on_thread(self.store.clone(), move |s| async move {
            let opts = GetOptions {
                range: Some(GetRange::Suffix(nbytes)),
                ..Default::default()
            };
            let result = s.get_opts(&path, opts).await?;
            result.bytes().await
        })
    }

    // ── Message scanning ─────────────────────────────────────────────────

    fn scan_messages(&mut self) -> Result<()> {
        let min_message_size = (PREAMBLE_SIZE + POSTAMBLE_SIZE) as u64;
        let mut pos: u64 = 0;

        // Fast path: assume contiguous messages starting at offset 0.
        // Each iteration: 1 GET for the preamble, then jump by total_length.
        // Cost: 1 GET per message (not per byte).
        while pos + min_message_size <= self.file_size {
            let preamble_bytes = self.get_range(pos..pos + PREAMBLE_SIZE as u64)?;
            if &preamble_bytes[..MAGIC.len()] != MAGIC {
                break;
            }

            let preamble = match Preamble::read_from(&preamble_bytes) {
                Ok(p) => p,
                Err(_) => break,
            };

            let msg_len = preamble.total_length;
            if msg_len == 0 || msg_len < min_message_size || pos + msg_len > self.file_size {
                break;
            }

            self.layouts.push(MessageLayout {
                offset: pos,
                length: msg_len,
                preamble,
                index: None,
                global_metadata: None,
            });

            pos += msg_len;
        }

        if self.layouts.is_empty() {
            return Err(TensogramError::Remote(
                "no valid messages found in remote file".to_string(),
            ));
        }

        Ok(())
    }

    // ── Layout discovery (metadata + index for a single message) ─────────

    fn ensure_layout(&mut self, msg_idx: usize) -> Result<()> {
        if msg_idx >= self.layouts.len() {
            return Err(TensogramError::Framing(format!(
                "message index {} out of range (count={})",
                msg_idx,
                self.layouts.len()
            )));
        }
        if self.layouts[msg_idx].global_metadata.is_some() {
            return Ok(());
        }

        let flags = self.layouts[msg_idx].preamble.flags;

        // PR1 scope: only header-indexed (buffered) messages are supported.
        // Footer-indexed (streaming) messages will be added in a follow-up PR
        // once the StreamingEncoder index lengths are verified as frame lengths.
        if flags.has(MessageFlags::HEADER_METADATA) && flags.has(MessageFlags::HEADER_INDEX) {
            self.discover_header_layout(msg_idx)?;
        } else {
            return Err(TensogramError::Remote(
                "remote access requires header-indexed messages (both HEADER_METADATA and \
                 HEADER_INDEX flags); header-metadata-only and footer-only layouts are not supported"
                    .to_string(),
            ));
        }

        Ok(())
    }

    fn discover_header_layout(&mut self, msg_idx: usize) -> Result<()> {
        let layout = &self.layouts[msg_idx];
        let msg_offset = layout.offset;
        let msg_len = layout.length;

        // Read a generous initial chunk: up to 256KB or the message size.
        // Header metadata + index are typically a few KB.
        let chunk_size = msg_len.min(256 * 1024);
        let header_bytes = self.get_range(msg_offset..msg_offset + chunk_size)?;

        self.parse_header_frames(msg_idx, &header_bytes)
    }

    fn parse_header_frames(&mut self, msg_idx: usize, buf: &[u8]) -> Result<()> {
        let min_frame_size = FRAME_HEADER_SIZE + FRAME_END.len();
        let mut pos = PREAMBLE_SIZE;

        while pos + FRAME_HEADER_SIZE <= buf.len() {
            if &buf[pos..pos + 2] != b"FR" {
                pos += 1;
                continue;
            }
            let fh = FrameHeader::read_from(&buf[pos..])?;
            let frame_total = fh.total_length as usize;

            if frame_total < min_frame_size {
                return Err(TensogramError::Remote(format!(
                    "frame total_length ({frame_total}) smaller than minimum ({min_frame_size})"
                )));
            }
            if pos + frame_total > buf.len() {
                break;
            }

            let frame_end = pos + frame_total;
            if &buf[frame_end - FRAME_END.len()..frame_end] != FRAME_END {
                return Err(TensogramError::Remote(
                    "frame missing ENDF trailer".to_string(),
                ));
            }

            let payload = &buf[pos + FRAME_HEADER_SIZE..frame_end - FRAME_END.len()];

            match fh.frame_type {
                FrameType::HeaderMetadata => {
                    let meta = metadata::cbor_to_global_metadata(payload)?;
                    self.layouts[msg_idx].global_metadata = Some(meta);
                }
                FrameType::HeaderIndex => {
                    let idx = metadata::cbor_to_index(payload)?;
                    self.layouts[msg_idx].index = Some(idx);
                }
                FrameType::DataObject | FrameType::PrecederMetadata => {
                    break;
                }
                _ => {}
            }

            let aligned = (pos + frame_total + 7) & !7;
            pos = aligned.min(buf.len());
        }

        if self.layouts[msg_idx].global_metadata.is_none() {
            return Err(TensogramError::Remote(
                "header region did not contain a metadata frame".to_string(),
            ));
        }
        if self.layouts[msg_idx].index.is_none() {
            return Err(TensogramError::Remote(
                "header region did not contain an index frame (header chunk may be too small)"
                    .to_string(),
            ));
        }

        Ok(())
    }

    // ── Public API used by TensogramFile ─────────────────────────────────

    #[allow(dead_code)]
    pub(crate) fn message_count(&self) -> usize {
        self.layouts.len()
    }

    pub(crate) fn message_offsets(&self) -> Result<Vec<(usize, usize)>> {
        self.layouts
            .iter()
            .map(|l| {
                let offset = usize::try_from(l.offset).map_err(|_| {
                    TensogramError::Remote(format!(
                        "message offset {} does not fit in usize",
                        l.offset
                    ))
                })?;
                let length = usize::try_from(l.length).map_err(|_| {
                    TensogramError::Remote(format!(
                        "message length {} does not fit in usize",
                        l.length
                    ))
                })?;
                Ok((offset, length))
            })
            .collect()
    }

    pub(crate) fn read_message(&self, msg_idx: usize) -> Result<Vec<u8>> {
        let layout = self.layouts.get(msg_idx).ok_or_else(|| {
            TensogramError::Framing(format!(
                "message index {} out of range (count={})",
                msg_idx,
                self.layouts.len()
            ))
        })?;
        let bytes = self.get_range(layout.offset..layout.offset + layout.length)?;
        Ok(bytes.to_vec())
    }

    pub(crate) fn read_metadata(&mut self, msg_idx: usize) -> Result<GlobalMetadata> {
        self.ensure_layout(msg_idx)?;
        self.layouts[msg_idx]
            .global_metadata
            .clone()
            .ok_or_else(|| TensogramError::Remote("metadata not found".to_string()))
    }

    pub(crate) fn read_descriptors(
        &mut self,
        msg_idx: usize,
    ) -> Result<(GlobalMetadata, Vec<DataObjectDescriptor>)> {
        self.ensure_layout(msg_idx)?;
        // If we have an index, fetch each indexed object frame and extract
        // the descriptor from it. Otherwise fall back to full message decode.
        let layout = &self.layouts[msg_idx];
        let msg_offset = layout.offset;

        if let Some(ref index) = layout.index {
            if index.offsets.len() != index.lengths.len() {
                return Err(TensogramError::Remote(format!(
                    "corrupt index: offsets.len()={} != lengths.len()={}",
                    index.offsets.len(),
                    index.lengths.len()
                )));
            }

            let meta = layout
                .global_metadata
                .clone()
                .ok_or_else(|| TensogramError::Remote("metadata not cached".to_string()))?;

            let msg_length = layout.length;
            let mut descriptors = Vec::with_capacity(index.offsets.len());
            for i in 0..index.offsets.len() {
                let range = Self::checked_frame_range(
                    msg_offset,
                    msg_length,
                    index.offsets[i],
                    index.lengths[i],
                )?;
                let frame_bytes = self.get_range(range)?;
                let (desc, _payload, _consumed) = framing::decode_data_object_frame(&frame_bytes)?;
                descriptors.push(desc);
            }
            Ok((meta, descriptors))
        } else {
            // No index — fall back to full message download
            let msg_bytes = self.read_message(msg_idx)?;
            crate::decode::decode_descriptors(&msg_bytes)
        }
    }

    pub(crate) fn read_object(
        &mut self,
        msg_idx: usize,
        obj_idx: usize,
        options: &DecodeOptions,
    ) -> Result<(GlobalMetadata, DataObjectDescriptor, Vec<u8>)> {
        self.ensure_layout(msg_idx)?;
        let layout = &self.layouts[msg_idx];
        let msg_offset = layout.offset;

        if let Some(ref index) = layout.index {
            Self::validate_index_access(index, obj_idx)?;

            let meta = layout
                .global_metadata
                .clone()
                .ok_or_else(|| TensogramError::Remote("metadata not cached".to_string()))?;

            let range = Self::checked_frame_range(
                msg_offset,
                layout.length,
                index.offsets[obj_idx],
                index.lengths[obj_idx],
            )?;
            let frame_bytes = self.get_range(range)?;

            let (desc, payload, _consumed) = framing::decode_data_object_frame(&frame_bytes)?;

            let decoded = crate::decode::decode_single_object(&desc, payload, options)?;

            Ok((meta, desc, decoded))
        } else {
            // No index — fall back to full message download
            let msg_bytes = self.read_message(msg_idx)?;
            crate::decode::decode_object(&msg_bytes, obj_idx, options)
        }
    }

    fn validate_index_access(index: &IndexFrame, obj_idx: usize) -> Result<()> {
        if index.offsets.len() != index.lengths.len() {
            return Err(TensogramError::Remote(format!(
                "corrupt index: offsets.len()={} != lengths.len()={}",
                index.offsets.len(),
                index.lengths.len()
            )));
        }
        if obj_idx >= index.offsets.len() {
            return Err(TensogramError::Object(format!(
                "object index {} out of range (count={})",
                obj_idx,
                index.offsets.len()
            )));
        }
        Ok(())
    }

    fn checked_frame_range(
        msg_offset: u64,
        msg_length: u64,
        frame_offset_in_msg: u64,
        frame_length: u64,
    ) -> Result<Range<u64>> {
        let start = msg_offset
            .checked_add(frame_offset_in_msg)
            .ok_or_else(|| TensogramError::Remote("frame offset overflow".to_string()))?;
        let end = start
            .checked_add(frame_length)
            .ok_or_else(|| TensogramError::Remote("frame end overflow".to_string()))?;
        let msg_end = msg_offset
            .checked_add(msg_length)
            .ok_or_else(|| TensogramError::Remote("message end overflow".to_string()))?;
        if end > msg_end {
            return Err(TensogramError::Remote(format!(
                "indexed frame {start}..{end} exceeds message boundary {msg_end}"
            )));
        }
        Ok(start..end)
    }
}
