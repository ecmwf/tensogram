//! Remote object store backend for `TensogramFile`.
//!
//! Enables reading `.tgm` files from S3, GCS, Azure, and HTTP(S) via
//! the `object_store` crate. Feature-gated behind `remote`.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, OnceLock};

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
    DataObjectFlags, FrameHeader, FrameType, MessageFlags, Postamble, Preamble,
    DATA_OBJECT_FOOTER_SIZE, FRAME_END, FRAME_HEADER_SIZE, MAGIC, POSTAMBLE_SIZE, PREAMBLE_SIZE,
};

// ── URL scheme detection ─────────────────────────────────────────────────────

const REMOTE_SCHEMES: &[&str] = &["s3", "s3a", "gs", "az", "azure", "http", "https"];

// ── Shared tokio runtime ─────────────────────────────────────────────────────

/// Process-wide shared tokio runtime for remote I/O.
///
/// Wrapped in `Result` so that `get_or_init` (which runs the closure
/// exactly once, no races) can cache a build failure without panicking.
static SHARED_RUNTIME: OnceLock<std::result::Result<tokio::runtime::Runtime, String>> =
    OnceLock::new();

fn shared_runtime() -> Result<&'static tokio::runtime::Runtime> {
    SHARED_RUNTIME
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .thread_name("tensogram-remote-io")
                .build()
                .map_err(|e| format!("tokio runtime: {e}"))
        })
        .as_ref()
        .map_err(|e| TensogramError::Remote(e.clone()))
}

/// Run an async operation synchronously using the shared runtime.
///
/// Three strategies based on the calling context:
///
/// - **Not in async context** (Python, CLI): direct `handle.block_on()`,
///   no extra thread creation.
/// - **Inside multi-thread tokio runtime** (`#[tokio::test]`, server
///   handler): `block_in_place` + `handle.block_on()`, which tells
///   tokio to spawn a replacement worker so the current one can block
///   without causing runtime starvation.
/// - **Inside current-thread tokio runtime**: scoped thread fallback,
///   since `block_in_place` is not supported on single-threaded
///   runtimes.
///
/// In all cases the shared runtime's event loop and connection pool
/// are reused, eliminating the per-call runtime creation overhead of
/// the old `block_on_thread` pattern.
fn block_on_shared<T, Fut>(future: Fut) -> Result<T>
where
    T: Send,
    Fut: std::future::Future<Output = std::result::Result<T, object_store::Error>> + Send,
{
    let rt = shared_runtime()?;
    let handle = rt.handle().clone();

    match tokio::runtime::Handle::try_current() {
        Err(_) => {
            // Not in an async context — drive the future directly.
            handle
                .block_on(future)
                .map_err(|e| TensogramError::Remote(e.to_string()))
        }
        Ok(current) if current.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            // Multi-thread runtime: block_in_place lets tokio spawn a
            // replacement worker so this one can block safely.
            tokio::task::block_in_place(|| {
                handle
                    .block_on(future)
                    .map_err(|e| TensogramError::Remote(e.to_string()))
            })
        }
        Ok(_) => {
            // Current-thread runtime: block_in_place is unsupported,
            // fall back to a scoped thread.
            std::thread::scope(|s| {
                match s
                    .spawn(|| {
                        handle
                            .block_on(future)
                            .map_err(|e| TensogramError::Remote(e.to_string()))
                    })
                    .join()
                {
                    Ok(result) => result,
                    Err(_) => Err(TensogramError::Remote(
                        "remote I/O thread panicked".to_string(),
                    )),
                }
            })
        }
    }
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
    next_scan_offset: u64,
    scan_complete: bool,
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

        let head_store = store.clone();
        let head_path = path.clone();
        let meta = block_on_shared(async move { head_store.head(&head_path).await })?;

        let file_size = meta.size;
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
            next_scan_offset: 0,
            scan_complete: false,
        };
        backend.scan_next()?;
        if backend.layouts.is_empty() {
            return Err(TensogramError::Remote(
                "no valid messages found in remote file".to_string(),
            ));
        }
        Ok(backend)
    }

    // ── Range reads ──────────────────────────────────────────────────────

    fn get_range(&self, range: Range<u64>) -> Result<Bytes> {
        let store = self.store.clone();
        let path = self.path.clone();
        block_on_shared(async move { store.get_range(&path, range).await })
    }

    #[allow(dead_code)]
    fn get_suffix(&self, nbytes: u64) -> Result<Bytes> {
        let store = self.store.clone();
        let path = self.path.clone();
        block_on_shared(async move {
            let opts = GetOptions {
                range: Some(GetRange::Suffix(nbytes)),
                ..Default::default()
            };
            let result = store.get_opts(&path, opts).await?;
            result.bytes().await
        })
    }

    // ── Message scanning ─────────────────────────────────────────────────

    fn scan_next(&mut self) -> Result<()> {
        if self.scan_complete {
            return Ok(());
        }
        let min_message_size = (PREAMBLE_SIZE + POSTAMBLE_SIZE) as u64;
        let pos = self.next_scan_offset;

        if pos + min_message_size > self.file_size {
            self.scan_complete = true;
            return Ok(());
        }

        let preamble_bytes = self.get_range(pos..pos + PREAMBLE_SIZE as u64)?;
        if &preamble_bytes[..MAGIC.len()] != MAGIC {
            self.scan_complete = true;
            return Ok(());
        }

        let preamble = match Preamble::read_from(&preamble_bytes) {
            Ok(p) => p,
            Err(_) => {
                self.scan_complete = true;
                return Ok(());
            }
        };

        let msg_len = preamble.total_length;

        if msg_len == 0 {
            let remaining = self.file_size - pos;
            if remaining < min_message_size {
                self.scan_complete = true;
                return Ok(());
            }
            let end_magic_pos = self.file_size - crate::wire::END_MAGIC.len() as u64;
            let end_bytes =
                self.get_range(end_magic_pos..end_magic_pos + crate::wire::END_MAGIC.len() as u64)?;
            if &end_bytes[..] != crate::wire::END_MAGIC {
                self.scan_complete = true;
                return Ok(());
            }
            self.layouts.push(MessageLayout {
                offset: pos,
                length: remaining,
                preamble,
                index: None,
                global_metadata: None,
            });
            self.scan_complete = true;
            return Ok(());
        }

        let msg_end = match pos.checked_add(msg_len) {
            Some(end) if msg_len >= min_message_size && end <= self.file_size => end,
            _ => {
                self.scan_complete = true;
                return Ok(());
            }
        };

        self.layouts.push(MessageLayout {
            offset: pos,
            length: msg_len,
            preamble,
            index: None,
            global_metadata: None,
        });
        self.next_scan_offset = msg_end;
        Ok(())
    }

    fn ensure_message(&mut self, msg_idx: usize) -> Result<()> {
        while msg_idx >= self.layouts.len() && !self.scan_complete {
            self.scan_next()?;
        }
        if msg_idx >= self.layouts.len() {
            return Err(TensogramError::Framing(format!(
                "message index {} out of range (count={})",
                msg_idx,
                self.layouts.len()
            )));
        }
        Ok(())
    }

    fn scan_all(&mut self) -> Result<()> {
        while !self.scan_complete {
            self.scan_next()?;
        }
        Ok(())
    }

    fn scan_and_discover_next(&mut self) -> Result<()> {
        if self.scan_complete {
            return Ok(());
        }
        let min_message_size = (PREAMBLE_SIZE + POSTAMBLE_SIZE) as u64;
        let pos = self.next_scan_offset;

        if pos + min_message_size > self.file_size {
            self.scan_complete = true;
            return Ok(());
        }

        let chunk_size = (self.file_size - pos).min(256 * 1024);
        let chunk = self.get_range(pos..pos + chunk_size)?;

        if chunk.len() < PREAMBLE_SIZE || &chunk[..MAGIC.len()] != MAGIC {
            self.scan_complete = true;
            return Ok(());
        }

        let preamble = match Preamble::read_from(&chunk[..PREAMBLE_SIZE]) {
            Ok(p) => p,
            Err(_) => {
                self.scan_complete = true;
                return Ok(());
            }
        };

        let msg_len = preamble.total_length;

        if msg_len == 0 {
            let remaining = self.file_size - pos;
            if remaining < min_message_size {
                self.scan_complete = true;
                return Ok(());
            }
            let end_magic_pos = self.file_size - crate::wire::END_MAGIC.len() as u64;
            let end_bytes =
                self.get_range(end_magic_pos..end_magic_pos + crate::wire::END_MAGIC.len() as u64)?;
            if &end_bytes[..] != crate::wire::END_MAGIC {
                self.scan_complete = true;
                return Ok(());
            }
            self.layouts.push(MessageLayout {
                offset: pos,
                length: remaining,
                preamble,
                index: None,
                global_metadata: None,
            });
            self.scan_complete = true;
            return Ok(());
        }

        let msg_end = match pos.checked_add(msg_len) {
            Some(end) if msg_len >= min_message_size && end <= self.file_size => end,
            _ => {
                self.scan_complete = true;
                return Ok(());
            }
        };

        let flags = preamble.flags;
        let msg_idx = self.layouts.len();

        self.layouts.push(MessageLayout {
            offset: pos,
            length: msg_len,
            preamble,
            index: None,
            global_metadata: None,
        });
        self.next_scan_offset = msg_end;

        if flags.has(MessageFlags::HEADER_METADATA) && flags.has(MessageFlags::HEADER_INDEX) {
            let chunk_end = (msg_len as usize).min(chunk.len());
            self.parse_header_frames(msg_idx, &chunk[..chunk_end])?;
        }

        Ok(())
    }

    fn ensure_layout_eager(&mut self, msg_idx: usize) -> Result<()> {
        while msg_idx >= self.layouts.len() && !self.scan_complete {
            self.scan_and_discover_next()?;
        }
        if msg_idx >= self.layouts.len() {
            return Err(TensogramError::Framing(format!(
                "message index {} out of range (count={})",
                msg_idx,
                self.layouts.len()
            )));
        }
        if self.layouts[msg_idx].global_metadata.is_some() && self.layouts[msg_idx].index.is_some()
        {
            return Ok(());
        }
        self.ensure_layout(msg_idx)
    }

    // ── Layout discovery (metadata + index for a single message) ─────────

    fn ensure_layout(&mut self, msg_idx: usize) -> Result<()> {
        self.ensure_message(msg_idx)?;
        if self.layouts[msg_idx].global_metadata.is_some() && self.layouts[msg_idx].index.is_some()
        {
            return Ok(());
        }

        let flags = self.layouts[msg_idx].preamble.flags;

        if flags.has(MessageFlags::HEADER_METADATA) && flags.has(MessageFlags::HEADER_INDEX) {
            self.discover_header_layout(msg_idx)?;
        } else if flags.has(MessageFlags::FOOTER_METADATA) && flags.has(MessageFlags::FOOTER_INDEX)
        {
            self.discover_footer_layout(msg_idx)?;
        } else {
            return Err(TensogramError::Remote(
                "remote access requires header-indexed or footer-indexed messages".to_string(),
            ));
        }

        Ok(())
    }

    fn discover_footer_layout(&mut self, msg_idx: usize) -> Result<()> {
        let msg_offset = self.layouts[msg_idx].offset;
        let msg_len = self.layouts[msg_idx].length;

        let pa_offset = msg_offset
            .checked_add(msg_len)
            .and_then(|end| end.checked_sub(POSTAMBLE_SIZE as u64))
            .ok_or_else(|| TensogramError::Remote("postamble offset overflow".to_string()))?;
        let pa_bytes = self.get_range(pa_offset..pa_offset + POSTAMBLE_SIZE as u64)?;
        let postamble = Postamble::read_from(&pa_bytes)?;

        if postamble.first_footer_offset < PREAMBLE_SIZE as u64 {
            return Err(TensogramError::Remote(format!(
                "first_footer_offset ({}) is before preamble end ({})",
                postamble.first_footer_offset, PREAMBLE_SIZE
            )));
        }
        let footer_start = msg_offset
            .checked_add(postamble.first_footer_offset)
            .ok_or_else(|| TensogramError::Remote("footer offset overflow".to_string()))?;
        let footer_end = pa_offset;
        if footer_start >= footer_end {
            return Err(TensogramError::Remote(
                "first_footer_offset points at or past postamble".to_string(),
            ));
        }
        let footer_bytes = self.get_range(footer_start..footer_end)?;

        self.parse_footer_frames(msg_idx, &footer_bytes)
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
            let frame_total = usize::try_from(fh.total_length).map_err(|_| {
                TensogramError::Remote("frame total_length does not fit in usize".to_string())
            })?;

            if frame_total < min_frame_size {
                return Err(TensogramError::Remote(format!(
                    "frame total_length ({frame_total}) smaller than minimum ({min_frame_size})"
                )));
            }
            let frame_end = match pos.checked_add(frame_total) {
                Some(end) if end <= buf.len() => end,
                _ => break,
            };

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

            let aligned = (frame_end.saturating_add(7)) & !7;
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

    fn parse_footer_frames(&mut self, msg_idx: usize, buf: &[u8]) -> Result<()> {
        let min_frame_size = FRAME_HEADER_SIZE + FRAME_END.len();
        let mut pos = 0;

        while pos + FRAME_HEADER_SIZE <= buf.len() {
            if &buf[pos..pos + 2] != b"FR" {
                pos += 1;
                continue;
            }
            let fh = FrameHeader::read_from(&buf[pos..])?;
            let frame_total = usize::try_from(fh.total_length).map_err(|_| {
                TensogramError::Remote(
                    "footer frame total_length does not fit in usize".to_string(),
                )
            })?;

            if frame_total < min_frame_size {
                return Err(TensogramError::Remote(format!(
                    "footer frame total_length ({frame_total}) smaller than minimum ({min_frame_size})"
                )));
            }
            let frame_end = match pos.checked_add(frame_total) {
                Some(end) if end <= buf.len() => end,
                _ => break,
            };

            if &buf[frame_end - FRAME_END.len()..frame_end] != FRAME_END {
                return Err(TensogramError::Remote(
                    "footer frame missing ENDF trailer".to_string(),
                ));
            }

            let payload = &buf[pos + FRAME_HEADER_SIZE..frame_end - FRAME_END.len()];

            match fh.frame_type {
                FrameType::FooterMetadata => {
                    let meta = metadata::cbor_to_global_metadata(payload)?;
                    self.layouts[msg_idx].global_metadata = Some(meta);
                }
                FrameType::FooterIndex => {
                    let idx = metadata::cbor_to_index(payload)?;
                    self.layouts[msg_idx].index = Some(idx);
                }
                _ => {}
            }

            let aligned = (frame_end.saturating_add(7)) & !7;
            pos = aligned.min(buf.len());
        }

        if self.layouts[msg_idx].global_metadata.is_none() {
            return Err(TensogramError::Remote(
                "footer region did not contain a metadata frame".to_string(),
            ));
        }
        if self.layouts[msg_idx].index.is_none() {
            return Err(TensogramError::Remote(
                "footer region did not contain an index frame".to_string(),
            ));
        }

        Ok(())
    }

    // ── Public API used by TensogramFile ─────────────────────────────────

    pub(crate) fn message_count(&mut self) -> Result<usize> {
        self.scan_all()?;
        Ok(self.layouts.len())
    }

    pub(crate) fn read_message(&mut self, msg_idx: usize) -> Result<Vec<u8>> {
        self.ensure_message(msg_idx)?;
        let layout = &self.layouts[msg_idx];
        let bytes = self.get_range(layout.offset..layout.offset + layout.length)?;
        Ok(bytes.to_vec())
    }

    pub(crate) fn read_metadata(&mut self, msg_idx: usize) -> Result<GlobalMetadata> {
        self.ensure_layout_eager(msg_idx)?;
        self.layouts[msg_idx]
            .global_metadata
            .clone()
            .ok_or_else(|| TensogramError::Remote("metadata not found".to_string()))
    }

    pub(crate) fn read_descriptors(
        &mut self,
        msg_idx: usize,
    ) -> Result<(GlobalMetadata, Vec<DataObjectDescriptor>)> {
        self.ensure_layout_eager(msg_idx)?;
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
                let desc = self.read_descriptor_only(
                    msg_offset,
                    msg_length,
                    index.offsets[i],
                    index.lengths[i],
                )?;
                descriptors.push(desc);
            }
            Ok((meta, descriptors))
        } else {
            let msg_bytes = self.read_message(msg_idx)?;
            crate::decode::decode_descriptors(&msg_bytes)
        }
    }

    /// Read only the CBOR descriptor from a data object frame, without
    /// downloading the full payload.
    ///
    /// For frames below `DESCRIPTOR_PREFIX_THRESHOLD` bytes, falls back
    /// to reading the entire frame (fewer round-trips).  For large frames,
    /// reads just the header, footer, and CBOR region — typically < 10 KB
    /// even when the payload is hundreds of megabytes.
    fn read_descriptor_only(
        &self,
        msg_offset: u64,
        msg_length: u64,
        frame_offset_in_msg: u64,
        frame_length: u64,
    ) -> Result<DataObjectDescriptor> {
        const DESCRIPTOR_PREFIX_THRESHOLD: u64 = 64 * 1024;

        let range =
            Self::checked_frame_range(msg_offset, msg_length, frame_offset_in_msg, frame_length)?;

        if frame_length <= DESCRIPTOR_PREFIX_THRESHOLD {
            let frame_bytes = self.get_range(range.clone())?;
            let (desc, _payload, _consumed) = framing::decode_data_object_frame(&frame_bytes)?;
            return Ok(desc);
        }

        let frame_start = range.start;
        let frame_end = range.end;

        let header_bytes = self.get_range(frame_start..frame_start + FRAME_HEADER_SIZE as u64)?;
        let fh = FrameHeader::read_from(&header_bytes)?;

        if fh.frame_type != FrameType::DataObject {
            return Err(TensogramError::Remote(format!(
                "expected DataObject frame, got {:?}",
                fh.frame_type
            )));
        }

        let footer_start = frame_end - DATA_OBJECT_FOOTER_SIZE as u64;
        let footer_bytes = self.get_range(footer_start..frame_end)?;

        if footer_bytes.len() < DATA_OBJECT_FOOTER_SIZE {
            return Err(TensogramError::Remote("frame footer too short".to_string()));
        }
        if &footer_bytes[8..] != FRAME_END {
            return Err(TensogramError::Remote(
                "frame missing ENDF trailer".to_string(),
            ));
        }

        let cbor_offset = u64::from_be_bytes(
            footer_bytes[..8]
                .try_into()
                .map_err(|_| TensogramError::Remote("footer cbor_offset truncated".to_string()))?,
        );

        if cbor_offset < FRAME_HEADER_SIZE as u64 {
            return Err(TensogramError::Remote(format!(
                "cbor_offset ({cbor_offset}) below frame header size ({FRAME_HEADER_SIZE})"
            )));
        }

        let cbor_after = fh.flags & DataObjectFlags::CBOR_AFTER_PAYLOAD != 0;
        let cbor_start = frame_start
            .checked_add(cbor_offset)
            .ok_or_else(|| TensogramError::Remote("cbor_start overflow".to_string()))?;

        if cbor_after {
            if cbor_start >= footer_start {
                return Err(TensogramError::Remote(
                    "cbor_offset points at or past footer".to_string(),
                ));
            }
            let cbor_bytes = self.get_range(cbor_start..footer_start)?;
            metadata::cbor_to_object_descriptor(&cbor_bytes)
        } else {
            if cbor_start >= footer_start {
                return Err(TensogramError::Remote(
                    "cbor_offset beyond frame body".to_string(),
                ));
            }
            let max_cbor_len = footer_start - cbor_start;
            let mut prefix_size: u64 = 8192;
            loop {
                let read_end = (cbor_start + prefix_size).min(footer_start);
                let prefix_bytes = self.get_range(cbor_start..read_end)?;
                match metadata::cbor_to_object_descriptor(&prefix_bytes) {
                    Ok(desc) => return Ok(desc),
                    Err(_) if prefix_size < max_cbor_len => {
                        prefix_size = (prefix_size * 2).min(max_cbor_len);
                    }
                    Err(e) => return Err(e),
                }
            }
        }
    }

    pub(crate) fn read_object(
        &mut self,
        msg_idx: usize,
        obj_idx: usize,
        options: &DecodeOptions,
    ) -> Result<(GlobalMetadata, DataObjectDescriptor, Vec<u8>)> {
        self.ensure_layout_eager(msg_idx)?;
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

    pub(crate) fn read_range(
        &mut self,
        msg_idx: usize,
        obj_idx: usize,
        ranges: &[(u64, u64)],
        options: &DecodeOptions,
    ) -> Result<(DataObjectDescriptor, Vec<Vec<u8>>)> {
        self.ensure_layout_eager(msg_idx)?;
        let layout = &self.layouts[msg_idx];
        let msg_offset = layout.offset;

        if let Some(ref index) = layout.index {
            Self::validate_index_access(index, obj_idx)?;

            let range = Self::checked_frame_range(
                msg_offset,
                layout.length,
                index.offsets[obj_idx],
                index.lengths[obj_idx],
            )?;
            let frame_bytes = self.get_range(range)?;
            let (desc, payload, _consumed) = framing::decode_data_object_frame(&frame_bytes)?;
            let parts = crate::decode::decode_range_from_payload(&desc, payload, ranges, options)?;
            Ok((desc, parts))
        } else {
            let msg_bytes = self.read_message(msg_idx)?;
            crate::decode::decode_range(&msg_bytes, obj_idx, ranges, options)
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

// ── Native async path (remote + async) ───────────────────────────────────────

#[cfg(feature = "async")]
impl RemoteBackend {
    async fn get_range_async(&self, range: Range<u64>) -> Result<Bytes> {
        self.store
            .get_range(&self.path, range)
            .await
            .map_err(|e| TensogramError::Remote(e.to_string()))
    }

    pub(crate) async fn open_async(
        source: &str,
        storage_options: &BTreeMap<String, String>,
    ) -> Result<Self> {
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
        let meta = store
            .head(&path)
            .await
            .map_err(|e| TensogramError::Remote(e.to_string()))?;

        let file_size = meta.size;
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
            next_scan_offset: 0,
            scan_complete: false,
        };
        backend.scan_next_async().await?;
        if backend.layouts.is_empty() {
            return Err(TensogramError::Remote(
                "no valid messages found in remote file".to_string(),
            ));
        }
        Ok(backend)
    }

    async fn scan_next_async(&mut self) -> Result<()> {
        if self.scan_complete {
            return Ok(());
        }
        let min_message_size = (PREAMBLE_SIZE + POSTAMBLE_SIZE) as u64;
        let pos = self.next_scan_offset;

        if pos + min_message_size > self.file_size {
            self.scan_complete = true;
            return Ok(());
        }

        let preamble_bytes = self
            .get_range_async(pos..pos + PREAMBLE_SIZE as u64)
            .await?;
        if &preamble_bytes[..MAGIC.len()] != MAGIC {
            self.scan_complete = true;
            return Ok(());
        }

        let preamble = match Preamble::read_from(&preamble_bytes) {
            Ok(p) => p,
            Err(_) => {
                self.scan_complete = true;
                return Ok(());
            }
        };

        let msg_len = preamble.total_length;

        if msg_len == 0 {
            let remaining = self.file_size - pos;
            if remaining < min_message_size {
                self.scan_complete = true;
                return Ok(());
            }
            let end_magic_pos = self.file_size - crate::wire::END_MAGIC.len() as u64;
            let end_bytes = self
                .get_range_async(end_magic_pos..end_magic_pos + crate::wire::END_MAGIC.len() as u64)
                .await?;
            if &end_bytes[..] != crate::wire::END_MAGIC {
                self.scan_complete = true;
                return Ok(());
            }
            self.layouts.push(MessageLayout {
                offset: pos,
                length: remaining,
                preamble,
                index: None,
                global_metadata: None,
            });
            self.scan_complete = true;
            return Ok(());
        }

        let msg_end = match pos.checked_add(msg_len) {
            Some(end) if msg_len >= min_message_size && end <= self.file_size => end,
            _ => {
                self.scan_complete = true;
                return Ok(());
            }
        };

        self.layouts.push(MessageLayout {
            offset: pos,
            length: msg_len,
            preamble,
            index: None,
            global_metadata: None,
        });
        self.next_scan_offset = msg_end;
        Ok(())
    }

    async fn scan_and_discover_next_async(&mut self) -> Result<()> {
        if self.scan_complete {
            return Ok(());
        }
        let min_message_size = (PREAMBLE_SIZE + POSTAMBLE_SIZE) as u64;
        let pos = self.next_scan_offset;

        if pos + min_message_size > self.file_size {
            self.scan_complete = true;
            return Ok(());
        }

        let chunk_size = (self.file_size - pos).min(256 * 1024);
        let chunk = self.get_range_async(pos..pos + chunk_size).await?;

        if chunk.len() < PREAMBLE_SIZE || &chunk[..MAGIC.len()] != MAGIC {
            self.scan_complete = true;
            return Ok(());
        }

        let preamble = match Preamble::read_from(&chunk[..PREAMBLE_SIZE]) {
            Ok(p) => p,
            Err(_) => {
                self.scan_complete = true;
                return Ok(());
            }
        };

        let msg_len = preamble.total_length;

        if msg_len == 0 {
            let remaining = self.file_size - pos;
            if remaining < min_message_size {
                self.scan_complete = true;
                return Ok(());
            }
            let end_magic_pos = self.file_size - crate::wire::END_MAGIC.len() as u64;
            let end_bytes = self
                .get_range_async(end_magic_pos..end_magic_pos + crate::wire::END_MAGIC.len() as u64)
                .await?;
            if &end_bytes[..] != crate::wire::END_MAGIC {
                self.scan_complete = true;
                return Ok(());
            }
            self.layouts.push(MessageLayout {
                offset: pos,
                length: remaining,
                preamble,
                index: None,
                global_metadata: None,
            });
            self.scan_complete = true;
            return Ok(());
        }

        let msg_end = match pos.checked_add(msg_len) {
            Some(end) if msg_len >= min_message_size && end <= self.file_size => end,
            _ => {
                self.scan_complete = true;
                return Ok(());
            }
        };

        let flags = preamble.flags;
        let msg_idx = self.layouts.len();

        self.layouts.push(MessageLayout {
            offset: pos,
            length: msg_len,
            preamble,
            index: None,
            global_metadata: None,
        });
        self.next_scan_offset = msg_end;

        if flags.has(MessageFlags::HEADER_METADATA) && flags.has(MessageFlags::HEADER_INDEX) {
            let chunk_end = (msg_len as usize).min(chunk.len());
            self.parse_header_frames(msg_idx, &chunk[..chunk_end])?;
        }

        Ok(())
    }

    async fn ensure_layout_eager_async(&mut self, msg_idx: usize) -> Result<()> {
        while msg_idx >= self.layouts.len() && !self.scan_complete {
            self.scan_and_discover_next_async().await?;
        }
        if msg_idx >= self.layouts.len() {
            return Err(TensogramError::Framing(format!(
                "message index {} out of range (count={})",
                msg_idx,
                self.layouts.len()
            )));
        }
        if self.layouts[msg_idx].global_metadata.is_some() && self.layouts[msg_idx].index.is_some()
        {
            return Ok(());
        }
        self.ensure_layout_async(msg_idx).await
    }

    async fn ensure_layout_async(&mut self, msg_idx: usize) -> Result<()> {
        self.ensure_message_async(msg_idx).await?;
        if self.layouts[msg_idx].global_metadata.is_some() && self.layouts[msg_idx].index.is_some()
        {
            return Ok(());
        }

        let flags = self.layouts[msg_idx].preamble.flags;

        if flags.has(MessageFlags::HEADER_METADATA) && flags.has(MessageFlags::HEADER_INDEX) {
            self.discover_header_layout_async(msg_idx).await?;
        } else if flags.has(MessageFlags::FOOTER_METADATA) && flags.has(MessageFlags::FOOTER_INDEX)
        {
            self.discover_footer_layout_async(msg_idx).await?;
        } else {
            return Err(TensogramError::Remote(
                "remote access requires header-indexed or footer-indexed messages".to_string(),
            ));
        }

        Ok(())
    }

    async fn discover_header_layout_async(&mut self, msg_idx: usize) -> Result<()> {
        let msg_offset = self.layouts[msg_idx].offset;
        let msg_len = self.layouts[msg_idx].length;
        let chunk_size = msg_len.min(256 * 1024);
        let header_bytes = self
            .get_range_async(msg_offset..msg_offset + chunk_size)
            .await?;
        self.parse_header_frames(msg_idx, &header_bytes)
    }

    async fn discover_footer_layout_async(&mut self, msg_idx: usize) -> Result<()> {
        let msg_offset = self.layouts[msg_idx].offset;
        let msg_len = self.layouts[msg_idx].length;
        let pa_offset = msg_offset
            .checked_add(msg_len)
            .and_then(|end| end.checked_sub(POSTAMBLE_SIZE as u64))
            .ok_or_else(|| TensogramError::Remote("postamble offset overflow".to_string()))?;
        let pa_bytes = self
            .get_range_async(pa_offset..pa_offset + POSTAMBLE_SIZE as u64)
            .await?;
        let postamble = Postamble::read_from(&pa_bytes)?;

        if postamble.first_footer_offset < PREAMBLE_SIZE as u64 {
            return Err(TensogramError::Remote(format!(
                "first_footer_offset ({}) is before preamble end ({})",
                postamble.first_footer_offset, PREAMBLE_SIZE
            )));
        }
        let footer_start = msg_offset
            .checked_add(postamble.first_footer_offset)
            .ok_or_else(|| TensogramError::Remote("footer offset overflow".to_string()))?;
        let footer_end = pa_offset;
        if footer_start >= footer_end {
            return Err(TensogramError::Remote(
                "first_footer_offset points at or past postamble".to_string(),
            ));
        }
        let footer_bytes = self.get_range_async(footer_start..footer_end).await?;
        self.parse_footer_frames(msg_idx, &footer_bytes)
    }

    async fn ensure_message_async(&mut self, msg_idx: usize) -> Result<()> {
        while msg_idx >= self.layouts.len() && !self.scan_complete {
            self.scan_next_async().await?;
        }
        if msg_idx >= self.layouts.len() {
            return Err(TensogramError::Framing(format!(
                "message index {} out of range (count={})",
                msg_idx,
                self.layouts.len()
            )));
        }
        Ok(())
    }

    pub(crate) async fn read_message_async(&mut self, msg_idx: usize) -> Result<Vec<u8>> {
        self.ensure_message_async(msg_idx).await?;
        let layout = &self.layouts[msg_idx];
        let bytes = self
            .get_range_async(layout.offset..layout.offset + layout.length)
            .await?;
        Ok(bytes.to_vec())
    }

    pub(crate) async fn read_metadata_async(&mut self, msg_idx: usize) -> Result<GlobalMetadata> {
        self.ensure_layout_eager_async(msg_idx).await?;
        self.layouts[msg_idx]
            .global_metadata
            .clone()
            .ok_or_else(|| TensogramError::Remote("metadata not found".to_string()))
    }

    pub(crate) async fn read_descriptors_async(
        &mut self,
        msg_idx: usize,
    ) -> Result<(GlobalMetadata, Vec<DataObjectDescriptor>)> {
        self.ensure_layout_eager_async(msg_idx).await?;
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
                let desc = self
                    .read_descriptor_only_async(
                        msg_offset,
                        msg_length,
                        index.offsets[i],
                        index.lengths[i],
                    )
                    .await?;
                descriptors.push(desc);
            }
            Ok((meta, descriptors))
        } else {
            let msg_bytes = self.read_message_async(msg_idx).await?;
            crate::decode::decode_descriptors(&msg_bytes)
        }
    }

    async fn read_descriptor_only_async(
        &self,
        msg_offset: u64,
        msg_length: u64,
        frame_offset_in_msg: u64,
        frame_length: u64,
    ) -> Result<DataObjectDescriptor> {
        const DESCRIPTOR_PREFIX_THRESHOLD: u64 = 64 * 1024;

        let range =
            Self::checked_frame_range(msg_offset, msg_length, frame_offset_in_msg, frame_length)?;

        if frame_length <= DESCRIPTOR_PREFIX_THRESHOLD {
            let frame_bytes = self.get_range_async(range.clone()).await?;
            let (desc, _payload, _consumed) = framing::decode_data_object_frame(&frame_bytes)?;
            return Ok(desc);
        }

        let frame_start = range.start;
        let frame_end = range.end;

        let header_bytes = self
            .get_range_async(frame_start..frame_start + FRAME_HEADER_SIZE as u64)
            .await?;
        let fh = FrameHeader::read_from(&header_bytes)?;

        if fh.frame_type != FrameType::DataObject {
            return Err(TensogramError::Remote(format!(
                "expected DataObject frame, got {:?}",
                fh.frame_type
            )));
        }

        let footer_start = frame_end - DATA_OBJECT_FOOTER_SIZE as u64;
        let footer_bytes = self.get_range_async(footer_start..frame_end).await?;

        if footer_bytes.len() < DATA_OBJECT_FOOTER_SIZE {
            return Err(TensogramError::Remote("frame footer too short".to_string()));
        }
        if &footer_bytes[8..] != FRAME_END {
            return Err(TensogramError::Remote(
                "frame missing ENDF trailer".to_string(),
            ));
        }

        let cbor_offset = u64::from_be_bytes(
            footer_bytes[..8]
                .try_into()
                .map_err(|_| TensogramError::Remote("footer cbor_offset truncated".to_string()))?,
        );

        if cbor_offset < FRAME_HEADER_SIZE as u64 {
            return Err(TensogramError::Remote(format!(
                "cbor_offset ({cbor_offset}) below frame header size ({FRAME_HEADER_SIZE})"
            )));
        }

        let cbor_after = fh.flags & DataObjectFlags::CBOR_AFTER_PAYLOAD != 0;
        let cbor_start = frame_start
            .checked_add(cbor_offset)
            .ok_or_else(|| TensogramError::Remote("cbor_start overflow".to_string()))?;

        if cbor_after {
            if cbor_start >= footer_start {
                return Err(TensogramError::Remote(
                    "cbor_offset points at or past footer".to_string(),
                ));
            }
            let cbor_bytes = self.get_range_async(cbor_start..footer_start).await?;
            metadata::cbor_to_object_descriptor(&cbor_bytes)
        } else {
            if cbor_start >= footer_start {
                return Err(TensogramError::Remote(
                    "cbor_offset beyond frame body".to_string(),
                ));
            }
            let max_cbor_len = footer_start - cbor_start;
            let mut prefix_size: u64 = 8192;
            loop {
                let read_end = (cbor_start + prefix_size).min(footer_start);
                let prefix_bytes = self.get_range_async(cbor_start..read_end).await?;
                match metadata::cbor_to_object_descriptor(&prefix_bytes) {
                    Ok(desc) => return Ok(desc),
                    Err(_) if prefix_size < max_cbor_len => {
                        prefix_size = (prefix_size * 2).min(max_cbor_len);
                    }
                    Err(e) => return Err(e),
                }
            }
        }
    }

    pub(crate) async fn read_object_async(
        &mut self,
        msg_idx: usize,
        obj_idx: usize,
        options: &DecodeOptions,
    ) -> Result<(GlobalMetadata, DataObjectDescriptor, Vec<u8>)> {
        self.ensure_layout_eager_async(msg_idx).await?;
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
            let frame_bytes = self.get_range_async(range).await?;
            let (desc, payload, _consumed) = framing::decode_data_object_frame(&frame_bytes)?;
            let decoded = crate::decode::decode_single_object(&desc, payload, options)?;
            Ok((meta, desc, decoded))
        } else {
            let msg_bytes = self.read_message_async(msg_idx).await?;
            crate::decode::decode_object(&msg_bytes, obj_idx, options)
        }
    }
}
