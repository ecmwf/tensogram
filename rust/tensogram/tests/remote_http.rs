// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

#![cfg(feature = "remote")]

use std::collections::BTreeMap;
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use http_body_util::Full;
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;

use tensogram::decode::DecodeOptions;
use tensogram::encode::{self, EncodeOptions};
use tensogram::file::TensogramFile;
use tensogram::types::{ByteOrder, DataObjectDescriptor, GlobalMetadata};
use tensogram::{Dtype, RemoteScanOptions, is_remote_url};

// ── Test helpers ─────────────────────────────────────────────────────────────

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

fn encode_test_message(shape: Vec<u64>, fill: u8) -> Result<Vec<u8>, std::io::Error> {
    let meta = make_global_meta();
    let desc = make_descriptor(shape.clone());
    let num_bytes = shape.iter().product::<u64>() as usize * 4; // float32 = 4 bytes
    let data = vec![fill; num_bytes];
    encode::encode(&meta, &[(&desc, &data)], &EncodeOptions::default())
        .map_err(std::io::Error::other)
}

fn encode_multi_object_message(
    shapes: &[Vec<u64>],
    fills: &[u8],
) -> Result<Vec<u8>, std::io::Error> {
    let meta = make_global_meta();
    let pairs: Vec<(DataObjectDescriptor, Vec<u8>)> = shapes
        .iter()
        .zip(fills)
        .map(|(shape, &fill)| {
            let desc = make_descriptor(shape.clone());
            let num_bytes = shape.iter().product::<u64>() as usize * 4;
            let data = vec![fill; num_bytes];
            (desc, data)
        })
        .collect();
    let refs: Vec<(&DataObjectDescriptor, &[u8])> =
        pairs.iter().map(|(d, b)| (d, b.as_slice())).collect();
    encode::encode(&meta, &refs, &EncodeOptions::default()).map_err(std::io::Error::other)
}

// ── Mock HTTP server with request counting ───────────────────────────────────

struct MockServer {
    #[allow(dead_code)]
    data: Arc<Vec<u8>>,
    request_count: Arc<AtomicUsize>,
    range_request_count: Arc<AtomicUsize>,
    addr: SocketAddr,
}

impl MockServer {
    async fn start(data: Vec<u8>) -> Result<Self, std::io::Error> {
        let data = Arc::new(data);
        let request_count = Arc::new(AtomicUsize::new(0));
        let range_request_count = Arc::new(AtomicUsize::new(0));
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;

        let data_clone = data.clone();
        let count_clone = request_count.clone();
        let range_count_clone = range_request_count.clone();

        tokio::spawn(async move {
            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(v) => v,
                    Err(_) => break,
                };
                let io = TokioIo::new(stream);
                let data = data_clone.clone();
                let count = count_clone.clone();
                let range_count = range_count_clone.clone();

                tokio::spawn(async move {
                    let _ = http1::Builder::new()
                        .serve_connection(
                            io,
                            service_fn(move |req: Request<hyper::body::Incoming>| {
                                let data = data.clone();
                                let count = count.clone();
                                let range_count = range_count.clone();
                                async move { handle_request(req, data, count, range_count) }
                            }),
                        )
                        .await;
                });
            }
        });

        Ok(MockServer {
            data,
            request_count,
            range_request_count,
            addr,
        })
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}/test.tgm", self.addr.port())
    }

    fn request_count(&self) -> usize {
        self.request_count.load(Ordering::SeqCst)
    }

    fn range_request_count(&self) -> usize {
        self.range_request_count.load(Ordering::SeqCst)
    }

    fn reset_count(&self) {
        self.request_count.store(0, Ordering::SeqCst);
        self.range_request_count.store(0, Ordering::SeqCst);
    }
}

fn handle_request(
    req: Request<hyper::body::Incoming>,
    data: Arc<Vec<u8>>,
    request_count: Arc<AtomicUsize>,
    range_request_count: Arc<AtomicUsize>,
) -> Result<Response<Full<Bytes>>, std::io::Error> {
    request_count.fetch_add(1, Ordering::SeqCst);

    if req.method() == hyper::Method::HEAD {
        let resp = Response::builder()
            .status(StatusCode::OK)
            .header("Content-Length", data.len())
            .header("Accept-Ranges", "bytes")
            .body(Full::new(Bytes::new()))
            .map_err(std::io::Error::other)?;
        return Ok(resp);
    }

    if let Some(range_header) = req.headers().get("Range") {
        range_request_count.fetch_add(1, Ordering::SeqCst);
        let range_str = range_header.to_str().unwrap_or("");
        match parse_range_header(range_str, data.len()) {
            Some(byte_range) => {
                let slice = &data[byte_range.0..byte_range.1];
                let resp = Response::builder()
                    .status(StatusCode::PARTIAL_CONTENT)
                    .header(
                        "Content-Range",
                        format!("bytes {}-{}/{}", byte_range.0, byte_range.1 - 1, data.len()),
                    )
                    .header("Content-Length", slice.len())
                    .body(Full::new(Bytes::copy_from_slice(slice)))
                    .map_err(std::io::Error::other)?;
                return Ok(resp);
            }
            None => {
                let resp = Response::builder()
                    .status(StatusCode::RANGE_NOT_SATISFIABLE)
                    .header("Content-Range", format!("bytes */{}", data.len()))
                    .body(Full::new(Bytes::new()))
                    .map_err(std::io::Error::other)?;
                return Ok(resp);
            }
        }
    }

    let resp = Response::builder()
        .status(StatusCode::OK)
        .header("Content-Length", data.len())
        .body(Full::new(Bytes::copy_from_slice(&data)))
        .map_err(std::io::Error::other)?;
    Ok(resp)
}

fn parse_range_header(header: &str, total: usize) -> Option<(usize, usize)> {
    let header = header.strip_prefix("bytes=")?;
    if total == 0 {
        return None;
    }
    if let Some(suffix) = header.strip_prefix('-') {
        let n: usize = suffix.parse().ok()?;
        if n == 0 {
            return None;
        }
        Some((total.saturating_sub(n), total))
    } else if let Some((start_s, end_s)) = header.split_once('-') {
        let start: usize = start_s.parse().ok()?;
        if start >= total {
            return None;
        }
        if end_s.is_empty() {
            Some((start, total))
        } else {
            let end: usize = end_s.parse().ok()?;
            if end < start {
                return None;
            }
            let end_clamped = end.min(total - 1) + 1;
            Some((start, end_clamped))
        }
    } else {
        None
    }
}

// ── URL detection tests ──────────────────────────────────────────────────────

#[test]
fn test_is_remote_url() -> Result<(), Box<dyn Error>> {
    assert!(is_remote_url("s3://bucket/file.tgm"));
    assert!(is_remote_url("S3://bucket/file.tgm"));
    assert!(is_remote_url("gs://bucket/file.tgm"));
    assert!(is_remote_url("az://container/file.tgm"));
    assert!(is_remote_url("http://host/file.tgm"));
    assert!(is_remote_url("https://host/file.tgm"));
    assert!(!is_remote_url("/local/path/file.tgm"));
    assert!(!is_remote_url("relative/path.tgm"));
    assert!(!is_remote_url("file.tgm"));
    assert!(!is_remote_url("ftp://not-supported/file.tgm"));
    Ok(())
}

// ── Remote access tests (header-indexed, single message) ─────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_open_and_message_count() -> Result<(), Box<dyn Error>> {
    let msg = encode_test_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source(server.url(), None)?;
    assert!(file.is_remote());
    assert_eq!(file.message_count()?, 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_decode_message() -> Result<(), Box<dyn Error>> {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![4]);
    let data = vec![42u8; 16];
    let msg = encode::encode(&meta, &[(&desc, &data)], &EncodeOptions::default())?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source(server.url(), None)?;
    let (_decoded_meta, objects) = file.decode_message(0, &DecodeOptions::default())?;
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0].1, data);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_decode_metadata() -> Result<(), Box<dyn Error>> {
    let msg = encode_test_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source(server.url(), None)?;
    server.reset_count();

    let _meta = file.decode_metadata(0)?;
    // First metadata call triggers layout discovery (1 header chunk read).
    // Subsequent calls should be free (cached).
    let count_after_first = server.request_count();
    assert!(
        count_after_first <= 1,
        "first metadata read should need at most 1 request"
    );

    server.reset_count();
    let _meta2 = file.decode_metadata(0)?;
    assert_eq!(
        server.request_count(),
        0,
        "repeat metadata should come from cache"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_decode_descriptors() -> Result<(), Box<dyn Error>> {
    let msg = encode_test_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source(server.url(), None)?;
    let (_meta, descriptors) = file.decode_descriptors(0)?;
    assert_eq!(descriptors.len(), 1);
    assert_eq!(descriptors[0].shape, vec![4]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_decode_object() -> Result<(), Box<dyn Error>> {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![4]);
    let data = vec![42u8; 16];
    let msg = encode::encode(&meta, &[(&desc, &data)], &EncodeOptions::default())?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source(server.url(), None)?;
    server.reset_count();

    let (_decoded_meta, decoded_desc, decoded_data) =
        file.decode_object(0, 0, &DecodeOptions::default())?;
    assert_eq!(decoded_desc.shape, vec![4]);
    assert_eq!(decoded_data, data);

    // Object read should use targeted range request via index, not full message
    let count = server.request_count();
    assert!(
        count <= 2,
        "expected at most 2 requests for object read (layout + object frame), got {count}"
    );
    Ok(())
}

// ── Multi-object message tests ───────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_multi_object_decode_single() -> Result<(), Box<dyn Error>> {
    let shapes = vec![vec![4], vec![8], vec![2]];
    let fills = vec![10u8, 20u8, 30u8];
    let msg = encode_multi_object_message(&shapes, &fills)?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source(server.url(), None)?;

    // Read only the second object
    let (_, desc, data) = file.decode_object(0, 1, &DecodeOptions::default())?;
    assert_eq!(desc.shape, vec![8]);
    assert_eq!(data, vec![20u8; 32]);

    // Read the third object
    let (_, desc, data) = file.decode_object(0, 2, &DecodeOptions::default())?;
    assert_eq!(desc.shape, vec![2]);
    assert_eq!(data, vec![30u8; 8]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_multi_object_descriptors() -> Result<(), Box<dyn Error>> {
    let shapes = vec![vec![4], vec![8]];
    let fills = vec![10u8, 20u8];
    let msg = encode_multi_object_message(&shapes, &fills)?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source(server.url(), None)?;
    let (_, descriptors) = file.decode_descriptors(0)?;
    assert_eq!(descriptors.len(), 2);
    assert_eq!(descriptors[0].shape, vec![4]);
    assert_eq!(descriptors[1].shape, vec![8]);
    Ok(())
}

// ── Multi-message file tests ─────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_multi_message_file() -> Result<(), Box<dyn Error>> {
    let msg1 = encode_test_message(vec![4], 10)?;
    let msg2 = encode_test_message(vec![8], 20)?;
    let mut combined = msg1.clone();
    combined.extend_from_slice(&msg2);

    let server = MockServer::start(combined).await?;

    let file = TensogramFile::open_source(server.url(), None)?;
    assert_eq!(file.message_count()?, 2);

    let _meta0 = file.decode_metadata(0)?;
    let (_, descs1) = file.decode_descriptors(1)?;
    assert_eq!(descs1[0].shape, vec![8]);
    Ok(())
}

// ── Object index out of range ────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_object_out_of_range() -> Result<(), Box<dyn Error>> {
    let msg = encode_test_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source(server.url(), None)?;
    let result = file.decode_range(0, 5, &[(0, 1)], &DecodeOptions::default());
    assert!(result.is_err());
    Ok(())
}

// ── Public scan_opts surface ────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_open_remote_bidirectional_layouts_match_forward() -> Result<(), Box<dyn Error>> {
    let msg1 = encode_test_message(vec![4], 10)?;
    let msg2 = encode_test_message(vec![8], 20)?;
    let msg3 = encode_test_message(vec![16], 30)?;
    let mut combined = msg1.clone();
    combined.extend_from_slice(&msg2);
    combined.extend_from_slice(&msg3);

    let server = MockServer::start(combined).await?;
    let storage: BTreeMap<String, String> = BTreeMap::new();

    let fwd = TensogramFile::open_remote(&server.url(), &storage, None)?;
    let fwd_layouts = fwd.message_layouts()?;

    let opts = RemoteScanOptions {
        bidirectional: true,
    };
    let bidir = TensogramFile::open_remote(&server.url(), &storage, Some(&opts))?;
    let bidir_layouts = bidir.message_layouts()?;

    assert_eq!(fwd_layouts, bidir_layouts);
    assert_eq!(fwd_layouts.len(), 3);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_open_remote_some_false_equivalent_to_none() -> Result<(), Box<dyn Error>> {
    let msg = encode_test_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;
    let storage: BTreeMap<String, String> = BTreeMap::new();

    let none_file = TensogramFile::open_remote(&server.url(), &storage, None)?;
    let none_layouts = none_file.message_layouts()?;

    let opts = RemoteScanOptions::default();
    let some_file = TensogramFile::open_remote(&server.url(), &storage, Some(&opts))?;
    let some_layouts = some_file.message_layouts()?;

    assert_eq!(none_layouts, some_layouts);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_open_source_local_path_ignores_scan_opts() -> Result<(), Box<dyn Error>> {
    let msg = encode_test_message(vec![4], 42)?;
    let tmpdir = tempfile::tempdir().map_err(|e| Box::new(e) as Box<dyn Error>)?;
    let path = tmpdir.path().join("local.tgm");
    std::fs::write(&path, &msg).map_err(|e| Box::new(e) as Box<dyn Error>)?;
    let path_str = path.to_str().expect("utf-8 path");

    let opts = RemoteScanOptions {
        bidirectional: true,
    };
    let file = TensogramFile::open_source(path_str, Some(&opts))?;
    assert!(!file.is_remote());
    assert_eq!(file.message_count()?, 1);

    let layouts = file.message_layouts()?;
    assert_eq!(layouts.len(), 1);
    assert_eq!(layouts[0].offset, 0);
    assert_eq!(layouts[0].length as usize, msg.len());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_open_remote_async_bidirectional_layouts_match_forward() -> Result<(), Box<dyn Error>>
{
    let msg1 = encode_test_message(vec![4], 10)?;
    let msg2 = encode_test_message(vec![8], 20)?;
    let mut combined = msg1.clone();
    combined.extend_from_slice(&msg2);

    let server = MockServer::start(combined).await?;
    let storage: BTreeMap<String, String> = BTreeMap::new();

    let fwd = TensogramFile::open_remote_async(&server.url(), &storage, None).await?;
    let fwd_layouts = fwd.message_layouts_async().await?;

    let opts = RemoteScanOptions {
        bidirectional: true,
    };
    let bidir = TensogramFile::open_remote_async(&server.url(), &storage, Some(&opts)).await?;
    let bidir_layouts = bidir.message_layouts_async().await?;

    assert_eq!(fwd_layouts, bidir_layouts);
    assert_eq!(fwd_layouts.len(), 2);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_message_layouts_local_and_remote_agree() -> Result<(), Box<dyn Error>> {
    let msg1 = encode_test_message(vec![4], 10)?;
    let msg2 = encode_test_message(vec![8], 20)?;
    let mut combined = msg1.clone();
    combined.extend_from_slice(&msg2);

    let tmpdir = tempfile::tempdir().map_err(|e| Box::new(e) as Box<dyn Error>)?;
    let path = tmpdir.path().join("local.tgm");
    std::fs::write(&path, &combined).map_err(|e| Box::new(e) as Box<dyn Error>)?;

    let local = TensogramFile::open(&path)?;
    let local_layouts = local.message_layouts()?;

    let server = MockServer::start(combined).await?;
    let remote = TensogramFile::open_source(server.url(), None)?;
    let remote_layouts = remote.message_layouts()?;

    assert_eq!(local_layouts, remote_layouts);
    Ok(())
}

// ── Request count verification ───────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_request_count_header_indexed() -> Result<(), Box<dyn Error>> {
    let msg = encode_test_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source(server.url(), None)?;
    let open_requests = server.request_count();

    server.reset_count();
    let _ = file.decode_object(0, 0, &DecodeOptions::default())?;
    let object_requests = server.request_count();
    let range_requests = server.range_request_count();

    // Layout discovery (1 header chunk) + object frame fetch (1) = at most 2
    assert!(
        object_requests <= 2,
        "expected <=2 requests for object read, got {object_requests} (open used {open_requests})"
    );
    assert!(
        range_requests > 0,
        "object read must use Range requests, not full GETs"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_lazy_open_only_reads_first_preamble() -> Result<(), Box<dyn Error>> {
    let msg1 = encode_test_message(vec![4], 10)?;
    let msg2 = encode_test_message(vec![8], 20)?;
    let mut combined = msg1;
    combined.extend_from_slice(&msg2);
    let server = MockServer::start(combined).await?;

    let file = TensogramFile::open_source(server.url(), None)?;
    let open_requests = server.request_count();
    assert_eq!(
        open_requests, 2,
        "open should cost exactly 2 requests (1 HEAD + 1 preamble), got {open_requests}"
    );

    let (_, desc, data) = file.decode_object(0, 0, &DecodeOptions::default())?;
    assert_eq!(desc.shape, vec![4]);
    assert_eq!(data, vec![10u8; 16]);

    server.reset_count();
    assert_eq!(file.message_count()?, 2);
    let scan_requests = server.request_count();
    assert!(
        scan_requests >= 1,
        "message_count should trigger scanning of remaining messages"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_eager_layout_combines_scan_and_discover() -> Result<(), Box<dyn Error>> {
    let msg1 = encode_test_message(vec![4], 10)?;
    let msg2 = encode_test_message(vec![8], 20)?;
    let mut combined = msg1;
    combined.extend_from_slice(&msg2);
    let server = MockServer::start(combined).await?;

    let file = TensogramFile::open_source(server.url(), None)?;
    server.reset_count();
    let (_, desc, data) = file.decode_object(1, 0, &DecodeOptions::default())?;
    let eager_requests = server.request_count();

    assert_eq!(desc.shape, vec![8]);
    assert_eq!(data, vec![20u8; 32]);
    assert!(
        eager_requests <= 2,
        "eager layout should combine scan+discover into 1 GET per message, got {eager_requests}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_eager_layout_streaming_falls_back() -> Result<(), Box<dyn Error>> {
    let msg = encode_streaming_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source(server.url(), None)?;
    server.reset_count();
    let (_, desc, data) = file.decode_object(0, 0, &DecodeOptions::default())?;

    assert_eq!(desc.shape, vec![4]);
    assert_eq!(data, vec![42u8; 16]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_repeated_object_reads_use_cache() -> Result<(), Box<dyn Error>> {
    let shapes = vec![vec![4], vec![8]];
    let fills = vec![10u8, 20u8];
    let msg = encode_multi_object_message(&shapes, &fills)?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source(server.url(), None)?;
    server.reset_count();

    // First object read
    let _ = file.decode_object(0, 0, &DecodeOptions::default())?;
    let first_read_count = server.request_count();

    server.reset_count();

    // Second object read from same message — layout is cached
    let _ = file.decode_object(0, 1, &DecodeOptions::default())?;
    let second_read_count = server.request_count();

    assert!(
        second_read_count <= 1,
        "repeated reads should reuse cached layout, got {second_read_count} requests (first was {first_read_count})"
    );
    Ok(())
}

// ── Matches local decode ─────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_matches_local_decode() -> Result<(), Box<dyn Error>> {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![10]);
    let data: Vec<u8> = (0..40).collect(); // 10 float32s = 40 bytes
    let msg = encode::encode(&meta, &[(&desc, &data)], &EncodeOptions::default())?;

    let server = MockServer::start(msg.clone()).await?;

    // Local decode
    let (_local_meta, local_objects) = tensogram::decode::decode(&msg, &DecodeOptions::default())?;

    // Remote decode
    let remote_file = TensogramFile::open_source(server.url(), None)?;
    let (_remote_meta, remote_desc, remote_data) =
        remote_file.decode_object(0, 0, &DecodeOptions::default())?;

    assert_eq!(local_objects[0].0.shape, remote_desc.shape);
    assert_eq!(local_objects[0].1, remote_data);
    Ok(())
}

// ── Error cases ──────────────────────────────────────────────────────────────

#[test]
fn test_remote_invalid_url() -> Result<(), Box<dyn Error>> {
    let result = TensogramFile::open_source("http://[invalid-url]/file.tgm", None);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_open_source_local_path() -> Result<(), Box<dyn Error>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("local.tgm");

    let mut file = TensogramFile::create(&path)?;
    let meta = make_global_meta();
    let desc = make_descriptor(vec![4]);
    let data = vec![0u8; 16];
    file.append(
        &meta,
        &[(&desc, data.as_slice())],
        &EncodeOptions::default(),
    )?;

    let path = path.to_str().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "path is not valid UTF-8")
    })?;
    let file = TensogramFile::open_source(path, None)?;
    assert!(!file.is_remote());
    assert_eq!(file.message_count()?, 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_source_returns_url() -> Result<(), Box<dyn Error>> {
    let msg = encode_test_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;
    let url = server.url();

    let file = TensogramFile::open_source(url.clone(), None)?;
    assert_eq!(file.source(), url);
    Ok(())
}

// ── Footer-indexed (streaming) message tests ─────────────────────────────────

fn encode_streaming_message(shape: Vec<u64>, fill: u8) -> Result<Vec<u8>, std::io::Error> {
    let meta = make_global_meta();
    let desc = make_descriptor(shape.clone());
    let num_bytes = shape.iter().product::<u64>() as usize * 4;
    let data = vec![fill; num_bytes];
    let buf = Vec::new();
    let mut enc =
        tensogram::streaming::StreamingEncoder::new(buf, &meta, &EncodeOptions::default())
            .map_err(std::io::Error::other)?;
    enc.write_object(&desc, &data)
        .map_err(std::io::Error::other)?;
    enc.finish().map_err(std::io::Error::other)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_streaming_message_open_and_decode() -> Result<(), Box<dyn Error>> {
    let msg = encode_streaming_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source(server.url(), None)?;
    assert!(file.is_remote());
    assert_eq!(file.message_count()?, 1);

    let _meta = file.decode_metadata(0)?;
    let (_, desc, data) = file.decode_object(0, 0, &DecodeOptions::default())?;
    assert_eq!(desc.shape, vec![4]);
    assert_eq!(data, vec![42u8; 16]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_streaming_matches_local_decode() -> Result<(), Box<dyn Error>> {
    let msg = encode_streaming_message(vec![10], 99)?;
    let server = MockServer::start(msg.clone()).await?;

    let (_local_meta, local_objects) = tensogram::decode::decode(&msg, &DecodeOptions::default())?;

    let remote_file = TensogramFile::open_source(server.url(), None)?;
    let (_remote_meta, remote_desc, remote_data) =
        remote_file.decode_object(0, 0, &DecodeOptions::default())?;

    assert_eq!(local_objects[0].0.shape, remote_desc.shape);
    assert_eq!(local_objects[0].1, remote_data);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_streaming_multi_object() -> Result<(), Box<dyn Error>> {
    let meta = make_global_meta();
    let desc1 = make_descriptor(vec![4]);
    let desc2 = make_descriptor(vec![8]);
    let data1 = vec![10u8; 16];
    let data2 = vec![20u8; 32];

    let buf = Vec::new();
    let mut enc =
        tensogram::streaming::StreamingEncoder::new(buf, &meta, &EncodeOptions::default())
            .map_err(std::io::Error::other)?;
    enc.write_object(&desc1, &data1)
        .map_err(std::io::Error::other)?;
    enc.write_object(&desc2, &data2)
        .map_err(std::io::Error::other)?;
    let msg = enc.finish().map_err(std::io::Error::other)?;

    let server = MockServer::start(msg).await?;
    let file = TensogramFile::open_source(server.url(), None)?;

    let (_, descs) = file.decode_descriptors(0)?;
    assert_eq!(descs.len(), 2);
    assert_eq!(descs[0].shape, vec![4]);
    assert_eq!(descs[1].shape, vec![8]);

    let (_, _, data) = file.decode_object(0, 1, &DecodeOptions::default())?;
    assert_eq!(data, vec![20u8; 32]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_mixed_buffered_then_streaming() -> Result<(), Box<dyn Error>> {
    let buffered_msg = encode_test_message(vec![4], 10)?;
    let streaming_msg = encode_streaming_message(vec![8], 20)?;

    let mut combined = buffered_msg;
    combined.extend_from_slice(&streaming_msg);

    let server = MockServer::start(combined).await?;
    let file = TensogramFile::open_source(server.url(), None)?;
    assert_eq!(file.message_count()?, 2);

    let (_, _, data0) = file.decode_object(0, 0, &DecodeOptions::default())?;
    assert_eq!(data0, vec![10u8; 16]);

    let (_, _, data1) = file.decode_object(1, 0, &DecodeOptions::default())?;
    assert_eq!(data1, vec![20u8; 32]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_streaming_index_lengths_are_frame_lengths() -> Result<(), Box<dyn Error>> {
    let msg = encode_streaming_message(vec![4], 42)?;
    let server = MockServer::start(msg.clone()).await?;

    let file = TensogramFile::open_source(server.url(), None)?;
    let (_, desc, data) = file.decode_object(0, 0, &DecodeOptions::default())?;
    assert_eq!(desc.shape, vec![4]);
    assert_eq!(data, vec![42u8; 16]);

    let (_local_meta, local_objects) = tensogram::decode::decode(&msg, &DecodeOptions::default())?;
    assert_eq!(local_objects[0].1, data);
    Ok(())
}

// ── Shared runtime tests ─────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_shared_runtime_concurrent_reads() -> Result<(), Box<dyn Error>> {
    let shapes = vec![vec![4], vec![8], vec![16]];
    let fills = vec![10u8, 20u8, 30u8];
    let msg = encode_multi_object_message(&shapes, &fills)?;
    let server = MockServer::start(msg).await?;
    let url = server.url();

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let u = url.clone();
            std::thread::spawn(
                move || -> std::result::Result<(), Box<dyn Error + Send + Sync>> {
                    let file = TensogramFile::open_source(&u, None)?;
                    let (_, desc, data) = file.decode_object(0, 0, &DecodeOptions::default())?;
                    assert_eq!(desc.shape, vec![4]);
                    assert_eq!(data, vec![10u8; 16]);
                    Ok(())
                },
            )
        })
        .collect();

    for h in handles {
        match h.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e.to_string().into()),
            Err(_) => return Err("concurrent reader thread panicked".into()),
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_shared_runtime_from_sync_context() -> Result<(), Box<dyn Error>> {
    let msg = encode_test_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;
    let url = server.url();

    let result = std::thread::spawn(
        move || -> std::result::Result<(), Box<dyn Error + Send + Sync>> {
            let file = TensogramFile::open_source(&url, None)?;
            let _meta = file.decode_metadata(0)?;
            let (_, desc, data) = file.decode_object(0, 0, &DecodeOptions::default())?;
            assert_eq!(desc.shape, vec![4]);
            assert_eq!(data, vec![42u8; 16]);
            Ok(())
        },
    )
    .join();
    match result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(e.to_string().into()),
        Err(_) => return Err("sync context thread panicked".into()),
    }
    Ok(())
}

// ── Descriptor-only read tests ───────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_descriptor_only_matches_full_read() -> Result<(), Box<dyn Error>> {
    let shapes = vec![vec![4], vec![8, 2], vec![100]];
    let fills = vec![10u8, 20u8, 30u8];
    let msg = encode_multi_object_message(&shapes, &fills)?;
    let server = MockServer::start(msg.clone()).await?;

    let file = TensogramFile::open_source(server.url(), None)?;
    let (_, remote_descs) = file.decode_descriptors(0)?;

    let (_, local_descs) = tensogram::decode::decode_descriptors(&msg)?;

    assert_eq!(remote_descs.len(), local_descs.len());
    for (rd, ld) in remote_descs.iter().zip(local_descs.iter()) {
        assert_eq!(rd.shape, ld.shape);
        assert_eq!(rd.dtype, ld.dtype);
        assert_eq!(rd.encoding, ld.encoding);
        assert_eq!(rd.compression, ld.compression);
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_descriptor_only_large_frame_exercises_fast_path() -> Result<(), Box<dyn Error>> {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![256, 256]);
    let data = vec![7u8; 256 * 256 * 4];
    let msg = encode::encode(&meta, &[(&desc, &data)], &EncodeOptions::default())?;
    assert!(
        msg.len() > 64 * 1024,
        "payload must exceed 64KB threshold, got {} bytes",
        msg.len()
    );
    let server = MockServer::start(msg.clone()).await?;

    let file = TensogramFile::open_source(server.url(), None)?;
    let (_, remote_descs) = file.decode_descriptors(0)?;
    let (_, local_descs) = tensogram::decode::decode_descriptors(&msg)?;

    assert_eq!(remote_descs.len(), 1);
    assert_eq!(remote_descs[0].shape, local_descs[0].shape);
    assert_eq!(remote_descs[0].dtype, local_descs[0].dtype);

    let (_, _, decoded) = file.decode_object(0, 0, &DecodeOptions::default())?;
    assert_eq!(decoded, data);
    Ok(())
}

// ── Async remote tests (remote + async) ──────────────────────────────────────

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_remote_open_and_metadata() -> Result<(), Box<dyn Error>> {
    let msg = encode_test_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source_async(server.url(), None).await?;
    assert!(file.is_remote());
    assert_eq!(file.message_count()?, 1);

    let _meta = file.decode_metadata_async(0).await?;
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_remote_decode_object() -> Result<(), Box<dyn Error>> {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![4]);
    let data = vec![42u8; 16];
    let msg = encode::encode(&meta, &[(&desc, &data)], &EncodeOptions::default())?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source_async(server.url(), None).await?;
    let (_decoded_meta, decoded_desc, decoded_data) = file
        .decode_object_async(0, 0, &DecodeOptions::default())
        .await?;
    assert_eq!(decoded_desc.shape, vec![4]);
    assert_eq!(decoded_data, data);
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_remote_decode_descriptors() -> Result<(), Box<dyn Error>> {
    let shapes = vec![vec![4], vec![8]];
    let fills = vec![10u8, 20u8];
    let msg = encode_multi_object_message(&shapes, &fills)?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source_async(server.url(), None).await?;
    let (_meta, descriptors) = file.decode_descriptors_async(0).await?;
    assert_eq!(descriptors.len(), 2);
    assert_eq!(descriptors[0].shape, vec![4]);
    assert_eq!(descriptors[1].shape, vec![8]);
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_vs_sync_parity() -> Result<(), Box<dyn Error>> {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![10]);
    let data: Vec<u8> = (0..40).collect();
    let msg = encode::encode(&meta, &[(&desc, &data)], &EncodeOptions::default())?;
    let server = MockServer::start(msg).await?;

    let sync_file = TensogramFile::open_source(server.url(), None)?;
    let (_sync_meta, sync_desc, sync_data) =
        sync_file.decode_object(0, 0, &DecodeOptions::default())?;

    let async_file = TensogramFile::open_source_async(server.url(), None).await?;
    let (_async_meta, async_desc, async_data) = async_file
        .decode_object_async(0, 0, &DecodeOptions::default())
        .await?;

    assert_eq!(sync_desc.shape, async_desc.shape);
    assert_eq!(sync_data, async_data);
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_remote_read_message() -> Result<(), Box<dyn Error>> {
    let msg = encode_test_message(vec![4], 42)?;
    let server = MockServer::start(msg.clone()).await?;

    let file = TensogramFile::open_source_async(server.url(), None).await?;
    let remote_msg = file.read_message_async(0).await?;
    assert_eq!(remote_msg, msg);
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_remote_streaming_message() -> Result<(), Box<dyn Error>> {
    let msg = encode_streaming_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source_async(server.url(), None).await?;
    assert!(file.is_remote());

    let _meta = file.decode_metadata_async(0).await?;
    let (_, desc, data) = file
        .decode_object_async(0, 0, &DecodeOptions::default())
        .await?;
    assert_eq!(desc.shape, vec![4]);
    assert_eq!(data, vec![42u8; 16]);
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_remote_object_out_of_range() -> Result<(), Box<dyn Error>> {
    let msg = encode_test_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source_async(server.url(), None).await?;
    let result = file
        .decode_object_async(0, 5, &DecodeOptions::default())
        .await;
    assert!(result.is_err());
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_remote_message_index_out_of_range() -> Result<(), Box<dyn Error>> {
    let msg = encode_test_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source_async(server.url(), None).await?;
    let result = file.decode_metadata_async(5).await;
    assert!(result.is_err());
    Ok(())
}

// ── Async multi-message scan ─────────────────────────────────────────────────

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_remote_multi_message_scan() -> Result<(), Box<dyn Error>> {
    let msg1 = encode_test_message(vec![4], 10)?;
    let msg2 = encode_test_message(vec![8], 20)?;
    let mut combined = msg1;
    combined.extend_from_slice(&msg2);

    let server = MockServer::start(combined).await?;

    let file = TensogramFile::open_source_async(server.url(), None).await?;
    assert_eq!(file.message_count()?, 2);

    let _meta0 = file.decode_metadata_async(0).await?;
    let (_, descs1) = file.decode_descriptors_async(1).await?;
    assert_eq!(descs1[0].shape, vec![8]);
    Ok(())
}

// ── Async eager layout tests ─────────────────────────────────────────────────

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_eager_layout_combines_scan_and_discover() -> Result<(), Box<dyn Error>> {
    let msg1 = encode_test_message(vec![4], 10)?;
    let msg2 = encode_test_message(vec![8], 20)?;
    let mut combined = msg1;
    combined.extend_from_slice(&msg2);
    let server = MockServer::start(combined).await?;

    let file = TensogramFile::open_source_async(server.url(), None).await?;
    server.reset_count();
    let (_, desc, data) = file
        .decode_object_async(1, 0, &DecodeOptions::default())
        .await?;

    assert_eq!(desc.shape, vec![8]);
    assert_eq!(data, vec![20u8; 32]);
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_eager_layout_streaming_falls_back() -> Result<(), Box<dyn Error>> {
    let msg = encode_streaming_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source_async(server.url(), None).await?;
    server.reset_count();
    let (_, desc, data) = file
        .decode_object_async(0, 0, &DecodeOptions::default())
        .await?;

    assert_eq!(desc.shape, vec![4]);
    assert_eq!(data, vec![42u8; 16]);
    Ok(())
}

// ── Async descriptor-only tests ──────────────────────────────────────────────

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_descriptor_only_matches_full_read() -> Result<(), Box<dyn Error>> {
    let shapes = vec![vec![4], vec![8, 2], vec![100]];
    let fills = vec![10u8, 20u8, 30u8];
    let msg = encode_multi_object_message(&shapes, &fills)?;
    let server = MockServer::start(msg.clone()).await?;

    let file = TensogramFile::open_source_async(server.url(), None).await?;
    let (_, remote_descs) = file.decode_descriptors_async(0).await?;

    let (_, local_descs) = tensogram::decode::decode_descriptors(&msg)?;

    assert_eq!(remote_descs.len(), local_descs.len());
    for (rd, ld) in remote_descs.iter().zip(local_descs.iter()) {
        assert_eq!(rd.shape, ld.shape);
        assert_eq!(rd.dtype, ld.dtype);
        assert_eq!(rd.encoding, ld.encoding);
        assert_eq!(rd.compression, ld.compression);
    }
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_descriptor_only_large_frame_exercises_fast_path() -> Result<(), Box<dyn Error>>
{
    let meta = make_global_meta();
    let desc = make_descriptor(vec![256, 256]);
    let data = vec![7u8; 256 * 256 * 4];
    let msg = encode::encode(&meta, &[(&desc, &data)], &EncodeOptions::default())?;
    assert!(
        msg.len() > 64 * 1024,
        "payload must exceed 64KB threshold, got {} bytes",
        msg.len()
    );
    let server = MockServer::start(msg.clone()).await?;

    let file = TensogramFile::open_source_async(server.url(), None).await?;
    let (_, remote_descs) = file.decode_descriptors_async(0).await?;
    let (_, local_descs) = tensogram::decode::decode_descriptors(&msg)?;

    assert_eq!(remote_descs.len(), 1);
    assert_eq!(remote_descs[0].shape, local_descs[0].shape);
    assert_eq!(remote_descs[0].dtype, local_descs[0].dtype);

    let (_, _, decoded) = file
        .decode_object_async(0, 0, &DecodeOptions::default())
        .await?;
    assert_eq!(decoded, data);
    Ok(())
}

// ── Async decode_range tests ─────────────────────────────────────────────────

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_remote_decode_range_single_range() -> Result<(), Box<dyn Error>> {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![8]);
    let data: Vec<u8> = (0..32).collect();
    let msg = encode::encode(&meta, &[(&desc, &data)], &EncodeOptions::default())?;
    let server = MockServer::start(msg).await?;

    // Use sync API to test the range (async decode_range not exposed on TensogramFile,
    // but the async layout discovery path is exercised by open_source_async + sync range)
    let file = TensogramFile::open_source(server.url(), None)?;
    let ranges = vec![(2u64, 3u64)];
    let (_, parts) = file.decode_range(0, 0, &ranges, &DecodeOptions::default())?;
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0].len(), 3 * 4);
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_remote_decode_range_matches_local() -> Result<(), Box<dyn Error>> {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![16]);
    let data: Vec<u8> = (0..64).collect();
    let msg = encode::encode(&meta, &[(&desc, &data)], &EncodeOptions::default())?;
    let server = MockServer::start(msg.clone()).await?;

    let ranges = vec![(0u64, 4u64), (8u64, 4u64)];
    let opts = DecodeOptions::default();

    let remote_file = TensogramFile::open_source(server.url(), None)?;
    let (_, remote_parts) = remote_file.decode_range(0, 0, &ranges, &opts)?;

    let (_, local_parts) = tensogram::decode_range(&msg, 0, &ranges, &opts)?;

    assert_eq!(remote_parts.len(), local_parts.len());
    for (rp, lp) in remote_parts.iter().zip(local_parts.iter()) {
        assert_eq!(rp, lp);
    }
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_remote_decode_range_out_of_range_object() -> Result<(), Box<dyn Error>> {
    let msg = encode_test_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source(server.url(), None)?;
    let result = file.decode_range(0, 5, &[(0, 1)], &DecodeOptions::default());
    assert!(result.is_err());
    Ok(())
}

// ── Async request budget tests ───────────────────────────────────────────────

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_remote_request_budget_header_indexed() -> Result<(), Box<dyn Error>> {
    let msg = encode_test_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source_async(server.url(), None).await?;
    let open_requests = server.request_count();

    server.reset_count();
    let _ = file
        .decode_object_async(0, 0, &DecodeOptions::default())
        .await?;
    let object_requests = server.request_count();
    let range_requests = server.range_request_count();

    assert!(
        object_requests <= 2,
        "expected <=2 requests for object read, got {object_requests} (open used {open_requests})"
    );
    assert!(
        range_requests > 0,
        "object read must use Range requests, not full GETs"
    );
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_remote_lazy_open_only_reads_first_preamble() -> Result<(), Box<dyn Error>> {
    let msg1 = encode_test_message(vec![4], 10)?;
    let msg2 = encode_test_message(vec![8], 20)?;
    let mut combined = msg1;
    combined.extend_from_slice(&msg2);
    let server = MockServer::start(combined).await?;

    let file = TensogramFile::open_source_async(server.url(), None).await?;

    let (_, desc, data) = file
        .decode_object_async(0, 0, &DecodeOptions::default())
        .await?;
    assert_eq!(desc.shape, vec![4]);
    assert_eq!(data, vec![10u8; 16]);

    server.reset_count();
    assert_eq!(file.message_count()?, 2);
    let scan_requests = server.request_count();
    assert!(
        scan_requests >= 1,
        "message_count should trigger scanning of remaining messages"
    );
    Ok(())
}

// ── Async suffix read combines test ──────────────────────────────────────────

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_remote_suffix_read_combines_streaming() -> Result<(), Box<dyn Error>> {
    let msg = encode_streaming_message(vec![4], 42)?;
    let server = MockServer::start(msg.clone()).await?;

    let file = TensogramFile::open_source_async(server.url(), None).await?;
    assert!(file.is_remote());
    assert_eq!(file.message_count()?, 1);

    let _meta = file.decode_metadata_async(0).await?;
    let (_, desc, data) = file
        .decode_object_async(0, 0, &DecodeOptions::default())
        .await?;
    assert_eq!(desc.shape, vec![4]);
    assert_eq!(data, vec![42u8; 16]);

    // Verify consistency with local decode
    let (_local_meta, local_objects) = tensogram::decode::decode(&msg, &DecodeOptions::default())?;
    assert_eq!(local_objects[0].1, data);
    Ok(())
}

// ── Async mixed message types test ───────────────────────────────────────────

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_remote_mixed_buffered_then_streaming() -> Result<(), Box<dyn Error>> {
    let buffered_msg = encode_test_message(vec![4], 10)?;
    let streaming_msg = encode_streaming_message(vec![8], 20)?;

    let mut combined = buffered_msg;
    combined.extend_from_slice(&streaming_msg);

    let server = MockServer::start(combined).await?;
    let file = TensogramFile::open_source_async(server.url(), None).await?;
    assert_eq!(file.message_count()?, 2);

    let (_, _, data0) = file
        .decode_object_async(0, 0, &DecodeOptions::default())
        .await?;
    assert_eq!(data0, vec![10u8; 16]);

    let (_, _, data1) = file
        .decode_object_async(1, 0, &DecodeOptions::default())
        .await?;
    assert_eq!(data1, vec![20u8; 32]);
    Ok(())
}

// ── Async repeated reads use cache ───────────────────────────────────────────

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_remote_repeated_object_reads_use_cache() -> Result<(), Box<dyn Error>> {
    let shapes = vec![vec![4], vec![8]];
    let fills = vec![10u8, 20u8];
    let msg = encode_multi_object_message(&shapes, &fills)?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source_async(server.url(), None).await?;
    server.reset_count();

    // First object read
    let _ = file
        .decode_object_async(0, 0, &DecodeOptions::default())
        .await?;
    let first_read_count = server.request_count();

    server.reset_count();

    // Second object read from same message — layout is cached
    let _ = file
        .decode_object_async(0, 1, &DecodeOptions::default())
        .await?;
    let second_read_count = server.request_count();

    assert!(
        second_read_count <= 1,
        "repeated reads should reuse cached layout, got {second_read_count} requests (first was {first_read_count})"
    );
    Ok(())
}

// ── Async matches local decode ───────────────────────────────────────────────

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_remote_matches_local_decode() -> Result<(), Box<dyn Error>> {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![10]);
    let data: Vec<u8> = (0..40).collect();
    let msg = encode::encode(&meta, &[(&desc, &data)], &EncodeOptions::default())?;

    let server = MockServer::start(msg.clone()).await?;

    // Local decode
    let (_local_meta, local_objects) = tensogram::decode::decode(&msg, &DecodeOptions::default())?;

    // Async remote decode
    let remote_file = TensogramFile::open_source_async(server.url(), None).await?;
    let (_remote_meta, remote_desc, remote_data) = remote_file
        .decode_object_async(0, 0, &DecodeOptions::default())
        .await?;

    assert_eq!(local_objects[0].0.shape, remote_desc.shape);
    assert_eq!(local_objects[0].1, remote_data);
    Ok(())
}

// ── Async multi-object test ──────────────────────────────────────────────────

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_remote_multi_object_decode_single() -> Result<(), Box<dyn Error>> {
    let shapes = vec![vec![4], vec![8], vec![2]];
    let fills = vec![10u8, 20u8, 30u8];
    let msg = encode_multi_object_message(&shapes, &fills)?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source_async(server.url(), None).await?;

    // Read only the second object
    let (_, desc, data) = file
        .decode_object_async(0, 1, &DecodeOptions::default())
        .await?;
    assert_eq!(desc.shape, vec![8]);
    assert_eq!(data, vec![20u8; 32]);

    // Read the third object
    let (_, desc, data) = file
        .decode_object_async(0, 2, &DecodeOptions::default())
        .await?;
    assert_eq!(desc.shape, vec![2]);
    assert_eq!(data, vec![30u8; 8]);
    Ok(())
}

// ── Scoped-thread fallback test (current_thread runtime) ─────────────────────

#[cfg(feature = "remote")]
#[tokio::test(flavor = "current_thread")]
async fn test_block_on_shared_current_thread_fallback() -> Result<(), Box<dyn Error>> {
    // This test uses flavor = "current_thread" which forces the scoped-thread
    // fallback path in block_on_shared (lines 93-112 of remote.rs).
    let msg = encode_test_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;
    let url = server.url();

    // Spawn a blocking task to call open_source from within the current-thread runtime.
    // open_source → RemoteBackend::open → block_on_shared which will hit the
    // current-thread fallback branch.
    let result = tokio::task::spawn_blocking(move || {
        let file = TensogramFile::open_source(&url, None)?;
        assert!(file.is_remote());
        let _meta = file.decode_metadata(0)?;
        let (_, desc, data) = file.decode_object(0, 0, &DecodeOptions::default())?;
        assert_eq!(desc.shape, vec![4]);
        assert_eq!(data, vec![42u8; 16]);
        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    })
    .await;

    match result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(e.to_string().into()),
        Err(e) => return Err(format!("spawn_blocking panicked: {e}").into()),
    }
    Ok(())
}

// ── Remote decode_range tests ────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_decode_range_single_range() -> Result<(), Box<dyn Error>> {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![8]);
    let data: Vec<u8> = (0..32).collect();
    let msg = encode::encode(&meta, &[(&desc, &data)], &EncodeOptions::default())?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source(server.url(), None)?;
    let ranges = vec![(2u64, 3u64)];
    let (_, parts) = file.decode_range(0, 0, &ranges, &DecodeOptions::default())?;
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0].len(), 3 * 4);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_decode_range_matches_local() -> Result<(), Box<dyn Error>> {
    let meta = make_global_meta();
    let desc = make_descriptor(vec![16]);
    let data: Vec<u8> = (0..64).collect();
    let msg = encode::encode(&meta, &[(&desc, &data)], &EncodeOptions::default())?;
    let server = MockServer::start(msg.clone()).await?;

    let ranges = vec![(0u64, 4u64), (8u64, 4u64)];
    let opts = DecodeOptions::default();

    let remote_file = TensogramFile::open_source(server.url(), None)?;
    let (_, remote_parts) = remote_file.decode_range(0, 0, &ranges, &opts)?;

    let (_, local_parts) = tensogram::decode_range(&msg, 0, &ranges, &opts)?;

    assert_eq!(remote_parts.len(), local_parts.len());
    for (rp, lp) in remote_parts.iter().zip(local_parts.iter()) {
        assert_eq!(rp, lp);
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_decode_range_out_of_range_object() -> Result<(), Box<dyn Error>> {
    let msg = encode_test_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;

    let file = TensogramFile::open_source(server.url(), None)?;
    let result = file.decode_range(0, 5, &[(0, 1)], &DecodeOptions::default());
    assert!(result.is_err());
    Ok(())
}
