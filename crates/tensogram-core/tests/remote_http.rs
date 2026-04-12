#![cfg(feature = "remote")]

use std::collections::BTreeMap;
use std::error::Error;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use http_body_util::Full;
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;

use tensogram_core::decode::DecodeOptions;
use tensogram_core::encode::{self, EncodeOptions};
use tensogram_core::file::TensogramFile;
use tensogram_core::types::{ByteOrder, DataObjectDescriptor, GlobalMetadata};
use tensogram_core::{is_remote_url, Dtype};

// ── Test helpers ─────────────────────────────────────────────────────────────

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

    let mut file = TensogramFile::open_source(server.url())?;
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

    let mut file = TensogramFile::open_source(server.url())?;
    let (decoded_meta, objects) = file.decode_message(0, &DecodeOptions::default())?;
    assert_eq!(decoded_meta.version, 2);
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0].1, data);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_decode_metadata() -> Result<(), Box<dyn Error>> {
    let msg = encode_test_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;

    let mut file = TensogramFile::open_source(server.url())?;
    server.reset_count();

    let meta = file.decode_metadata(0)?;
    assert_eq!(meta.version, 2);

    // First metadata call triggers layout discovery (1 header chunk read).
    // Subsequent calls should be free (cached).
    let count_after_first = server.request_count();
    assert!(
        count_after_first <= 1,
        "first metadata read should need at most 1 request"
    );

    server.reset_count();
    let meta2 = file.decode_metadata(0)?;
    assert_eq!(meta2.version, 2);
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

    let mut file = TensogramFile::open_source(server.url())?;
    let (meta, descriptors) = file.decode_descriptors(0)?;
    assert_eq!(meta.version, 2);
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

    let mut file = TensogramFile::open_source(server.url())?;
    server.reset_count();

    let (decoded_meta, decoded_desc, decoded_data) =
        file.decode_object(0, 0, &DecodeOptions::default())?;
    assert_eq!(decoded_meta.version, 2);
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

    let mut file = TensogramFile::open_source(server.url())?;

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

    let mut file = TensogramFile::open_source(server.url())?;
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

    let mut file = TensogramFile::open_source(server.url())?;
    assert_eq!(file.message_count()?, 2);

    let meta0 = file.decode_metadata(0)?;
    assert_eq!(meta0.version, 2);

    let (_, descs1) = file.decode_descriptors(1)?;
    assert_eq!(descs1[0].shape, vec![8]);
    Ok(())
}

// ── Object index out of range ────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_object_out_of_range() -> Result<(), Box<dyn Error>> {
    let msg = encode_test_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;

    let mut file = TensogramFile::open_source(server.url())?;
    let result = file.decode_object(0, 5, &DecodeOptions::default());
    assert!(result.is_err());
    Ok(())
}

// ── Request count verification ───────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_request_count_header_indexed() -> Result<(), Box<dyn Error>> {
    let msg = encode_test_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;

    let mut file = TensogramFile::open_source(server.url())?;
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

    let mut file = TensogramFile::open_source(server.url())?;
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

    let mut file = TensogramFile::open_source(server.url())?;
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
async fn test_remote_repeated_object_reads_use_cache() -> Result<(), Box<dyn Error>> {
    let shapes = vec![vec![4], vec![8]];
    let fills = vec![10u8, 20u8];
    let msg = encode_multi_object_message(&shapes, &fills)?;
    let server = MockServer::start(msg).await?;

    let mut file = TensogramFile::open_source(server.url())?;
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
    let (local_meta, local_objects) =
        tensogram_core::decode::decode(&msg, &DecodeOptions::default())?;

    // Remote decode
    let mut remote_file = TensogramFile::open_source(server.url())?;
    let (remote_meta, remote_desc, remote_data) =
        remote_file.decode_object(0, 0, &DecodeOptions::default())?;

    assert_eq!(local_meta.version, remote_meta.version);
    assert_eq!(local_objects[0].0.shape, remote_desc.shape);
    assert_eq!(local_objects[0].1, remote_data);
    Ok(())
}

// ── Error cases ──────────────────────────────────────────────────────────────

#[test]
fn test_remote_invalid_url() -> Result<(), Box<dyn Error>> {
    let result = TensogramFile::open_source("http://[invalid-url]/file.tgm");
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
    let mut file = TensogramFile::open_source(path)?;
    assert!(!file.is_remote());
    assert_eq!(file.message_count()?, 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_source_returns_url() -> Result<(), Box<dyn Error>> {
    let msg = encode_test_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;
    let url = server.url();

    let file = TensogramFile::open_source(url.clone())?;
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
        tensogram_core::streaming::StreamingEncoder::new(buf, &meta, &EncodeOptions::default())
            .map_err(std::io::Error::other)?;
    enc.write_object(&desc, &data)
        .map_err(std::io::Error::other)?;
    enc.finish().map_err(std::io::Error::other)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_streaming_message_open_and_decode() -> Result<(), Box<dyn Error>> {
    let msg = encode_streaming_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;

    let mut file = TensogramFile::open_source(server.url())?;
    assert!(file.is_remote());
    assert_eq!(file.message_count()?, 1);

    let meta = file.decode_metadata(0)?;
    assert_eq!(meta.version, 2);

    let (_, desc, data) = file.decode_object(0, 0, &DecodeOptions::default())?;
    assert_eq!(desc.shape, vec![4]);
    assert_eq!(data, vec![42u8; 16]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_streaming_matches_local_decode() -> Result<(), Box<dyn Error>> {
    let msg = encode_streaming_message(vec![10], 99)?;
    let server = MockServer::start(msg.clone()).await?;

    let (local_meta, local_objects) =
        tensogram_core::decode::decode(&msg, &DecodeOptions::default())?;

    let mut remote_file = TensogramFile::open_source(server.url())?;
    let (remote_meta, remote_desc, remote_data) =
        remote_file.decode_object(0, 0, &DecodeOptions::default())?;

    assert_eq!(local_meta.version, remote_meta.version);
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
        tensogram_core::streaming::StreamingEncoder::new(buf, &meta, &EncodeOptions::default())
            .map_err(std::io::Error::other)?;
    enc.write_object(&desc1, &data1)
        .map_err(std::io::Error::other)?;
    enc.write_object(&desc2, &data2)
        .map_err(std::io::Error::other)?;
    let msg = enc.finish().map_err(std::io::Error::other)?;

    let server = MockServer::start(msg).await?;
    let mut file = TensogramFile::open_source(server.url())?;

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
    let mut file = TensogramFile::open_source(server.url())?;
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

    let mut file = TensogramFile::open_source(server.url())?;
    let (_, desc, data) = file.decode_object(0, 0, &DecodeOptions::default())?;
    assert_eq!(desc.shape, vec![4]);
    assert_eq!(data, vec![42u8; 16]);

    let (local_meta, local_objects) =
        tensogram_core::decode::decode(&msg, &DecodeOptions::default())?;
    assert_eq!(local_objects[0].1, data);
    assert_eq!(local_meta.version, 2);
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
                    let mut file = TensogramFile::open_source(&u)?;
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
            let mut file = TensogramFile::open_source(&url)?;
            let meta = file.decode_metadata(0)?;
            assert_eq!(meta.version, 2);
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

    let mut file = TensogramFile::open_source(server.url())?;
    let (_, remote_descs) = file.decode_descriptors(0)?;

    let (_, local_descs) = tensogram_core::decode::decode_descriptors(&msg)?;

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

    let mut file = TensogramFile::open_source(server.url())?;
    let (_, remote_descs) = file.decode_descriptors(0)?;
    let (_, local_descs) = tensogram_core::decode::decode_descriptors(&msg)?;

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

    let mut file = TensogramFile::open_source_async(server.url()).await?;
    assert!(file.is_remote());
    assert_eq!(file.message_count()?, 1);

    let meta = file.decode_metadata_async(0).await?;
    assert_eq!(meta.version, 2);
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

    let mut file = TensogramFile::open_source_async(server.url()).await?;
    let (decoded_meta, decoded_desc, decoded_data) = file
        .decode_object_async(0, 0, &DecodeOptions::default())
        .await?;
    assert_eq!(decoded_meta.version, 2);
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

    let mut file = TensogramFile::open_source_async(server.url()).await?;
    let (meta, descriptors) = file.decode_descriptors_async(0).await?;
    assert_eq!(meta.version, 2);
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

    let mut sync_file = TensogramFile::open_source(server.url())?;
    let (sync_meta, sync_desc, sync_data) =
        sync_file.decode_object(0, 0, &DecodeOptions::default())?;

    let mut async_file = TensogramFile::open_source_async(server.url()).await?;
    let (async_meta, async_desc, async_data) = async_file
        .decode_object_async(0, 0, &DecodeOptions::default())
        .await?;

    assert_eq!(sync_meta.version, async_meta.version);
    assert_eq!(sync_desc.shape, async_desc.shape);
    assert_eq!(sync_data, async_data);
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_remote_read_message() -> Result<(), Box<dyn Error>> {
    let msg = encode_test_message(vec![4], 42)?;
    let server = MockServer::start(msg.clone()).await?;

    let mut file = TensogramFile::open_source_async(server.url()).await?;
    let remote_msg = file.read_message_async(0).await?;
    assert_eq!(remote_msg, msg);
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_async_remote_streaming_message() -> Result<(), Box<dyn Error>> {
    let msg = encode_streaming_message(vec![4], 42)?;
    let server = MockServer::start(msg).await?;

    let mut file = TensogramFile::open_source_async(server.url()).await?;
    assert!(file.is_remote());

    let meta = file.decode_metadata_async(0).await?;
    assert_eq!(meta.version, 2);

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

    let mut file = TensogramFile::open_source_async(server.url()).await?;
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

    let mut file = TensogramFile::open_source_async(server.url()).await?;
    let result = file.decode_metadata_async(5).await;
    assert!(result.is_err());
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

    let mut file = TensogramFile::open_source(server.url())?;
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

    let mut remote_file = TensogramFile::open_source(server.url())?;
    let (_, remote_parts) = remote_file.decode_range(0, 0, &ranges, &opts)?;

    let (_, local_parts) = tensogram_core::decode_range(&msg, 0, &ranges, &opts)?;

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

    let mut file = TensogramFile::open_source(server.url())?;
    let result = file.decode_range(0, 5, &[(0, 1)], &DecodeOptions::default());
    assert!(result.is_err());
    Ok(())
}
