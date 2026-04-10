# Remote Access

Enable the `remote` feature to open `.tgm` files on HTTP, S3, GCS, or Azure without downloading the whole file. Individual objects are fetched via targeted range requests.

```toml
[dependencies]
tensogram-core = { path = "...", features = ["remote"] }
```

## Opening a Remote File

```rust
use tensogram_core::TensogramFile;

// Auto-detect: local path or remote URL
let mut file = TensogramFile::open_source("https://example.com/forecast.tgm")?;

// S3
let mut file = TensogramFile::open_source("s3://bucket/forecast.tgm")?;
```

`open_source` inspects the URL scheme and routes to the remote backend for `s3://`, `s3a://`, `gs://`, `az://`, `azure://`, `http://`, `https://`. Everything else is treated as a local path.

The original `open()` method is unchanged and always opens a local file.

You can also check whether a string is a remote URL without opening:

```rust
use tensogram_core::is_remote_url;

assert!(is_remote_url("s3://bucket/file.tgm"));
assert!(!is_remote_url("/local/path/file.tgm"));
```

## Storage Options (Credentials, Region, etc.)

Pass an explicit options map for fine-grained control:

```rust
use std::collections::BTreeMap;
use tensogram_core::TensogramFile;

let mut opts = BTreeMap::new();
opts.insert("aws_access_key_id".to_string(), "AKIA...".to_string());
opts.insert("aws_secret_access_key".to_string(), "...".to_string());
opts.insert("region".to_string(), "eu-west-1".to_string());

let mut file = TensogramFile::open_remote("s3://bucket/forecast.tgm", &opts)?;
```

When no options are passed, credentials are read from the environment (e.g. `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `GOOGLE_APPLICATION_CREDENTIALS`).

## Supported Schemes

| Scheme | Backend | Notes |
|--------|---------|-------|
| `http://`, `https://` | HTTP | `allow_http` is set automatically for `http://` |
| `s3://`, `s3a://` | Amazon S3 | Env-based or explicit credentials |
| `gs://` | Google Cloud Storage | Service account or env |
| `az://`, `azure://` | Azure Blob Storage | MSI or env |

All backends are provided by the [`object_store`](https://crates.io/crates/object_store) crate.

## Object-Level Access

Three methods provide selective access without downloading full messages:

```rust
use tensogram_core::DecodeOptions;

// Metadata only — triggers layout discovery on first call, then cached
let meta = file.file_decode_metadata(0)?;

// Descriptors — fetches each object frame to extract its descriptor
let (meta, descriptors) = file.file_decode_descriptors(0)?;

// Single object by index — fetches only the target object frame
let (meta, desc, data) = file.file_decode_object(0, 2, &DecodeOptions::default())?;
```

These methods also work on local files, where they read the full message and decode the requested parts.

## Request Budget

For header-indexed files (the default for non-streaming writes):

| Phase | Operation | HTTP Requests |
|-------|-----------|:---:|
| **Open** | `open_source` / `open_remote` | 1 HEAD + 1 GET per message (preamble read) |
| **First access** | `file_decode_metadata(i)` | 1 GET (header chunk, discovers metadata + index) |
| **Cached** | `file_decode_metadata(i)` again | 0 (served from cache) |
| **Object read** | `file_decode_object(i, j)` | 1 GET per object (if layout already cached) |
| **Descriptors** | `file_decode_descriptors(i)` | 1 GET per object in message |

The layout (metadata + index) is discovered per-message on first access to that message, then cached. Subsequent calls reuse the cached layout.

## How It Works

```mermaid
sequenceDiagram
    participant App
    participant TensogramFile
    participant ObjectStore

    App->>TensogramFile: open_source("s3://bucket/file.tgm")
    TensogramFile->>ObjectStore: HEAD (get file size)
    TensogramFile->>ObjectStore: GET range 0..24 (preamble)
    Note right of TensogramFile: Discover message offsets

    App->>TensogramFile: file_decode_object(0, 2)
    TensogramFile->>ObjectStore: GET range 24..N (header chunk, up to 256KB)
    Note right of TensogramFile: First access: parse metadata + index, cache layout
    TensogramFile->>ObjectStore: GET range offset..offset+len (object frame 2)
    TensogramFile-->>App: (metadata, descriptor, decoded_bytes)
```

## Checking if a File is Remote

```rust
use tensogram_core::TensogramFile;

let file = TensogramFile::open_source("s3://bucket/data.tgm")?;
assert!(file.is_remote());
println!("source: {}", file.source()); // "s3://bucket/data.tgm"
```

`source()` returns the original URL for remote files and the file path for local files.

## Error Handling

Remote operations return `TensogramError::Remote` for transport-level failures:

| Error condition | When it happens |
|-----------------|-----------------|
| Invalid URL | `open_source` / `open_remote` with a malformed URL |
| Connection failure | Network unreachable, DNS failure, timeout |
| File not found | HTTP 404, S3 NoSuchKey |
| No valid messages | File contains only streaming (`total_length=0`) or corrupt messages |
| Header-only required | Message has no header metadata flag |
| Object index out of range | `file_decode_object(i, j)` where `j >= object_count` |

All errors are returned as `Result` — no panics in library code.

## Limitations

- **Header-indexed messages only.** Footer-indexed messages (produced by `StreamingEncoder`) are not yet supported remotely. Use buffered `encode()` for files destined for remote storage.
- **Read-only.** Remote writes are not supported.
- **Header probe size.** Layout discovery reads up to 256 KB of the header region. If the header metadata + index exceeds this (extremely rare), metadata may still be found but the index could be missing, causing object reads to fall back to downloading the full message.
- **Rust core only.** Python bindings, xarray backend, and zarr store do not yet support remote URLs. This will be added in a follow-up PR.
- **`read_message()` and `decode_message()` download the full message** even for remote files. Use the `file_decode_*` methods for selective access.
- **Thread-per-request.** Each range request spawns a thread with a temporary tokio runtime. This is correct but not optimal for many small reads. A shared runtime will be added in a follow-up.
