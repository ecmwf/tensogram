# Remote Access

Enable the `remote` feature to open `.tgm` files on HTTP, S3, GCS, or Azure without downloading the whole file. Individual objects are fetched via targeted range requests.

```toml
[dependencies]
tensogram = { path = "...", features = ["remote"] }
```

## Opening a Remote File

```rust
use tensogram::TensogramFile;

// Auto-detect: local path or remote URL.  The second argument
// is `scan_opts: Option<RemoteScanOptions>` — pass `None` for
// the forward-only walker (see "Bidirectional scan" below).
let mut file = TensogramFile::open_source("https://example.com/data.tgm", None)?;

// S3
let mut file = TensogramFile::open_source("s3://bucket/data.tgm", None)?;
```

`open_source` inspects the URL scheme and routes to the remote backend for `s3://`, `s3a://`, `gs://`, `az://`, `azure://`, `http://`, `https://`. Everything else is treated as a local path.

The Rust `open()` method is unchanged and always opens a local file. In Python, `TensogramFile.open()` auto-detects remote URLs.

You can also check whether a string is a remote URL without opening:

```rust
use tensogram::is_remote_url;

assert!(is_remote_url("s3://bucket/file.tgm"));
assert!(!is_remote_url("/local/path/file.tgm"));
```

## Storage Options (Credentials, Region, etc.)

Pass an explicit options map for fine-grained control:

```rust
use std::collections::BTreeMap;
use tensogram::TensogramFile;

let mut opts = BTreeMap::new();
opts.insert("aws_access_key_id".to_string(), "AKIA...".to_string());
opts.insert("aws_secret_access_key".to_string(), "...".to_string());
opts.insert("region".to_string(), "eu-west-1".to_string());

let mut file = TensogramFile::open_remote("s3://bucket/data.tgm", &opts, None)?;
```

When no options are passed, credentials are read from the environment (e.g. `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `GOOGLE_APPLICATION_CREDENTIALS`).

## Bidirectional Scan

`open_source`, `open_remote`, and their async siblings accept a `scan_opts: Option<RemoteScanOptions>` argument. Passing `Some(RemoteScanOptions { bidirectional: true })` enables a walker that alternates forward and backward hops across the file, using the v3 postamble's mirrored `total_length` field to walk inward from EOF in parallel with the forward sweep. On well-formed header-indexed files this roughly halves the number of HTTP `GET` requests needed for tail / full-scan access.

`None` (or `Some(RemoteScanOptions::default())`) keeps the forward-only walker. Both walkers produce identical layouts; the difference is only in the discovery path.

```rust
use tensogram::{RemoteScanOptions, TensogramFile};

let file = TensogramFile::open_source(
    "https://example.com/data.tgm",
    Some(RemoteScanOptions { bidirectional: true }),
)?;
```

In Python the same opt-in is a keyword argument, accepted on both sync and async entry points:

```python
import tensogram

with tensogram.TensogramFile.open_remote(
    "https://example.com/data.tgm",
    bidirectional=True,
) as f:
    ...

# Async sibling
f = await tensogram.AsyncTensogramFile.open_remote(
    "https://example.com/data.tgm",
    bidirectional=True,
)
```

In TypeScript the same opt-in is a `FromUrlOptions` field, accepted by `TensogramFile.fromUrl` against any Range-capable HTTP server:

```typescript
import { TensogramFile, init } from '@ecmwf.int/tensogram';

await init();
const file = await TensogramFile.fromUrl('https://example.com/data.tgm', {
    bidirectional: true,
});
console.log('messageCount:', file.messageCount);
console.log('messageLayouts:', file.messageLayouts);
file.close();
```

Set `debug: true` alongside `bidirectional: true` to emit `console.debug` events on every walker state transition (`tensogram:scan:mode`, `tensogram:scan:fallback`, `tensogram:scan:fwd-terminated`, `tensogram:scan:gap-closed`, `tensogram:scan:hop`, `tensogram:scan:footer-eager`) — same vocabulary as the Rust `tracing` events at `target = "tensogram::remote_scan"`.

The default is forward-only across all bindings until benchmarks confirm the win across every workload tier. Local-file backends accept the option and silently ignore it (a single forward sweep is the only sensible strategy locally).

### Eager footer-indexed backward discovery

When the bidirectional walker discovers a **footer-indexed** message via its postamble, the dispatcher folds an eager footer-region fetch into the same paired round as the candidate-preamble validation. The metadata + index frames land in the cached layout inline, so a subsequent `read_metadata` / `messageMetadata` accessor short-circuits without issuing a separate footer-region GET.

The optimisation fires only on **footer-indexed messages discovered backward**:
- Header-indexed messages on backward keep the lazy path. The forward walker fetches the header chunk in one GET; backward fetching postamble + preamble + separate header chunk would be net-worse (3 GETs vs 1).
- Footer-indexed messages discovered forward keep the existing forward-only optimisation (one suffix-chunk fetch per message).
- Header-indexed messages with footer hash frames have a non-empty footer region; the dispatcher fetches the bytes speculatively but discards them after the preamble flags reveal `HEADER_INDEX`. Cost: a few hundred bytes per such message; benefit: zero extra GETs.

The footer fetch is **best-effort**: a transport failure or footer parse error never poisons preamble validation. The layout still commits via the validated preamble alone, and the lazy `ensure_layout` path picks up footer discovery on first metadata access. Behaviour is symmetric across the Rust sync, Rust async, and TypeScript dispatchers.

## Python Usage

```python
import tensogram

# Auto-detect remote URL
with tensogram.TensogramFile.open("s3://bucket/data.tgm") as f:
    meta = f.file_decode_metadata(0)
    result = f.file_decode_object(0, 0)
    data = result["data"]  # numpy array

# With explicit storage options
with tensogram.TensogramFile.open_remote(
    "s3://bucket/data.tgm",
    {"region": "eu-west-1"}
) as f:
    print(f.source())   # "s3://bucket/data.tgm"
    print(f.is_remote()) # True

# With the bidirectional walker (see "Bidirectional Scan" above)
with tensogram.TensogramFile.open_remote(
    "s3://bucket/data.tgm",
    {"region": "eu-west-1"},
    bidirectional=True,
) as f:
    print(f.message_count())
```

## xarray Usage

```python
import xarray as xr

ds = xr.open_dataset(
    "s3://bucket/data.tgm",
    engine="tensogram",
    storage_options={"region": "eu-west-1"},
)
```

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
use tensogram::DecodeOptions;

// Metadata only — triggers layout discovery on first call, then cached
let meta = file.decode_metadata(0)?;

// Descriptors — reads only the descriptor data needed for each object
let (meta, descriptors) = file.decode_descriptors(0)?;

// Single object by index — fetches only the target object frame
let (meta, desc, data) = file.decode_object(0, 2, &DecodeOptions::default())?;
```

These methods also work on local files, where they read the full message and decode the requested parts.

## Request Budget

### Header-indexed files (buffered writes)

| Phase | Operation | HTTP Requests |
|-------|-----------|:---:|
| **Open** | `open_source` / `open_remote` | 1 HEAD + 1 GET (first preamble only, 24 B) |
| **Next message** | first data access to message *i* | 1 GET (preamble + layout combined) |
| **Cached** | `decode_metadata(i)` again | 0 (served from cache) |
| **Object read** | `decode_object(i, j)` | 1 GET per object (if layout already cached) |
| **Descriptors** | `decode_descriptors(i)` | 1–3 GETs per object (descriptor-only reads for large frames) |
| **Message count** | `message_count()` | 1 GET per undiscovered message (24 B each, preamble only) |

### Footer-indexed files (buffered with known total_length)

| Phase | Operation | HTTP Requests |
|-------|-----------|:---:|
| **Open** | `open_source` / `open_remote` | 1 HEAD + 1 GET (first preamble only, 24 B) |
| **Next message** | first data access to message *i* | 1 GET (preamble) + 1 GET (suffix) |
| **Cached** | `decode_metadata(i)` again | 0 (served from cache) |
| **Object read** | `decode_object(i, j)` | 1 GET per object (if layout already cached) |
| **Descriptors** | `decode_descriptors(i)` | 1–3 GETs per object |
| **Message count** | `message_count()` | 1 GET per undiscovered message (24 B each) |

### Streaming files (total_length=0)

| Phase | Operation | HTTP Requests |
|-------|-----------|:---:|
| **Open** | `open_source` / `open_remote` | 1 HEAD + 1 GET (preamble) + 1 GET (END_MAGIC check) |
| **First access** | `decode_metadata(0)` | 2 GETs (postamble + footer region) |
| **Object read** | `decode_object(0, j)` | 1 GET per object |
| **Message count** | `message_count()` | 0 (streaming is always the last message) |

Layout discovery is combined with message scanning for both header-indexed and footer-indexed messages — the library reads the preamble and layout in one GET (header-indexed) or two GETs (footer-indexed suffix read). `message_count()` uses a lean scan path (24 bytes per preamble). Streaming messages (`total_length=0`) must be the last message in a multi-message file.

## How It Works (Header-Indexed Example)

```mermaid
sequenceDiagram
    participant App
    participant TensogramFile
    participant ObjectStore

    App->>TensogramFile: open_source("s3://bucket/file.tgm")
    TensogramFile->>ObjectStore: HEAD (get file size)
    TensogramFile->>ObjectStore: GET range 0..24 (preamble)
    Note right of TensogramFile: Discover message offsets

    App->>TensogramFile: decode_object(0, 2)
    TensogramFile->>ObjectStore: GET range 24..N (header chunk, up to 256KB)
    Note right of TensogramFile: First access: parse metadata + index, cache layout
    TensogramFile->>ObjectStore: GET range offset..offset+len (object frame 2)
    TensogramFile-->>App: (metadata, descriptor, decoded_bytes)
```

## Checking if a File is Remote

```rust
use tensogram::TensogramFile;

let file = TensogramFile::open_source("s3://bucket/data.tgm", None)?;
assert!(file.is_remote());
println!("source: {}", file.source()); // "s3://bucket/data.tgm"
```

`source()` returns the original URL for remote files and the file path for local files.

## Error Handling

Remote access can return different `TensogramError` variants depending on the failure:

| Error condition | Error type | When it happens |
|-----------------|------------|-----------------|
| Invalid URL | `Remote` | `open_source` / `open_remote` with a malformed URL |
| Connection failure | `Remote` | Network unreachable, DNS failure, timeout |
| File not found | `Remote` | HTTP 404, S3 NoSuchKey |
| No valid messages | `Remote` | File contains no parseable messages |
| Unsupported layout | `Remote` | Message lacks both header-index and footer-index flags |
| Object index out of range | `Object` | `decode_object(i, j)` where `j >= object_count` |

All errors are returned as `Result`. The library avoids panics.

## Shared Runtime

Remote I/O uses a process-wide shared tokio runtime (multi-thread, 2 workers) created on first use. All `RemoteBackend` instances share the same runtime, so TCP connection pools and DNS caches are reused across calls.

The sync bridge adapts to the calling context:

- **Not in a tokio runtime** (Python, CLI): the shared runtime's handle drives the future directly — no extra thread creation.
- **Inside a multi-thread tokio runtime** (`#[tokio::test]`, server handler): `block_in_place` tells tokio to spawn a replacement worker so the blocked thread doesn't cause runtime starvation.
- **Inside a current-thread tokio runtime**: falls back to a scoped thread, since `block_in_place` is not supported on single-threaded runtimes.

## Async API

The `async` feature enables async methods for decode, read, and metadata extraction. These work for both local and remote files:

```rust
use tensogram::{TensogramFile, DecodeOptions};

// Async decode methods (feature = "async")
let meta = file.decode_metadata_async(0).await?;
let (meta, descs) = file.decode_descriptors_async(0).await?;
let (meta, desc, data) = file.decode_object_async(0, 0, &DecodeOptions::default()).await?;
let msg = file.read_message_async(0).await?;
```

When both `remote` and `async` features are enabled, async open methods are also available:

```rust
// Async open (auto-detects local vs remote) — requires remote + async
let file = TensogramFile::open_source_async("s3://bucket/data.tgm", None).await?;

// Async open with explicit storage options
let file = TensogramFile::open_remote_async(
    "s3://bucket/data.tgm",
    &opts,
    None,
).await?;
```

For remote backends, async methods directly `await` object store operations, bypassing the sync bridge entirely. For local backends, they use `spawn_blocking` for file I/O.

```toml
[dependencies]
tensogram = { path = "...", features = ["remote", "async"] }
```

## Range Reads

`TensogramFile::decode_range()` supports partial object decoding for both local and remote files. It takes an object index and a list of `(offset, count)` element ranges, returning only the requested elements without decoding the entire object.

For remote files, it fetches the full object frame (via indexed access) then runs the range decode pipeline on the raw payload. This is most beneficial with szip-compressed objects that have `szip_block_offsets`, where only the compressed blocks covering the requested range are decompressed.

```rust
// Rust: decode elements 100..200 from object 0
let ranges = vec![(100, 100)];
let (desc, parts) = file.decode_range(0, 0, &ranges, &DecodeOptions::default())?;
```

```python
# Python: decode elements 100..200 from object 0
arr = file.file_decode_range(0, 0, [(100, 100)], join=True)
```

The xarray backend uses `file_decode_range` automatically when slicing remote arrays that support partial decode (uncompressed or szip-compressed objects without shuffle filters).

## Descriptor-Only Reads

`decode_descriptors()` fetches only the CBOR descriptor from each data object frame, not the full payload. For large objects (hundreds of MB), this avoids downloading the entire frame just to extract a few hundred bytes of metadata.

For frames smaller than 64 KB, the full frame is read in a single request (fewer round-trips). For larger frames, the library reads only the frame header (16 bytes), footer (12 bytes), and the CBOR descriptor region.

## Limitations

- **Streaming messages must be last.** In multi-message files, streaming-encoded messages (`total_length=0`) must be the last message. The remote scanner assumes the streaming message extends to the end of the file.
- **Optimistic scan for buffered messages.** Remote message scanning validates preamble magic and `total_length` plausibility but does not verify end-of-message markers for buffered messages. Streaming messages (`total_length=0`) do validate the END_MAGIC at EOF.
- **Read-only.** Remote writes are not supported.
- **Header probe size.** Layout discovery reads a single chunk of up to 256 KB from the header region. If the metadata or index frame does not fit in this chunk, `decode_metadata()` will error (it does not retry with a larger read).
- **HTTP server requirements.** The remote HTTP server must support `HEAD` requests (for file size) and `Range` request headers (for partial reads).
- **`read_message()` and `decode_message()` download the full message** even for remote files. Use `decode_metadata()`, `decode_descriptors()`, or `decode_object()` for selective access.
- **Zarr remote reads are lazy per-chunk.** The zarr store fetches only metadata at open time; individual chunks are decoded on first access. Local files still use eager decode for lower latency.
- **Concurrent async access.** Async methods take `&self`, so a single `TensogramFile` handle can serve concurrent async reads. The remote backend serialises forward-walker scan rounds via an internal mutex, but reads of already-discovered messages run truly concurrently — `asyncio.gather` / `tokio::join!` achieve real I/O overlap on a single handle.
