# Python API

Tensogram provides native Python bindings via [PyO3](https://pyo3.rs). All tensor data crosses the boundary as NumPy arrays.

## Installation

```bash
# From PyPI (once published)
pip install tensogram

# From source
pip install maturin numpy
cd crates/tensogram-python && maturin develop
```

## Quick Start

```python
import numpy as np
import tensogram

# Encode a 2D temperature field
temps = np.random.randn(100, 200).astype(np.float32) + 273.15
meta = {"version": 2}
desc = {"type": "ntensor", "shape": [100, 200], "dtype": "float32"}

msg = tensogram.encode(meta, [(desc, temps)])

# Decode it back
meta, objects = tensogram.decode(msg)
desc, array = objects[0]
print(array.shape)  # (100, 200)
```

## Encoding

### Basic encoding

`tensogram.encode()` takes metadata, a list of `(descriptor, array)` pairs, and returns wire-format bytes:

```python
msg = tensogram.encode(
    {"version": 2},
    [({"type": "ntensor", "shape": [3], "dtype": "float32"}, np.array([1, 2, 3], dtype=np.float32))],
    hash="xxh3",  # default; use None to skip hashing
)
```

### Descriptor keys

Every object in a message is described by a dict. The three required keys define what the tensor looks like; the optional keys control how it is stored on the wire.

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `"type"` | yes | — | Object type, e.g. `"ntensor"` |
| `"shape"` | yes | — | Tensor dimensions, e.g. `[100, 200]` |
| `"dtype"` | yes | — | Data type name (see [Data Types](../format/dtypes.md)) |
| `"strides"` | no | row-major | Element strides; computed automatically if omitted |
| `"byte_order"` | no | native | `"little"` or `"big"`; defaults to host byte order |
| `"encoding"` | no | `"none"` | Encoding stage — see below |
| `"filter"` | no | `"none"` | Filter stage — see below |
| `"compression"` | no | `"none"` | Compression stage — see below |

Any additional keys (e.g. `"reference_value"`, `"bits_per_value"`) are stored in the descriptor's `.params` dict and passed through to the encoding pipeline.

### The encoding pipeline

Each object passes through a three-stage pipeline before it is stored. You control each stage via descriptor keys:

```
raw bytes → encoding → filter → compression → wire payload
```

**Encoding** transforms the data representation:

| Value | What it does | Use case |
|-------|-------------|----------|
| `"none"` | Pass-through (default) | Exact values, integer data |
| `"simple_packing"` | Quantize floats to packed integers | Weather fields (GRIB-style) |

**Filter** rearranges bytes to improve compressibility:

| Value | What it does | Use case |
|-------|-------------|----------|
| `"none"` | Pass-through (default) | Most cases |
| `"shuffle"` | Byte-transpose by element width (requires `"shuffle_element_size"`) | Improves lz4/zstd ratio on typed data |

**Compression** reduces the payload size:

| Value | Random access | Type | Use case |
|-------|--------------|------|----------|
| `"none"` | yes | — | No compression |
| `"zstd"` | no | lossless | General-purpose, best ratio/speed tradeoff |
| `"lz4"` | no | lossless | Fastest decompression |
| `"szip"` | yes (RSI blocks) | lossless | Integer/packed data (CCSDS 121.0-B-3) |
| `"blosc2"` | yes (chunks) | lossless | Large tensors, multi-codec |
| `"zfp"` | yes (fixed-rate) | lossy | Floating-point arrays |
| `"sz3"` | no | lossy | Error-bounded scientific data |

Compression parameters are passed as extra descriptor keys. For example, zstd level:

```python
desc = {
    "type": "ntensor", "shape": [1000], "dtype": "float32",
    "compression": "zstd", "zstd_level": 9,
}
```

For the full list of compressor parameters, see [Compression](../encodings/compression.md).

### Common pipeline combinations

```python
# Lossless, fast decompression
desc = {"type": "ntensor", "shape": shape, "dtype": "float32",
        "compression": "lz4"}

# Lossless, best ratio (shuffle_element_size must match dtype byte width)
desc = {"type": "ntensor", "shape": shape, "dtype": "float32",
        "filter": "shuffle", "shuffle_element_size": 4, "compression": "zstd", "zstd_level": 12}

# GRIB-style: quantize to 16-bit packed ints, then compress
# compute_packing_params expects a flat float64 array
values = data.astype(np.float64).ravel()
params = tensogram.compute_packing_params(values, bits_per_value=16, decimal_scale_factor=0)
desc = {"type": "ntensor", "shape": shape, "dtype": "float64",
        "encoding": "simple_packing", "compression": "zstd", **params}

# Lossy float compression with error bound (zfp operates on float64)
desc = {"type": "ntensor", "shape": shape, "dtype": "float64",
        "compression": "zfp", "zfp_mode": "fixed_accuracy", "zfp_tolerance": 0.01}
```

> **Invalid combinations:** Some pipeline combinations are rejected at encode time — e.g. `zfp` + `shuffle` (ZFP operates on typed floats, not byte-shuffled data) or `simple_packing` + `sz3` (both are encoding stages). See [Compression — Invalid Combinations](../encodings/compression.md#invalid-combinations).

### Multiple objects per message

A single message can contain multiple tensors, each with its own descriptor:

```python
spectrum = np.random.randn(256).astype(np.float64)
mask = np.array([1, 0, 1, 1, 0], dtype=np.uint8)

msg = tensogram.encode(
    {"version": 2},
    [
        ({"type": "ntensor", "shape": [256], "dtype": "float64", "compression": "zstd"}, spectrum),
        ({"type": "ntensor", "shape": [5], "dtype": "uint8"}, mask),
    ],
)
```

### Pre-encoded data

If you already have compressed/packed payloads (e.g. from another system), use `tensogram.encode_pre_encoded()` with the same interface. The library skips the encoding pipeline and writes the bytes as-is:

```python
msg = tensogram.encode_pre_encoded(meta, [(desc, pre_compressed_bytes)])
```

See [Pre-Encoded Data API](encode-pre-encoded.md) for details.

## Decoding

### Full decode

```python
meta, objects = tensogram.decode(msg)
```

Returns a `Message` namedtuple with `.metadata` and `.objects`. Tuple unpacking works directly.

By default, decoded arrays are in the caller's **native byte order** — the library handles byte-swapping automatically. Pass `native_byte_order=False` to receive the raw wire byte order instead:

```python
meta, objects = tensogram.decode(msg, native_byte_order=False)
```

### Metadata

`meta` is a `Metadata` object:

```python
meta.version     # int — always 2
meta.base        # list[dict] — per-object metadata (one entry per object)
meta.extra       # dict — message-level annotations (_extra_ in CBOR)
meta.reserved    # dict — library internals (_reserved_ in CBOR, read-only)
meta["key"]      # dict-style access (checks base entries, then extra)
```

To read metadata without decoding any payloads:

```python
meta = tensogram.decode_metadata(msg)
```

To read metadata and descriptors (no payload decode):

```python
meta, descriptors = tensogram.decode_descriptors(msg)
for desc in descriptors:
    print(desc.shape, desc.dtype, desc.compression)
```

### Selective decode

Decode a single object without touching the others — O(1) seek via the binary header's offset table:

```python
meta, desc, array = tensogram.decode_object(msg, index=2)
```

Decode a sub-range of elements from one object (for compressors that support random access):

```python
# Elements 100-149 and 300-324 from object 0
parts = tensogram.decode_range(msg, object_index=0, ranges=[(100, 50), (300, 25)])
# parts is a list of numpy arrays, one per range

# Or join into a single contiguous array
joined = tensogram.decode_range(msg, object_index=0, ranges=[(100, 50), (300, 25)], join=True)
# joined is a single flat numpy array of shape (75,)
```

> `decode_range` works with uncompressed data, `simple_packing`, `szip`, `blosc2`, and `zfp` fixed-rate mode. It returns an error for stream compressors (`zstd`, `lz4`, `sz3`) and for the `shuffle` filter. See [Decoding Data](decoding.md) for details.

### Scanning and iteration

To find message boundaries in a buffer without decoding:

```python
offsets = tensogram.scan(buf)  # list of (offset, length) pairs
```

To iterate messages in a multi-message buffer:

```python
for meta, objects in tensogram.iter_messages(buf):
    print(meta.version, len(objects))
```

### Hash verification

```python
meta, objects = tensogram.decode(msg, verify_hash=True)
```

Raises `RuntimeError` if any object's payload hash doesn't match. If the message was encoded without a hash (`hash=None`), verification is silently skipped.

## File API

### Writing

```python
with tensogram.TensogramFile.create("forecast.tgm") as f:
    for step in range(24):
        data = model.run(step)
        desc = {"type": "ntensor", "shape": list(data.shape), "dtype": "float32",
                "compression": "zstd"}
        f.append({"version": 2, "base": [{"step": step}]}, [(desc, data)])
```

Each `append` encodes one message and writes it to the end of the file. Messages are independent and self-describing.

### Reading

```python
with tensogram.TensogramFile.open("forecast.tgm") as f:
    print(len(f))                    # message count

    meta, objects = f[0]             # index (supports negative indices)
    subset = f[1:10:2]              # slice → list[Message]

    for meta, objects in f:          # iterate all messages
        for desc, array in objects:
            print(desc.shape, array.dtype)

    raw = f.read_message(0)          # raw bytes for forwarding/caching
```

The first access triggers a streaming scan that records message offsets. After that, every read is an O(1) seek.

### Streaming encoder

For building a message one object at a time in memory:

```python
enc = tensogram.StreamingEncoder({"version": 2}, hash="xxh3")
for desc, data in objects:
    enc.write_object(desc, data)
msg = enc.finish()  # returns complete message as bytes
```

For pre-encoded payloads, use `enc.write_object_pre_encoded(desc, raw_bytes)`.

## Validation

Two functions check whether messages and files are well-formed without consuming the data. See also the [CLI reference](../cli/validate.md).

```python
report = tensogram.validate(msg)
file_report = tensogram.validate_file("data.tgm")
```

### Levels

| Level | Checks | `hash_verified` |
|-------|--------|-----------------|
| `"quick"` | Structure only: magic bytes, frame layout, lengths | always `False` |
| `"default"` | + metadata (CBOR) + integrity (hash verification, decompression) | `True` only if hash succeeds and no errors |
| `"checksum"` | Hash verification only, structural warnings suppressed | `True` only if hash succeeds and no errors |
| `"full"` | + fidelity (full decode, decoded-size check, NaN/Inf scan) | `True` only if hash succeeds and no errors |

```python
# Full validation with canonical CBOR key-order checking
report = tensogram.validate(msg, level="full", check_canonical=True)
```

### Return values

`validate()` returns:

```python
{
    "issues": [
        {
            "code": "hash_mismatch",   # stable snake_case string
            "level": "integrity",      # which validation level found it
            "severity": "error",       # "error" or "warning"
            "description": "...",      # human-readable message
            "object_index": 0,         # optional — which object
            "byte_offset": 1234,       # optional — position in buffer
        }
    ],
    "object_count": 1,
    "hash_verified": False,
}
```

`validate_file()` returns file-level issues plus per-message reports:

```python
{
    "file_issues": [
        {"byte_offset": 100, "length": 19, "description": "trailing bytes after last message"}
    ],
    "messages": [
        {"issues": [], "object_count": 1, "hash_verified": True}
    ],
}
```

### Interpreting results

```python
report = tensogram.validate(msg)
if not report["issues"]:
    print(f"OK — {report['object_count']} objects, hash verified")
else:
    for issue in report["issues"]:
        print(f"[{issue['severity']}] {issue['code']}: {issue['description']}")
```

## Error Handling

| Exception | When |
|-----------|------|
| `ValueError` | Invalid parameters, unknown dtype, NaN in simple packing, unknown validation level |
| `OSError` | File I/O errors, including missing paths |
| `RuntimeError` | Hash mismatch during `decode(..., verify_hash=True)` |
| `KeyError` | Missing metadata key via `meta["key"]` |

## Supported dtypes

| Category | Types |
|----------|-------|
| Floating point | `float16`, `bfloat16`, `float32`, `float64` |
| Complex | `complex64`, `complex128` |
| Signed integer | `int8`, `int16`, `int32`, `int64` |
| Unsigned integer | `uint8`, `uint16`, `uint32`, `uint64` |
| Special | `bitmask` |

`bfloat16` is returned as `ml_dtypes.bfloat16` when `ml_dtypes` is installed; otherwise it falls back to `np.uint16`.

See [Data Types](../format/dtypes.md) for byte widths and wire-format details.

## Examples

See `examples/python/` for complete working examples:

| Example | Topic |
|---------|-------|
| `01_encode_decode.py` | Basic round-trip |
| `02_mars_metadata.py` | MARS-style per-object metadata |
| `03_simple_packing.py` | Simple-packing encoding |
| `04_multi_object.py` | Multi-object messages, selective decode |
| `05_file_api.py` | Multi-message `.tgm` files |
| `06_hash_and_errors.py` | Hash verification and error handling |
| `07_iterators.py` | File iteration, indexing, slicing |
| `08_xarray_integration.py` | Opening `.tgm` as xarray Datasets |
| `08_zarr_backend.py` | Reading/writing through Zarr v3 |
| `09_dask_distributed.py` | Dask distributed computing over 4-D tensors |
| `09_streaming_consumer.py` | Streaming consumer pattern |
| `11_encode_pre_encoded.py` | Pre-encoded data API |
| `12_convert_netcdf.py` | NetCDF → Tensogram conversion via CLI |
| `13_validate.py` | Message and file validation |
