# Python Examples

## Installation

```bash
pip install tensogram          # once published to PyPI
# or
maturin develop                # build from source (from crates/tensogram-python/)
```

For the xarray and dask examples, also install:

```bash
pip install tensogram-xarray              # or: pip install -e tensogram-xarray/
pip install "tensogram-xarray[dask]"      # adds dask[array] for example 09
```

## Examples

| File | Topic |
|------|-------|
| `01_encode_decode.py` | Basic encode/decode round-trip |
| `02_mars_metadata.py` | MARS-style metadata in common and payload |
| `03_simple_packing.py` | Simple-packing encoding for integer quantization |
| `04_multi_object.py` | Multi-object messages, `decode_object`, `decode_range` |
| `05_file_api.py` | `TensogramFile` for multi-message `.tgm` files |
| `06_hash_and_errors.py` | Hash verification and error handling |
| `07_iterators.py` | Scanning and iterating over messages |
| `08_xarray_integration.py` | Opening `.tgm` files as xarray Datasets |
| `09_dask_distributed.py` | Dask distributed computing over 4-D tensors |

## Module Structure

```
tensogram
├── encode(metadata, descriptors_and_data, hash="xxh3") -> bytes
├── decode(buf, verify_hash=False) -> (Metadata, list[(DataObjectDescriptor, ndarray)])
├── decode_metadata(buf) -> Metadata
├── decode_object(buf, index, verify_hash=False) -> (Metadata, DataObjectDescriptor, ndarray)
├── decode_range(buf, object_index, ranges, join=False, verify_hash=False) -> list[ndarray] | ndarray
├── scan(buf) -> list[tuple[int, int]]
├── compute_packing_params(values, bits_per_value, decimal_scale_factor) -> dict
└── TensogramFile
    ├── open(path) -> TensogramFile
    ├── create(path) -> TensogramFile
    ├── append(metadata, descriptors_and_data, hash="xxh3")
    ├── message_count() -> int
    ├── read_message(index) -> bytes
    ├── decode_message(index, verify_hash=False) -> (Metadata, list[(DataObjectDescriptor, ndarray)])
    └── context manager (__enter__ / __exit__)

tensogram.Metadata
    .version -> int
    .common  -> dict              # message-level common metadata
    .payload -> list[dict]        # per-object metadata (one dict per object)
    .extra   -> dict              # non-standard top-level keys
    ["key"]  -> value             # dict-style access (checks common, then extra)

tensogram.DataObjectDescriptor
    .obj_type, .ndim, .shape, .strides, .dtype
    .byte_order, .encoding, .filter, .compression
    .params -> dict               # extra descriptor keys (user metadata)
    .hash   -> dict | None
```

## NumPy Integration

The Python API returns `numpy.ndarray` objects directly. The dtype mapping is:

| Tensogram dtype | NumPy dtype |
|---|---|
| `float16` | `np.float16` |
| `bfloat16` | `np.uint16` (reinterpret via `view`) |
| `float32` | `np.float32` |
| `float64` | `np.float64` |
| `int8`/`int16`/`int32`/`int64` | `np.int8` etc. |
| `uint8`/`uint16`/`uint32`/`uint64` | `np.uint8` etc. |
| `complex64` | `np.complex64` |
| `complex128` | `np.complex128` |
