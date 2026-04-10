# Python Examples

## Installation

```bash
uv venv .venv && source .venv/bin/activate   # if not already in a virtualenv
uv pip install tensogram          # once published to PyPI
# or, build from source:
uv pip install maturin numpy      # install build dependencies
(cd crates/tensogram-python && maturin develop)
```

For the xarray and dask examples (08, 09), also install:

```bash
uv pip install "tensogram-xarray[dask]"   # includes xarray + dask[array]
# or from source:
uv pip install -e "tensogram-xarray/[dask]"
```

For the Zarr example, install:

```bash
uv pip install tensogram-zarr
# or from source:
uv pip install -e tensogram-zarr/
```

## Examples

| File | Topic |
|------|-------|
| `01_encode_decode.py` | Basic encode/decode round-trip |
| `02_mars_metadata.py` | MARS-style metadata in per-object `base` entries |
| `03_simple_packing.py` | Simple-packing encoding for integer quantization |
| `04_multi_object.py` | Multi-object messages, `decode_object`, `decode_range` |
| `05_file_api.py` | `TensogramFile` for multi-message `.tgm` files |
| `06_hash_and_errors.py` | Hash verification and error handling |
| `07_iterators.py` | File iteration, indexing, slicing, `iter_messages` |
| `08_xarray_integration.py` | Opening `.tgm` files as xarray Datasets |
| `08_zarr_backend.py` | Reading and writing `.tgm` files through Zarr v3 |
| `09_dask_distributed.py` | Dask distributed computing over 4-D tensors |
| `11_encode_pre_encoded.py` | Pre-encoded data API for already-framed payloads |
| `12_convert_netcdf.py` | Convert NetCDF → Tensogram via the CLI (uses `netCDF4` + `subprocess`) |
| `13_validate.py` | Message and file validation at different levels |

## Module Structure

```
tensogram
├── encode(metadata, descriptors_and_data, hash="xxh3") -> bytes
├── decode(buf, verify_hash=False) -> Message
├── decode_metadata(buf) -> Metadata
├── decode_object(buf, index, verify_hash=False) -> (Metadata, DataObjectDescriptor, ndarray)
├── decode_range(buf, object_index, ranges, join=False, verify_hash=False) -> list[ndarray] | ndarray
├── scan(buf) -> list[tuple[int, int]]
├── iter_messages(buf, verify_hash=False) -> MessageIter
├── compute_packing_params(values, bits_per_value, decimal_scale_factor) -> dict
├── validate(buf, level="default", check_canonical=False) -> dict
├── validate_file(path, level="default", check_canonical=False) -> dict
└── TensogramFile
    ├── open(path) -> TensogramFile
    ├── create(path) -> TensogramFile
    ├── append(metadata, descriptors_and_data, hash="xxh3")
    ├── message_count() -> int
    ├── read_message(index) -> bytes
    ├── decode_message(index, verify_hash=False) -> Message
    ├── for msg in file: ...         # iteration
    ├── file[i], file[-1]            # indexing
    ├── file[1:10:2]                 # slicing -> list[Message]
    ├── len(file)                    # message count
    └── context manager (__enter__ / __exit__)

tensogram.Message (namedtuple)
    .metadata -> Metadata
    .objects  -> list[(DataObjectDescriptor, ndarray)]
    # tuple unpacking: meta, objects = msg

tensogram.Metadata
    .version  -> int
    .base     -> list[dict]       # per-object metadata (one dict per object, independent entries)
    .reserved -> dict             # library internals (_reserved_ in CBOR, read-only)
    .extra    -> dict             # client-writable annotations (_extra_ in CBOR)
    ["key"]   -> value            # dict-style access (checks base entries, then extra)

tensogram.DataObjectDescriptor
    .obj_type, .ndim, .shape, .strides, .dtype
    .byte_order, .encoding, .filter, .compression
    .params -> dict               # encoding parameters (e.g. reference_value, bits_per_value)
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
