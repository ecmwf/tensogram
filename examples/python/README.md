# Python Examples

> **Status:** The Python bindings (`tensogram-python` / PyO3 + maturin) are not
> yet implemented. These examples show the **intended API** once the bindings are
> complete. They document the design contract for implementors.

## Planned Installation

```bash
pip install tensogram          # once published to PyPI
# or
maturin develop                # build from source (from crates/tensogram-python/)
```

## Intended Module Structure

```
tensogram
├── encode(metadata, *arrays, hash="xxh3") -> bytes
├── decode(buf) -> (Metadata, list[numpy.ndarray])
├── decode_metadata(buf) -> Metadata
├── decode_object(buf, index) -> (ObjectDescriptor, numpy.ndarray)
├── decode_range(buf, object_index, ranges) -> numpy.ndarray
├── scan(buf) -> list[tuple[int, int]]
└── TensogramFile
    ├── open(path) -> TensogramFile
    ├── create(path) -> TensogramFile
    ├── append(metadata, *arrays, hash="xxh3")
    ├── message_count() -> int
    ├── read_message(index) -> bytes
    ├── decode_message(index) -> (Metadata, list[numpy.ndarray])
    └── messages() -> list[bytes]

tensogram.Metadata         — top-level metadata dict-like object
tensogram.ObjectDescriptor — per-object shape/dtype/extra
tensogram.simple_packing
    ├── compute_params(values, bits_per_value, decimal_scale_factor) -> PackingParams
    ├── encode(values, params) -> bytes
    └── decode(packed_bytes, num_values, params) -> numpy.ndarray
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
