# Tensogram Python API Quick Reference

## Module: `tensogram`

### Classes

| Class | Purpose | Key Attributes |
|-------|---------|-----------------|
| **`Message`** | Decoded message (namedtuple) | `.metadata`, `.objects`, tuple unpacking |
| **`Metadata`** | Global message metadata | `.version`, `.base`, `.reserved`, `.extra`, `[key]` |
| **`DataObjectDescriptor`** | Tensor descriptor (shape, dtype, encoding) | `.shape`, `.dtype`, `.encoding`, `.params`, `.compression` |
| **`TensogramFile`** | File I/O API | `.open()`, `.create()`, `.append()`, `.decode_message()` |
| **`StreamingEncoder`** | Progressive encode to file | `.write_object()`, `.write_object_pre_encoded()`, `.finish()` |

### Module Functions

| Function | Signature | Returns |
|----------|-----------|---------|
| **`encode`** | `encode(meta, descriptors_and_data, hash="xxh3")` | `bytes` |
| **`encode_pre_encoded`** | `encode_pre_encoded(meta, descriptors_and_data, hash="xxh3")` | `bytes` |
| **`decode`** | `decode(buf, verify_hash=False)` | `Message` |
| **`decode_metadata`** | `decode_metadata(buf)` | `Metadata` |
| **`decode_object`** | `decode_object(buf, index, verify_hash=False)` | `(Metadata, DataObjectDescriptor, array)` |
| **`decode_range`** | `decode_range(buf, object_index, ranges, join=False, verify_hash=False)` | `list[ndarray] \| ndarray` |
| **`scan`** | `scan(buf)` | `list[(offset, length)]` |
| **`iter_messages`** | `iter_messages(buf, verify_hash=False)` | `MessageIter` |
| **`compute_packing_params`** | `compute_packing_params(values, bits_per_value, decimal_scale_factor)` | `dict` |

### TensogramFile Methods

```python
# Static
TensogramFile.open(path: str) → TensogramFile
TensogramFile.create(path: str) → TensogramFile

# Instance
file.message_count() → int
file.append(global_meta_dict, descriptors_and_data, hash="xxh3")
file.decode_message(index, verify_hash=False) → Message
file.read_message(index) → bytes
file.messages() → list[bytes]

# Iteration, indexing, slicing
for meta, objects in file: ...        # iterate all messages
meta, objects = file[i]               # index (supports negative)
subset = file[1:10:2]                # slice → list[Message]
len(file)                             # message count
```

### Supported dtypes

```
float16, bfloat16, float32, float64, complex64, complex128,
int8, int16, int32, int64, uint8, uint16, uint32, uint64, bitmask
```

### Descriptor Fields

**Required:**
- `"type"` (str): Object type, e.g. "ntensor"
- `"shape"` (list[int]): Tensor dimensions
- `"dtype"` (str): Data type name

**Optional:**
- `"strides"` (list[int]): Computed if omitted
- `"byte_order"` (str): "little" (default) or "big"
- `"encoding"` (str): "none" (default), "simple_packing", etc.
- `"filter"` (str): "none" (default), "shuffle", etc.
- `"compression"` (str): "none" (default), "zstd", etc.
- Any other keys → stored in `.params` dict

### Common Patterns

**Basic encode/decode:**
```python
import numpy as np
import tensogram

data = np.array([1, 2, 3], dtype=np.float32)
meta = {"version": 2}
desc = {"type": "ntensor", "shape": [3], "dtype": "float32"}

msg = tensogram.encode(meta, [(desc, data)])
meta_out, objs_out = tensogram.decode(msg)
```

**File API:**
```python
# Write
with tensogram.TensogramFile.create("data.tgm") as f:
    f.append({"version": 2}, [(descriptor, data)])

# Read
with tensogram.TensogramFile.open("data.tgm") as f:
    for meta, objects in f:
        for desc, array in objects:
            print(array.shape)
```

**Metadata access:**
```python
meta, _ = tensogram.decode(message)
print(meta.version)
print(meta['custom_key'])  # KeyError if missing
print(meta.extra)          # Message-level annotations (_extra_ in CBOR)
print(meta.base)           # Per-object metadata list (independent entries)
print(meta.reserved)       # Library internals (_reserved_ in CBOR)
```

**Hash verification:**
```python
# Encode with hash
msg = tensogram.encode(meta, objs, hash="xxh3")

# Verify on decode
meta, objs = tensogram.decode(msg, verify_hash=True)
```

### Error Handling

- `ValueError`: Invalid parameters, dtype errors, NaN in packing
- `IOError`: File I/O errors
- `RuntimeError`: Hash mismatch
- `KeyError`: Missing metadata key

---

## Source Code Location

- **Implementation:** `crates/tensogram-python/src/lib.rs`
- **Build config:** `crates/tensogram-python/Cargo.toml`, `pyproject.toml`
- **Examples:** `examples/python/`
