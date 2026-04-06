# COMPREHENSIVE ZARR RESEARCH REPORT
## Storage Format Versions 2 & 3, Store Interfaces, and Implementation Patterns

**Date**: April 2026  
**Source**: Official Zarr specifications, zarr-python repository, technical documentation

---

## TABLE OF CONTENTS

1. [Zarr v2 Specification](#zarr-v2-specification)
2. [Zarr v3 Specification](#zarr-v3-specification)
3. [Key Differences v2 vs v3](#key-differences-v2-vs-v3)
4. [Zarr-Python Store Interface (v3)](#zarr-python-store-interface-v3)
5. [Zarr v2 Store Interface (Compatibility)](#zarr-v2-store-interface)
6. [Lazy Loading Mechanism](#lazy-loading-mechanism)
7. [Codecs System](#codecs-system)
8. [Read-Only Stores](#read-only-stores)
9. [Example Store Implementations](#example-store-implementations)

---

## ZARR V2 SPECIFICATION

### Storage Model

Zarr v2 provides a **key/value storage interface** abstraction layer:

- **Keys**: ASCII strings
- **Values**: Arbitrary byte sequences
- **Operations**: Read, Write, Delete
- **Examples of backends**: Local filesystem, S3 buckets, ZIP files, in-memory stores

### Array Metadata (.zarray)

Every Zarr v2 array requires a JSON metadata file at key `.zarray`:

```json
{
    "zarr_format": 2,
    "shape": [10000, 10000],
    "chunks": [1000, 1000],
    "dtype": "<f8",
    "compressor": {
        "id": "blosc",
        "cname": "lz4",
        "clevel": 5,
        "shuffle": 1
    },
    "fill_value": "NaN",
    "filters": [
        {"id": "delta", "dtype": "<f8", "astype": "<f4"}
    ],
    "order": "C",
    "dimension_separator": "."
}
```

**Required metadata fields:**

| Field | Type | Description |
|-------|------|-------------|
| `zarr_format` | integer | Must be `2` |
| `shape` | array of int | Dimensions of the array |
| `chunks` | array of int | Size of each chunk dimension |
| `dtype` | string or array | Data type (NumPy typestr format) |
| `compressor` | object or null | Primary compression codec config, must contain `"id"` key |
| `fill_value` | scalar or null | Default value for uninitialized chunks |
| `order` | string | `"C"` (row-major) or `"F"` (column-major) |
| `filters` | array or null | List of filter codecs to apply before compression |

**Optional fields:**

- `dimension_separator`: `"."` (default) or `"/"` - separator between chunk indices in keys

### Data Type Encoding

Follows **NumPy array protocol format (typestr)**:

**Format**: `[byteorder][typecode][size][optional-units]`

- **Byteorder**: `<` (little-endian), `>` (big-endian), `|` (not-relevant)
- **Typecode**: `b` (bool), `i` (int), `u` (unsigned), `f` (float), `c` (complex), `m` (timedelta), `M` (datetime), `S` (byte string), `U` (unicode), `V` (void)
- **Size**: Byte count

**Examples:**
- `"<f8"` - little-endian 64-bit float
- `">i4"` - big-endian 32-bit int
- `"<M8[ns]"` - datetime64 with nanosecond precision

**Structured types** (with multiple fields):
```json
[["r", "|u1"], ["g", "|u1"], ["b", "|u1"]]
```

Nested structures supported:
```json
[["x", "<f4"], ["y", "<f4"], ["z", "<f4", [2, 2]]]
```

### Chunk Organization

**Chunk naming convention:**
- Chunk indices converted to strings and concatenated with separator (default: ".")
- Example: Array with chunks (1000, 1000) → chunks stored at keys: `"0.0"`, `"0.1"`, `"1.0"`, `"1.1"`, etc.
- With `"/"` separator: `"0/0"`, `"0/1"`, `"1/0"`, `"1/1"` (hierarchical)

**Chunk storage format:**
```
[compressed bytes from chunk encoder]
```

No header added; compression format depends on codec (e.g., Blosc includes its own 16-byte header).

**Fill value encoding:**
- Floating point special values: `"NaN"`, `"Infinity"`, `"-Infinity"` (JSON strings)
- Fixed-length byte strings: Base64 encoded (if fill_value ≠ null)
- Structured types: Base64 encoded

### Group Metadata (.zgroup)

Groups organize arrays and sub-groups:

```json
{
    "zarr_format": 2
}
```

**Only required field**: `zarr_format: 2`

**Hierarchical paths:**
- Root group: key `.zgroup` at root
- Sub-group: key `foo/bar/.zgroup` for path `foo/bar`
- Parent groups **automatically created** when creating child arrays/groups
- Direct children found by listing keys matching `{prefix}/*` pattern

### Attributes (.zattrs)

Custom metadata stored as JSON at key `.zattrs`:

```json
{
    "foo": 42,
    "bar": "apples",
    "baz": [1, 2, 3, 4]
}
```

- File is **optional** - if absent, attributes are empty
- Can contain any JSON-serializable values
- Applied to both arrays and groups

### Logical Paths & Normalization

Paths normalized as follows:
1. Replace `\\` with `/`
2. Strip leading `/`
3. Strip trailing `/`
4. Collapse `//` to `/`

**Validation**: Path segments cannot be `.` or `..`

**Key construction**: Logical path becomes prefix: `{normalized_path}/{key}`

Example:
```
Path: "foo/bar"
Keys: 
  - "foo/bar/.zarray"
  - "foo/bar/.zattrs"
  - "foo/bar/0.0" (chunk data)
  - "foo/bar/0.1" (chunk data)
```

---

## ZARR V3 SPECIFICATION

### Overview

Zarr v3 **consolidates all metadata into a single `zarr.json` file** at the root of each array/group, moving away from distributed `.zarray`, `.zgroup`, `.zattrs` files.

### Array Metadata (zarr.json)

**Single consolidated metadata file** contains all array information:

```json
{
    "zarr_format": 3,
    "node_type": "array",
    "shape": [10000, 10000],
    "data_type": "float64",
    "chunk_grid": {
        "name": "regular",
        "configuration": {
            "chunk_shape": [1000, 1000]
        }
    },
    "codecs": [
        {
            "name": "bytes",
            "configuration": {
                "endian": "little"
            }
        },
        {
            "name": "blosc",
            "configuration": {
                "cname": "lz4",
                "clevel": 5,
                "shuffle": "bitshuffle"
            }
        }
    ],
    "fill_value": null,
    "attributes": {
        "foo": 42,
        "bar": "apples"
    }
}
```

**Key schema differences from v2:**

| v2 | v3 | Change |
|----|----|----|
| `.zarray` file (separate) | `zarr.json` (consolidated) | Single metadata file |
| `.zattrs` file (separate) | `attributes` (in zarr.json) | Attributes embedded |
| `dtype` (string) | `data_type` (string) | Field renamed |
| `compressor` + `filters` (separate) | `codecs` (unified array) | Unified codec chain |
| `chunks` | `chunk_grid.configuration.chunk_shape` | Nested structure |
| `order` (C/F) | codec order or byte codec config | Endianness in bytes codec |
| `dimension_separator` (v2 feature) | Implicit in chunk naming | Different path structure |

### Group Metadata (zarr.json)

Similar consolidated format for groups:

```json
{
    "zarr_format": 3,
    "node_type": "group",
    "attributes": {
        "group_attr": "value"
    }
}
```

### Chunk Naming (v3)

- **Default**: `"c/{chunk_indices}"` or similar structure
- Path behavior varies; check codec configuration
- Can support different chunk grids (regular, ragged, variable, etc.)

### Data Type Encoding (v3)

Uses **explicit data type descriptors** instead of NumPy typestr:

```json
{
    "name": "float64",
    "endian": "little"
}
```

Or structured:
```json
{
    "name": "structured",
    "fields": [
        {"name": "r", "data_type": "uint8"},
        {"name": "g", "data_type": "uint8"},
        {"name": "b", "data_type": "uint8"}
    ]
}
```

### Consolidated Metadata

**v3 Feature**: One-time consolidation of all metadata into a single file (e.g., `zarr.json` at root), eliminating need to traverse directory structures or make multiple key lookups for group listings.

---

## KEY DIFFERENCES V2 VS V3

### Metadata Storage

| Aspect | v2 | v3 |
|--------|----|----|
| **Metadata files** | Distributed: `.zarray`, `.zgroup`, `.zattrs` | Consolidated: single `zarr.json` |
| **Attributes storage** | Separate `.zattrs` file | Embedded in `zarr.json` |
| **Multiple reads needed** | Yes (3+ keys per array) | No (single key) |
| **Consolidated metadata support** | Optional feature | Native design |

### Codec System

| Aspect | v2 | v3 |
|--------|----|----|
| **Codec chain** | `compressor` + `filters` | `codecs[]` (ordered array) |
| **Codec format** | JSON objects with `"id"` | JSON objects with `"name"` |
| **Byte encoding** | Implicit via dtype | Explicit `bytes` codec in chain |
| **Extensibility** | Limited to registered codecs | Registry-based, more flexible |

### Chunk Grid

| Aspect | v2 | v3 |
|--------|----|----|
| **Configuration** | Flat `chunks` field | Nested `chunk_grid` object |
| **Grid types** | Only regular/uniform | Regular, ragged, variable, custom |
| **Separator config** | `dimension_separator` | Varies by chunk grid type |

### Data Types

| Aspect | v2 | v3 |
|--------|----|----|
| **Format** | NumPy typestr (e.g., `"<f8"`) | Explicit name (e.g., `"float64"`) |
| **Byte order** | In typestr | Separate field or in bytes codec |
| **Structured types** | Nested arrays | Dedicated `fields` structure |

### Storage Efficiency

**v2**: Multiple HTTP/S3 calls for array metadata discovery
```
GET .zarray
GET .zattrs
GET 0.0 (chunk)
→ 3+ calls
```

**v3**: Single consolidated metadata call
```
GET zarr.json
GET 0.0 (chunk)
→ 2 calls (or 1 if metadata pre-loaded)
```

### Backward Compatibility

- **v3 can read v2**: zarr-python includes compatibility layer
- **v2 cannot read v3**: Different format entirely
- Migration: Convert v2→v3 or maintain both

---

## ZARR-PYTHON STORE INTERFACE V3

### Abstract Base Class: `zarr.abc.store.Store`

Located in: `src/zarr/abc/store.py`

All Zarr stores must inherit from `Store` and implement abstract methods.

### Core Initialization

```python
class Store(ABC):
    def __init__(self, *, read_only: bool = False) -> None:
        self._is_open = False
        self._read_only = read_only
```

### Life Cycle Methods

```python
@classmethod
async def open(cls, *args: Any, **kwargs: Any) -> Self:
    """Create and open the store."""
    store = cls(*args, **kwargs)
    await store._open()
    return store

async def _open(self) -> None:
    """Open the store. Raises ValueError if already open."""
    if self._is_open:
        raise ValueError("store is already open")
    self._is_open = True

async def _ensure_open(self) -> None:
    """Open if not already open."""
    if not self._is_open:
        await self._open()

def close(self) -> None:
    """Close the store."""
    self._is_open = False

def __enter__(self) -> Self:
    """Context manager entry."""
    return self

def __exit__(self, exc_type, exc_value, traceback) -> None:
    """Context manager exit; closes store."""
    self.close()
```

### Byte Request Types

```python
@dataclass(frozen=True, slots=True)
class RangeByteRequest:
    start: int  # inclusive
    end: int    # exclusive

@dataclass(frozen=True, slots=True)
class OffsetByteRequest:
    offset: int  # start from this offset to end of object

@dataclass(frozen=True, slots=True)
class SuffixByteRequest:
    suffix: int  # last N bytes of object

# Union type:
ByteRequest = RangeByteRequest | OffsetByteRequest | SuffixByteRequest
```

### Abstract Methods (MUST IMPLEMENT)

#### 1. Data Retrieval

```python
@abstractmethod
async def get(
    self,
    key: str,
    prototype: BufferPrototype,
    byte_range: ByteRequest | None = None,
) -> Buffer | None:
    """
    Retrieve value at key.
    
    Parameters
    ----------
    key : str
        The key to retrieve
    prototype : BufferPrototype
        Buffer prototype for output
    byte_range : ByteRequest, optional
        If provided, only fetch specified byte range
        
    Returns
    -------
    Buffer | None
        The buffer, or None if key doesn't exist
    """
    ...

@abstractmethod
async def get_partial_values(
    self,
    prototype: BufferPrototype,
    key_ranges: Iterable[tuple[str, ByteRequest | None]],
) -> list[Buffer | None]:
    """
    Retrieve multiple (key, byte_range) pairs.
    
    Returns list in order of key_ranges.
    May contain None for missing keys.
    """
    ...
```

#### 2. Existence Check

```python
@abstractmethod
async def exists(self, key: str) -> bool:
    """Check if key exists in store."""
    ...
```

#### 3. Data Writing

```python
@property
@abstractmethod
def supports_writes(self) -> bool:
    """Does store support writes?"""
    ...

@abstractmethod
async def set(self, key: str, value: Buffer) -> None:
    """Store (key, value) pair."""
    ...

async def set_if_not_exists(self, key: str, value: Buffer) -> None:
    """Store key only if not already present."""
    # Default implementation; can override for atomicity
    if not await self.exists(key):
        await self.set(key, value)

async def _set_many(self, values: Iterable[tuple[str, Buffer]]) -> None:
    """Insert multiple (key, value) pairs."""
    # Default: gather concurrent sets
    ...
```

#### 4. Data Deletion

```python
@property
@abstractmethod
def supports_deletes(self) -> bool:
    """Does store support deletes?"""
    ...

@abstractmethod
async def delete(self, key: str) -> None:
    """Remove key from store."""
    ...

async def delete_dir(self, prefix: str) -> None:
    """Remove all keys with given prefix."""
    # Requires both deletes and listing support
    ...
```

#### 5. Listing

```python
@property
@abstractmethod
def supports_listing(self) -> bool:
    """Does store support listing?"""
    ...

@abstractmethod
def list(self) -> AsyncIterator[str]:
    """
    Retrieve all keys in store.
    Returns async generator of key strings.
    """
    ...

@abstractmethod
def list_prefix(self, prefix: str) -> AsyncIterator[str]:
    """
    Retrieve all keys beginning with prefix.
    Keys returned relative to store root.
    """
    ...

@abstractmethod
def list_dir(self, prefix: str) -> AsyncIterator[str]:
    """
    Retrieve keys and prefixes with given prefix
    that don't contain "/" after the prefix.
    Similar to directory listing.
    """
    ...
```

### Property Methods (MUST IMPLEMENT)

```python
@property
def read_only(self) -> bool:
    """Is store read-only?"""
    return self._read_only

@property
@abstractmethod
def supports_writes(self) -> bool: ...

@property
@abstractmethod
def supports_deletes(self) -> bool: ...

@property
@abstractmethod
def supports_listing(self) -> bool: ...

@property
def supports_partial_writes(self) -> Literal[False]:
    """Always False (deprecated feature)."""
    return False

@property
def supports_consolidated_metadata(self) -> bool:
    """Can store support consolidated metadata? Default True."""
    return True
```

### Equality & Utility

```python
@abstractmethod
def __eq__(self, value: object) -> bool:
    """Equality comparison between stores."""
    ...

def with_read_only(self, read_only: bool = False) -> Store:
    """Return new store instance with new read_only setting."""
    raise NotImplementedError(...)

def _check_writable(self) -> None:
    """Raise if store is read-only."""
    if self.read_only:
        raise ValueError("store was opened in read-only mode...")
```

### Convenience Methods (Provided by Base Class)

```python
async def _get_bytes(
    self, 
    key: str, 
    *, 
    prototype: BufferPrototype,
    byte_range: ByteRequest | None = None
) -> bytes:
    """Get and convert to bytes."""
    buffer = await self.get(key, prototype, byte_range)
    if buffer is None:
        raise FileNotFoundError(key)
    return buffer.to_bytes()

async def _get_json(
    self,
    key: str,
    *,
    prototype: BufferPrototype,
    byte_range: ByteRequest | None = None
) -> Any:
    """Get, parse as JSON."""
    return json.loads(await self._get_bytes(key, prototype=prototype, byte_range=byte_range))

async def is_empty(self, prefix: str) -> bool:
    """Check if prefix has any keys."""
    async for _ in self.list_prefix(prefix):
        return False
    return True

async def clear(self) -> None:
    """Remove all keys from store."""
    await self.delete_dir("")

async def getsize(self, key: str) -> int:
    """Get size of value in bytes."""
    value = await self.get(key, prototype=default_buffer_prototype())
    if value is None:
        raise FileNotFoundError(key)
    return len(value)

async def getsize_prefix(self, prefix: str) -> int:
    """Get total size of all values under prefix."""
    ...
```

### Synchronous Compatibility Protocols

For stores with synchronous implementations, optional protocols:

```python
@runtime_checkable
class SupportsGetSync(Protocol):
    def get_sync(
        self,
        key: str,
        *,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None: ...

@runtime_checkable
class SupportsSetSync(Protocol):
    def set_sync(self, key: str, value: Buffer) -> None: ...

@runtime_checkable
class SupportsDeleteSync(Protocol):
    def delete_sync(self, key: str) -> None: ...
```

---

## ZARR V2 STORE INTERFACE

Zarr v2 uses a simpler, dict-like store interface (deprecated in v3):

```python
class Store:
    # Must implement dict-like access
    def __getitem__(self, key: str) -> bytes:
        """Retrieve bytes at key."""
        
    def __setitem__(self, key: str, value: bytes) -> None:
        """Store bytes at key."""
        
    def __delitem__(self, key: str) -> None:
        """Delete key."""
        
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        
    def __iter__(self) -> Iterator[str]:
        """Iterate over all keys."""
        
    def keys(self) -> Iterator[str]:
        """List all keys."""
        
    def listdir(self, path: str) -> List[str]:
        """List immediate children of path."""
```

**Example**: DirectoryStore, MemoryStore, ZipStore (v2 versions)

---

## LAZY LOADING MECHANISM

### How Zarr Defers Reading Chunk Data

1. **Array Creation (metadata-only)**
   ```python
   arr = zarr.open_array("data.zarr", mode="r")
   # At this point:
   # - zarr.json (or .zarray in v2) is loaded
   # - Chunk shape, dtype, codecs known
   # - NO chunks loaded into memory
   # - Shape: (10000, 10000) registered but chunks remain on disk
   ```

2. **Chunk Indexing Triggers Load**
   ```python
   # Accessing a slice triggers lazy loading
   data = arr[0:100, 0:100]  # Only this chunk fetched
   
   # Internally:
   # - Identify which chunk indices needed (0:100, 0:100) maps to chunk(0, 0)
   # - Call store.get("c/0/0")  # Fetch chunk bytes
   # - Decompress & decode chunk (applies codecs in reverse)
   # - Extract requested slice from decompressed chunk
   # - Return as ndarray
   ```

3. **Codec Chain Applied (Deferred)**
   ```
   Stored bytes → codec[n].decode() → ... → codec[1].decode() → codec[0].decode() → raw data
   ```
   
   Example for v3:
   ```python
   codecs = [
       {"name": "bytes", "config": {"endian": "little"}},
       {"name": "blosc", "config": {"cname": "lz4"}}
   ]
   # Reading: stored_bytes → blosc_decompress → bytes_endian_decode → ndarray
   ```

4. **Fill Values Applied**
   - Uninitialized chunks (not in store): treated as filled with `fill_value`
   - Check `exists(chunk_key)` before fetch
   - If not exists: return array filled with fill_value

5. **Partial Reads (Byte Range)**
   ```python
   # Some stores support partial chunk reads
   store.get(key, prototype, byte_range=RangeByteRequest(0, 1000))
   # Fetch only first 1000 bytes, avoiding full chunk load
   # Requires codec support (e.g., HTTP range requests)
   ```

### Concrete Implementation Flow

**zarr-python Array Access:**

```python
# File: src/zarr/core/array.py (conceptual)
class Array:
    async def __getitem__(self, selection):
        # 1. Parse selection into chunk indices
        chunk_indices = self._get_chunk_indices(selection)
        
        # 2. For each chunk, request from store
        chunk_keys = [self._get_chunk_key(idx) for idx in chunk_indices]
        
        # 3. Batch fetch from store
        buffers = await self.store.get_partial_values(
            [(key, byte_range) for key in chunk_keys]
        )
        
        # 4. Decode each chunk
        decoded_chunks = []
        for buf in buffers:
            if buf is None:
                # Uninitialized chunk
                decoded_chunks.append(self._filled_chunk())
            else:
                # Apply codec chain in reverse
                decoded = self._decode_chunk(buf)
                decoded_chunks.append(decoded)
        
        # 5. Assemble and slice result
        return self._assemble_chunks(decoded_chunks, selection)
```

---

## CODECS SYSTEM

### Codec Structure (v3)

Codecs represent transformations applied in order:

```python
codecs: list[CodecConfig] = [
    {
        "name": "bytes",
        "configuration": {"endian": "little"}
    },
    {
        "name": "blosc",
        "configuration": {
            "cname": "lz4",
            "clevel": 5,
            "shuffle": "bitshuffle",
            "typesize": 8
        }
    }
]
```

### Encoding vs Decoding

**Encoding (Write):**
```
raw_data → bytes_codec → blosc_compress → stored_bytes
```

**Decoding (Read):**
```
stored_bytes → blosc_decompress → bytes_codec → raw_data
```

**Codecs applied in reverse order** during decode!

### Standard Codecs (zarr-python)

```python
# src/zarr/codecs/__init__.py

# Compression codecs
- blosc (Blosc algorithm with multiple sub-codecs)
- gzip (GZip compression)
- zstd (Zstandard compression)
- numcodecs.bz2 (BZ2)
- numcodecs.lz4 (LZ4)
- numcodecs.lzma (LZMA)

# Filters/Transformations
- bytes (Byte order, endianness)
- delta (Delta encoding)
- shuffle (Shuffle bytes)
- transpose (Transpose data)
- vlen-utf8 (Variable-length UTF-8 strings)
- vlen-bytes (Variable-length bytes)

# Checksums
- crc32c (CRC32C checksum)
- numcodecs.crc32 (CRC32)
- numcodecs.fletcher32 (Fletcher32)

# Specialized
- sharding_indexed (Sharding/super-chunks)
- numcodecs.quantize (Quantization)
- numcodecs.fixedscaleoffset (Scale/offset)
```

### Codec Configuration Examples

**Blosc with LZ4:**
```json
{
    "name": "blosc",
    "configuration": {
        "cname": "lz4",
        "clevel": 5,
        "shuffle": "bitshuffle",
        "typesize": 8
    }
}
```

**Bytes codec (endianness):**
```json
{
    "name": "bytes",
    "configuration": {
        "endian": "little"
    }
}
```

**Sharding (super-chunks):**
```json
{
    "name": "sharding_indexed",
    "configuration": {
        "chunk_shape": [100, 100],
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "blosc", "configuration": {"cname": "lz4"}}
        ],
        "index_location": "start"
    }
}
```

### Codec Registry

Codecs registered dynamically:

```python
# src/zarr/registry.py (conceptual)
def register_codec(name: str, codec_class: Type[Codec], qualname: str = None) -> None:
    """Register codec for use in zarr.json"""
    CODEC_REGISTRY[name] = codec_class
```

**Finding codec at runtime:**
```python
config = {"name": "blosc", "configuration": {...}}
codec_class = CODEC_REGISTRY[config["name"]]
codec_instance = codec_class.from_dict(config)
```

---

## READ-ONLY STORES

### Minimal Interface for Read-Only Store

```python
from zarr.abc.store import Store
from typing import AsyncIterator

class ReadOnlyStore(Store):
    """Minimal read-only store implementation."""
    
    def __init__(self, *, read_only: bool = True) -> None:
        if not read_only:
            raise ValueError("This store is read-only only")
        super().__init__(read_only=True)
    
    # ===== REQUIRED (for read operations) =====
    
    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """Implement actual data retrieval."""
        raise NotImplementedError
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        raise NotImplementedError
    
    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        """Get multiple values."""
        # Default implementation
        result = []
        for key, byte_range in key_ranges:
            result.append(await self.get(key, prototype, byte_range))
        return result
    
    @property
    def supports_writes(self) -> bool:
        """Read-only stores don't support writes."""
        return False
    
    async def set(self, key: str, value: Buffer) -> None:
        """Not supported."""
        self._check_writable()
    
    @property
    def supports_deletes(self) -> bool:
        """Read-only stores don't support deletes."""
        return False
    
    async def delete(self, key: str) -> None:
        """Not supported."""
        self._check_writable()
    
    @property
    def supports_listing(self) -> bool:
        """Override based on whether store supports listing."""
        return False  # Can be True if listing implemented
    
    def list(self) -> AsyncIterator[str]:
        if not self.supports_listing:
            raise NotImplementedError
        return self._list_impl()
    
    async def _list_impl(self) -> AsyncIterator[str]:
        raise NotImplementedError
    
    def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        if not self.supports_listing:
            raise NotImplementedError
        return self._list_prefix_impl(prefix)
    
    async def _list_prefix_impl(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError
    
    def list_dir(self, prefix: str) -> AsyncIterator[str]:
        if not self.supports_listing:
            raise NotImplementedError
        return self._list_dir_impl(prefix)
    
    async def _list_dir_impl(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError
    
    @abstractmethod
    def __eq__(self, value: object) -> bool:
        """Equality check."""
        ...
```

### Read-Only URL Store Example

```python
class ReadOnlyHTTPStore(Store):
    """Read-only store backed by HTTP URLs."""
    
    def __init__(self, base_url: str, *, read_only: bool = True) -> None:
        super().__init__(read_only=True)
        self.base_url = base_url.rstrip("/")
        self.session = None  # HTTP client
    
    async def _open(self) -> None:
        await super()._open()
        self.session = aiohttp.ClientSession()
    
    def close(self) -> None:
        if self.session:
            # asyncio.run(self.session.close())
            pass
        super().close()
    
    async def get(self, key: str, prototype, byte_range=None):
        """Fetch from HTTP."""
        url = f"{self.base_url}/{key}"
        headers = {}
        if byte_range:
            if isinstance(byte_range, RangeByteRequest):
                headers["Range"] = f"bytes={byte_range.start}-{byte_range.end-1}"
            # ... handle other types
        
        async with self.session.get(url, headers=headers) as resp:
            if resp.status == 404:
                return None
            if resp.status != 200 and resp.status != 206:
                raise IOError(f"HTTP {resp.status}")
            return prototype.buffer.from_bytes(await resp.read())
    
    async def exists(self, key: str) -> bool:
        """Check existence with HEAD."""
        url = f"{self.base_url}/{key}"
        async with self.session.head(url) as resp:
            return resp.status != 404
    
    @property
    def supports_writes(self) -> bool:
        return False
    
    @property
    def supports_deletes(self) -> bool:
        return False
    
    @property
    def supports_listing(self) -> bool:
        return False  # Unless backend provides listing
    
    # ... (other required methods stubbed)
```

---

## EXAMPLE STORE IMPLEMENTATIONS

### 1. MemoryStore (In-Memory)

```python
# src/zarr/storage/_memory.py

class MemoryStore(Store):
    """In-memory key/value store."""
    
    supports_writes = True
    supports_deletes = True
    supports_listing = True
    
    def __init__(self, *, read_only: bool = False) -> None:
        super().__init__(read_only=read_only)
        self._data: dict[str, bytes] = {}
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, MemoryStore)
    
    async def get(self, key: str, prototype, byte_range=None):
        if key not in self._data:
            return None
        data = self._data[key]
        if byte_range is None:
            return prototype.buffer.from_bytes(data)
        # Handle byte range
        if isinstance(byte_range, RangeByteRequest):
            data = data[byte_range.start:byte_range.end]
        elif isinstance(byte_range, OffsetByteRequest):
            data = data[byte_range.offset:]
        elif isinstance(byte_range, SuffixByteRequest):
            data = data[-byte_range.suffix:]
        return prototype.buffer.from_bytes(data)
    
    async def exists(self, key: str) -> bool:
        return key in self._data
    
    async def get_partial_values(self, prototype, key_ranges):
        result = []
        for key, byte_range in key_ranges:
            result.append(await self.get(key, prototype, byte_range))
        return result
    
    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable()
        self._data[key] = value.to_bytes()
    
    async def delete(self, key: str) -> None:
        self._check_writable()
        self._data.pop(key, None)
    
    async def list(self) -> AsyncIterator[str]:
        for key in self._data:
            yield key
    
    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        for key in self._data:
            if key.startswith(prefix):
                yield key
    
    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        for key in self._data:
            if key.startswith(prefix):
                suffix = key[len(prefix):]
                if "/" not in suffix:
                    yield suffix
```

### 2. LocalStore (Filesystem)

```python
# src/zarr/storage/_local.py (simplified)

class LocalStore(Store):
    """File system store."""
    
    supports_writes = True
    supports_deletes = True
    supports_listing = True
    
    def __init__(self, root: Path | str, *, read_only: bool = False) -> None:
        super().__init__(read_only=read_only)
        self.root = Path(root)
    
    async def get(self, key: str, prototype, byte_range=None):
        path = self.root / key
        if not path.exists():
            return None
        if byte_range is None:
            return prototype.buffer.from_bytes(path.read_bytes())
        # Handle partial reads
        with path.open("rb") as f:
            if isinstance(byte_range, RangeByteRequest):
                f.seek(byte_range.start)
                data = f.read(byte_range.end - byte_range.start)
            elif isinstance(byte_range, OffsetByteRequest):
                f.seek(byte_range.offset)
                data = f.read()
            elif isinstance(byte_range, SuffixByteRequest):
                f.seek(0, 2)  # end of file
                size = f.tell()
                f.seek(max(0, size - byte_range.suffix))
                data = f.read()
        return prototype.buffer.from_bytes(data)
    
    async def exists(self, key: str) -> bool:
        return (self.root / key).exists()
    
    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable()
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write with temp file
        path.write_bytes(value.to_bytes())
    
    async def delete(self, key: str) -> None:
        self._check_writable()
        path = self.root / key
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    
    async def list(self) -> AsyncIterator[str]:
        for path in self.root.rglob("*"):
            if path.is_file():
                yield str(path.relative_to(self.root))
    
    # ... (similar for list_prefix, list_dir)
```

### 3. ZipStore

```python
# src/zarr/storage/_zip.py (conceptual)

class ZipStore(Store):
    """Store backed by ZIP file."""
    
    supports_writes: bool
    supports_deletes: bool
    supports_listing = True
    
    def __init__(self, path: str, mode: str = "r", *, read_only: bool = False) -> None:
        super().__init__(read_only=read_only)
        self.path = path
        self.mode = mode
        self.zf = None
    
    async def _open(self) -> None:
        self.zf = zipfile.ZipFile(self.path, mode=self.mode)
        self.supports_writes = self.mode in ("a", "w")
        self.supports_deletes = self.mode in ("a", "w")
        await super()._open()
    
    async def get(self, key: str, prototype, byte_range=None):
        try:
            data = self.zf.read(key)
            if byte_range is None:
                return prototype.buffer.from_bytes(data)
            # Handle byte range on data
            if isinstance(byte_range, RangeByteRequest):
                data = data[byte_range.start:byte_range.end]
            elif isinstance(byte_range, OffsetByteRequest):
                data = data[byte_range.offset:]
            elif isinstance(byte_range, SuffixByteRequest):
                data = data[-byte_range.suffix:]
            return prototype.buffer.from_bytes(data)
        except KeyError:
            return None
    
    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable()
        self.zf.writestr(key, value.to_bytes())
    
    async def list(self) -> AsyncIterator[str]:
        for name in self.zf.namelist():
            yield name
    
    # ... (other methods)
    
    def close(self) -> None:
        if self.zf:
            self.zf.close()
        super().close()
```

### 4. FsspecStore (S3, GCS, etc.)

```python
# src/zarr/storage/_fsspec.py (simplified)

class FsspecStore(Store):
    """Store backed by fsspec (S3, GCS, SFTP, etc.)."""
    
    supports_writes = True
    supports_deletes = True
    supports_listing = True
    
    def __init__(self, url: str, *, read_only: bool = False) -> None:
        super().__init__(read_only=read_only)
        self.url = url
        self.fs = None
    
    async def _open(self) -> None:
        import fsspec
        self.fs = fsspec.filesystem(self.url.split(":")[0])
        await super()._open()
    
    async def get(self, key: str, prototype, byte_range=None):
        path = f"{self.url}/{key}"
        try:
            if byte_range is None:
                data = self.fs.cat_file(path)
            else:
                data = self.fs.cat_file(path, start=..., end=...)
            return prototype.buffer.from_bytes(data)
        except FileNotFoundError:
            return None
    
    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable()
        path = f"{self.url}/{key}"
        self.fs.pipe_file(path, value.to_bytes())
    
    # ... (other methods)
```

---

## SUMMARY TABLE: Store Implementation Requirements

| Feature | Required | Async | Notes |
|---------|----------|-------|-------|
| `__init__` | ✓ | No | Accept `read_only=bool` kwarg |
| `_open()` | ✓ | Yes | Called by `.open()` classmethod |
| `close()` | ✓ | No | Cleanup resources |
| `get()` | ✓ | Yes | Fetch value, support byte_range |
| `exists()` | ✓ | Yes | Check key existence |
| `get_partial_values()` | ✓ | Yes | Batch fetch with ranges |
| `set()` | ✓ | Yes | If `supports_writes=True` |
| `delete()` | ✓ | Yes | If `supports_deletes=True` |
| `list()` | ✓ | Yes (async generator) | If `supports_listing=True` |
| `list_prefix()` | ✓ | Yes (async generator) | If `supports_listing=True` |
| `list_dir()` | ✓ | Yes (async generator) | If `supports_listing=True` |
| `__eq__()` | ✓ | No | Compare store instances |
| `with_read_only()` | Optional | No | Return new store with read_only flag |
| `supports_writes` | ✓ | Property | Return bool |
| `supports_deletes` | ✓ | Property | Return bool |
| `supports_listing` | ✓ | Property | Return bool |
| `read_only` | ✓ | Property | Return `self._read_only` |

---

## CONCLUSION

**Zarr Storage Format** provides:
- **Flexible key/value abstraction** for any backend
- **Chunked, compressed N-dimensional arrays** with metadata
- **Hierarchical organization** via groups
- **Extensible codec system** for compression & transformation
- **Lazy loading** of chunks on-demand
- **Read-only access** patterns for remote/immutable data

**zarr-python v3** simplifies:
- Consolidated metadata → single `zarr.json` file
- Unified codec chain → cleaner pipeline
- Improved async/await API
- Better remote store support (S3, HTTP, etc.)

For **custom backend implementation**, implement the `Store` ABC with the ~11 core async methods and 4-5 properties to unlock Zarr functionality on any storage medium.
