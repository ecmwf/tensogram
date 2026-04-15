# Free-Threaded Python

Tensogram supports [free-threaded Python](https://docs.python.org/3.14/whatsnew/3.13.html#free-threaded-cpython) (CPython 3.13t / 3.14t), which removes the Global Interpreter Lock (GIL) and allows true multi-threaded parallelism from Python.

## What This Means

On standard CPython, the GIL serializes access to the interpreter — only one thread runs Python code at a time. Tensogram already releases the GIL during Rust computation (`py.detach()`), which helps, but the GIL is still re-acquired for numpy array construction and Python object creation.

On free-threaded CPython (3.13t / 3.14t), there is no GIL at all. Multiple threads can call `tensogram.encode()` and `tensogram.decode()` in true parallel. Use the included benchmark (`rust/benchmarks/python/bench_threading.py`) to measure scaling on your hardware.

## Building for Free-Threaded Python

Install a free-threaded Python build:

```bash
# uv (recommended)
uv python install cpython-3.14+freethreaded

# Or via pyenv
pyenv install 3.14t
```

Build tensogram:

```bash
uv venv .venv --python python3.14t
source .venv/bin/activate
uv pip install maturin "numpy>=2.1"
cd python/bindings && maturin develop --release
```

Verify the GIL is disabled:

```python
import sys
print(sys._is_gil_enabled())  # False
```

## Thread-Safe API

All tensogram read operations are safe to call from multiple threads simultaneously:

```python
import threading
import numpy as np
import tensogram

data = np.random.randn(1_000_000).astype(np.float32)
meta = {"version": 2, "base": [{}]}
desc = {"type": "ntensor", "shape": [1_000_000], "dtype": "float32"}
msg = tensogram.encode(meta, [(desc, data)])

def decode_worker():
    for _ in range(100):
        result = tensogram.decode(msg)

threads = [threading.Thread(target=decode_worker) for _ in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

Each thread can independently:
- Encode and decode messages
- Scan buffers
- Validate messages and files
- Read from `TensogramFile` instances (same handle or separate handles)
- Use `StreamingEncoder` (separate instances per thread)

## `TensogramFile` Thread Safety

All read methods on `TensogramFile` (`decode_message`, `read_message`, `decode_metadata`, `decode_descriptors`, `decode_object`, `decode_range`, `__getitem__`, `__len__`, `__iter__`) use `&self` and support concurrent access from multiple threads on the **same** handle:

```python
f = tensogram.TensogramFile.open("data.tgm")

def worker(thread_id):
    # Multiple threads can read from the same handle concurrently
    msg = f.decode_message(thread_id % len(f))

threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

Only `append()` requires exclusive access — calling it while other threads are reading will raise `RuntimeError` (PyO3 runtime borrow check).

## Benchmark Results

Measured on Linux x86_64 (20 cores), NumPy 2.4.4, release build.
Same-version paired comparisons to isolate the GIL effect.

All scaling below comes from Python-level threading (`threading.Thread`). Each call into Rust is single-threaded — there is no rayon or internal parallelism within a single encode/decode. The speedups reflect multiple Python threads entering Rust concurrently via `py.detach()`. A future Rust-level parallel pipeline would multiply on top of these numbers.

### Headline: Decode Throughput (1M float32, no codec)

| Threads | 3.13 (GIL) | 3.13t (free) | 3.14 (GIL) | 3.14t (free) |
|--------:|-----------:|-------------:|-----------:|-------------:|
| 1       |  932 op/s  |  972 op/s    |  971 op/s  |  954 op/s    |
| 2       |  985 (1.06x) | 1,819 (1.87x) | 1,001 (1.03x) | 1,754 (1.84x) |
| 4       |  905 (0.97x) | 2,764 (2.84x) |  899 (0.93x) | 2,745 (2.88x) |
| 8       |  902 (0.97x) | 2,136 (2.20x) |  901 (0.93x) | 2,134 (2.24x) |

### Headline: Encode Throughput (1M float32, no codec)

| Threads | 3.13 (GIL) | 3.13t (free) | 3.14 (GIL) | 3.14t (free) |
|--------:|-----------:|-------------:|-----------:|-------------:|
| 1       |  679 op/s  |  644 op/s    |  704 op/s  |  596 op/s    |
| 2       |  829 (1.22x) |  847 (1.31x) |  858 (1.22x) |  793 (1.33x) |
| 4       |  777 (1.14x) |  845 (1.31x) |  778 (1.10x) |  801 (1.34x) |
| 8       |  776 (1.14x) |  743 (1.15x) |  782 (1.11x) |  704 (1.18x) |

### Small Messages (16K float32, no codec)

| Threads | 3.13 (GIL) | 3.13t (free) | 3.14 (GIL) | 3.14t (free) |
|--------:|-----------:|-------------:|-----------:|-------------:|
| 1       | 41,453 op/s | 38,989 op/s | 39,702 op/s | 34,398 op/s |
| 2       | 52,803 (1.27x) | 74,776 (1.92x) | 50,870 (1.28x) | 70,337 (2.04x) |
| 4       | 51,810 (1.25x) | 151,869 (3.90x) | 50,094 (1.26x) | 138,838 (4.04x) |
| 8       | 53,011 (1.28x) | 290,683 (7.46x) | 50,996 (1.28x) | 284,758 (8.28x) |
| 16      | 51,290 (1.24x) | 310,577 (7.97x) | 51,468 (1.30x) | 289,111 (8.40x) |

### Other Operations (1M float32)

**Scan** (message boundary detection — ~0.2µs/call, GIL overhead dominates):

| Threads | 3.14 (GIL) | 3.14t (free) |
|--------:|-----------:|-------------:|
| 1       | 5,049,944 op/s | 1,195,612 op/s |
| 2       | 2,457,685 (0.49x) | 2,497,771 (2.09x) |
| 4       | 1,009,050 (0.20x) | 3,301,490 (2.76x) |
| 8       | 797,019 (0.16x) | 3,936,237 (3.29x) |
| 16      | 747,419 (0.15x) | 4,420,334 (3.70x) |

**Validate** (full message validation — CPU-bound, scales well on both):

| Threads | 3.14 (GIL) | 3.14t (free) |
|--------:|-----------:|-------------:|
| 1       | 5,591 op/s | 5,022 op/s |
| 2       | 11,121 (1.99x) | 10,071 (2.01x) |
| 4       | 21,769 (3.89x) | 19,697 (3.92x) |
| 8       | 44,115 (7.89x) | 39,057 (7.78x) |
| 16      | 51,518 (9.21x) | 48,573 (9.67x) |

**Decode-range** (sub-array extraction, 2x1K slices from 1M):

| Threads | 3.14 (GIL) | 3.14t (free) |
|--------:|-----------:|-------------:|
| 1       | 92,670 op/s | 86,293 op/s |
| 2       | 175,087 (1.89x) | 173,262 (2.01x) |
| 4       | 167,350 (1.81x) | 343,264 (3.98x) |
| 8       | 160,692 (1.73x) | 665,837 (7.72x) |
| 16      | 159,898 (1.73x) | 742,405 (8.60x) |

**Iter-messages** (3 messages, 100K f32 each):

| Threads | 3.14 (GIL) | 3.14t (free) |
|--------:|-----------:|-------------:|
| 1       | 2,443 op/s | 2,560 op/s |
| 2       | 2,793 (1.14x) | 4,838 (1.89x) |
| 4       | 2,715 (1.11x) | 9,244 (3.61x) |
| 8       | 2,357 (0.96x) | 8,564 (3.35x) |
| 16      | 2,066 (0.85x) | 4,847 (1.89x) |

### Key Takeaways

Methodology: 5 runs per configuration, median reported. 200–500 warmup iterations for fast operations.

- **Validate scales near-linearly on both GIL and free-threaded** — 9.2x (GIL) and 9.7x (free-threaded) at 16 threads. This is the most CPU-bound operation and benefits fully from `py.detach()` regardless of GIL.
- **Free-threaded decode scales to 2.9x at 4 threads** for the headline workload (1M f32, no codec). GIL-enabled stays near 1.0x because numpy array construction dominates and serializes under the GIL.
- **GIL-enabled decode-range plateaus at ~1.9x** — `py.detach()` allows 2 threads of overlap but the lightweight result construction can't overlap further. Free-threaded reaches **8.6x at 16 threads**.
- **Scan degrades under the GIL** — scan is so fast (~0.2µs/call) that GIL acquisition overhead dominates, yielding 0.15x at 16 threads. Free-threaded scales to 3.7x.
- **Small messages (16K) reach 8.4x at 16 threads** on free-threaded vs 1.3x on GIL-enabled.
- **iter_messages scales to 3.6x at 4 threads** on free-threaded, then drops due to contention. GIL-enabled stays flat (~1.1x).
- **Single-thread trade-off** — for heavyweight operations (decode, encode, validate), free-threaded builds perform within ~10% of GIL-enabled. For scan, free-threaded single-thread is ~4x slower due to per-object biased reference counting overhead on the returned Python list, but this is recovered by 2 threads.

> These numbers are machine-specific. Run the benchmark on your hardware:
> ```bash
> python rust/benchmarks/python/bench_threading.py              # full suite
> python rust/benchmarks/python/bench_threading.py --headline   # quick comparison
> python rust/benchmarks/python/bench_threading.py --quick      # CI smoke test
> ```

## Tensogram Python vs ecCodes C

This section compares **Tensogram called from Python** against **ecCodes called from C** (via Rust FFI) on the same workload: 10 million float64 weather values (80 MiB), 24-bit simple packing + szip compression. This is the standard pipeline used in operational weather forecasting.

### What we measured

Both sides are measured **end-to-end**: from a float64 array to serialized compressed bytes (encode), and back to a float64 array (decode). Both include metadata serialization, framing, and integrity overhead — not just the raw packing step.

**ecCodes (C, single-threaded)**: The Rust benchmark (`rust/benchmarks/src/bin/grib_comparison.rs`) calls ecCodes' C library directly via FFI. Encode: allocate a GRIB handle, configure the grid (10M regular lat/lon), set packing type to CCSDS at 24 bits, write the values array, serialize to GRIB bytes. Decode: load the GRIB message from bytes, extract the values array. No Python involved. Median of 10 iterations, 3 warmup.

**Tensogram (Python, multi-threaded)**: The same 10M float64 values, same 24-bit quantization, same szip compression. Encode: pass a numpy array + CBOR metadata dict to `tensogram.encode()`, which crosses the PyO3 boundary, quantizes, compresses, frames, computes the integrity hash, and returns Python `bytes`. Decode: pass `bytes` to `tensogram.decode()`, which deframes, decompresses, dequantizes, and returns a numpy array. Each Python thread makes independent encode/decode calls. The GIL is released during the Rust computation.

### Why scaling depends on the codec

Threading helps most when the Rust computation (compression, quantization) is the dominant cost. With simple packing + szip, each encode/decode spends ~170 ms in Rust and ~20 ms in Python/numpy — so ~89% of the time runs with the GIL released and threads scale well. Without compression, the Rust work is trivial (~1 ms) and the Python overhead limits parallelism.

The tables above measure uncompressed data to isolate the threading mechanism. The results below use the production pipeline (24-bit packing + szip) and show what real workloads achieve.

### Results

ecCodes CCSDS (Rust FFI, single-threaded): **870 MB/s encode, 531 MB/s decode**.

Tensogram from Python (free-threaded 3.14t, 5-run median, 10M float64 24-bit packing+szip):

**Decode:**

| Threads | Throughput | vs ecCodes C |
|--------:|-----------:|-------------:|
| 1       | 446 MB/s   | 0.84x        |
| 2       | 858 MB/s   | **1.62x**    |
| 4       | 1,596 MB/s | **3.01x**    |
| 8       | 2,602 MB/s | **4.90x**    |

**Encode:**

| Threads | Throughput | vs ecCodes C |
|--------:|-----------:|-------------:|
| 1       | 435 MB/s   | 0.50x        |
| 2       | 833 MB/s   | 0.96x        |
| 4       | 1,516 MB/s | **1.74x**    |
| 8       | 2,353 MB/s | **2.71x**    |

Single-threaded Tensogram from Python is slower than ecCodes from C (the PyO3 boundary costs ~10-15% on decode, ~50% on encode due to numpy data extraction for 80 MiB). But at 2 threads, decode already surpasses ecCodes. At 4 threads, both encode and decode exceed ecCodes. At 8 threads, decode reaches **4.9x ecCodes throughput** — from Python.

## Requirements

- Python >= 3.13t for free-threaded mode (3.12/3.13 GIL-enabled also works)
- NumPy >= 2.1 (free-threaded support)
- maturin >= 1.8 (free-threaded wheel building)

## Known Limitations

**Inherent:**
- Shared mutable numpy arrays across threads can cause data races (same as any Python threading)
- xarray and zarr backends have their own threading models (dask, zarr locking)

**By design:**
- `TensogramFile` read methods (`decode_message`, `read_message`, `__getitem__`, etc.) support concurrent access from multiple threads on the same handle. Only `append()` requires exclusive access.
- `bytes` inputs to decode/scan/validate are zero-copy across the GIL release. `bytearray` inputs are copied once internally by PyO3.
- `iter_messages` / `PyBufferIter` own a full buffer copy (the buffer must outlive iteration).
