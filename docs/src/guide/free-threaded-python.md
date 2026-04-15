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
| 1       |  416 op/s  |  391 op/s    |  408 op/s  |  396 op/s    |
| 2       |  432 (1.04x) |  775 (1.98x) |  432 (1.06x) |  776 (1.96x) |
| 4       |  427 (1.03x) | 1,356 (3.47x) |  425 (1.04x) | 1,352 (3.41x) |
| 8       |  309 (0.74x) | 1,507 (3.85x) |  293 (0.72x) | 1,841 (4.65x) |

### Headline: Encode Throughput (1M float32, no codec)

| Threads | 3.13 (GIL) | 3.13t (free) | 3.14 (GIL) | 3.14t (free) |
|--------:|-----------:|-------------:|-----------:|-------------:|
| 1       |  608 op/s  |  572 op/s    |  504 op/s  |  595 op/s    |
| 2       |  761 (1.25x) |  709 (1.24x) |  664 (1.32x) |  702 (1.18x) |
| 4       |  659 (1.08x) |  726 (1.27x) |  468 (0.93x) |  725 (1.22x) |
| 8       |  520 (0.86x) |  706 (1.23x) |  351 (0.70x) |  717 (1.20x) |

### Small Messages (16K float32, no codec)

| Threads | 3.13 (GIL) | 3.13t (free) | 3.14 (GIL) | 3.14t (free) |
|--------:|-----------:|-------------:|-----------:|-------------:|
| 1       | 20,765 op/s | 17,085 op/s | 20,174 op/s | 12,951 op/s |
| 2       | 23,689 (1.14x) | 35,642 (2.09x) | 23,093 (1.14x) | 35,176 (2.72x) |
| 4       | 22,629 (1.09x) | 36,483 (2.14x) | 22,839 (1.13x) | 61,583 (4.75x) |
| 8       | 23,664 (1.14x) | 79,539 (4.66x) | 22,487 (1.11x) | 73,549 (5.68x) |
| 16      | 23,418 (1.13x) | 93,627 (5.48x) | 23,369 (1.16x) | 168,786 (13.03x) |

### Other Operations (1M float32)

**Scan** (message boundary detection — ~0.2µs/call, GIL overhead dominates):

| Threads | 3.14 (GIL) | 3.14t (free) |
|--------:|-----------:|-------------:|
| 1       | 312,930 op/s | 79,431 op/s |
| 2       | 421,701 (1.35x) | 266,103 (3.35x) |
| 4       | 629,505 (2.01x) | 811,096 (10.21x) |
| 8       | 522,940 (1.67x) | 389,106 (4.90x) |
| 16      | 516,342 (1.65x) | 1,231,777 (15.51x) |

**Validate** (full message validation — CPU-bound, scales well on both):

| Threads | 3.14 (GIL) | 3.14t (free) |
|--------:|-----------:|-------------:|
| 1       | 5,457 op/s | 4,347 op/s |
| 2       | 10,860 (1.99x) | 9,440 (2.17x) |
| 4       | 20,249 (3.71x) | 18,752 (4.31x) |
| 8       | 39,766 (7.29x) | 23,048 (5.30x) |
| 16      | 48,560 (8.90x) | 45,455 (10.46x) |

**Decode-range** (sub-array extraction, 2x1K slices from 1M):

| Threads | 3.14 (GIL) | 3.14t (free) |
|--------:|-----------:|-------------:|
| 1       | 66,488 op/s | 40,265 op/s |
| 2       | 111,544 (1.68x) | 98,319 (2.44x) |
| 4       | 103,191 (1.55x) | 167,786 (4.17x) |
| 8       | 104,752 (1.58x) | 325,101 (8.07x) |
| 16      | 103,236 (1.55x) | 475,755 (11.82x) |

**Iter-messages** (3 messages, 100K f32 each):

| Threads | 3.14 (GIL) | 3.14t (free) |
|--------:|-----------:|-------------:|
| 1       | 1,214 op/s | 1,195 op/s |
| 2       | 1,291 (1.06x) | 2,327 (1.95x) |
| 4       | 1,211 (1.00x) | 4,548 (3.81x) |
| 8       | 1,194 (0.98x) | 5,589 (4.68x) |
| 16      | 1,106 (0.91x) | 4,432 (3.71x) |

### Key Takeaways

Methodology: 5 runs per configuration, median reported. 200–500 warmup iterations for fast operations.

- **Validate scales near-linearly on both GIL and free-threaded** — 8.9x (GIL) and 10.5x (free-threaded) at 16 threads. This is the most CPU-bound operation and benefits fully from `py.detach()` regardless of GIL.
- **Free-threaded decode scales to 4.7x at 8 threads** for the headline workload (1M f32, no codec). GIL-enabled stays near 1.0x because numpy array construction dominates and serializes under the GIL.
- **GIL-enabled decode-range plateaus at ~1.7x** — `py.detach()` allows 2 threads of overlap but the lightweight result construction can't overlap further. Free-threaded reaches **11.8x at 16 threads**.
- **Scan shows dramatic free-threaded scaling** — free-threaded reaches **15.5x at 16 threads**. GIL-enabled scales to 2.0x at 4 threads but drops back at higher thread counts due to contention.
- **Small messages (16K) reach 13.0x at 16 threads** on free-threaded (3.14t) vs 1.2x on GIL-enabled.
- **iter_messages scales to 4.7x at 8 threads** on free-threaded, then drops due to contention. GIL-enabled stays flat (~1.0x).
- **Single-thread trade-off** — for heavyweight operations (decode, encode, validate), free-threaded builds perform within ~5% of GIL-enabled. For scan, free-threaded single-thread is ~4x slower due to per-object biased reference counting overhead on the returned Python list, but this is recovered by 2 threads.

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
