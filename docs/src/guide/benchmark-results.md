# Benchmark Results

This page is a snapshot of benchmark results recorded on a specific machine.
For methodology, flags, and how to re-run, see [Benchmarks](benchmarks.md).

> **Note:** Timing and throughput are machine-specific. Compression ratios,
> sizes, and fidelity metrics are determined by the codec and are reproducible.

## Run metadata

| Field | Value |
|-------|-------|
| **Date** | 2026-04-16 |
| **Tensogram version** | 0.13.0 |
| **CPU** | Apple M4, 10 cores / 10 threads |
| **OS** | macOS 26.3 (Darwin 25.3.0) |
| **Rust** | rustc 1.94.1 |
| **ecCodes** | 2.46.0 |
| **Methodology** | 10 timed iterations, 3 warmup, median reported |

## Codec Matrix

16 million float64 values (122 MiB). The test data is a synthetic smooth
scientific-like field with values in the range 250–310 (a profile that also
matches real temperature grids and other bounded-range physical measurements).

#### How fidelity is measured

After each encode→decode round-trip, the decoded values are compared to the
original. Three error norms are reported, all absolute in the same units as
the input:

- **Linf** — the largest error for any single value. Answers: *"what is the
  worst case?"*
- **L1** — the average error across all values. Answers: *"how far off are
  values on average?"*
- **L2 (RMSE)** — root mean square error. Like L1 but penalizes large outliers
  more heavily. Answers: *"how large are the typical errors, weighted toward
  the worst ones?"*

For lossless codecs all three are zero.

### Lossless compressors on raw floats

No encoding step — raw 64-bit floats compressed directly. Decoded values are
bit-identical to the original.

| Method | Enc (ms) | Dec (ms) | Enc MB/s | Dec MB/s | Ratio | Size (MiB) |
|--------|----------|----------|----------|----------|-------|------------|
| no compression **[REF]** | 3.7 | 3.7 | 32818 | 33226 | 100.0% | 122.1 |
| zstd level 3 | 128.5 | 114.5 | 950 | 1066 | 90.3% | 110.2 |
| LZ4 | 8.5 | 7.4 | 14328 | 16535 | 100.4% | 122.6 |
| Blosc2 | 51.9 | 26.6 | 2350 | 4584 | 75.2% | 91.8 |
| szip | 69.7 | 206.8 | 1753 | 590 | 100.9% | 123.2 |

Raw 64-bit floats have high entropy, so most lossless compressors cannot reduce
their size. LZ4 and szip slightly expand the data. Blosc2 is the exception — its
byte-shuffle step exposes compressible patterns (75%).

### SimplePacking (quantization) + lossless compressors

Values are quantized to N bits, then compressed. Fidelity depends only on the
bit width, not on the compressor — see the fidelity table below.

| Method | Enc (ms) | Dec (ms) | Enc MB/s | Dec MB/s | Ratio | Size (MiB) |
|--------|----------|----------|----------|----------|-------|------------|
| 16-bit only | 17.3 | 15.1 | 7039 | 8078 | 25.0% | 30.5 |
| 16-bit + zstd | 54.2 | 36.2 | 2254 | 3375 | 24.4% | 29.7 |
| 16-bit + LZ4 | 19.7 | 22.2 | 6204 | 5493 | 25.1% | 30.6 |
| 16-bit + Blosc2 | 115.2 | 31.5 | 1060 | 3873 | 20.3% | 24.8 |
| 16-bit + szip | 53.9 | 99.3 | 2263 | 1229 | 14.6% | 17.8 |
| 24-bit only | 19.2 | 17.1 | 6347 | 7135 | 37.5% | 45.8 |
| 24-bit + zstd | 67.3 | 41.1 | 1813 | 2969 | 37.2% | 45.4 |
| 24-bit + LZ4 | 31.5 | 23.5 | 3871 | 5188 | 37.6% | 46.0 |
| 24-bit + Blosc2 | 124.9 | 40.0 | 978 | 3052 | 32.8% | 40.0 |
| 24-bit + szip | 63.3 | 133.5 | 1928 | 914 | 27.2% | 33.2 |
| 32-bit only | 21.2 | 25.3 | 5771 | 4825 | 50.0% | 61.0 |
| 32-bit + zstd | 97.8 | 37.0 | 1248 | 3299 | 49.8% | 60.8 |
| 32-bit + LZ4 | 37.1 | 45.1 | 3287 | 2706 | 50.2% | 61.3 |
| 32-bit + Blosc2 | 141.0 | 38.3 | 866 | 3183 | 45.3% | 55.3 |
| 32-bit + szip | 69.8 | 157.4 | 1748 | 775 | 39.7% | 48.4 |

#### Fidelity by bit width

| Bit width | Linf (max abs) | L1 (mean abs) | L2 (RMSE) |
|-----------|----------------|---------------|-----------|
| 16 bits | 4.9 × 10⁻⁴ | 2.4 × 10⁻⁴ | 2.8 × 10⁻⁴ |
| 24 bits | 1.9 × 10⁻⁶ | 9.5 × 10⁻⁷ | 1.1 × 10⁻⁶ |
| 32 bits | 7.5 × 10⁻⁹ | 3.7 × 10⁻⁹ | 4.3 × 10⁻⁹ |

For context: with input values around 280, a Linf of 1.9 × 10⁻⁶ means the worst-case
relative error at 24 bits is roughly 7 parts per billion.

### Lossy floating-point compressors

These operate directly on raw f64 bytes without quantization.

| Method | Enc (ms) | Dec (ms) | Enc MB/s | Dec MB/s | Ratio | Size (MiB) |
|--------|----------|----------|----------|----------|-------|------------|
| ZFP rate 16 | 220.1 | 304.2 | 555 | 401 | 25.0% | 30.5 |
| ZFP rate 24 | 248.0 | 468.5 | 492 | 261 | 37.5% | 45.8 |
| ZFP rate 32 | 288.0 | 581.0 | 424 | 210 | 50.0% | 61.0 |
| SZ3 abs 0.01 | 131.4 | 141.0 | 929 | 865 | 6.5% | 7.9 |

#### Fidelity by lossy codec

| Method | Linf (max abs) | L1 (mean abs) | L2 (RMSE) |
|--------|----------------|---------------|-----------|
| ZFP rate 16 | 1.3 × 10⁻² | 1.6 × 10⁻³ | 2.0 × 10⁻³ |
| ZFP rate 24 | 5.6 × 10⁻⁵ | 6.1 × 10⁻⁶ | 7.9 × 10⁻⁶ |
| ZFP rate 32 | 1.9 × 10⁻⁷ | 2.4 × 10⁻⁸ | 3.1 × 10⁻⁸ |
| SZ3 abs 0.01 | 1.0 × 10⁻² | 5.0 × 10⁻³ | 5.8 × 10⁻³ |

### Notable observations

- **16-bit + szip** achieves the best compression ratio (14.6%) among the
  SimplePacking combinations.
- **SZ3** achieves the smallest output overall (6.5%) with a max error of 0.01.
  If your application tolerates that error bound, this gives the best compression
  in this benchmark.
- In this benchmark, higher ZFP rates gave proportionally smaller errors.
  ZFP fixed-rate modes always hit their target ratio exactly (25% / 37.5% / 50%).

## Reference Comparison: ecCodes GRIB Encoding

[GRIB](https://en.wikipedia.org/wiki/GRIB) is a binary format widely used in
operational weather forecasting, and
[ecCodes](https://confluence.ecmwf.int/display/ECC) (from ECMWF) is a common
implementation. Comparing against it gives a concrete, reproducible reference
point for Tensogram's quantisation + entropy-coding pipeline.

This benchmark runs Tensogram's 24-bit SimplePacking + szip and ecCodes'
built-in packing methods on the same input. Both sides are timed end-to-end:
from a float64 array to serialised compressed bytes (encode), and back (decode).

10 million float64 values (76 MiB), 24-bit packing. Different dataset size from the
codec matrix above.

| Method | Enc (ms) | Dec (ms) | Enc MB/s | Dec MB/s | Ratio | Size (MiB) |
|--------|----------|----------|----------|----------|-------|------------|
| ecCodes CCSDS **[REF]** | 47.9 | 84.8 | 1594 | 900 | 27.2% | 20.8 |
| ecCodes simple packing | 32.6 | 7.9 | 2339 | 9660 | 37.5% | 28.6 |
| Tensogram 24-bit + szip | 43.7 | 80.4 | 1745 | 950 | 27.4% | 20.9 |

All three methods produce identical fidelity: Linf = 1.9 × 10⁻⁶,
L1 = 9.5 × 10⁻⁷, L2 = 1.1 × 10⁻⁶.

### Notable observations

- **Tensogram and ecCodes CCSDS achieve nearly identical compression** (27.4% vs 27.2%)
  and identical fidelity at 24 bits.
- **Tensogram encode is now slightly faster than ecCodes CCSDS** (43.7 vs 47.9 ms) on this
  machine; decode is comparable (80.4 vs 84.8 ms).
- **ecCodes simple packing** decodes fastest (7.9 ms) but produces a larger file (37.5% vs 27%).

## Threading Scaling

The v0.13.0 multi-threaded coding pipeline lets callers spend a
`threads` budget on encode/decode work. Results here show the effect
of sweeping `threads ∈ {0, 1, 2, 4, 8}` on 16M f64 values (122 MiB)
for seven representative codec combinations. `threads=0` is the
sequential baseline; speedups are measured against it.

> **Reminder:** Transparent codecs (no codec, simple_packing, szip,
> lz4, zfp, sz3, shuffle) produce byte-identical encoded payloads
> across thread counts. Opaque codecs (blosc2, zstd with
> `nb_workers > 0`) may produce different compressed bytes while
> always round-tripping losslessly.

### Lossless (no encoding)

| Method | Metric | threads=0 | threads=1 | threads=2 | threads=4 | threads=8 |
|--------|--------|-----------|-----------|-----------|-----------|-----------|
| none+none | enc MB/s | 32818 | 35929 | 36801 | 35173 | 35520 |
| none+none | speedup | 1.00x | 1.09x | 1.12x | 1.07x | 1.08x |
| none+lz4 | enc MB/s | 7733 | 3619 | 3559 | 2029 | 2513 |
| none+lz4 | speedup | 1.00x | 0.47x | 0.46x | 0.26x | 0.32x |
| none+zstd(3) | enc MB/s | 942 | 1163 | 2075 | 2259 | 1839 |
| none+zstd(3) | speedup | 1.00x | 1.23x | 2.20x | 2.40x | 1.95x |
| none+blosc2(lz4) | enc MB/s | 3150 | 3140 | 5030 | 7458 | 8906 |
| none+blosc2(lz4) | speedup | 1.00x | 1.00x | 1.60x | 2.37x | 2.83x |

### SimplePacking + compression

| Method | Metric | threads=0 | threads=1 | threads=2 | threads=4 | threads=8 |
|--------|--------|-----------|-----------|-----------|-----------|-----------|
| sp(16)+none | enc MB/s | 12964 | 13268 | 15584 | 15643 | 14612 |
| sp(16)+none | enc speedup | 1.00x | 1.02x | 1.20x | 1.21x | 1.13x |
| sp(16)+none | dec speedup | 1.00x | 1.14x | 2.37x | 2.34x | 2.18x |
| sp(24)+szip | enc MB/s | 2273 | 2263 | 2351 | 2389 | 2427 |
| sp(24)+szip | speedup | 1.00x | 1.00x | 1.03x | 1.05x | 1.07x |
| sp(24)+blosc2(lz4) | enc MB/s | 2371 | 2350 | 3965 | 5554 | 6388 |
| sp(24)+blosc2(lz4) | enc speedup | 1.00x | 0.99x | 1.67x | 2.34x | 2.69x |

### Notable observations

- **Memory-bound baselines (none+none, none+lz4) do not scale.**
  The parallel dispatch overhead outweighs any gain when the work
  per task is already at memory bandwidth. `none+lz4` actually
  *regresses* — leave `threads=0` for lz4-only workloads.
- **blosc2 scales best.** Encoding with blosc2+lz4 reaches 2.8×
  on 8 threads; the sp(24)+blosc2 combination reaches 2.7× on
  encode and 1.3× on decode.
- **zstd scales ~2.4× on encode** at 4 threads via libzstd's
  `NbWorkers`. Beyond 4 threads the benefit plateaus on this CPU.
- **simple_packing decode is 2.3× faster at 2+ threads** — the
  internal chunk-parallel scatter saturates memory bandwidth
  quickly.
- **szip is single-threaded.** The marginal gains shown for
  `sp(24)+szip` come from parallelising the `simple_packing`
  stage only; szip itself runs sequentially in v0.13.0.

The raw numbers above were produced by the `threads-scaling` binary
in `rust/benchmarks`. Re-run locally with:

```bash
cargo build --release -p tensogram-benchmarks
./target/release/threads-scaling \
    --num-points 16000000 \
    --iterations 5 \
    --warmup 2 \
    --threads 0,1,2,4,8
```
