# Benchmark Results

This page is a snapshot of benchmark results recorded on a specific machine.
For methodology, flags, and how to re-run, see [Benchmarks](benchmarks.md).

> **Note:** Timing and throughput are machine-specific. Compression ratios,
> sizes, and fidelity metrics are determined by the codec and are reproducible.

## Run metadata

| Field | Value |
|-------|-------|
| **Date** | 2026-04-16 |
| **Tensogram version** | 0.11.0 |
| **CPU** | Intel Core i9-10850K @ 3.60 GHz, 10 cores / 20 threads |
| **OS** | Linux 6.19.10 (CachyOS) x86_64 |
| **Rust** | rustc 1.94.0 |
| **ecCodes** | 2.46.0 |
| **Methodology** | 10 timed iterations, 3 warmup, median reported |

## Codec Matrix

16 million float64 values (122 MiB). The test data is a synthetic temperature-like
field with values in the range 250–310.

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
| no compression **[REF]** | 34.6 | 35.2 | 3525 | 3465 | 100% | 122.1 |
| zstd level 3 | 179.7 | 114.3 | 680 | 1068 | 90.3% | 110.2 |
| LZ4 | 40.2 | 36.6 | 3038 | 3337 | 100.4% | 122.6 |
| Blosc2 | 162.9 | 64.4 | 749 | 1894 | 75.2% | 91.8 |
| szip | 186.9 | 314.3 | 653 | 388 | 100.9% | 123.2 |

Raw 64-bit floats have high entropy, so most lossless compressors cannot reduce
their size. LZ4 and szip slightly expand the data. Blosc2 is the exception — its
byte-shuffle step exposes compressible patterns (75%).

### SimplePacking (quantization) + lossless compressors

Values are quantized to N bits, then compressed. Fidelity depends only on the
bit width, not on the compressor — see the fidelity table below.

| Method | Enc (ms) | Dec (ms) | Enc MB/s | Dec MB/s | Ratio | Size (MiB) |
|--------|----------|----------|----------|----------|-------|------------|
| 16-bit only | 96.8 | 73.5 | 1261 | 1662 | 25.0% | 30.5 |
| 16-bit + zstd | 146.9 | 88.1 | 831 | 1385 | 24.4% | 29.7 |
| 16-bit + LZ4 | 107.8 | 76.1 | 1132 | 1604 | 25.1% | 30.6 |
| 16-bit + Blosc2 | 230.3 | 92.4 | 530 | 1321 | 20.3% | 24.8 |
| 16-bit + szip | 167.0 | 188.6 | 731 | 647 | 14.6% | 17.8 |
| 24-bit only | 107.8 | 78.6 | 1133 | 1553 | 37.5% | 45.8 |
| 24-bit + zstd | 178.6 | 102.0 | 684 | 1197 | 37.2% | 45.4 |
| 24-bit + LZ4 | 124.8 | 91.2 | 978 | 1339 | 37.6% | 46.0 |
| 24-bit + Blosc2 | 272.8 | 135.7 | 448 | 900 | 32.8% | 40.0 |
| 24-bit + szip | 199.7 | 230.8 | 611 | 529 | 27.2% | 33.2 |
| 32-bit only | 107.3 | 74.5 | 1137 | 1638 | 50.0% | 61.0 |
| 32-bit + zstd | 203.7 | 104.4 | 599 | 1169 | 49.8% | 60.8 |
| 32-bit + LZ4 | 131.6 | 94.6 | 928 | 1291 | 50.2% | 61.3 |
| 32-bit + Blosc2 | 291.2 | 126.0 | 419 | 969 | 45.3% | 55.3 |
| 32-bit + szip | 212.4 | 269.7 | 575 | 453 | 39.7% | 48.4 |

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
| ZFP rate 16 | 492.9 | 508.5 | 248 | 240 | 25.0% | 30.5 |
| ZFP rate 24 | 615.1 | 742.7 | 199 | 164 | 37.5% | 45.8 |
| ZFP rate 32 | 711.7 | 919.2 | 172 | 133 | 50.0% | 61.0 |
| SZ3 abs 0.01 | 459.9 | 277.1 | 266 | 441 | 6.5% | 7.9 |

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

## GRIB Comparison

[GRIB](https://en.wikipedia.org/wiki/GRIB) is the standard binary format used in
operational weather forecasting. The most widely used GRIB encoder is
[ecCodes](https://confluence.ecmwf.int/display/ECC) from ECMWF.

This benchmark compares Tensogram's 24-bit SimplePacking + szip against ecCodes'
built-in packing methods on the same data. Both sides are timed end-to-end: from a
float64 array to serialized compressed bytes (encode), and back (decode).

10 million float64 values (76 MiB), 24-bit packing. Different dataset size from the
codec matrix above.

| Method | Enc (ms) | Dec (ms) | Enc MB/s | Dec MB/s | Ratio | Size (MiB) |
|--------|----------|----------|----------|----------|-------|------------|
| ecCodes CCSDS **[REF]** | 84.8 | 141.5 | 900 | 539 | 27.2% | 20.8 |
| ecCodes simple packing | 65.6 | 40.8 | 1163 | 1870 | 37.5% | 28.6 |
| Tensogram 24-bit + szip | 119.2 | 138.0 | 640 | 553 | 27.4% | 20.9 |

All three methods produce identical fidelity: Linf = 1.9 × 10⁻⁶,
L1 = 9.5 × 10⁻⁷, L2 = 1.1 × 10⁻⁶.

### Notable observations

- **Tensogram and ecCodes CCSDS achieve nearly identical compression** (27.4% vs 27.2%)
  and identical fidelity at 24 bits.
- **Tensogram encode is 1.4× slower** (119 vs 85 ms); decode is comparable (138 vs 142 ms).
- **ecCodes simple packing** decodes fastest (41 ms) but produces a larger file (37.5% vs 27%).
