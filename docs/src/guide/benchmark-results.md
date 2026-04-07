# Benchmark Results

This page is a snapshot of benchmark results recorded on a specific machine.
For methodology, flags, and how to re-run, see [Benchmarks](benchmarks.md).

> **Note:** Timing and throughput are machine-specific. Compression ratios,
> sizes, and fidelity metrics are determined by the codec and are reproducible.

## Run metadata

| Field | Value |
|-------|-------|
| **Date** | 2026-04-08 |
| **Tensogram version** | 0.6.0 |
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
| no compression **[REF]** | 35.2 | 35.0 | 3466 | 3489 | 100% | 122.1 |
| zstd level 3 | 178.9 | 113.8 | 682 | 1073 | 90.3% | 110.2 |
| LZ4 | 41.5 | 37.5 | 2942 | 3251 | 100.4% | 122.5 |
| Blosc2 | 160.6 | 63.9 | 760 | 1912 | 75.2% | 91.8 |
| szip | 186.4 | 316.5 | 655 | 386 | 100.9% | 123.2 |

Raw 64-bit floats have high entropy, so most lossless compressors cannot reduce
their size. LZ4 and szip slightly expand the data. Blosc2 is the exception — its
byte-shuffle step exposes compressible patterns (75%).

### SimplePacking (quantization) + lossless compressors

Values are quantized to N bits, then compressed. Fidelity depends only on the
bit width, not on the compressor — see the fidelity table below.

| Method | Enc (ms) | Dec (ms) | Enc MB/s | Dec MB/s | Ratio | Size (MiB) |
|--------|----------|----------|----------|----------|-------|------------|
| 16-bit only | 88.7 | 73.3 | 1376 | 1665 | 25.0% | 30.5 |
| 16-bit + zstd | 138.6 | 87.6 | 881 | 1393 | 24.4% | 29.7 |
| 16-bit + LZ4 | 98.9 | 74.4 | 1234 | 1640 | 25.1% | 30.6 |
| 16-bit + Blosc2 | 221.3 | 91.9 | 552 | 1329 | 20.3% | 24.8 |
| 16-bit + szip | 158.6 | 187.8 | 770 | 650 | 14.6% | 17.8 |
| 24-bit only | 99.9 | 78.3 | 1222 | 1560 | 37.5% | 45.8 |
| 24-bit + zstd | 170.8 | 102.5 | 715 | 1191 | 37.2% | 45.4 |
| 24-bit + LZ4 | 117.0 | 91.8 | 1044 | 1329 | 37.7% | 46.0 |
| 24-bit + Blosc2 | 265.2 | 135.9 | 460 | 898 | 32.8% | 40.0 |
| 24-bit + szip | 191.8 | 230.1 | 637 | 531 | 27.2% | 33.2 |
| 32-bit only | 99.3 | 73.4 | 1229 | 1664 | 50.0% | 61.0 |
| 32-bit + zstd | 192.6 | 100.0 | 634 | 1221 | 49.8% | 60.8 |
| 32-bit + LZ4 | 121.7 | 91.8 | 1003 | 1330 | 50.2% | 61.3 |
| 32-bit + Blosc2 | 279.5 | 124.0 | 437 | 985 | 45.3% | 55.3 |
| 32-bit + szip | 203.2 | 268.6 | 601 | 455 | 39.7% | 48.4 |

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
| ZFP rate 16 | 493.5 | 507.3 | 247 | 241 | 25.0% | 30.5 |
| ZFP rate 24 | 621.8 | 743.4 | 196 | 164 | 37.5% | 45.8 |
| ZFP rate 32 | 717.5 | 920.0 | 170 | 133 | 50.0% | 61.0 |
| SZ3 abs 0.01 | 464.1 | 276.2 | 263 | 442 | 6.5% | 7.9 |

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
| ecCodes CCSDS **[REF]** | 84.7 | 139.1 | 900 | 549 | 27.2% | 20.7 |
| ecCodes simple packing | 67.0 | 39.0 | 1139 | 1954 | 37.5% | 28.6 |
| Tensogram 24-bit + szip | 114.2 | 136.5 | 668 | 559 | 27.4% | 20.9 |

All three methods produce identical fidelity: Linf = 1.9 × 10⁻⁶,
L1 = 9.5 × 10⁻⁷, L2 = 1.1 × 10⁻⁶.

### Notable observations

- **Tensogram and ecCodes CCSDS achieve nearly identical compression** (27.4% vs 27.2%)
  and identical fidelity at 24 bits.
- **Tensogram encode is 1.3× slower** (114 vs 85 ms); decode is slightly faster (137 vs 139 ms).
- **ecCodes simple packing** decodes fastest (39 ms) but produces a larger file (37.5% vs 27%).
