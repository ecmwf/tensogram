# Benchmark Results

This page is a snapshot of benchmark results recorded on a specific machine at a specific date.
For methodology, flag descriptions, and how to re-run the benchmarks yourself, see [Benchmarks](benchmarks.md).

> **Note:** Absolute timing values (`Encode (ms)`, `Decode (ms)`, `vs Ref` columns) are
> machine-specific and will differ on other hardware. The `Ratio (%)` and `Size (KiB)` columns
> are derived from compressed byte counts and are portable across machines.

## Run metadata

| Field | Value |
|-------|-------|
| **Date** | 2026-04-07 |
| **Tensogram version** | 0.6.0 |
| **Machine** | MacBook Pro (Mac16,1) |
| **Chip** | Apple M4, 10 cores (4 Performance + 6 Efficiency) |
| **Memory** | 16 GB |
| **OS** | macOS 26.3.1 (Build 25D2128) |
| **Rust** | rustc 1.94.1 (e408947bf 2026-03-25) |
| **ecCodes** | 2.46.0 (Homebrew) |

## Codec Matrix

**Command:**
```
cargo run --release -p tensogram-benchmarks --bin codec-matrix \
    -- --num-points 16000000 --iterations 5 --seed 42
```

**Results** (24 combinations, 16 000 000 float64 values, 5 iterations, median):

```
Tensogram Codec Matrix (16000000 float64 values, 5 iterations, median)
Reference: none+none

 Combo                  | Encode (ms) | Decode (ms) | Ratio (%) | Size (KiB) | vs Ref Enc | vs Ref Dec
------------------------+--------------+--------------+------------+-------------+-------------+-------------
 none+none [REF]        |       3.075 |       4.825 |    100.00 |   125000.0 |      1.00x |      1.00x
 none+zstd(3)           |     124.317 |     110.565 |     90.30 |   112870.0 |     40.42x |     22.91x
 none+lz4               |      14.860 |      17.728 |    100.39 |   125490.2 |      4.83x |      3.67x
 none+blosc2(blosclz)   |      51.608 |       8.807 |     75.22 |    94023.5 |     16.78x |      1.83x
 none+szip(32)          |      62.116 |     230.228 |    100.96 |   126202.8 |     20.20x |     47.71x
 sp(16)+none            |      28.108 |      28.126 |     25.00 |    31250.0 |      9.14x |      5.83x
 sp(16)+zstd(3)         |      62.453 |      46.609 |     24.35 |    30437.5 |     20.31x |      9.66x
 sp(16)+lz4             |      30.858 |      30.103 |     25.10 |    31372.6 |     10.03x |      6.24x
 sp(16)+blosc2(blosclz) |     116.136 |      42.672 |     20.28 |    25354.9 |     37.76x |      8.84x
 sp(16)+szip            |      50.469 |      90.414 |     25.39 |    31736.8 |     16.41x |     18.74x
 sp(24)+none            |      33.928 |      31.987 |     37.50 |    46875.0 |     11.03x |      6.63x
 sp(24)+zstd(3)         |      76.825 |      51.890 |     37.18 |    46477.1 |     24.98x |     10.75x
 sp(24)+lz4             |      37.208 |      33.731 |     37.65 |    47058.8 |     12.10x |      6.99x
 sp(24)+blosc2(blosclz) |     137.165 |      55.272 |     32.78 |    40980.9 |     44.60x |     11.45x
 sp(24)+szip            |      53.804 |     112.939 |     28.49 |    35614.0 |     17.49x |     23.40x
 sp(32)+none            |      40.802 |      36.103 |     50.00 |    62500.0 |     13.27x |      7.48x
 sp(32)+zstd(3)         |     115.953 |      54.272 |     49.81 |    62264.9 |     37.70x |     11.25x
 sp(32)+lz4             |      45.954 |      41.120 |     50.20 |    62745.1 |     14.94x |      8.52x
 sp(32)+blosc2(blosclz) |     139.815 |      52.611 |     45.29 |    56606.8 |     45.46x |     10.90x
 sp(32)+szip            |      72.659 |     151.333 |     50.49 |    63109.1 |     23.63x |     31.36x
 none+zfp(rate=16)      |     214.038 |     296.643 |     25.00 |    31250.0 |     69.60x |     61.47x
 none+zfp(rate=24)      |     246.089 |     470.858 |     37.50 |    46875.0 |     80.02x |     97.58x
 none+zfp(rate=32)      |     282.528 |     593.183 |     50.00 |    62500.0 |     91.87x |    122.93x
 none+sz3(abs=0.01)     |     133.714 |     142.765 |      6.46 |     8069.4 |     43.48x |     29.59x
```

## GRIB Comparison

**Command:**
```
cargo run --release -p tensogram-benchmarks --bin grib-comparison --features eccodes \
    -- --num-points 10000000 --iterations 5 --seed 42
```

**Results** (10 000 000 float64 values, 24-bit packing, 5 iterations, median):

```
GRIB vs Tensogram Comparison (10000000 float64 values, 24 bit, 5 iterations)
Reference: eccodes grid_ccsds

 Combo                    | Encode (ms) | Decode (ms) | Ratio (%) | Size (KiB) | vs Ref Enc | vs Ref Dec
--------------------------+--------------+--------------+------------+-------------+-------------+-------------
 eccodes grid_ccsds [REF] |      45.559 |      83.248 |     27.20 |    21247.1 |      1.00x |      1.00x
 eccodes grid_simple      |      30.993 |       7.678 |     37.50 |    29297.0 |      0.68x |      0.09x
 tensogram sp(24)+szip    |      36.137 |      72.323 |     28.49 |    22258.8 |      0.79x |      0.87x
```
