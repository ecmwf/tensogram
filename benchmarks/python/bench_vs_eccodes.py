#!/usr/bin/env python3
# (C) Copyright 2024- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Compare Tensogram Python multi-threaded throughput against ecCodes C.

Encodes and decodes 10M float64 weather values with 24-bit simple
packing + szip — the standard pipeline for operational weather data.

The ecCodes reference numbers come from the Rust benchmark binary
(benchmarks/src/bin/grib_comparison.rs) which calls ecCodes C via FFI.
Run it first to get your machine's baseline:

    cargo build --release -p tensogram-benchmarks --bin grib-comparison --features eccodes
    ./target/release/grib-comparison

Then run this script to measure Tensogram from Python at the same workload:

    python benchmarks/python/bench_vs_eccodes.py
    python benchmarks/python/bench_vs_eccodes.py --eccodes-encode 870 --eccodes-decode 531
"""

import argparse
import os
import sys
import threading
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["BLOSC_NTHREADS"] = "1"

import numpy as np
import tensogram

NUM_POINTS = 10_000_000
ORIG_BYTES = NUM_POINTS * 8
RUNS = 5
ITERS = 5
WARMUP = 2
THREAD_COUNTS = [1, 2, 4, 8]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--eccodes-encode",
        type=float,
        default=870,
        help="ecCodes encode MB/s from Rust benchmark",
    )
    p.add_argument(
        "--eccodes-decode",
        type=float,
        default=531,
        help="ecCodes decode MB/s from Rust benchmark",
    )
    return p.parse_args()


def prepare():
    np.random.seed(42)
    values = (280.0 + 15.0 * np.random.randn(NUM_POINTS)).astype(np.float64)
    meta = {"version": 2, "base": [{}]}
    params = tensogram.compute_packing_params(values, 24, 0)
    desc = {
        "type": "ntensor",
        "shape": [NUM_POINTS],
        "dtype": "float64",
        "encoding": "simple_packing",
        "compression": "szip",
        "szip_rsi": 128,
        "szip_block_size": 16,
        "szip_flags": 128,
        **params,
    }
    msg = tensogram.encode(meta, [(desc, values)])
    return meta, desc, values, msg


def bench_median(operation, args):
    results = {}
    for n in THREAD_COUNTS:
        medians = []
        for _ in range(RUNS):
            for _ in range(WARMUP):
                operation(*args)
            barrier = threading.Barrier(n + 1)

            def worker(tid, _barrier=barrier):
                _barrier.wait(timeout=60)
                for _ in range(ITERS):
                    operation(*args)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
            for t in threads:
                t.start()
            barrier.wait(timeout=60)
            wall = time.perf_counter_ns()
            for t in threads:
                t.join(timeout=300)
            for t in threads:
                if t.is_alive():
                    raise RuntimeError(f"worker thread did not finish for {n} thread(s)")
            wall_ns = time.perf_counter_ns() - wall
            medians.append(n * ITERS / (wall_ns / 1e9))
        medians.sort()
        results[n] = medians[len(medians) // 2]
    return results


def print_table(label, results, eccodes_mbs):
    print(f"\n  {label}")
    print(f"  {'Thr':>4} | {'op/s':>8} | {'MB/s':>10} | vs ecCodes C")
    print(f"  {'-' * 4} | {'-' * 8} | {'-' * 10} | {'-' * 12}")
    for n in THREAD_COUNTS:
        ops = results[n]
        mbs = ops * ORIG_BYTES / 1e6
        ratio = mbs / eccodes_mbs
        print(f"  {n:>4} | {ops:>8.1f} | {mbs:>7.0f}    | {ratio:.2f}x")


def main():
    args = parse_args()
    meta, desc, values, msg = prepare()

    is_ft = hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled()
    ft_label = "yes" if is_ft else "no"
    print(f"Python {sys.version.split()[0]} (free-threaded: {ft_label})")
    print(f"{NUM_POINTS:,} float64 = {ORIG_BYTES / 1e6:.0f} MiB, 24-bit packing + szip")
    print(
        f"ecCodes C reference: {args.eccodes_encode:.0f} MB/s enc, "
        f"{args.eccodes_decode:.0f} MB/s dec"
    )
    print(f"{RUNS} runs x {ITERS} iterations per thread count, median reported")

    enc = bench_median(tensogram.encode, (meta, [(desc, values)]))
    print_table("Encode", enc, args.eccodes_encode)

    dec = bench_median(tensogram.decode, (msg,))
    print_table("Decode", dec, args.eccodes_decode)

    print()


if __name__ == "__main__":
    main()
