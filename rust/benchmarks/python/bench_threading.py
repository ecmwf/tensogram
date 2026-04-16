#!/usr/bin/env python3
# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Multi-threaded scaling benchmark for tensogram.

Two orthogonal dimensions of parallelism are measured:

1. **Python-thread scaling** (``--dimension=python``, default) — N
   Python threads each run their own encode/decode. Each call goes
   through tensogram with ``threads=0`` so no internal pool is
   built. This measures GIL-release + library thread-safety. Still
   relevant for free-threaded Python (3.13t+).

2. **Tensogram-thread scaling** (``--dimension=tensogram``) — one
   Python thread runs repeated encode/decode calls with
   ``threads=N`` kwarg, measuring v0.13.0's internal pool.

3. **Combined** (``--dimension=combined``) — N Python threads each
   using ``threads=M`` internally (defaults: M=2). Over-subscription
   risk; useful for measuring the combined ceiling.

Usage:
    python bench_threading.py              # full benchmark (python dim)
    python bench_threading.py --quick      # quick smoke test (CI)
    python bench_threading.py --headline   # headline numbers only
    python bench_threading.py --dimension tensogram
    python bench_threading.py --dimension combined --inner-threads 2
"""

import argparse
import os
import platform
import sys
import threading
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["BLOSC_NTHREADS"] = "1"

import numpy as np
import tensogram


def detect_environment():
    is_ft = hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled()
    return {
        "python": sys.version.split()[0],
        "free_threaded": is_ft,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def bench_scaling(
    operation,
    args,
    thread_counts,
    iterations,
    warmup=3,
    dimension="python",
    inner_threads=1,
):
    """Run `operation(*args)` under one of three scaling dimensions.

    dimension="python" — N Python threads, each calls operation with
                         whatever its kwargs say.
    dimension="tensogram" — 1 Python thread, operation receives
                            threads=N via its last positional arg.
    dimension="combined" — N Python threads × inner_threads inside
                           tensogram.
    """
    # Warm up any one-shot compile / allocation cost.
    for _ in range(warmup):
        operation(*args)

    results = {}
    for n in thread_counts:
        if dimension == "tensogram":
            # Single-threaded Python driver, tensogram uses n threads.
            t0 = time.perf_counter_ns()
            for _ in range(iterations):
                operation(*args, threads=n)
            wall_ns = time.perf_counter_ns() - t0
            total_ops = iterations
            results[n] = {
                "wall_ms": wall_ns / 1e6,
                "throughput": total_ops / (wall_ns / 1e9),
            }
            continue

        # python / combined: use N Python threads.
        barrier = threading.Barrier(n + 1)
        thread_times = [0.0] * n
        inner = inner_threads if dimension == "combined" else 0

        def worker(tid, _barrier=barrier, _times=thread_times, _inner=inner):
            _barrier.wait(timeout=30)
            t0 = time.perf_counter_ns()
            for _ in range(iterations):
                if _inner:
                    operation(*args, threads=_inner)
                else:
                    operation(*args)
            _times[tid] = time.perf_counter_ns() - t0

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        barrier.wait(timeout=30)
        wall_t0 = time.perf_counter_ns()
        for t in threads:
            t.join(timeout=120)
        wall_ns = time.perf_counter_ns() - wall_t0
        for t in threads:
            if t.is_alive():
                raise RuntimeError(f"worker thread did not finish for {n} thread(s)")

        total_ops = n * iterations
        results[n] = {
            "wall_ms": wall_ns / 1e6,
            "throughput": total_ops / (wall_ns / 1e9),
        }
    return results


def print_scaling_table(label, results, thread_counts):
    baseline = results[thread_counts[0]]["throughput"]
    print(f"\n  {label}")
    print(f"  {'Threads':>7} | {'Throughput':>14} | {'Speedup':>8}")
    print(f"  {'-' * 7} | {'-' * 14} | {'-' * 8}")
    for n in thread_counts:
        r = results[n]
        sp = r["throughput"] / baseline
        print(f"  {n:>7} | {r['throughput']:>10.0f} op/s | {sp:>7.2f}x")


def make_msg(size, encoding="none", compression="none", dtype="float32"):
    data = np.random.randn(size).astype(getattr(np, dtype))
    meta = {"version": 2, "base": [{}]}
    desc = {
        "type": "ntensor",
        "shape": [size],
        "dtype": dtype,
        "encoding": encoding,
        "compression": compression,
    }
    if encoding == "simple_packing":
        params = tensogram.compute_packing_params(
            data.astype(np.float64).ravel(), 24, 0
        )
        desc.update(params)
    msg = tensogram.encode(meta, [(desc, data)])
    return meta, desc, data, msg


def run_headline(thread_counts, iters, dimension="python", inner_threads=1):
    """Decode + encode scaling for the primary workload (1M f32, none+none)."""
    meta, desc, data, msg = make_msg(1_000_000)
    kw = {"dimension": dimension, "inner_threads": inner_threads}
    dec = bench_scaling(tensogram.decode, (msg,), thread_counts, iters, **kw)
    enc = bench_scaling(
        tensogram.encode, (meta, [(desc, data)]), thread_counts, iters, **kw
    )
    print_scaling_table(
        f"Decode (1M f32, none+none, dim={dimension})", dec, thread_counts
    )
    print_scaling_table(
        f"Encode (1M f32, none+none, dim={dimension})", enc, thread_counts
    )
    return dec, enc


def run_codec_sweep(thread_counts, iters):
    """Codec matrix: none, zstd, lz4, simple_packing+zstd."""
    codecs = [
        ("none", "zstd", "float32"),
        ("none", "lz4", "float32"),
        ("simple_packing", "zstd", "float64"),
    ]
    for encoding, compression, dtype in codecs:
        meta, desc, data, msg = make_msg(1_000_000, encoding, compression, dtype)
        label = f"{encoding}+{compression}"
        dec = bench_scaling(tensogram.decode, (msg,), thread_counts, iters)
        enc = bench_scaling(
            tensogram.encode, (meta, [(desc, data)]), thread_counts, iters
        )
        print_scaling_table(f"Decode (1M {dtype}, {label})", dec, thread_counts)
        print_scaling_table(f"Encode (1M {dtype}, {label})", enc, thread_counts)


def run_size_sweep(thread_counts, iters):
    """Payload size sweep: 16K (64KB), 1M (4MB), 16M (64MB)."""
    sizes = [
        (16_000, "64KB"),
        (1_000_000, "4MB"),
        (16_000_000, "64MB"),
    ]
    for size, label in sizes:
        adjusted_iters = max(iters // max(size // 100_000, 1), 3)
        _, _, _, msg = make_msg(size)
        dec = bench_scaling(tensogram.decode, (msg,), thread_counts, adjusted_iters)
        print_scaling_table(f"Decode ({label} f32, none+none)", dec, thread_counts)


def run_operations(thread_counts, iters):
    """Benchmark non-encode/decode operations: scan, validate, decode_range, iter_messages."""
    _, _, _, msg = make_msg(1_000_000)

    scan_res = bench_scaling(tensogram.scan, (msg,), thread_counts, iters)
    print_scaling_table("Scan (1M f32)", scan_res, thread_counts)

    val_res = bench_scaling(tensogram.validate, (msg,), thread_counts, iters)
    print_scaling_table("Validate (1M f32)", val_res, thread_counts)

    dr_res = bench_scaling(
        tensogram.decode_range,
        (msg, 0, [(0, 1000), (500_000, 1000)]),
        thread_counts,
        iters,
    )
    print_scaling_table("Decode-range (2x1K slices from 1M f32)", dr_res, thread_counts)

    _, _, _, m1 = make_msg(100_000)
    _, _, _, m2 = make_msg(100_000)
    combined = m1 + m2 + m1
    iter_res = bench_scaling(
        lambda buf: list(tensogram.iter_messages(buf)),
        (combined,),
        thread_counts,
        iters,
    )
    print_scaling_table(
        "Iter-messages (3 msgs, 100K f32 each)", iter_res, thread_counts
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="CI smoke test")
    mode.add_argument("--headline", action="store_true", help="Headline numbers only")
    parser.add_argument(
        "--dimension",
        choices=("python", "tensogram", "combined"),
        default="python",
        help="Parallelism dimension to sweep (default: python).",
    )
    parser.add_argument(
        "--inner-threads",
        type=int,
        default=2,
        help="For --dimension=combined, the tensogram threads= value.",
    )
    args = parser.parse_args()

    env = detect_environment()
    ft_label = "yes" if env["free_threaded"] else "no"
    print("=" * 64)
    print("  Tensogram Python Threading Benchmark")
    print(f"  Python: {env['python']} (free-threaded: {ft_label})")
    print(f"  NumPy: {env['numpy']} | CPU: {env['cpu_count']} cores")
    print(f"  Platform: {env['platform']}")
    print("=" * 64)

    dim = args.dimension
    inner = args.inner_threads
    kw = {"dimension": dim, "inner_threads": inner}

    print(
        f"Dimension: {dim}" + (f" (inner_threads={inner})" if dim == "combined" else "")
    )

    if args.quick:
        tc = [1, 2, 4]
        run_headline(tc, 10, **kw)
        return

    tc = [1, 2, 4, 8, 16]

    if args.headline:
        run_headline(tc, 50, **kw)
        return

    print(f"\n── Headline: 1M f32, none+none (dim={dim}) ──")
    run_headline(tc, 50, **kw)

    print("\n── Codec sweep: 1M f32 ──")
    run_codec_sweep(tc, 30)

    print("\n── Size sweep: none+none ──")
    run_size_sweep(tc, 30)

    print("\n── Other operations: 1M f32 ──")
    run_operations(tc, 30)

    print()


if __name__ == "__main__":
    main()
