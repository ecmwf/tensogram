"""
Example 15 — Async operations with AsyncTensogramFile
=====================================================

Demonstrates:
  1. Async open and decode
  2. Concurrent decodes with asyncio.gather
  3. Reading one grid point from many messages concurrently
  4. Async context manager and iteration
"""

import asyncio
import tempfile

import numpy as np
import tensogram


def create_test_file(path: str) -> None:
    with tensogram.TensogramFile.create(path) as f:
        for i in range(20):
            meta = {"version": 2, "base": [{"step": i}]}
            desc = {
                "type": "ntensor",
                "shape": [100, 200],
                "dtype": "float32",
                "encoding": "none",
                "compression": "none",
            }
            data = np.random.randn(100, 200).astype(np.float32) + i * 10
            f.append(meta, [(desc, data)])


async def main():
    with tempfile.NamedTemporaryFile(suffix=".tgm", delete=False) as tmp:
        path = tmp.name

    create_test_file(path)

    f = await tensogram.AsyncTensogramFile.open(path)
    print(f"Opened: {f}")
    print(f"  is_remote = {f.is_remote()}")
    print(f"  source    = {f.source()}")
    print(f"  len       = {len(f)}")

    # 1. Single decode
    meta, objects = await f.decode_message(0)
    print(f"\nDecode message 0: version={meta.version}, objects={len(objects)}")
    print(f"  shape={objects[0][1].shape}, dtype={objects[0][1].dtype}")

    # 2. Concurrent decodes with asyncio.gather
    results = await asyncio.gather(
        f.file_decode_object(0, 0),
        f.file_decode_object(1, 0),
        f.file_decode_object(2, 0),
    )
    print(f"\nGathered {len(results)} objects concurrently:")
    for i, r in enumerate(results):
        print(f"  message {i}: mean={r['data'].mean():.1f}")

    # 3. Read one grid point from all 20 messages concurrently
    row, col = 50, 100
    offset = row * 200 + col
    values = await asyncio.gather(
        *[f.file_decode_range(i, 0, [(offset, 1)], join=True) for i in range(20)]
    )
    print(f"\nGrid point [{row},{col}] across 20 messages:")
    for i, arr in enumerate(values):
        print(f"  message {i}: value={float(arr[0]):.1f}")

    # 6. Async context manager and iteration
    async with await tensogram.AsyncTensogramFile.open(path) as f2:
        count = 0
        async for _meta, _objects in f2:
            count += 1
        print(f"\nAsync iteration: {count} messages")


if __name__ == "__main__":
    asyncio.run(main())
