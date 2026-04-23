# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Example 07 — Iterators and Indexing

decode(), decode_message(), file iteration, and iter_messages() return Message
namedtuples with .metadata and .objects:

    with tensogram.TensogramFile.open("data.tgm") as f:
        msg = f[0]
        msg.metadata          # Metadata object
        msg.objects           # list[(DataObjectDescriptor, ndarray)]
        meta, objects = msg   # tuple unpacking also works

TensogramFile supports standard Python iteration and indexing:

    for msg in file:                  # iterate all messages
    msg = file[i]                     # single message by index
    msg = file[-1]                    # negative indexing
    subset = file[10:20]              # slice a range
    subset = file[::5]               # every 5th message

For raw byte buffers (pipes, network streams):

    for msg in tensogram.iter_messages(buf):
        ...

This example also shows multi-object messages (multiple arrays per message).

NOTE: Requires building tensogram-python first:
    cd crates/tensogram-python && maturin develop
"""

import os
import tempfile

import numpy as np
import tensogram


def create_test_file(path, n=5):
    """Write n messages, each with a 4x4 grid, to a .tgm file."""
    with tensogram.TensogramFile.create(path) as f:
        for i in range(n):
            data = np.full((4, 4), float(i), dtype=np.float32)
            # Free-form CBOR metadata: `step` / `time` are just
            # caller-supplied annotations that flow into `_extra_`.
            meta = {"step": i, "time": i * 0.1}
            desc = {"type": "ntensor", "shape": [4, 4], "dtype": "float32"}
            f.append(meta, [(desc, data)])
    print(f"Created {path} with {n} messages")


def demo_message_access(path):
    """Show both attribute access and tuple unpacking."""
    print("\n--- Message namedtuple ---")
    with tensogram.TensogramFile.open(path) as f:
        # Attribute access
        msg = f[0]
        print(f"  msg.metadata['step'] = {msg.metadata['step']}")
        print(f"  msg.objects[0] shape = {msg.objects[0][1].shape}")

        # Tuple unpacking (same data, different style)
        meta, objects = f[0]
        print(f"  unpacked: step={meta['step']}, shape={objects[0][1].shape}")


def demo_iteration(path):
    """Iterate over all messages with a for loop."""
    print("\n--- for meta, objects in file ---")
    with tensogram.TensogramFile.open(path) as f:
        for meta, objects in f:
            desc, arr = objects[0]
            print(
                f"  step={meta['step']}, shape={arr.shape}, "
                f"dtype={desc.dtype}, val={arr[0, 0]:.1f}"
            )


def demo_indexing(path):
    """Access messages by index and slice."""
    print("\n--- file[i] and file[start:stop:step] ---")
    with tensogram.TensogramFile.open(path) as f:
        # Single index
        meta, _ = f[0]
        print(f"  f[0]:    step={meta['step']}")
        meta, _ = f[-1]
        print(f"  f[-1]:   step={meta['step']}")

        # Slicing
        steps = [m["step"] for m, _ in f[1:4]]
        print(f"  f[1:4]:  steps={steps}")
        steps = [m["step"] for m, _ in f[::2]]
        print(f"  f[::2]:  steps={steps}")
        steps = [m["step"] for m, _ in f[::-1]]
        print(f"  f[::-1]: steps={steps}")


def demo_multi_object(path):
    """Iterate over objects within a single multi-object message."""
    print("\n--- multi-object messages ---")
    with tensogram.TensogramFile.create(path) as f:
        temperature = np.random.rand(3, 3).astype(np.float32)
        humidity = np.random.rand(3, 3).astype(np.float32)
        meta = {"source": "sensor"}
        f.append(
            meta,
            [
                ({"type": "ntensor", "shape": [3, 3], "dtype": "float32"}, temperature),
                ({"type": "ntensor", "shape": [3, 3], "dtype": "float32"}, humidity),
            ],
        )

    with tensogram.TensogramFile.open(path) as f:
        meta, objects = f[0]
        print(f"  message has {len(objects)} objects:")
        for i, (_desc, arr) in enumerate(objects):
            print(f"    object {i}: shape={arr.shape}, mean={arr.mean():.4f}")


def demo_buffer_iteration():
    """Iterate over messages in raw bytes using iter_messages()."""
    print("\n--- tensogram.iter_messages(buf) ---")

    # Encode three messages into raw bytes
    msgs = []
    for i in range(3):
        data = np.full(8, float(i), dtype=np.float32)
        meta = {"step": i}
        desc = {"type": "ntensor", "shape": [8], "dtype": "float32"}
        msgs.append(bytes(tensogram.encode(meta, [(desc, data)])))

    buf = b"".join(msgs)
    print(f"  buffer: {len(buf)} bytes, {len(msgs)} messages")

    for msg in tensogram.iter_messages(buf):
        _, arr = msg.objects[0]
        print(f"    step={msg.metadata['step']}, arr[:3]={arr[:3]}")


def demo_len_and_iter(path):
    """Use len() and manual iterator control."""
    print("\n--- len() and iter() ---")
    with tensogram.TensogramFile.open(path) as f:
        print(f"  len(f) = {len(f)}")
        it = iter(f)
        print(f"  len(iter) = {len(it)}")
        meta, _ = next(it)
        print(f"  next(it): step={meta['step']}, remaining={len(it)}")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.tgm")
        create_test_file(path, n=5)
        demo_message_access(path)
        demo_iteration(path)
        demo_indexing(path)
        demo_multi_object(os.path.join(tmpdir, "multi.tgm"))
        demo_buffer_iteration()
        demo_len_and_iter(path)

    print("\nDone.")


if __name__ == "__main__":
    main()
