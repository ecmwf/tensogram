# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Example 11 — Pre-encoded payloads (Python)

Demonstrates tensogram.encode_pre_encoded(), which lets callers hand
already-encoded payload bytes to the library without re-running the
encoding pipeline.

This is useful for GPU pipelines, HPC frameworks, or any system that
produces encoded data outside the library — the caller packs the data
themselves and Tensogram wraps it into the wire format.

IMPORTANT — bit-vs-byte gotcha:
  szip_block_offsets in the descriptor are BIT offsets, not byte offsets.
  For example, if a compressed block starts at byte 16, the offset is 128
  (= 16 * 8). Getting this wrong silently breaks decode_range().

NOTE: Requires building tensogram-python first:
    cd crates/tensogram-python && maturin develop
"""

import numpy as np
import tensogram

# ── 1. Synthesise pre-encoded payload (simple_packing) ──────────────────────

n = 1000
temps = np.linspace(249.15, 349.05, n, dtype=np.float64)
print(f"Source: {n} float64 values  raw={temps.nbytes} bytes")

# Compute packing parameters via the library helper
params = tensogram.compute_packing_params(temps, bits_per_value=16, decimal_scale_factor=0)
ref_val = params["reference_value"]
bsf = params["binary_scale_factor"]
dsf = params["decimal_scale_factor"]
bpv = params["bits_per_value"]

print(f"Packing params: ref={ref_val:.4f}  bsf={bsf}  dsf={dsf}  bpv={bpv}")

# Manual simple packing in Python:
#   packed_int = round((value - ref_val) * 10^dsf / 2^bsf)
#   then bit-pack into a byte stream
scale = (10.0**dsf) / (2.0**bsf)
packed_ints = np.round((temps - ref_val) * scale).astype(np.uint64)


def bit_pack(values: np.ndarray, bits_per_value: int) -> bytes:
    """Pack integers into a bit stream (big-endian bit order, MSB first)."""
    total_bits = len(values) * bits_per_value
    total_bytes = (total_bits + 7) // 8
    buf = bytearray(total_bytes)

    bit_pos = 0
    for val in values:
        remaining = bits_per_value
        v = int(val)
        while remaining > 0:
            byte_idx = bit_pos // 8
            bit_offset = bit_pos % 8
            space = 8 - bit_offset
            write_bits = min(remaining, space)
            shift = remaining - write_bits
            bits = (v >> shift) & ((1 << write_bits) - 1)
            buf[byte_idx] |= bits << (space - write_bits)
            bit_pos += write_bits
            remaining -= write_bits

    return bytes(buf)


packed_bytes = bit_pack(packed_ints, bpv)
print(f"Packed payload: {len(packed_bytes)} bytes  ({bpv} bits x {n} values)")

# ── 2. Build descriptor and encode ──────────────────────────────────────────

metadata = {"version": 3, "source": "pre-encoded example"}
descriptor = {
    "type": "ntensor",
    "shape": [n],
    "dtype": "float64",
    "encoding": "simple_packing",
    "filter": "none",
    "compression": "none",
    **params,
}

message = bytes(tensogram.encode_pre_encoded(metadata, [(descriptor, packed_bytes)]))
print(f"Wire message: {len(message)} bytes")

# ── 3. Decode and verify ────────────────────────────────────────────────────

msg = tensogram.decode(message)
desc_out, decoded = msg.objects[0]

max_err = float(np.abs(temps - decoded).max())
print(f"Max quantisation error: {max_err:.6f}")
assert max_err < 0.01, f"error {max_err} exceeds tolerance"
assert decoded.dtype == np.float64
assert decoded.shape == (n,)
assert msg.metadata["source"] == "pre-encoded example"

# ── 4. encoding=none variant ────────────────────────────────────────────────
#
# The simplest case: raw payload bytes, no encoding applied.

raw_data = np.arange(50, dtype=np.float32)
raw_bytes = raw_data.tobytes()

raw_desc = {
    "type": "ntensor",
    "shape": [50],
    "dtype": "float32",
    "encoding": "none",
    "filter": "none",
    "compression": "none",
}

msg_raw = bytes(tensogram.encode_pre_encoded({"version": 3}, [(raw_desc, raw_bytes)]))
_, objs_raw = tensogram.decode(msg_raw)
_, decoded_raw = objs_raw[0]
np.testing.assert_array_equal(decoded_raw, raw_data)
print("Raw encoding=none round-trip: OK")

# ── 5. StreamingEncoder variant ─────────────────────────────────────────────
#
# The streaming encoder also supports write_object_pre_encoded().

enc = tensogram.StreamingEncoder({"version": 3})
enc.write_object_pre_encoded(descriptor, packed_bytes)
enc.write_object_pre_encoded(raw_desc, raw_bytes)
stream_msg = bytes(enc.finish())

_, stream_objs = tensogram.decode(stream_msg)
assert len(stream_objs) == 2

_, s0 = stream_objs[0]
assert np.abs(temps - s0).max() < 0.01

_, s1 = stream_objs[1]
np.testing.assert_array_equal(s1, raw_data)
print("Streaming pre-encoded: OK")

# ── Done ────────────────────────────────────────────────────────────────────
print("\nOK: all pre-encoded round-trips succeeded")
