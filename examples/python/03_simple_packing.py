"""
Example 03 — Simple packing (Python)

NOTE: Requires tensogram Python bindings (not yet implemented).
      This file documents the intended API.

simple_packing quantises float64 values into N-bit integers (GRIB-style),
delivering 4-8x payload size reduction with precision loss typically well
below instrument noise.
"""

import numpy as np
import tensogram
import tensogram.simple_packing as sp

# ── 1. Source data ────────────────────────────────────────────────────────────
n = 1000
temps = np.linspace(249.15, 349.05, n, dtype=np.float64)
print(f"Source: {n} float64 values  raw={temps.nbytes} bytes")
print(f"  range: [{temps.min():.2f}, {temps.max():.2f}]")

# ── 2a. High-level API: request packing in metadata ───────────────────────────
#
# The simplest path: put bits_per_value in the payload descriptor.
# The library computes reference_value / binary_scale_factor from the data.
metadata_hl = tensogram.Metadata(
    version=1,
    objects=[
        tensogram.ObjectDescriptor(
            type="ntensor", shape=[n], dtype="float64", byte_order="big"
        )
    ],
    payload=[
        tensogram.PayloadDescriptor(
            encoding="simple_packing",
            bits_per_value=16,
            decimal_scale_factor=0,
        )
    ]
)

msg_hl = tensogram.encode(metadata_hl, temps)
print(f"\nHigh-level API message: {len(msg_hl)} bytes")

_, [decoded_hl] = tensogram.decode(msg_hl)
max_err = np.abs(temps - decoded_hl).max()
print(f"Max error: {max_err:.6f} K")
assert max_err < 0.01

# ── 2b. Low-level API: compute params explicitly ───────────────────────────────
#
# Use this when you need to inspect or store the packing parameters,
# or when you want to use the same params for many fields.
params = sp.compute_params(temps, bits_per_value=16, decimal_scale_factor=0)

print(f"\nLow-level params:")
print(f"  reference_value      = {params.reference_value:.6f}")
print(f"  binary_scale_factor  = {params.binary_scale_factor}")
print(f"  decimal_scale_factor = {params.decimal_scale_factor}")
print(f"  bits_per_value       = {params.bits_per_value}")

# Encode the packed bytes directly
packed: bytes = sp.encode(temps, params)
print(f"  packed: {len(packed)} bytes  (~{temps.nbytes / len(packed):.1f}x smaller)")

# Decode them back
decoded_ll: np.ndarray = sp.decode(packed, n, params)
assert decoded_ll.dtype == np.float64
assert len(decoded_ll) == n
max_err = np.abs(temps - decoded_ll).max()
print(f"  max error: {max_err:.6f} K")
assert max_err < 0.01

# ── 3. NaN rejection ──────────────────────────────────────────────────────────
#
# simple_packing cannot represent NaN. compute_params() raises ValueError.
temps_with_nan = temps.copy()
temps_with_nan[42] = np.nan

try:
    sp.compute_params(temps_with_nan, bits_per_value=16)
    assert False, "expected ValueError"
except ValueError as e:
    print(f"\nNaN rejected: {e}")

# ── 4. Constant field ─────────────────────────────────────────────────────────
#
# All-same values: range = 0. Packs correctly; all values decode to reference.
constant = np.full(100, 273.15, dtype=np.float64)
params_c = sp.compute_params(constant, bits_per_value=16)
decoded_c = sp.decode(sp.encode(constant, params_c), 100, params_c)
np.testing.assert_allclose(decoded_c, 273.15, atol=1e-9)
print("Constant field: OK")
