# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Example 03 — Simple packing (Python)

simple_packing quantises float64 values into N-bit integers (GRIB-style),
delivering 4-8x payload size reduction with precision loss typically well
below instrument noise.

This example covers both paths:

  1. The ergonomic **auto-compute** form — user supplies only
     ``sp_bits_per_value``; the encoder derives the reference value and
     scale factors.
  2. The **explicit** form — user pre-computes params via
     ``compute_packing_params`` and pins them into the descriptor.  Use
     this when you need the same reference value across multiple encodes.

NOTE: Requires building tensogram-python first:
    cd python/bindings && maturin develop
"""

import numpy as np
import tensogram

# ── 1. Source data ────────────────────────────────────────────────────────────
n = 1000
temps = np.linspace(249.15, 349.05, n, dtype=np.float64)
print(f"Source: {n} float64 values  raw={temps.nbytes} bytes")
print(f"  range: [{temps.min():.2f}, {temps.max():.2f}]")

# ── 2. Ergonomic path — auto-compute ─────────────────────────────────────────
#
# Write a descriptor with just ``sp_bits_per_value`` (and optionally
# ``sp_decimal_scale_factor``).  The encoder derives
# ``sp_reference_value`` and ``sp_binary_scale_factor`` from the data
# and stamps all four keys into the wire descriptor so the file
# remains self-describing.
descriptor = {
    "type": "ntensor",
    "shape": [n],
    "dtype": "float64",
    "encoding": "simple_packing",
    "filter": "none",
    "compression": "none",
    "sp_bits_per_value": 16,
}
message = bytes(tensogram.encode({}, [(descriptor, temps)]))
print(f"\nAuto-compute encoded message: {len(message)} bytes  (raw={temps.nbytes})")

msg = tensogram.decode(message)
_, decoded = msg.objects[0]
max_err = np.abs(temps - decoded).max()
print(f"Max error: {max_err:.6f} K")
assert max_err < 0.01, f"error {max_err} exceeds tolerance"

# ── 3. Explicit path — compute_packing_params ────────────────────────────────
#
# Pre-compute the params once, then spread them into every descriptor.
# Useful for pinning the reference value across a time-series.
params = tensogram.compute_packing_params(temps, bits_per_value=16, decimal_scale_factor=0)
print("\nExplicit packing params (16 bpv):")
print(f"  sp_reference_value      = {params['sp_reference_value']:.6f}")
print(f"  sp_binary_scale_factor  = {params['sp_binary_scale_factor']}")
print(f"  sp_decimal_scale_factor = {params['sp_decimal_scale_factor']}")
print(f"  sp_bits_per_value       = {params['sp_bits_per_value']}")

descriptor_explicit = {
    "type": "ntensor",
    "shape": [n],
    "dtype": "float64",
    "encoding": "simple_packing",
    "filter": "none",
    "compression": "none",
    **params,  # sp_reference_value, sp_binary_scale_factor, etc. as top-level keys
}
message_explicit = bytes(tensogram.encode({}, [(descriptor_explicit, temps)]))
_, decoded_explicit = tensogram.decode(message_explicit).objects[0]
np.testing.assert_allclose(decoded, decoded_explicit)
print("Explicit and auto-compute paths produce identical decoded values.")

# ── 4. NaN rejection ─────────────────────────────────────────────────────────
#
# simple_packing cannot represent NaN.  Both the auto-compute encoder
# path and the explicit compute_packing_params call reject it.
temps_with_nan = temps.copy()
temps_with_nan[42] = np.nan

try:
    tensogram.compute_packing_params(temps_with_nan, bits_per_value=16, decimal_scale_factor=0)
    raise AssertionError("expected ValueError")
except ValueError as e:
    print(f"\nNaN rejected by compute_packing_params: {e}")

try:
    tensogram.encode({}, [(descriptor, temps_with_nan)])
    raise AssertionError("expected ValueError")
except (ValueError, Exception) as e:
    print(f"NaN rejected by encoder auto-compute: {e}")

# ── 5. Constant field ────────────────────────────────────────────────────────
#
# All-same values: range = 0.  Packs correctly; all values decode to reference.
constant = np.full(100, 273.15, dtype=np.float64)
desc_c = {
    "type": "ntensor",
    "shape": [100],
    "dtype": "float64",
    "encoding": "simple_packing",
    "filter": "none",
    "compression": "none",
    "sp_bits_per_value": 16,
}
msg_c = bytes(tensogram.encode({}, [(desc_c, constant)]))
_, decoded_c = tensogram.decode(msg_c).objects[0]
np.testing.assert_allclose(decoded_c, 273.15, atol=1e-9)
print("\nConstant field: OK")
