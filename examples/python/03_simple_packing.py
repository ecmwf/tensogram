# (C) Copyright 2024- ECMWF and individual contributors.
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

NOTE: Requires building tensogram-python first:
    cd crates/tensogram-python && maturin develop
"""

import numpy as np
import tensogram

# ── 1. Source data ────────────────────────────────────────────────────────────
n = 1000
temps = np.linspace(249.15, 349.05, n, dtype=np.float64)
print(f"Source: {n} float64 values  raw={temps.nbytes} bytes")
print(f"  range: [{temps.min():.2f}, {temps.max():.2f}]")

# ── 2. Compute packing parameters ────────────────────────────────────────────
#
# compute_packing_params() determines the reference_value and scale factors
# needed to quantise the data into the requested number of bits.
params = tensogram.compute_packing_params(temps, bits_per_value=16, decimal_scale_factor=0)

print(f"\nPacking params (16 bpv):")
print(f"  reference_value      = {params['reference_value']:.6f}")
print(f"  binary_scale_factor  = {params['binary_scale_factor']}")
print(f"  decimal_scale_factor = {params['decimal_scale_factor']}")
print(f"  bits_per_value       = {params['bits_per_value']}")

# ── 3. Encode with simple_packing ────────────────────────────────────────────
metadata = {"version": 2}
descriptor = {
    "type": "ntensor",
    "shape": [n],
    "dtype": "float64",
    "byte_order": "little",
    "encoding": "simple_packing",
    "filter": "none",
    "compression": "none",
    **params,  # reference_value, binary_scale_factor, etc. as top-level keys
}

message = bytes(tensogram.encode(metadata, [(descriptor, temps)]))
print(f"\nEncoded message: {len(message)} bytes  (raw={temps.nbytes})")

# ── 4. Decode and verify ─────────────────────────────────────────────────────
msg = tensogram.decode(message)
_, decoded = msg.objects[0]
max_err = np.abs(temps - decoded).max()
print(f"Max error: {max_err:.6f} K")
assert max_err < 0.01, f"error {max_err} exceeds tolerance"

# ── 5. NaN rejection ─────────────────────────────────────────────────────────
#
# simple_packing cannot represent NaN. compute_packing_params() raises ValueError.
temps_with_nan = temps.copy()
temps_with_nan[42] = np.nan

try:
    tensogram.compute_packing_params(temps_with_nan, bits_per_value=16, decimal_scale_factor=0)
    raise AssertionError("expected ValueError")
except ValueError as e:
    print(f"\nNaN rejected: {e}")

# ── 6. Constant field ────────────────────────────────────────────────────────
#
# All-same values: range = 0. Packs correctly; all values decode to reference.
constant = np.full(100, 273.15, dtype=np.float64)
params_c = tensogram.compute_packing_params(constant, bits_per_value=16, decimal_scale_factor=0)
desc_c = {
    "type": "ntensor",
    "shape": [100],
    "dtype": "float64",
    "byte_order": "little",
    "encoding": "simple_packing",
    "filter": "none",
    "compression": "none",
    **params_c,
}
msg_c = bytes(tensogram.encode({"version": 2}, [(desc_c, constant)]))
_, decoded_c = tensogram.decode(msg_c).objects[0]
np.testing.assert_allclose(decoded_c, 273.15, atol=1e-9)
print("Constant field: OK")
