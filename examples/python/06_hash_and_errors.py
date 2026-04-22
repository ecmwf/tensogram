# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Example 06 — Integrity and error handling (wire-format v3)

Every frame carries an inline xxh3-64 hash slot.  When the preamble's
``HASHES_PRESENT`` flag is set (the default), those slots are populated
at encode time and ``tensogram.validate(..., level='default')``
recomputes the per-frame body hashes and compares them to the inline
values.

Importantly, ``tensogram.decode()`` is **not** the integrity surface in
v3.  ``verify_hash=True`` is accepted for API compatibility but no
longer performs integrity checking — use ``tensogram.validate()``
instead.  See ``plans/WIRE_FORMAT.md §11`` for the rationale.
"""

import numpy as np
import tensogram

# Large enough that a byte-flip near the message midpoint reliably
# lands in the data-object frame body (hashed region).
data = np.full(4096, 42, dtype=np.float32)
descriptor = {
    "type": "ntensor",
    "shape": [4096],
    "dtype": "float32",
    "encoding": "none",
    "filter": "none",
    "compression": "none",
}
metadata = {"version": 3}


# ── 1. Default encode populates the inline hash slots ────────────────────────
hashed = bytes(tensogram.encode(metadata, [(descriptor, data)], hash="xxh3"))
print(f"Hashed message:   {len(hashed)} bytes")

# ── 2. Encode with hashing turned off ────────────────────────────────────────
unhashed = bytes(tensogram.encode(metadata, [(descriptor, data)], hash=None))
print(f"Unhashed message: {len(unhashed)} bytes")

# ── 3. Decode is hash-agnostic in v3 ─────────────────────────────────────────
#
# Both messages round-trip through ``decode``.  ``verify_hash=True`` is
# accepted for API compatibility but no longer performs integrity
# checking — use ``tensogram.validate()`` for that.
tensogram.decode(hashed, verify_hash=True)
tensogram.decode(unhashed, verify_hash=True)
print("\nBoth messages decode cleanly (decode is hash-agnostic in v3).")

# ── 4. validate() at the default level checks inline hashes ─────────────────
print("\n=== validate(clean, hashed) ===")
report = tensogram.validate(hashed)
print(
    f"  issues={len(report['issues'])}  "
    f"hash_verified={report['hash_verified']}  "
    f"objects={report['object_count']}"
)
assert report["hash_verified"]
assert len(report["issues"]) == 0

print("\n=== validate(clean, unhashed) ===")
report = tensogram.validate(unhashed)
print(
    f"  issues={len(report['issues'])}  "
    f"hash_verified={report['hash_verified']} "
    f"(HASHES_PRESENT=0, nothing to verify)"
)
assert not report["hash_verified"]
# The integrity level emits a ``no_hash_available`` warning when
# ``HASHES_PRESENT=0``, not an error.  Pinning the code here documents
# the v3 contract the example is teaching.
assert any(i["code"] == "no_hash_available" for i in report["issues"]), (
    "expected no_hash_available warning on unhashed clean message"
)

# ── 5. Corruption detection via validate() ──────────────────────────────────
#
# Flip one byte near the middle of the message.  With a 16 KiB payload
# the midpoint is deep inside the data-object frame body, so the
# recomputed xxh3 disagrees with the inline slot and validate reports
# a ``hash_mismatch`` error at level ``integrity``.  Had the flip
# landed in a header or CBOR region instead, validate would still
# flag it — as a structural or metadata issue at level 1 or 2.
corrupted = bytearray(hashed)
mid = len(corrupted) // 2
corrupted[mid] ^= 0xFF

print(f"\n=== validate(corrupted at byte {mid}) ===")
report = tensogram.validate(bytes(corrupted))
for issue in report["issues"]:
    print(f"  [{issue['severity']}/{issue['level']}] {issue['code']}: {issue['description']}")
# At 16 KiB the message midpoint lands deep inside the single data
# object frame body, so the recomputed xxh3 must disagree with the
# inline slot.  Assert the exact v3 contract this example teaches.
assert any(i["code"] == "hash_mismatch" for i in report["issues"]), (
    "expected hash_mismatch on a payload-byte flip"
)
print("  -> inline xxh3 slot disagreed with recomputed body hash.")

# ── 6. Error handling for malformed input ───────────────────────────────────
#
# ``decode`` still raises on frankly malformed buffers (truncated
# preamble, garbage magic, object index out of range, ...).  These are
# not integrity failures but structural ones, and they continue to
# surface as ``ValueError`` from the Python API.

print("\n=== Error handling ===")
for label, call in [
    ("garbage input", lambda: tensogram.decode(b"GARBAGE!")),
    ("empty input", lambda: tensogram.decode(b"")),
    ("out-of-range object", lambda: tensogram.decode_object(hashed, index=99)),
]:
    try:
        call()
    except ValueError as e:
        print(f"  {label:22s}: ValueError raised ({type(e).__name__})")
    else:
        raise AssertionError(f"{label}: expected ValueError, got clean return")

nan_data = np.array([1.0, float("nan"), 3.0])
try:
    tensogram.compute_packing_params(nan_data, bits_per_value=16, decimal_scale_factor=0)
except ValueError:
    print(f"  {'NaN in packing params':22s}: ValueError raised")
else:
    raise AssertionError("expected ValueError on NaN input")

print("\nAll checks passed.")
