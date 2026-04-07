"""Cross-language SHA256 golden test helper for encode_pre_encoded().

Generates the SAME deterministic float64[1024] payload as the Rust driver,
encodes via tensogram.encode_pre_encoded(), decodes, and prints the SHA-256
hex digest of the decoded payload to stdout.

The Rust test driver spawns this script and compares the output.
"""

import hashlib
import sys

import numpy as np
import tensogram


def main() -> None:
    # Same deterministic input as the Rust driver.
    values = np.array([200.0 + i * 0.125 for i in range(1024)], dtype=np.float64)
    raw_bytes = values.astype("<f8", copy=False).tobytes()  # explicit little-endian

    global_meta = {"version": 2}
    descriptor = {
        "type": "ndarray",
        "ndim": 1,
        "shape": [1024],
        "strides": [8],
        "dtype": "float64",
        "byte_order": "little",
        "encoding": "none",
        "filter": "none",
        "compression": "none",
    }

    wire = tensogram.encode_pre_encoded(global_meta, [(descriptor, raw_bytes)])

    # Decode and extract payload.
    msg = tensogram.decode(wire)
    assert len(msg.objects) == 1, f"Expected 1 object, got {len(msg.objects)}"
    _desc, payload_array = msg.objects[0]
    # payload_array is a numpy array; get its raw bytes.
    payload_bytes = payload_array.tobytes()

    sha = hashlib.sha256(payload_bytes).hexdigest()
    sys.stdout.write(sha)


if __name__ == "__main__":
    main()
