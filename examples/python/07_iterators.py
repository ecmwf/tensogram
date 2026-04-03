"""Example 07 — Iterator APIs (Python)

Tensogram Python bindings provide iterator protocols for ergonomic traversal:

    for msg in tensogram.TensogramFile.open("data.tgm"):
        for tensor in msg:
            value = tensor[i, j, k]

This example demonstrates:
  1. File iteration — for msg in file: ...
  2. Object iteration — for tensor in msg: ...
  3. N-dimensional indexing — tensor[i, j, k]
  4. Buffer iteration — iter_messages(buf)

NOTE: Requires building tensogram-python first:
    cd crates/tensogram-python && maturin develop
"""

import tempfile
import os
import numpy as np

# Uncomment after building with maturin:
# import tensogram

def main():
    """
    The iterator API is designed for this natural pattern:

        f = tensogram.TensogramFile.open("forecast.tgm")
        for msg in f:                     # yields Message objects
            print(msg.metadata)           # lazy metadata decode
            for tensor in msg:            # yields Tensor objects
                print(tensor.shape)       # shape, dtype, ndim
                val = tensor[0, 1, 2]     # N-d indexing → scalar

    Buffer-based iteration works similarly:

        buf = open("data.tgm", "rb").read()
        for msg in tensogram.iter_messages(buf):
            for tensor in msg:
                data = tensor.to_numpy()  # full numpy array
    """

    print("=== Python iterator API design ===")
    print()
    print("File iteration:")
    print('  f = tensogram.TensogramFile.open("data.tgm")')
    print("  for msg in f:")
    print("      print(len(msg))          # number of objects")
    print("      for tensor in msg:")
    print("          print(tensor.shape)  # e.g. (721, 1440)")
    print("          print(tensor.dtype)  # e.g. float32")
    print("          val = tensor[0, 0]   # scalar indexing")
    print()
    print("Buffer iteration:")
    print("  for msg in tensogram.iter_messages(buf):")
    print("      for tensor in msg:")
    print("          arr = tensor.to_numpy()")
    print()

    # The following shows what the API will look like once
    # tensogram-python is built with `maturin develop`:
    #
    # import tensogram
    #
    # # Encode a few test messages
    # messages = []
    # for i in range(3):
    #     meta = {
    #         "version": 1,
    #         "objects": [{
    #             "type": "ntensor",
    #             "ndim": 2,
    #             "shape": [4, 4],
    #             "strides": [4, 1],
    #             "dtype": "float32",
    #         }],
    #         "payload": [{
    #             "byte_order": "little",
    #             "encoding": "none",
    #             "filter": "none",
    #             "compression": "none",
    #         }],
    #     }
    #     data = np.zeros((4, 4), dtype=np.float32)
    #     messages.append(tensogram.encode(meta, [data.tobytes()]))
    #
    # # Write to a temp file
    # with tempfile.NamedTemporaryFile(suffix=".tgm", delete=False) as tmp:
    #     for msg in messages:
    #         tmp.write(msg)
    #     path = tmp.name
    #
    # # Iterate over file
    # f = tensogram.TensogramFile.open(path)
    # for msg in f:
    #     print(f"  Message with {len(msg)} objects")
    #     for tensor in msg:
    #         print(f"    shape={tensor.shape}  dtype={tensor.dtype}")
    #         print(f"    tensor[0,0] = {tensor[0, 0]}")
    #
    # # Iterate over buffer
    # buf = open(path, "rb").read()
    # for msg in tensogram.iter_messages(buf):
    #     for tensor in msg:
    #         arr = tensor.to_numpy()
    #         print(f"    array shape: {arr.shape}")
    #
    # os.unlink(path)

    print("Iterator example complete.")
    print("Build with: cd crates/tensogram-python && maturin develop")


if __name__ == "__main__":
    main()
