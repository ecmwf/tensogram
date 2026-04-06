"""Example 09 — Streaming consumer: decode tensogram from an HTTP stream

Demonstrates consumer-side streaming: downloading a .tgm file over HTTP
and decoding data objects as they arrive, without buffering the entire
message in memory.

The example:
1. Creates a multi-object .tgm file using the streaming encoder
2. Serves it via a local HTTP server (mock)
3. Downloads it in small chunks, scanning for complete messages
4. Decodes each message's objects into xarray Datasets on-the-fly

This proves that the wire format supports progressive decoding: the
PRECEDER METADATA FRAME arrives before each data object, so the consumer
knows the shape/dtype before the payload bytes arrive.

Requirements:
    pip install tensogram xarray numpy

NOTE: Requires building tensogram-python first:
    cd crates/tensogram-python && maturin develop
"""

import http.server
import pathlib
import tempfile
import threading

import numpy as np
import tensogram

# Check xarray availability (optional for this example)
try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    print("xarray not installed — will print arrays instead of building Datasets")


# ── 1. Create a multi-object .tgm file ───────────────────────────────────────
#
# We create a file with 4 "forecast fields" — each a separate data object.
# In a real scenario this would be a forecast model output file.

PARAMS = [
    ("2t", "2m temperature", (181, 360)),
    ("10u", "10m U wind", (181, 360)),
    ("10v", "10m V wind", (181, 360)),
    ("msl", "mean sea level pressure", (181, 360)),
]


def create_streaming_tgm(path: str) -> int:
    """Create a .tgm file with multiple objects, return byte count."""
    with tensogram.TensogramFile.create(path) as f:
        for param, long_name, shape in PARAMS:
            meta = {
                "version": 2,
                "base": [
                    {
                        "mars": {
                            "class": "od",
                            "date": "20260401",
                            "step": 0,
                            "type": "fc",
                            "param": param,
                        },
                        "long_name": long_name,
                    },
                ],
            }
            desc = {
                "type": "ntensor",
                "shape": list(shape),
                "dtype": "float32",
                "byte_order": "little",
                "encoding": "none",
                "filter": "none",
                "compression": "none",
            }
            # Use zlib.crc32 for a stable seed (Python hash() is randomized)
            import zlib

            seed = zlib.crc32(param.encode()) & 0xFFFFFFFF
            data = np.random.default_rng(seed).random(shape, dtype=np.float32)
            f.append(meta, [(desc, data)])
    return pathlib.Path(path).stat().st_size


# ── 2. Mock HTTP server ──────────────────────────────────────────────────────


class TGMHandler(http.server.BaseHTTPRequestHandler):
    """Serve a .tgm file with small chunk sizes to simulate slow network."""

    tgm_path: str = ""

    def do_GET(self):
        tgm_path = pathlib.Path(self.tgm_path)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(tgm_path.stat().st_size))
        self.end_headers()
        # Stream from disk in small chunks — no full-file buffer
        chunk_size = 4096
        with tgm_path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def log_message(self, *_args):
        pass  # suppress server logs


# ── 3. Streaming consumer ────────────────────────────────────────────────────


def consume_stream(url: str):
    """Download from URL and decode messages as they arrive.

    Strategy: accumulate bytes in a rolling buffer, scan for complete
    messages, decode each one immediately, and discard consumed bytes.
    Peak memory = one complete message (not the whole file).
    """
    import urllib.request

    buffer = bytearray()
    messages_decoded = 0
    datasets = []

    print(f"\nStreaming from {url}")
    print(f"{'─' * 60}")

    with urllib.request.urlopen(url) as response:
        while True:
            chunk = response.read(4096)
            if not chunk:
                break
            buffer.extend(chunk)

            # Take a single bytes snapshot to avoid repeated copies
            buf_snapshot = bytes(buffer)

            # Scan for complete messages in the snapshot
            entries = tensogram.scan(buf_snapshot)

            # Decode each complete message found
            for offset, length in entries:
                msg_bytes = buf_snapshot[offset : offset + length]
                msg = tensogram.decode(msg_bytes)

                messages_decoded += 1
                _desc, arr = msg.objects[0]

                param = "unknown"
                if msg.metadata.base:
                    mars = msg.metadata.base[0].get("mars", {})
                    if isinstance(mars, dict):
                        param = mars.get("param", "unknown")

                print(
                    f"  Message {messages_decoded}: "
                    f"param={param:<5} shape={arr.shape} "
                    f"dtype={arr.dtype} "
                    f"range=[{arr.min():.3f}, {arr.max():.3f}]"
                )

                # Build xarray Dataset if available
                if HAS_XARRAY:
                    ds = xr.Dataset({param: xr.DataArray(arr, dims=["latitude", "longitude"])})
                    datasets.append(ds)

            # Discard consumed bytes (everything up to end of last complete message)
            if entries:
                last_offset, last_length = entries[-1]
                consumed = last_offset + last_length
                buffer = buffer[consumed:]

    # Process any remaining bytes in buffer
    if buffer:
        entries = tensogram.scan(bytes(buffer))
        for offset, length in entries:
            msg_bytes = bytes(buffer[offset : offset + length])
            msg = tensogram.decode(msg_bytes)
            messages_decoded += 1
            _desc, arr = msg.objects[0]
            print(f"  Message {messages_decoded}: shape={arr.shape} (trailing)")

    print(f"{'─' * 60}")
    print(f"Total: {messages_decoded} messages decoded from stream")
    print(f"Peak buffer: single message (not whole file)")

    if HAS_XARRAY and datasets:
        merged = xr.merge(datasets)
        print(f"\nMerged xarray Dataset:")
        print(f"  Variables: {list(merged.data_vars)}")
        print(f"  Dims: {dict(merged.sizes)}")

    return messages_decoded


# ── 4. Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        tgm_path = str(pathlib.Path(tmpdir) / "forecast.tgm")

        # Create test file
        file_size = create_streaming_tgm(tgm_path)
        print(f"Created {tgm_path}: {file_size:,} bytes, {len(PARAMS)} objects")

        # Start mock HTTP server
        TGMHandler.tgm_path = tgm_path
        server = http.server.HTTPServer(("127.0.0.1", 0), TGMHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            # Stream and decode
            url = f"http://127.0.0.1:{port}/forecast.tgm"
            count = consume_stream(url)
            assert count == len(PARAMS), f"expected {len(PARAMS)}, got {count}"
        finally:
            server.shutdown()

    print("\nStreaming consumer example complete.")
