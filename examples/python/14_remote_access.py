"""Example 14 — Remote Access (Python)

Opens a tensogram file over HTTP using the remote backend.  A local
HTTP server with Range-request support is started in a background
thread so the example is fully self-contained.

NOTE: Requires building tensogram-python first:
    cd crates/tensogram-python && maturin develop
"""

from __future__ import annotations

import http.server
import threading

import numpy as np
import tensogram

# ── Minimal Range-capable HTTP server ─────────────────────────────────────────


def _make_range_handler(file_data: bytes):
    """Create an HTTP handler that serves *file_data* with Range support."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(file_data)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self):
            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                spec = range_header[6:]
                if spec.startswith("-"):
                    n = int(spec[1:])
                    start = max(0, len(file_data) - n)
                    end = len(file_data)
                else:
                    parts = spec.split("-")
                    start = int(parts[0])
                    end = int(parts[1]) + 1 if parts[1] else len(file_data)
                    end = min(end, len(file_data))
                if start >= len(file_data) or start >= end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{len(file_data)}")
                    self.end_headers()
                    return
                chunk = file_data[start:end]
                self.send_response(206)
                self.send_header(
                    "Content-Range", f"bytes {start}-{end - 1}/{len(file_data)}"
                )
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(file_data)))
                self.end_headers()
                self.wfile.write(file_data)

    return Handler


# ── Create a sample TGM file ─────────────────────────────────────────────────

meta = {
    "version": 2,
    "base": [
        {"mars": {"param": "2t", "step": 0, "date": "20260401"}},
        {"mars": {"param": "msl", "step": 0, "date": "20260401"}},
    ],
}
descriptors = [
    {
        "type": "ntensor",
        "shape": [72, 144],
        "dtype": "float32",
        "byte_order": "little",
        "encoding": "none",
        "filter": "none",
        "compression": "none",
    },
    {
        "type": "ntensor",
        "shape": [72, 144],
        "dtype": "float32",
        "byte_order": "little",
        "encoding": "none",
        "filter": "none",
        "compression": "none",
    },
]
rng = np.random.default_rng(42)
temperature = rng.standard_normal((72, 144)).astype(np.float32) + 293.15
pressure = rng.standard_normal((72, 144)).astype(np.float32) * 100 + 101325

tgm_bytes = tensogram.encode(
    meta, [(descriptors[0], temperature), (descriptors[1], pressure)]
)
print(f"Encoded {len(tgm_bytes)} bytes with 2 objects")

# ── Start local HTTP server ──────────────────────────────────────────────────

handler = _make_range_handler(tgm_bytes)
server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
port = server.server_address[1]
url = f"http://127.0.0.1:{port}/forecast.tgm"
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()

print(f"Serving at {url}")

# ── URL detection ────────────────────────────────────────────────────────────

assert tensogram.is_remote_url(url)
assert not tensogram.is_remote_url("/tmp/local.tgm")
print(f"is_remote_url({url!r}) = True")

# ── Open remote file ─────────────────────────────────────────────────────────

f = tensogram.TensogramFile.open_remote(url, {})
print(f"\nOpened remote: source={f.source()}")
print(f"  is_remote = {f.is_remote()}")
print(f"  messages  = {f.message_count()}")

# ── Decode metadata (only fetches header, not full payload) ──────────────────

meta_obj = f.file_decode_metadata(0)
print(f"\nMetadata: version={meta_obj.version}")
print(f"  base[0] = {meta_obj.base[0]}")

# ── Decode descriptors (only fetches CBOR descriptor frames) ─────────────────

result = f.file_decode_descriptors(0)
descs = result["descriptors"]
print(f"\nDescriptors: {len(descs)} objects")
for i, d in enumerate(descs):
    print(f"  [{i}] shape={d.shape}  dtype={d.dtype}")

# ── Decode a single object (fetches only that object's frame) ────────────────

result = f.file_decode_object(0, 0)
arr = result["data"]
print(f"\nObject 0: shape={arr.shape}  dtype={arr.dtype}  mean={arr.mean():.2f}")
np.testing.assert_array_equal(arr, temperature)
print("  matches original temperature data")

result = f.file_decode_object(0, 1)
arr = result["data"]
print(f"Object 1: shape={arr.shape}  dtype={arr.dtype}  mean={arr.mean():.2f}")
np.testing.assert_array_equal(arr, pressure)
print("  matches original pressure data")

# ── Clean up ─────────────────────────────────────────────────────────────────

server.shutdown()
server.server_close()
thread.join(timeout=5)
print("\nRemote access example complete.")
