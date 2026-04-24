# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Mock HTTP server for the remote-parity harness.

Serves checked-in .tgm fixtures with Range + HEAD support and logs every
incoming request in the `ScanEvent` schema (see `schema.json`).

URL shape: ``http://127.0.0.1:<port>/<run_id>/<fixture>.tgm``.
The first path segment is the opaque run identifier used to segregate
request logs per driver invocation. The remaining segments name a file
under the configured fixture directory.

Control endpoints (underscore-prefixed so they can't collide with a
``run_id``, which must start with ``[A-Za-z0-9]``):

- ``POST /_reset`` — clears all in-memory logs.
- ``GET /_log/<run_id>`` — returns that run's log as newline-delimited JSON.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import NamedTuple
from urllib.parse import urlsplit

_RUN_ID_RE = re.compile(r"^/(?P<run_id>[A-Za-z0-9][A-Za-z0-9_-]*)/(?P<rest>.+)$")
_CONTROL_LOG_RE = re.compile(r"^/_log/(?P<run_id>[A-Za-z0-9][A-Za-z0-9_-]*)/?$")


class RequestRecord(NamedTuple):
    """One observed HTTP request.

    Category / direction / scan_round / logical_range fields from
    `schema.json` are NOT set here — the orchestrator fills them in
    after observing the log, because only the orchestrator knows which
    request belongs to which scan round for each language's protocol.
    """

    method: str
    path: str
    range_header: str | None
    status: int
    response_bytes: int


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        pass

    @property
    def server_state(self) -> _ServerState:
        return self.server.state  # type: ignore[attr-defined]

    def do_POST(self) -> None:
        if urlsplit(self.path).path == "/_reset":
            self.server_state.reset()
            self._send_empty(204)
            return
        self._send_empty(404)

    def do_HEAD(self) -> None:
        self._serve(method="HEAD")

    def do_GET(self) -> None:
        parsed_path = urlsplit(self.path).path
        log_match = _CONTROL_LOG_RE.match(parsed_path)
        if log_match:
            self._send_log(log_match.group("run_id"))
            return
        self._serve(method="GET")

    def _serve(self, *, method: str) -> None:
        parsed_path = urlsplit(self.path).path
        match = _RUN_ID_RE.match(parsed_path)
        if not match:
            self._send_empty(404)
            self._record(method, parsed_path, 404, 0)
            return

        run_id = match.group("run_id")
        fixture_path = self.server_state.fixtures_dir / match.group("rest")
        if not self._fixture_is_safe(fixture_path):
            self._send_empty(404)
            self._record(method, parsed_path, 404, 0, run_id=run_id)
            return

        try:
            data = fixture_path.read_bytes()
        except FileNotFoundError:
            self._send_empty(404)
            self._record(method, parsed_path, 404, 0, run_id=run_id)
            return

        total = len(data)
        range_header = self.headers.get("Range")

        if method == "HEAD":
            self.send_response(200)
            self.send_header("Content-Length", str(total))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self._record(method, parsed_path, 200, 0, run_id=run_id, range_header=range_header)
            return

        if range_header is None:
            self.send_response(200)
            self.send_header("Content-Length", str(total))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(data)
            self._record(method, parsed_path, 200, total, run_id=run_id, range_header=None)
            return

        parsed_range = self._parse_range(range_header, total)
        if parsed_range is None:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{total}")
            self.end_headers()
            self._record(method, parsed_path, 416, 0, run_id=run_id, range_header=range_header)
            return

        start, end_exclusive = parsed_range
        chunk = data[start:end_exclusive]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end_exclusive - 1}/{total}")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)
        self._record(
            method,
            parsed_path,
            206,
            len(chunk),
            run_id=run_id,
            range_header=range_header,
        )

    def _send_log(self, run_id: str) -> None:
        records = self.server_state.snapshot(run_id)
        body = "\n".join(json.dumps(r._asdict()) for r in records)
        encoded = body.encode("utf-8") + b"\n"
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_empty(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _fixture_is_safe(self, candidate: pathlib.Path) -> bool:
        """Reject path traversal by requiring `candidate` to resolve under fixtures_dir."""
        try:
            resolved = candidate.resolve()
            base = self.server_state.fixtures_dir.resolve()
            resolved.relative_to(base)
        except (ValueError, OSError):
            return False
        return True

    @staticmethod
    def _parse_range(header: str, total: int) -> tuple[int, int] | None:
        """Parse ``bytes=a-b`` or ``bytes=-n`` into ``[start, end_exclusive)``.

        Returns ``None`` for an unsatisfiable or malformed range.
        """
        if not header.startswith("bytes="):
            return None
        spec = header[len("bytes=") :].strip()
        if spec.startswith("-"):
            try:
                suffix_len = int(spec[1:])
            except ValueError:
                return None
            if suffix_len <= 0:
                return None
            return max(0, total - suffix_len), total

        if "-" not in spec:
            return None
        start_s, end_s = spec.split("-", 1)
        try:
            start = int(start_s)
        except ValueError:
            return None
        if start < 0 or start >= total:
            return None
        if not end_s:
            return start, total
        try:
            end_inclusive = int(end_s)
        except ValueError:
            return None
        if end_inclusive < start:
            return None
        return start, min(end_inclusive + 1, total)

    def _record(
        self,
        method: str,
        path: str,
        status: int,
        response_bytes: int,
        *,
        run_id: str | None = None,
        range_header: str | None = None,
    ) -> None:
        self.server_state.append(
            run_id,
            RequestRecord(
                method=method,
                path=path,
                range_header=range_header,
                status=status,
                response_bytes=response_bytes,
            ),
        )


class _ServerState:
    def __init__(self, fixtures_dir: pathlib.Path) -> None:
        self.fixtures_dir = fixtures_dir
        self._lock = threading.Lock()
        self._log: dict[str, list[RequestRecord]] = {}

    def append(self, run_id: str | None, record: RequestRecord) -> None:
        key = run_id or "__untagged__"
        with self._lock:
            self._log.setdefault(key, []).append(record)

    def snapshot(self, run_id: str) -> list[RequestRecord]:
        with self._lock:
            return list(self._log.get(run_id, []))

    def reset(self) -> None:
        with self._lock:
            self._log.clear()


class MockServer:
    def __init__(self, fixtures_dir: pathlib.Path, port: int = 0) -> None:
        self._fixtures_dir = fixtures_dir
        self._requested_port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        if self._httpd is None:
            raise RuntimeError("MockServer not started")
        return self._httpd.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def url_for(self, run_id: str, fixture_name: str) -> str:
        return f"{self.base_url}/{run_id}/{fixture_name}"

    def log_for(self, run_id: str) -> list[RequestRecord]:
        if self._httpd is None:
            raise RuntimeError("MockServer not started")
        return self._httpd.state.snapshot(run_id)  # type: ignore[attr-defined]

    def start(self) -> None:
        httpd = ThreadingHTTPServer(("127.0.0.1", self._requested_port), _Handler)
        httpd.state = _ServerState(self._fixtures_dir)  # type: ignore[attr-defined]
        self._httpd = httpd
        self._thread = threading.Thread(
            target=httpd.serve_forever, name="parity-mock-server", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._httpd = None
        self._thread = None

    def __enter__(self) -> MockServer:
        self.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument(
        "--fixtures-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent / "fixtures",
        help="Directory containing .tgm fixtures",
    )
    parser.add_argument("--port", type=int, default=0, help="Port (0 = auto)")
    args = parser.parse_args(argv)

    if not args.fixtures_dir.exists():
        print(
            f"error: fixtures dir {args.fixtures_dir} does not exist. "
            "Run tools/gen_fixtures.py first.",
            file=sys.stderr,
        )
        return 2

    server = MockServer(args.fixtures_dir, args.port)
    server.start()
    try:
        print(f"serving {args.fixtures_dir} on {server.base_url}")
        print("press Ctrl-C to stop")
        threading.Event().wait()
    except KeyboardInterrupt:
        # Ctrl-C is the documented way to stop this stand-alone server;
        # swallow the interrupt so the user gets a clean exit. Cleanup
        # happens in the `finally` block below.
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
