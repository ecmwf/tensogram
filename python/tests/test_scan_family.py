# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for the scan family: scan_file, scan_with_options,
scan_file_with_options, and the ScanOptions class."""

# pyright: basic, reportAttributeAccessIssue=false, reportMissingTypeStubs=false

import numpy as np
import pytest
import tensogram

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(index: int, fill: float, n: int = 6) -> bytes:
    data = np.full(n, fill, dtype=np.float32)
    desc = {"type": "ntensor", "shape": [n], "dtype": "float32"}
    return tensogram.encode({"base": [{"index": index}]}, [(desc, data)])


def _multi_buffer(count: int = 3) -> bytes:
    return b"".join(_msg(i, float(i)) for i in range(count))


def _write_tgm(tmp_path, count: int = 3) -> str:
    path = str(tmp_path / "scan.tgm")
    with open(path, "wb") as f:
        f.write(_multi_buffer(count))
    return path


# ---------------------------------------------------------------------------
# ScanOptions
# ---------------------------------------------------------------------------


class TestScanOptions:
    def test_defaults(self):
        opts = tensogram.ScanOptions()
        assert opts.bidirectional is True
        assert opts.max_message_size == 4 * 1024 * 1024 * 1024

    def test_kwargs(self):
        opts = tensogram.ScanOptions(bidirectional=False, max_message_size=1024)
        assert opts.bidirectional is False
        assert opts.max_message_size == 1024

    def test_setters(self):
        opts = tensogram.ScanOptions()
        opts.bidirectional = False
        opts.max_message_size = 42
        assert opts.bidirectional is False
        assert opts.max_message_size == 42

    def test_repr(self):
        opts = tensogram.ScanOptions(bidirectional=False)
        assert "ScanOptions(" in repr(opts)
        assert "bidirectional=False" in repr(opts)

    def test_keyword_only(self):
        # bidirectional / max_message_size are keyword-only.
        with pytest.raises(TypeError):
            tensogram.ScanOptions(True)


# ---------------------------------------------------------------------------
# scan_with_options (in-memory)
# ---------------------------------------------------------------------------


class TestScanWithOptions:
    def test_matches_scan_default(self):
        buf = _multi_buffer(3)
        assert tensogram.scan_with_options(buf, tensogram.ScanOptions()) == tensogram.scan(buf)

    def test_forward_only_matches_bidirectional(self):
        buf = _multi_buffer(4)
        forward = tensogram.scan_with_options(buf, tensogram.ScanOptions(bidirectional=False))
        bidir = tensogram.scan_with_options(buf, tensogram.ScanOptions(bidirectional=True))
        assert forward == bidir
        assert len(forward) == 4

    def test_spans_are_contiguous(self):
        buf = _multi_buffer(3)
        spans = tensogram.scan_with_options(buf, tensogram.ScanOptions())
        assert spans[0][0] == 0
        for (off, length), (next_off, _) in zip(spans, spans[1:]):
            assert off + length == next_off

    def test_empty_buffer(self):
        assert tensogram.scan_with_options(b"", tensogram.ScanOptions()) == []

    def test_rejects_non_options(self):
        with pytest.raises(TypeError):
            tensogram.scan_with_options(_multi_buffer(1), {"bidirectional": True})


# ---------------------------------------------------------------------------
# scan_file (streaming I/O)
# ---------------------------------------------------------------------------


class TestScanFile:
    def test_matches_in_memory_scan(self, tmp_path):
        path = _write_tgm(tmp_path, 3)
        with open(path, "rb") as f:
            buf = f.read()
        assert tensogram.scan_file(path) == tensogram.scan(buf)

    def test_count(self, tmp_path):
        path = _write_tgm(tmp_path, 5)
        assert len(tensogram.scan_file(path)) == 5

    def test_single_message(self, tmp_path):
        path = str(tmp_path / "one.tgm")
        with open(path, "wb") as f:
            f.write(_msg(0, 1.0))
        spans = tensogram.scan_file(path)
        assert len(spans) == 1
        assert spans[0][0] == 0

    def test_nonexistent_file_raises_oserror(self):
        with pytest.raises(OSError, match="No such file"):
            tensogram.scan_file("/nonexistent/path/scan.tgm")

    def test_trailing_garbage_ignored(self, tmp_path):
        path = str(tmp_path / "garbage.tgm")
        with open(path, "wb") as f:
            f.write(_msg(0, 1.0))
            f.write(b"NOT_A_MESSAGE_AT_ALL")
        # scan only reports valid message spans; garbage is skipped.
        spans = tensogram.scan_file(path)
        assert len(spans) == 1


# ---------------------------------------------------------------------------
# scan_file_with_options
# ---------------------------------------------------------------------------


class TestScanFileWithOptions:
    def test_matches_scan_file(self, tmp_path):
        path = _write_tgm(tmp_path, 4)
        assert tensogram.scan_file_with_options(path, tensogram.ScanOptions()) == (
            tensogram.scan_file(path)
        )

    def test_forward_only(self, tmp_path):
        path = _write_tgm(tmp_path, 3)
        forward = tensogram.scan_file_with_options(
            path, tensogram.ScanOptions(bidirectional=False)
        )
        assert len(forward) == 3
        assert forward == tensogram.scan_file(path)

    def test_nonexistent_file_raises_oserror(self):
        with pytest.raises(OSError, match="No such file"):
            tensogram.scan_file_with_options("/nonexistent/path/scan.tgm", tensogram.ScanOptions())
