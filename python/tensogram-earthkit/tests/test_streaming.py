# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Stream-reader path: ``ekd.from_source("stream", <file-like>)``.

The stream reader drains the stream to bytes, routes through the
memory-reader path, and exposes the same data-object surface as the
file and memory paths.  Two invariants are tested here:

* Parity: a file-like stream of tensogram bytes decodes identically to
  the same bytes fed through the memory path and the same content
  stored on disk.
* The stream reader handles both ``BytesIO`` and objects that only
  expose ``read``/``peek`` (the earthkit-data stream contract).

True progressive ("yield fields as they arrive") decode is explicitly
out of scope for Cycle 5 — the stream backs a materialised FieldList /
Dataset.  This mirrors earthkit-data's default ``stream_reader`` with
``memory=False`` returning the reader and ``memory=True`` materialising,
with our default being materialised because the downstream xarray
backend needs a path anyway.
"""

from __future__ import annotations

import io

import numpy as np
import xarray as xr

from tensogram_earthkit.readers.stream import stream_reader


class _FakeSource:
    """Minimal stand-in for the earthkit-data ``Source`` contract.

    The stream reader is allowed to weakref-hold the source; we only
    need the attributes Reader.__init__ reads.
    """

    def __init__(self) -> None:
        self.source_filename = None


def _stream_like(buf: bytes) -> io.BufferedReader:
    """Wrap *buf* in a peekable :class:`io.BufferedReader`."""
    return io.BufferedReader(io.BytesIO(buf))


class TestStreamReaderDirect:
    """Invoke the stream_reader callable directly — covers the plumbing."""

    def test_returns_none_for_non_tensogram_stream(self) -> None:
        stream = _stream_like(b"GRIB\x02\x00\x00\x00" + b"\x00" * 32)
        result = stream_reader(
            _FakeSource(), stream, magic=b"GRIB\x02\x00\x00\x00", deeper_check=False
        )
        assert result is None

    def test_reads_tensogram_stream_to_xarray(self, nonmars_tensogram_bytes) -> None:
        stream = _stream_like(nonmars_tensogram_bytes)
        reader = stream_reader(_FakeSource(), stream, magic=nonmars_tensogram_bytes[:8])
        assert reader is not None
        ds = reader.to_xarray()
        assert isinstance(ds, xr.Dataset)
        var = next(iter(ds.data_vars.values()))
        assert var.shape == (2, 3, 4)

    def test_reads_mars_stream_to_fieldlist(self, mars_tensogram_bytes) -> None:
        stream = _stream_like(mars_tensogram_bytes)
        reader = stream_reader(_FakeSource(), stream, magic=mars_tensogram_bytes[:8])
        assert reader is not None
        fl = reader.to_fieldlist()
        assert len(fl) == 2


class TestStreamReaderParity:
    """Stream path must match file / memory paths byte-for-byte."""

    def test_stream_matches_memory_nonmars(self, nonmars_tensogram_bytes) -> None:
        stream = _stream_like(nonmars_tensogram_bytes)
        reader = stream_reader(_FakeSource(), stream, magic=nonmars_tensogram_bytes[:8])
        via_stream = reader.to_numpy()

        import earthkit.data as ekd

        via_memory = ekd.from_source("tensogram", nonmars_tensogram_bytes).to_numpy()
        np.testing.assert_array_equal(via_stream, via_memory)

    def test_stream_matches_file_mars(self, mars_tensogram_bytes, mars_tensogram_file) -> None:
        import earthkit.data as ekd

        stream = _stream_like(mars_tensogram_bytes)
        reader = stream_reader(_FakeSource(), stream, magic=mars_tensogram_bytes[:8])
        via_stream = reader.to_fieldlist()
        via_file = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()

        assert len(via_stream) == len(via_file)
        for a, b in zip(via_stream, via_file, strict=True):
            assert a.metadata("param") == b.metadata("param")
            np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


class TestStreamReaderMagicHandling:
    def test_no_magic_provided_peeks_the_stream(self, nonmars_tensogram_bytes) -> None:
        """When earthkit-data cannot pre-peek, the reader peeks itself."""
        stream = _stream_like(nonmars_tensogram_bytes)
        reader = stream_reader(_FakeSource(), stream, magic=None)
        assert reader is not None
        ds = reader.to_xarray()
        assert len(ds.data_vars) == 1

    def test_empty_stream_returns_none(self) -> None:
        stream = _stream_like(b"")
        result = stream_reader(_FakeSource(), stream, magic=b"")
        assert result is None


class TestStreamReaderDoesNotLeak:
    """The reader must close the stream once it has drained it."""

    def test_stream_closed_after_read(self, nonmars_tensogram_bytes) -> None:
        stream = _stream_like(nonmars_tensogram_bytes)
        reader = stream_reader(_FakeSource(), stream, magic=nonmars_tensogram_bytes[:8])
        reader.to_numpy()  # force the drain / decode
        # The underlying BytesIO exposes `.closed`; once the stream reader
        # is done with it, it should be closed to release the buffer.
        assert stream.closed, "stream should be closed after decode"


class TestStreamThroughEarthkit:
    """End-to-end via ``ekd.from_source("stream", …)`` with our reader."""

    def test_from_source_stream(self, nonmars_tensogram_bytes) -> None:
        """Drain a stream, then route the bytes through ``ekd.from_source``.

        ``ekd.from_source("stream", …)`` itself dispatches across
        earthkit-data's built-in readers/ tree, where our reader does
        not (yet) live, so it would resolve to ``UnknownStreamReader``.
        The supported invocation is the ``tensogram`` source with the
        drained bytes — internally it uses the memory path.
        """
        import earthkit.data as ekd

        stream = _stream_like(nonmars_tensogram_bytes)
        data = ekd.from_source("tensogram", stream.read())
        arr = data.to_numpy()
        assert arr.shape == (2, 3, 4)
