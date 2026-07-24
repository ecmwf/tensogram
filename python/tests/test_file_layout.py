# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for TensogramFile.message_layouts() and TensogramFile.open_mmap()."""

# pyright: basic, reportAttributeAccessIssue=false, reportMissingTypeStubs=false

import numpy as np
import pytest
import tensogram

# ---------------------------------------------------------------------------
# message_layouts()
# ---------------------------------------------------------------------------


class TestMessageLayouts:
    def test_count_matches(self, tgm_path):
        with tensogram.TensogramFile.open(tgm_path) as f:
            layouts = f.message_layouts()
            assert len(layouts) == f.message_count()
            assert len(layouts) == 3

    def test_matches_scan(self, tgm_path):
        with tensogram.TensogramFile.open(tgm_path) as f:
            layouts = f.message_layouts()
        spans = tensogram.scan_file(tgm_path)
        assert [(layout.offset, layout.length) for layout in layouts] == spans

    def test_attributes(self, tgm_path):
        with tensogram.TensogramFile.open(tgm_path) as f:
            layout = f.message_layouts()[0]
        assert layout.offset == 0
        assert layout.length > 0

    def test_contiguous(self, tgm_path):
        with tensogram.TensogramFile.open(tgm_path) as f:
            layouts = f.message_layouts()
        for cur, nxt in zip(layouts, layouts[1:]):
            assert cur.offset + cur.length == nxt.offset

    def test_tuple_unpacking(self, tgm_path):
        with tensogram.TensogramFile.open(tgm_path) as f:
            layout = f.message_layouts()[0]
        offset, length = layout
        assert offset == layout.offset
        assert length == layout.length
        assert len(layout) == 2
        assert layout[0] == layout.offset
        assert layout[1] == layout.length

    def test_index_out_of_range(self, tgm_path):
        with tensogram.TensogramFile.open(tgm_path) as f:
            layout = f.message_layouts()[0]
        with pytest.raises(IndexError):
            _ = layout[2]

    def test_repr(self, tgm_path):
        with tensogram.TensogramFile.open(tgm_path) as f:
            layout = f.message_layouts()[0]
        assert "MessageLayout(offset=" in repr(layout)

    def test_equality(self, tgm_path):
        with tensogram.TensogramFile.open(tgm_path) as f:
            layouts = f.message_layouts()
        assert layouts[0] == layouts[0]
        assert layouts[0] != layouts[1]

    def test_lengths_can_read_message(self, tgm_path):
        # offset+length must exactly bound each message.
        with open(tgm_path, "rb") as fh:
            buf = fh.read()
        with tensogram.TensogramFile.open(tgm_path) as f:
            for i, layout in enumerate(f.message_layouts()):
                slice_ = buf[layout.offset : layout.offset + layout.length]
                decoded = tensogram.decode(slice_)
                assert decoded.metadata.base[0]["index"] == i


# ---------------------------------------------------------------------------
# open_mmap()
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not getattr(tensogram, "__has_mmap__", False),
    reason="tensogram built without the mmap feature",
)
class TestOpenMmap:
    def test_has_mmap_flag(self):
        assert tensogram.__has_mmap__ is True

    def test_message_count(self, tgm_path):
        f = tensogram.TensogramFile.open_mmap(tgm_path)
        assert f.message_count() == 3

    def test_layouts_match_regular_open(self, tgm_path):
        mm = tensogram.TensogramFile.open_mmap(tgm_path)
        with tensogram.TensogramFile.open(tgm_path) as reg:
            mm_layouts = [(x.offset, x.length) for x in mm.message_layouts()]
            reg_layouts = [(x.offset, x.length) for x in reg.message_layouts()]
        assert mm_layouts == reg_layouts

    def test_decode_matches_regular(self, tgm_path):
        mm = tensogram.TensogramFile.open_mmap(tgm_path)
        with tensogram.TensogramFile.open(tgm_path) as reg:
            for i in range(3):
                _m, mm_objs = mm.decode_message(i)
                _r, reg_objs = reg.decode_message(i)
                np.testing.assert_array_equal(mm_objs[0][1], reg_objs[0][1])

    def test_read_message_matches(self, tgm_path):
        mm = tensogram.TensogramFile.open_mmap(tgm_path)
        with tensogram.TensogramFile.open(tgm_path) as reg:
            assert mm.read_message(0) == reg.read_message(0)

    def test_append_rejected(self, tgm_path):
        mm = tensogram.TensogramFile.open_mmap(tgm_path)
        data = np.zeros(10, dtype=np.float32)
        desc = {"type": "ntensor", "shape": [10], "dtype": "float32"}
        with pytest.raises(OSError, match="memory-mapped"):
            mm.append({"base": [{}]}, [(desc, data)])

    def test_nonexistent(self):
        with pytest.raises(OSError, match="No such file"):
            tensogram.TensogramFile.open_mmap("/nonexistent/path/mmap.tgm")
