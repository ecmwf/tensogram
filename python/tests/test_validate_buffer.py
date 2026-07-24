# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for tensogram.validate_buffer() — multi-message in-memory
validation (sibling of validate_file, distinct from single-message
validate)."""

# pyright: basic, reportAttributeAccessIssue=false, reportMissingTypeStubs=false

import numpy as np
import pytest
import tensogram

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_msg(fill: float = 1.0) -> bytes:
    data = np.array([fill, fill + 1, fill + 2], dtype=np.float32)
    desc = {"type": "ntensor", "shape": [3], "dtype": "float32"}
    return tensogram.encode({"version": 3}, [(desc, data)])


# ---------------------------------------------------------------------------
# Shape / basics
# ---------------------------------------------------------------------------


class TestValidateBufferBasics:
    def test_return_shape(self):
        buf = _valid_msg()
        report = tensogram.validate_buffer(buf)
        assert isinstance(report, dict)
        assert "file_issues" in report
        assert "messages" in report

    def test_single_message(self):
        report = tensogram.validate_buffer(_valid_msg())
        assert len(report["messages"]) == 1
        assert report["messages"][0]["issues"] == []
        assert report["messages"][0]["object_count"] == 1
        assert report["file_issues"] == []

    def test_multi_message(self):
        buf = _valid_msg(1.0) + _valid_msg(2.0) + _valid_msg(3.0)
        report = tensogram.validate_buffer(buf)
        assert len(report["messages"]) == 3
        assert all(m["issues"] == [] for m in report["messages"])
        assert report["file_issues"] == []

    def test_per_message_shape_matches_validate(self):
        # Each entry in `messages` has the same shape as validate()'s return.
        buf = _valid_msg()
        report = tensogram.validate_buffer(buf)
        msg_report = report["messages"][0]
        assert set(msg_report.keys()) >= {"issues", "object_count", "hash_verified"}
        assert msg_report["hash_verified"] is True

    def test_empty_buffer(self):
        report = tensogram.validate_buffer(b"")
        assert report["messages"] == []
        # No messages, no bytes → no file-level issues.
        assert report["file_issues"] == []


# ---------------------------------------------------------------------------
# File-level issues
# ---------------------------------------------------------------------------


class TestValidateBufferFileIssues:
    def test_trailing_garbage(self):
        buf = _valid_msg() + b"TRAILING_GARBAGE"
        report = tensogram.validate_buffer(buf)
        assert len(report["messages"]) == 1
        assert len(report["file_issues"]) > 0
        assert any("trailing" in i["description"].lower() for i in report["file_issues"])

    def test_garbage_between_messages(self):
        buf = _valid_msg() + b"GARBAGE!!" + _valid_msg()
        report = tensogram.validate_buffer(buf)
        assert len(report["messages"]) == 2
        assert len(report["file_issues"]) > 0

    def test_garbage_only(self):
        report = tensogram.validate_buffer(b"this is not a tensogram buffer")
        assert report["messages"] == []
        assert len(report["file_issues"]) > 0
        assert any("no valid messages" in i["description"] for i in report["file_issues"])

    def test_file_issue_fields(self):
        buf = _valid_msg() + b"TRAILING_GARBAGE"
        report = tensogram.validate_buffer(buf)
        issue = report["file_issues"][0]
        assert "byte_offset" in issue
        assert "length" in issue
        assert "description" in issue


# ---------------------------------------------------------------------------
# Levels / options
# ---------------------------------------------------------------------------


class TestValidateBufferLevels:
    def test_quick_skips_hash(self):
        report = tensogram.validate_buffer(_valid_msg(), level="quick")
        assert report["messages"][0]["hash_verified"] is False

    def test_checksum_level(self):
        report = tensogram.validate_buffer(_valid_msg(), level="checksum")
        assert report["messages"][0]["hash_verified"] is True

    def test_full_level_detects_nan(self):
        nan_data = np.array([1.0, float("nan"), 3.0], dtype=np.float64).tobytes()
        msg = tensogram.encode_pre_encoded(
            {"version": 3},
            [({"type": "ntensor", "shape": [3], "dtype": "float64"}, nan_data)],
        )
        report = tensogram.validate_buffer(msg, level="full")
        issues = report["messages"][0]["issues"]
        assert any(i["code"] == "nan_detected" for i in issues)

    def test_canonical(self):
        report = tensogram.validate_buffer(_valid_msg(), check_canonical=True)
        assert report["file_issues"] == []
        assert report["messages"][0]["issues"] == []

    def test_invalid_level(self):
        with pytest.raises(ValueError, match="unknown validation level"):
            tensogram.validate_buffer(_valid_msg(), level="bogus")


# ---------------------------------------------------------------------------
# Parity with validate_file
# ---------------------------------------------------------------------------


class TestValidateBufferParity:
    def test_matches_validate_file(self, tmp_path):
        buf = _valid_msg(1.0) + _valid_msg(2.0)
        path = str(tmp_path / "parity.tgm")
        with open(path, "wb") as f:
            f.write(buf)
        from_buf = tensogram.validate_buffer(buf)
        from_file = tensogram.validate_file(path)
        assert len(from_buf["messages"]) == len(from_file["messages"])
        assert from_buf["file_issues"] == from_file["file_issues"]
