"""Tests for tensogram.validate() and tensogram.validate_file()."""

# pyright: basic, reportAttributeAccessIssue=false, reportMissingTypeStubs=false

import numpy as np
import pytest
import tensogram

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def encode_valid_message():
    """Encode a simple valid message for testing."""
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    meta = {"version": 2}
    desc = {"type": "ntensor", "shape": [4], "dtype": "float32"}
    return tensogram.encode(meta, [(desc, data)])


def encode_nan_message():
    """Encode a message containing NaN values."""
    data = np.array([1.0, float("nan"), 3.0], dtype=np.float64)
    meta = {"version": 2}
    desc = {"type": "ntensor", "shape": [3], "dtype": "float64"}
    return tensogram.encode(meta, [(desc, data)])


def encode_inf_message():
    """Encode a message containing Inf values."""
    data = np.array([1.0, float("inf"), 3.0], dtype=np.float64)
    meta = {"version": 2}
    desc = {"type": "ntensor", "shape": [3], "dtype": "float64"}
    return tensogram.encode(meta, [(desc, data)])


def encode_no_hash_message():
    """Encode a message without hash."""
    data = np.array([1.0, 2.0], dtype=np.float32)
    meta = {"version": 2}
    desc = {"type": "ntensor", "shape": [2], "dtype": "float32"}
    return tensogram.encode(meta, [(desc, data)], hash=None)


# ---------------------------------------------------------------------------
# validate() — basic
# ---------------------------------------------------------------------------


class TestValidateBasic:
    def test_valid_message(self):
        msg = encode_valid_message()
        report = tensogram.validate(msg)
        assert isinstance(report, dict)
        assert report["object_count"] == 1
        assert report["hash_verified"] is True
        assert report["issues"] == []

    def test_return_type_keys(self):
        msg = encode_valid_message()
        report = tensogram.validate(msg)
        assert "issues" in report
        assert "object_count" in report
        assert "hash_verified" in report

    def test_empty_buffer(self):
        report = tensogram.validate(b"")
        assert report["object_count"] == 0
        assert report["hash_verified"] is False
        assert len(report["issues"]) > 0
        assert any(i["code"] == "buffer_too_short" for i in report["issues"])

    def test_corrupted_magic(self):
        msg = bytearray(encode_valid_message())
        msg[0:8] = b"WRONGMAG"
        report = tensogram.validate(bytes(msg))
        assert any(i["code"] == "invalid_magic" for i in report["issues"])

    def test_multi_object(self):
        data = np.array([1.0, 2.0], dtype=np.float32)
        meta = {"version": 2}
        desc = {"type": "ntensor", "shape": [2], "dtype": "float32"}
        msg = tensogram.encode(meta, [(desc, data), (desc, data)])
        report = tensogram.validate(msg)
        assert report["object_count"] == 2
        assert report["hash_verified"] is True
        assert report["issues"] == []


# ---------------------------------------------------------------------------
# validate() — levels
# ---------------------------------------------------------------------------


class TestValidateLevels:
    def test_quick_mode(self):
        msg = encode_valid_message()
        report = tensogram.validate(msg, level="quick")
        assert report["hash_verified"] is False
        assert report["issues"] == []

    def test_default_mode(self):
        msg = encode_valid_message()
        report = tensogram.validate(msg, level="default")
        assert report["hash_verified"] is True

    def test_checksum_mode(self):
        msg = encode_valid_message()
        report = tensogram.validate(msg, level="checksum")
        assert report["hash_verified"] is True

    def test_full_mode_valid(self):
        msg = encode_valid_message()
        report = tensogram.validate(msg, level="full")
        assert report["hash_verified"] is True
        assert report["issues"] == []

    def test_full_mode_nan(self):
        msg = encode_nan_message()
        report = tensogram.validate(msg, level="full")
        assert any(i["code"] == "nan_detected" for i in report["issues"])
        assert report["hash_verified"] is False

    def test_full_mode_inf(self):
        msg = encode_inf_message()
        report = tensogram.validate(msg, level="full")
        assert any(i["code"] == "inf_detected" for i in report["issues"])

    def test_default_mode_skips_fidelity(self):
        msg = encode_nan_message()
        report = tensogram.validate(msg, level="default")
        assert not any(i["code"] == "nan_detected" for i in report["issues"])

    def test_invalid_level(self):
        msg = encode_valid_message()
        with pytest.raises(ValueError, match="unknown validation level"):
            tensogram.validate(msg, level="bogus")


# ---------------------------------------------------------------------------
# validate() — canonical
# ---------------------------------------------------------------------------


class TestValidateCanonical:
    def test_canonical_valid(self):
        msg = encode_valid_message()
        report = tensogram.validate(msg, check_canonical=True)
        assert report["issues"] == []

    def test_canonical_with_quick(self):
        msg = encode_valid_message()
        report = tensogram.validate(msg, level="quick", check_canonical=True)
        assert report["issues"] == []

    def test_canonical_with_checksum(self):
        msg = encode_valid_message()
        report = tensogram.validate(msg, level="checksum", check_canonical=True)
        assert report["hash_verified"] is True

    def test_canonical_with_full(self):
        msg = encode_valid_message()
        report = tensogram.validate(msg, level="full", check_canonical=True)
        assert report["hash_verified"] is True
        assert report["issues"] == []


# ---------------------------------------------------------------------------
# validate() — hash
# ---------------------------------------------------------------------------


class TestValidateHash:
    def test_no_hash_warning(self):
        msg = encode_no_hash_message()
        report = tensogram.validate(msg)
        assert report["hash_verified"] is False
        no_hash = [i for i in report["issues"] if i["code"] == "no_hash_available"]
        assert len(no_hash) > 0
        assert all(i["severity"] == "warning" for i in no_hash)

    def test_corrupted_payload_hash_mismatch(self):
        msg = bytearray(encode_valid_message())
        target = len(msg) * 7 // 10
        msg[target] ^= 0xFF
        report = tensogram.validate(bytes(msg))
        assert report["hash_verified"] is False


# ---------------------------------------------------------------------------
# validate() — issue structure
# ---------------------------------------------------------------------------


class TestIssueStructure:
    def test_issue_has_required_fields(self):
        report = tensogram.validate(b"")
        assert len(report["issues"]) > 0
        issue = report["issues"][0]
        assert "code" in issue
        assert "level" in issue
        assert "severity" in issue
        assert "description" in issue

    def test_optional_fields_absent_or_null(self):
        msg = bytearray(encode_valid_message())
        msg[0:8] = b"WRONGMAG"
        report = tensogram.validate(bytes(msg))
        issue = next(i for i in report["issues"] if i["code"] == "invalid_magic")
        obj_idx = issue.get("object_index")
        assert obj_idx is None


# ---------------------------------------------------------------------------
# validate_file()
# ---------------------------------------------------------------------------


class TestValidateFile:
    def test_valid_file(self, tmp_path):
        path = str(tmp_path / "test.tgm")
        with tensogram.TensogramFile.create(path) as f:
            data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            desc = {"type": "ntensor", "shape": [3], "dtype": "float32"}
            f.append({"version": 2}, [(desc, data)])
        report = tensogram.validate_file(path)
        assert isinstance(report, dict)
        assert "file_issues" in report
        assert "messages" in report
        assert len(report["messages"]) == 1
        assert report["messages"][0]["issues"] == []
        assert report["file_issues"] == []

    def test_multi_message_file(self, tmp_path):
        path = str(tmp_path / "multi.tgm")
        with tensogram.TensogramFile.create(path) as f:
            data = np.array([1.0, 2.0], dtype=np.float32)
            desc = {"type": "ntensor", "shape": [2], "dtype": "float32"}
            f.append({"version": 2}, [(desc, data)])
            f.append({"version": 2}, [(desc, data)])
        report = tensogram.validate_file(path)
        assert len(report["messages"]) == 2
        assert all(m["issues"] == [] for m in report["messages"])

    def test_nonexistent_file(self):
        with pytest.raises(OSError, match=r"No such file|nonexistent"):
            tensogram.validate_file("/nonexistent/path/to/file.tgm")

    def test_trailing_garbage(self, tmp_path):
        """File with trailing garbage after valid message."""
        path = str(tmp_path / "garbage.tgm")
        msg = encode_valid_message()
        with open(path, "wb") as f:
            f.write(msg)
            f.write(b"TRAILING_GARBAGE")
        report = tensogram.validate_file(path)
        assert len(report["file_issues"]) > 0
        assert any("trailing" in issue["description"].lower() for issue in report["file_issues"])

    def test_empty_file(self, tmp_path):
        path = str(tmp_path / "empty.tgm")
        with open(path, "wb"):
            pass
        report = tensogram.validate_file(path)
        assert len(report["messages"]) == 0

    def test_file_with_full_level(self, tmp_path):
        path = str(tmp_path / "full.tgm")
        with tensogram.TensogramFile.create(path) as f:
            data = np.array([1.0, 2.0], dtype=np.float32)
            desc = {"type": "ntensor", "shape": [2], "dtype": "float32"}
            f.append({"version": 2}, [(desc, data)])
        report = tensogram.validate_file(path, level="full")
        assert len(report["messages"]) == 1
        assert report["messages"][0]["issues"] == []

    def test_garbage_only_file(self, tmp_path):
        """File with only garbage bytes — no valid messages."""
        path = str(tmp_path / "garbage_only.tgm")
        with open(path, "wb") as f:
            f.write(b"this is not a tensogram file at all")
        report = tensogram.validate_file(path)
        assert len(report["messages"]) == 0
        assert len(report["file_issues"]) > 0
        assert any("no valid messages" in issue["description"] for issue in report["file_issues"])

    def test_garbage_between_messages(self, tmp_path):
        """File with garbage bytes between two valid messages."""
        path = str(tmp_path / "gap.tgm")
        msg = encode_valid_message()
        with open(path, "wb") as f:
            f.write(msg)
            f.write(b"GARBAGE")
            f.write(msg)
        report = tensogram.validate_file(path)
        assert len(report["messages"]) == 2
        assert len(report["file_issues"]) > 0

    def test_truncated_second_message(self, tmp_path):
        """File with a valid message followed by a truncated one."""
        path = str(tmp_path / "truncated.tgm")
        msg = encode_valid_message()
        with open(path, "wb") as f:
            f.write(msg)
            f.write(msg[: len(msg) // 2])
        report = tensogram.validate_file(path)
        assert len(report["messages"]) >= 1
        has_issue = len(report["file_issues"]) > 0 or any(m["issues"] for m in report["messages"])
        assert has_issue

    def test_validate_file_invalid_level(self, tmp_path):
        """validate_file with an invalid level string raises ValueError."""
        path = str(tmp_path / "valid.tgm")
        with tensogram.TensogramFile.create(path) as f:
            data = np.array([1.0], dtype=np.float32)
            f.append(
                {"version": 2},
                [({"type": "ntensor", "shape": [1], "dtype": "float32"}, data)],
            )
        with pytest.raises(ValueError, match="unknown validation level"):
            tensogram.validate_file(path, level="bogus")

    def test_file_with_checksum_level(self, tmp_path):
        path = str(tmp_path / "checksum.tgm")
        with tensogram.TensogramFile.create(path) as f:
            data = np.array([1.0, 2.0], dtype=np.float32)
            desc = {"type": "ntensor", "shape": [2], "dtype": "float32"}
            f.append({"version": 2}, [(desc, data)])
        report = tensogram.validate_file(path, level="checksum")
        assert len(report["messages"]) == 1
        assert report["messages"][0]["hash_verified"] is True

    def test_file_with_canonical(self, tmp_path):
        path = str(tmp_path / "canonical.tgm")
        with tensogram.TensogramFile.create(path) as f:
            data = np.array([1.0, 2.0], dtype=np.float32)
            desc = {"type": "ntensor", "shape": [2], "dtype": "float32"}
            f.append({"version": 2}, [(desc, data)])
        report = tensogram.validate_file(path, check_canonical=True)
        assert report["file_issues"] == []
        assert report["messages"][0]["issues"] == []

    def test_file_full_level_nan(self, tmp_path):
        path = str(tmp_path / "nan.tgm")
        nan_data = np.array([1.0, float("nan"), 3.0], dtype=np.float64)
        nan_msg = tensogram.encode(
            {"version": 2},
            [({"type": "ntensor", "shape": [3], "dtype": "float64"}, nan_data)],
        )
        with open(path, "wb") as f:
            f.write(nan_msg)
        report = tensogram.validate_file(path, level="full")
        assert len(report["messages"]) == 1
        issues = report["messages"][0]["issues"]
        assert any(i["code"] == "nan_detected" for i in issues)
