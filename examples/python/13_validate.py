#!/usr/bin/env python3
# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Example 13: Message and file validation.

Demonstrates how to validate Tensogram messages and files at different
levels of depth — from quick structural checks to full data fidelity
verification with NaN/Inf detection.
"""

# pyright: basic, reportAttributeAccessIssue=false, reportMissingTypeStubs=false

import os
import tempfile

import numpy as np
import tensogram


def main():
    # ── 1. Validate a single message ──────────────────────────────────────────

    print("=== Single message validation ===\n")

    data = np.random.randn(100).astype(np.float32)
    msg = tensogram.encode(
        {"version": 2},
        [({"type": "ntensor", "shape": [100], "dtype": "float32"}, data)],
    )

    report = tensogram.validate(msg)
    print(
        f"Valid message:  issues={len(report['issues'])}, objects={report['object_count']}, "
        f"hash_verified={report['hash_verified']}"
    )

    # ── 2. Validation levels ─────────────────────────────────────────────────

    print("\n=== Validation levels ===\n")

    for level in ("quick", "default", "checksum", "full"):
        report = tensogram.validate(msg, level=level)
        print(
            f"  {level:10s}: issues={len(report['issues'])}, "
            f"hash_verified={report['hash_verified']}"
        )

    # ── 3. Detect structural corruption ──────────────────────────────────────

    print("\n=== Corrupted message ===\n")

    corrupted = bytearray(msg)
    corrupted[0:8] = b"WRONGMAG"
    report = tensogram.validate(bytes(corrupted))
    for issue in report["issues"]:
        print(f"  [{issue['severity']}] {issue['code']}: {issue['description']}")

    # ── 4. NaN/Inf detection (full mode) ─────────────────────────────────────

    print("\n=== NaN/Inf detection (full mode) ===\n")

    nan_data = np.array([1.0, float("nan"), 3.0], dtype=np.float64)
    nan_msg = tensogram.encode(
        {"version": 2},
        [({"type": "ntensor", "shape": [3], "dtype": "float64"}, nan_data)],
    )

    report = tensogram.validate(nan_msg, level="full")
    for issue in report["issues"]:
        print(f"  [{issue['severity']}] {issue['code']}: {issue['description']}")

    # ── 5. File validation ───────────────────────────────────────────────────

    print("\n=== File validation ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "example.tgm")

        with tensogram.TensogramFile.create(path) as f:
            for i in range(3):
                arr = np.random.randn(50).astype(np.float32)
                desc = {"type": "ntensor", "shape": [50], "dtype": "float32"}
                f.append({"version": 2, "base": [{"index": i}]}, [(desc, arr)])

        report = tensogram.validate_file(path)
        print(
            f"File: {len(report['messages'])} messages, "
            f"{len(report['file_issues'])} file-level issues"
        )
        for i, message_report in enumerate(report["messages"]):
            status = (
                "OK"
                if not message_report["issues"]
                else f"{len(message_report['issues'])} issues"
            )
            print(
                f"  Message {i}: {status}, "
                f"objects={message_report['object_count']}, "
                f"hash_verified={message_report['hash_verified']}"
            )


if __name__ == "__main__":
    main()
