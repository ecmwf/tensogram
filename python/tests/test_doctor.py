# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for :func:`tensogram.doctor`.

Locks in the cross-language parity contract (see
``docs/src/cli/doctor.md``) — Rust, WASM/TypeScript, and C FFI all
expose the same ``doctor()`` entry point with identical JSON shape.

Core invariants exercised here:

- Top-level keys: ``build``, ``features``, ``self_test``.
- ``build``: ``version``, ``wire_version``, ``target``, ``profile``.
- ``features``: at least one row; every row carries ``name`` / ``kind``
  / ``state``.  ``state == "on"`` rows additionally carry ``backend``
  and ``linkage``; ``state == "off"`` rows carry only the three
  mandatory keys.
- ``self_test``: at least one row; every row carries ``label`` /
  ``outcome`` (``"ok"`` / ``"failed"`` / ``"skipped"``).
- The wheel's compiled-in codecs round-trip cleanly — no ``failed``
  rows in the default build.
"""

from __future__ import annotations

import tensogram


def test_doctor_returns_three_top_level_sections() -> None:
    report = tensogram.doctor()
    assert isinstance(report, dict)
    assert set(report.keys()) == {"build", "features", "self_test"}


def test_doctor_build_section_carries_expected_fields() -> None:
    build = tensogram.doctor()["build"]
    assert isinstance(build, dict)
    assert isinstance(build["version"], str)
    assert build["version"]
    assert isinstance(build["wire_version"], int)
    assert build["wire_version"] > 0
    assert isinstance(build["target"], str)
    assert build["target"]
    assert build["profile"] in {"release", "debug"}


def test_doctor_build_wire_version_matches_module_constant() -> None:
    # Cross-check against `tensogram.WIRE_VERSION` so a future bump
    # cannot silently desync the doctor view from the rest of the API.
    assert tensogram.doctor()["build"]["wire_version"] == tensogram.WIRE_VERSION


def test_doctor_features_rows_have_name_kind_state() -> None:
    features = tensogram.doctor()["features"]
    assert isinstance(features, list)
    assert len(features) > 0
    for f in features:
        assert isinstance(f["name"], str)
        assert f["name"]
        assert f["kind"] in {"compression", "threading", "io", "converter"}
        assert f["state"] in {"on", "off"}


def test_doctor_features_on_rows_carry_backend_and_linkage() -> None:
    features = tensogram.doctor()["features"]
    on_rows = [f for f in features if f["state"] == "on"]
    # Default wheel has at least one On feature (zstd is in default features).
    assert len(on_rows) > 0
    for f in on_rows:
        assert isinstance(f["backend"], str)
        assert f["backend"]
        assert f["linkage"] in {"ffi", "pure-rust"}


def test_doctor_features_off_rows_have_minimal_shape() -> None:
    features = tensogram.doctor()["features"]
    for f in features:
        if f["state"] == "off":
            # Off rows MUST NOT carry backend / linkage / version, by contract.
            assert set(f.keys()) == {"name", "kind", "state"}, (
                f"Off row carries unexpected keys: {f}"
            )


def test_doctor_self_test_rows_have_label_and_outcome() -> None:
    self_test = tensogram.doctor()["self_test"]
    assert isinstance(self_test, list)
    assert len(self_test) > 0
    for r in self_test:
        assert isinstance(r["label"], str)
        assert r["label"]
        assert r["outcome"] in {"ok", "failed", "skipped"}


def test_doctor_passes_self_test_on_default_wheel() -> None:
    # Every codec the wheel was compiled with should round-trip cleanly.
    # Skipped rows are fine (no codec compiled in); failures are not.
    self_test = tensogram.doctor()["self_test"]
    failures = [r for r in self_test if r["outcome"] == "failed"]
    assert failures == [], f"unexpected self-test failures: {failures}"


def test_doctor_self_test_includes_core_round_trip() -> None:
    # The `none/none/none` round-trip is unconditional — every wheel
    # exercises the basic encode/decode path regardless of features.
    labels = [r["label"] for r in tensogram.doctor()["self_test"]]
    assert any("none/none/none" in label for label in labels), (
        f"core round-trip missing from self-test: {labels}"
    )


def test_doctor_is_idempotent() -> None:
    # Successive calls return structurally equal reports (the only
    # field that could vary is the order of features, which is also
    # fixed by `collect_features`).
    a = tensogram.doctor()
    b = tensogram.doctor()
    assert a == b
