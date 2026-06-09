#!/usr/bin/env python3
# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.
"""Bump the project version across every manifest that carries it.

The ``VERSION`` file at the repo root is the single source of truth. This
script reads the *current* version from ``VERSION`` and rewrites it to the
new version everywhere it appears, then greps the tree to prove no straggler
copies of the old version remain.

Usage:
    python3 scripts/bump_version.py X.Y.Z [--check]

``--check`` runs the post-bump consistency scan only (no edits) against the
version currently in ``VERSION`` — useful as a CI guard.

The set of edited locations mirrors the "SINGLE SOURCE OF TRUTH FOR VERSION"
section of ``AGENTS.md``. Generated/git-ignored manifests (e.g.
``typescript/wasm/package.json``, produced by wasm-pack) are deliberately
NOT edited here — they are regenerated from the crate version at build time.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")

# Workspace members inherit the version from the root Cargo.toml
# [workspace.package] block, so a single edit there covers all of them.
# The list below is the set of files that carry an *independent* version
# string and must each be rewritten.
#
# Each entry is (path, [list of (regex-with-one-capture-group-for-the-old-version)]).
# The capture group identifies the exact old-version token to replace, so we
# never touch an unrelated version string (e.g. a dependency pin on a
# third-party crate).


def _cargo_workspace_version() -> list[tuple[str, str]]:
    # root [workspace.package] version = "X.Y.Z"
    return [(r'(?m)^(version\s*=\s*")(?:\d+\.\d+\.\d+)(")', r"\g<1>{new}\g<2>")]


def _cargo_pinned_internal_deps() -> list[tuple[str, str]]:
    # Pinned internal workspace deps: version = "=X.Y.Z"
    return [(r'(version\s*=\s*"=)(?:\d+\.\d+\.\d+)(")', r"\g<1>{new}\g<2>")]


def _toml_version() -> list[tuple[str, str]]:
    return [(r'(?m)^(version\s*=\s*")(?:\d+\.\d+\.\d+)(")', r"\g<1>{new}\g<2>")]


def _json_version() -> list[tuple[str, str]]:
    return [(r'("version"\s*:\s*")(?:\d+\.\d+\.\d+)(")', r"\g<1>{new}\g<2>")]


def _cmake_project_version() -> list[tuple[str, str]]:
    return [(r"(project\([^)]*VERSION\s+)(?:\d+\.\d+\.\d+)", r"\g<1>{new}")]


# Every Cargo.toml that pins an internal workspace crate does so with
# `version = "=X.Y.Z"`. Those pins live in member *and* excluded crates, so
# we rewrite the `=`-pin token in all of them. The root Cargo.toml also
# carries the [workspace.package] version that the members inherit.
_INTERNAL_CARGO_TOMLS = [
    "rust/tensogram/Cargo.toml",
    "rust/tensogram-cli/Cargo.toml",
    "rust/tensogram-ffi/Cargo.toml",
    "rust/tensogram-sz3/Cargo.toml",
    "rust/tensogram-grib/Cargo.toml",
    "rust/tensogram-netcdf/Cargo.toml",
    "rust/tensogram-wasm/Cargo.toml",
]

# (relative path, edit-spec factory)
TARGETS: list[tuple[str, list[tuple[str, str]]]] = [
    # Root Cargo.toml carries BOTH the [workspace.package] version (covers all
    # inheriting members) AND the pinned internal dependency strings.
    ("Cargo.toml", _cargo_workspace_version() + _cargo_pinned_internal_deps()),
    # Cargo packages excluded from the default workspace (own version field).
    ("python/bindings/Cargo.toml", _toml_version()),
    ("rust/tensogram-grib/Cargo.toml", _toml_version()),
    ("rust/tensogram-netcdf/Cargo.toml", _toml_version()),
    ("rust/tensogram-wasm/Cargo.toml", _toml_version()),
    # Internal `version = "=X.Y.Z"` pins in every crate that references a
    # sibling crate (covers tensogram-cli/ffi/sz3/grib/netcdf/wasm/tensogram).
    *((p, _cargo_pinned_internal_deps()) for p in _INTERNAL_CARGO_TOMLS),
    # Python packages.
    ("python/bindings/pyproject.toml", _toml_version()),
    ("python/tensogram-xarray/pyproject.toml", _toml_version()),
    ("python/tensogram-zarr/pyproject.toml", _toml_version()),
    ("python/tensogram-anemoi/pyproject.toml", _toml_version()),
    ("python/tensogram-earthkit/pyproject.toml", _toml_version()),
    ("examples/jupyter/pyproject.toml", _toml_version()),
    # JS packages that are version-controlled (NOT the generated wasm output).
    ("typescript/package.json", _json_version()),
    ("examples/typescript/package.json", _json_version()),
    # Fortran binding.
    ("fortran/fpm.toml", _toml_version()),
    ("fortran/CMakeLists.txt", _cmake_project_version()),
]


def read_current_version() -> str:
    return (REPO / "VERSION").read_text().strip()


def apply_edits(new: str) -> list[str]:
    """Rewrite every target file. Returns the list of files actually changed."""
    changed: list[str] = []
    for rel, specs in TARGETS:
        path = REPO / rel
        if not path.exists():
            print(f"  WARNING: target not found, skipping: {rel}", file=sys.stderr)
            continue
        text = path.read_text()
        original = text
        for pattern, repl in specs:
            text = re.sub(pattern, repl.replace("{new}", new), text)
        if text != original:
            path.write_text(text)
            changed.append(rel)
    return changed


def write_version_file(new: str) -> None:
    (REPO / "VERSION").write_text(new + "\n")


def scan_for_stragglers(old: str) -> tuple[list[str], list[str]]:
    """git grep for the old version.

    Returns ``(needs_review, unexpected)``:
    - ``needs_review`` — Python dependency *constraint ranges*
      (``tensogram>=X.Y.Z,<X.Y+1``). The lower bound mirrors the version but
      the ``<`` ceiling encodes a deliberate compatibility policy, so these
      are surfaced for a human to decide rather than rewritten blindly.
    - ``unexpected`` — anything else still referencing the old version, which
      most likely IS a straggler that should have been updated.
    """
    if not old:
        return [], []
    try:
        out = subprocess.run(
            ["git", "grep", "-n", "--fixed-strings", old],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("  WARNING: git not available, skipping straggler scan", file=sys.stderr)
        return [], []
    # git grep exits 1 when there are no matches — that's success for us.
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    ignore_substrings = (
        "CHANGELOG.md",  # historical release entries legitimately mention old versions
        "Cargo.lock",  # regenerated by cargo; not a source of truth
        "/SZ3/",  # vendored upstream sources
        "scripts/bump_version.py",  # this script's own doc examples
    )
    constraint_re = re.compile(r">=\s*" + re.escape(old) + r"\s*,\s*<")
    needs_review: list[str] = []
    unexpected: list[str] = []
    for ln in lines:
        if any(s in ln for s in ignore_substrings):
            continue
        if constraint_re.search(ln):
            needs_review.append(ln)
        else:
            unexpected.append(ln)
    return needs_review, unexpected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("version", nargs="?", help="new version X.Y.Z")
    ap.add_argument(
        "--check",
        action="store_true",
        help="scan only: verify no straggler differs from VERSION (no edits)",
    )
    args = ap.parse_args()

    current = read_current_version()

    if args.check:
        # In check mode we assert that every target already matches VERSION.
        # We do that by dry-applying edits and seeing if anything *would* change.
        changed = []
        for rel, specs in TARGETS:
            path = REPO / rel
            if not path.exists():
                continue
            text = path.read_text()
            new_text = text
            for pattern, repl in specs:
                new_text = re.sub(pattern, repl.replace("{new}", current), new_text)
            if new_text != text:
                changed.append(rel)
        if changed:
            print(
                f"Version mismatch — these files do not match VERSION ({current}):",
                file=sys.stderr,
            )
            for c in changed:
                print(f"  {c}", file=sys.stderr)
            return 1
        print(f"All version strings match VERSION ({current}).")
        return 0

    if not args.version:
        ap.error("a new version is required unless --check is given")
    new = args.version
    if not SEMVER_RE.match(new):
        ap.error(f"version must be MAJOR.MINOR.MICRO (got {new!r})")

    print(f"Bumping version {current} -> {new}")
    write_version_file(new)
    changed = apply_edits(new)
    print(f"Updated {len(changed)} manifest(s):")
    for c in changed:
        print(f"  {c}")

    needs_review, unexpected = scan_for_stragglers(current)
    print()
    if needs_review:
        print(
            f"Dependency CONSTRAINT RANGES still pinned to {current} — review by "
            "hand (the '<' ceiling is a compatibility policy, not a mirror):"
        )
        for ln in needs_review:
            print(f"  {ln}")
        print()
    if unexpected:
        print(
            f"UNEXPECTED references to the old version ({current}) — likely "
            "stragglers, please check:"
        )
        for ln in unexpected:
            print(f"  {ln}")
        print()
    if not needs_review and not unexpected:
        print(f"No remaining references to {current} found outside ignored paths.")
        print()

    print("Next steps (not done automatically):")
    print(
        "  - review/update Python dependency constraint ceilings if this is "
        "a minor bump"
    )
    print("  - add a new release entry header to CHANGELOG.md")
    print("  - review the diff, commit, tag (no 'v' prefix), and push")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
