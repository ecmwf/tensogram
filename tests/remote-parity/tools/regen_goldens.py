# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Fail-closed golden regeneration for the remote-parity harness.

Never runs in CI. The explicit ``--regen`` flag guards against accidental
invocation; without it the tool prints what would be regenerated and
exits non-zero.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from run_parity import (
    _FIXTURES,
    _OPS,
    DriverCase,
    collect_events,
    write_goldens,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regen",
        action="store_true",
        help="Actually write goldens. Without this flag, prints a dry-run and exits non-zero.",
    )
    parser.add_argument(
        "--fixture",
        action="append",
        help="Restrict to one or more fixtures (default: all)",
    )
    args = parser.parse_args(argv)

    fixtures = tuple(args.fixture) if args.fixture else _FIXTURES
    unknown = [f for f in fixtures if f not in _FIXTURES]
    if unknown:
        print(f"unknown fixtures: {unknown}", file=sys.stderr)
        return 2

    cases = [
        DriverCase(fixture=fixture, language=language, op=op)
        for fixture in fixtures
        for language in ("rust", "ts")
        for op in _OPS
    ]

    if not args.regen:
        print("dry-run (pass --regen to actually write):")
        for case in cases:
            print(f"  would regenerate: {case.run_id}.json")
        return 1

    events = collect_events(cases)
    write_goldens(events)
    print(f"regenerated {len(cases)} goldens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
