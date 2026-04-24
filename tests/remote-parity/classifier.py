# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Request classification and scan-round assignment.

Two responsibilities, kept deliberately separate so future backward-scan
and eager footer-discovery work can extend them without touching the
rest of the harness:

- `Classifier` maps one raw HTTP observation to `(category, direction,
  logical_range)` based on status + method + range length + response
  bytes. Status-aware: non-206/non-200 responses become ``error``.
- `RoundBuilder` walks classified events in observed order and assigns
  `scan_round`. Forward-only today (one scan event per round);
  bidirectional will pair forward + backward events per round.

Schema contract is defined by `schema.json`. Non-scan events use
``scan_round = -1`` and ``direction = "none"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Category = Literal["probe", "scan", "payload", "fallback", "error"]
Direction = Literal["forward", "backward", "none"]

PREAMBLE_BYTES = 24
POSTAMBLE_BYTES = 24
END_MAGIC_BYTES = 8
NON_SCAN_ROUND = -1


@dataclass(frozen=True)
class Observation:
    """One raw observation from the mock server, pre-classification."""

    method: str
    range_header: str | None
    status: int
    response_bytes: int


@dataclass(frozen=True)
class Classified:
    """Result of classifying one `Observation`."""

    category: Category
    direction: Direction
    logical_range: tuple[int, int]
    physical: dict


def classify(observation: Observation) -> Classified:
    physical = _physical_dict(observation)
    logical_range = _logical_from_range(observation.range_header, observation.response_bytes)

    if observation.method == "HEAD":
        return Classified("probe", "none", (0, 0), physical)

    if observation.method != "GET":
        return Classified("error", "none", logical_range, physical)

    if observation.status == 200:
        return Classified("fallback", "none", logical_range, physical)

    if observation.status != 206:
        return Classified("error", "none", logical_range, physical)

    length = logical_range[1] - logical_range[0]
    if length in (PREAMBLE_BYTES, POSTAMBLE_BYTES, END_MAGIC_BYTES):
        return Classified("scan", "forward", logical_range, physical)

    return Classified("payload", "none", logical_range, physical)


def _logical_from_range(range_header: str | None, response_bytes: int) -> tuple[int, int]:
    if range_header is None:
        return (0, response_bytes)
    if not range_header.startswith("bytes="):
        return (0, response_bytes)
    spec = range_header[len("bytes=") :].strip()

    if spec.startswith("-"):
        try:
            int(spec[1:])
        except ValueError:
            return (0, response_bytes)
        # Suffix ranges (`bytes=-n`) identify the last n bytes of the
        # object, but without the total object size or a parsed
        # `Content-Range` header we can't recover the true logical
        # start offset here. Fall back to the observed response span;
        # capturing total size is tracked for when bidirectional scan
        # adds suffix-range usage.
        return (0, response_bytes)

    if "-" not in spec:
        return (0, response_bytes)
    start_s, end_s = spec.split("-", 1)
    try:
        start = int(start_s)
    except ValueError:
        return (0, response_bytes)
    if end_s:
        try:
            end_inclusive = int(end_s)
        except ValueError:
            return (start, start + response_bytes)
        return (start, end_inclusive + 1)
    return (start, start + response_bytes)


def _physical_dict(observation: Observation) -> dict:
    headers: dict[str, str] = {}
    if observation.range_header is not None:
        headers["Range"] = observation.range_header
    return {"method": observation.method, "headers": headers, "status": observation.status}


class RoundBuilder:
    """Assigns `scan_round` to classified events in observation order.

    Forward-only contract: every ``scan`` event starts a new round.
    Non-scan events get `scan_round = NON_SCAN_ROUND`.

    Bidirectional (future): forward + backward events belonging to the
    same logical round share a `scan_round`. This class is the
    extension point — override or subclass without touching callers.
    """

    def __init__(self) -> None:
        self._next_round = 0

    def assign(self, classified: Classified) -> int:
        if classified.category != "scan":
            return NON_SCAN_ROUND
        round_id = self._next_round
        self._next_round += 1
        return round_id
