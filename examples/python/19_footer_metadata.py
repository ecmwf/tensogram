# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Example 19 — Message metadata known only at the end of a stream

A streaming producer often cannot describe its own message up front: which of
the requested parameters actually had data, the observed value range, how many
objects ended up written.  ``StreamingEncoder.set_footer_metadata`` records
those facts while encoding; they are merged into the footer metadata frame,
which supersedes the header frame on decode.

The header can therefore carry a provisional value and the footer the final
one — footer ``_extra_`` keys win on conflict.

Run:
    python examples/python/19_footer_metadata.py
"""

import numpy as np
import tensogram

# The parameters a consumer asked for.  Only some of them turn out to have
# data — which ones is not known until the last field has been fetched.
REQUESTED = ["2t", "10u", "tp"]

DESCRIPTOR = {
    "type": "ntensor",
    "ndim": 1,
    "shape": [4],
    "dtype": "float32",
    "encoding": "none",
    "filter": "none",
    "compression": "none",
}


def main() -> None:
    # The header metadata states the request and marks the result provisional:
    # at this point the producer does not yet know what it will be able to
    # write.
    enc = tensogram.StreamingEncoder(
        {"_extra_": {"requested_params": REQUESTED, "status": "provisional"}}
    )

    written = []
    maximum = float("-inf")
    for i, param in enumerate(REQUESTED):
        if param == "tp":
            print(f"  {param}: no data — skipped")
            continue
        values = np.arange(4, dtype=np.float32) + 250.0 + 4 * i
        enc.write_object(DESCRIPTOR, values)
        written.append(param)
        maximum = max(maximum, float(values.max()))
        print(f"  {param}: wrote {values.size} values")

    # Now the message-level facts are known.  Record them for the footer
    # frame; `status` replaces the provisional header value.
    enc.set_footer_metadata(
        {
            "status": "complete",
            "written_params": written,
            "value_maximum": maximum,
        }
    )
    # Repeated calls accumulate, so late additions need no bookkeeping of
    # their own.
    enc.set_footer_metadata({"producer": "footer-metadata-example"})

    message = enc.finish()
    print(f"\nStreamed {len(message)} bytes")

    # ── Decoding sees the final values ──────────────────────────────────────
    decoded = tensogram.decode(message)
    extra = decoded.metadata.extra
    print(f"Decoded {len(decoded.objects)} object(s)")
    print(f"  status:           {extra['status']}")
    print(f"  written_params:   {extra['written_params']}")
    print(f"  value_maximum:    {extra['value_maximum']}")
    print(f"  requested_params: {extra['requested_params']}")

    # A reader that parses only the header metadata frame — a forward scan
    # that never reaches the footer — keeps seeing the provisional value.
    header_only = tensogram.decode_metadata(message)
    print(f"\nHeader-only read still reports status={header_only.extra['status']}")


if __name__ == "__main__":
    main()
