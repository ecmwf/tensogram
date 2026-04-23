# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Reader-shaped modules for tensogram content.

Each submodule exposes the three callables earthkit-data's own reader
registry expects — ``reader`` (file), ``memory_reader`` (bytes), and
``stream_reader`` (stream).  They are consumed internally by
:class:`tensogram_earthkit.source.TensogramSource` but are also shaped
so that moving them into ``earthkit-data/readers/tensogram/`` upstream
later is a verbatim copy.
"""

from __future__ import annotations

from tensogram_earthkit.readers.file import reader
from tensogram_earthkit.readers.memory import memory_reader
from tensogram_earthkit.readers.stream import stream_reader

__all__ = ["memory_reader", "reader", "stream_reader"]
