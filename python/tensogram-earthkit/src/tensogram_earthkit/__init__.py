# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""earthkit-data source + encoder plugins for Tensogram ``.tgm`` files.

This package ships two earthkit-data plugins:

* A **source** registered under ``earthkit.data.sources.tensogram`` that lets
  users load ``.tgm`` content via ``ekd.from_source("tensogram", ...)``.
  Local files, remote URLs, in-memory bytes buffers, and byte streams are all
  supported.

* An **encoder** registered under ``earthkit.data.encoders.tensogram`` so that
  any earthkit-data FieldList or xarray object can be written out as a
  ``.tgm`` file via ``data.to_target("file", path, encoder="tensogram")``.

Internally the package is organised in reader-shaped modules
(:mod:`.readers.file`, :mod:`.readers.memory`, :mod:`.readers.stream`) so
the code can be lifted into ``ecmwf/earthkit-data``'s ``readers/`` tree
upstream later without re-structuring.

The public names below are convenience re-exports; the earthkit-data
integration goes through entry points and does not require importing
anything from this module directly.
"""

from __future__ import annotations

from tensogram_earthkit.data import TensogramData
from tensogram_earthkit.detection import TENSOGRM_MAGIC, is_mars_tensogram
from tensogram_earthkit.encoder import TensogramEncodedData, TensogramEncoder
from tensogram_earthkit.fieldlist import TensogramSimpleFieldList, build_fieldlist_from_path
from tensogram_earthkit.source import TensogramSource

__all__ = [
    "TENSOGRM_MAGIC",
    "TensogramData",
    "TensogramEncodedData",
    "TensogramEncoder",
    "TensogramSimpleFieldList",
    "TensogramSource",
    "build_fieldlist_from_path",
    "is_mars_tensogram",
]
