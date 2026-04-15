# (C) Copyright 2024- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""tensogram-xarray: xarray backend engine for tensogram .tgm files.

Provides ``engine="tensogram"`` for ``xr.open_dataset()`` and a top-level
``open_datasets()`` function for multi-message .tgm files that auto-groups
incompatible objects into separate Datasets.
"""

from __future__ import annotations

from tensogram_xarray.backend import TensogramBackendEntrypoint
from tensogram_xarray.merge import open_datasets

__all__ = [
    "TensogramBackendEntrypoint",
    "open_datasets",
]
