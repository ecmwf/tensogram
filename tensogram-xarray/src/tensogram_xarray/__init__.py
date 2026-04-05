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
