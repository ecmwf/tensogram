# (C) Copyright 2024- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Zarr v3 store backend for tensogram .tgm files.

Provides ``TensogramStore`` — a Zarr v3 ``Store`` that reads and writes
Tensogram wire-format (``.tgm``) files through the standard Zarr API.

Usage::

    import zarr
    from tensogram_zarr import TensogramStore

    # Read existing .tgm through Zarr
    store = TensogramStore.open_tgm("data.tgm")
    root = zarr.open_group(store=store, mode="r")
    arr = root["temperature"][:]

    # Write new .tgm through Zarr
    import numpy as np
    store = TensogramStore("output.tgm", mode="w")
    root = zarr.open_group(store=store, mode="w")
    root.create_array("temperature", data=np.random.rand(100, 200).astype(np.float32))
    store.close()
"""

from tensogram_zarr.store import TensogramStore

__all__ = ["TensogramStore"]
