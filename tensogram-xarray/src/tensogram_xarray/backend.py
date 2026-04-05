"""xarray backend entry point for tensogram ``.tgm`` files.

Registers ``engine="tensogram"`` with xarray via the ``xarray.backends``
entry point in ``pyproject.toml``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence

import xarray as xr
from xarray.backends import BackendEntrypoint

from tensogram_xarray.store import TensogramDataStore


class TensogramBackendEntrypoint(BackendEntrypoint):
    """Open tensogram ``.tgm`` files as xarray Datasets.

    Usage::

        import xarray as xr

        # Simple open (single message, generic dim names)
        ds = xr.open_dataset("file.tgm", engine="tensogram")

        # With user-specified dimension names
        ds = xr.open_dataset("file.tgm", engine="tensogram",
                             dim_names=["latitude", "longitude"])

        # With variable naming from metadata
        ds = xr.open_dataset("file.tgm", engine="tensogram",
                             variable_key="mars.param")
    """

    description = "Open tensogram .tgm files in xarray"
    url = "https://github.com/ecmwf/tensogram"

    def open_dataset(  # type: ignore[override]
        self,
        filename_or_obj: str | os.PathLike,
        *,
        drop_variables: Iterable[str] | None = None,
        dim_names: Sequence[str] | None = None,
        variable_key: str | None = None,
        message_index: int = 0,
        merge_objects: bool = False,
        verify_hash: bool = False,
        range_threshold: float = 0.5,
    ) -> xr.Dataset:
        """Open a single tensogram message as an :class:`xr.Dataset`.

        Parameters
        ----------
        filename_or_obj
            Path to a ``.tgm`` file.
        drop_variables
            Variable names to exclude from the Dataset.
        dim_names
            Explicit dimension names for data variables. Must have exactly
            as many entries as the tensor has axes.
        variable_key
            Dotted metadata path (e.g. ``"mars.param"``) whose value at each
            data object becomes the xarray variable name.
        message_index
            Which message to open when the file contains multiple messages.
        merge_objects
            If *True*, attempt to merge objects across messages by stacking
            along metadata dimensions that vary.  When *False* (default),
            only the single message at *message_index* is opened.
        verify_hash
            Whether to verify xxh3 hashes during decode.
        range_threshold
            Maximum fraction of total array elements (0.0-1.0) for which
            partial ``decode_range()`` is used instead of a full
            ``decode_object()``.  Default ``0.5`` (50%).

        Returns
        -------
        xr.Dataset
        """
        file_path = str(filename_or_obj)

        if message_index < 0:
            msg = f"message_index must be >= 0, got {message_index}"
            raise ValueError(msg)

        if merge_objects:
            # Delegate to open_datasets and return the first result.
            from tensogram_xarray.merge import open_datasets

            datasets = open_datasets(
                file_path,
                dim_names=dim_names,
                variable_key=variable_key,
                verify_hash=verify_hash,
                range_threshold=range_threshold,
            )
            if not datasets:
                return xr.Dataset()
            return datasets[0]

        store = TensogramDataStore(
            file_path=file_path,
            msg_index=message_index,
            dim_names=dim_names,
            variable_key=variable_key,
            verify_hash=verify_hash,
            range_threshold=range_threshold,
        )

        drop_set = set(drop_variables) if drop_variables else None
        ds = store.build_dataset(drop_variables=drop_set)
        ds.set_close(store.close)
        return ds

    def guess_can_open(self, filename_or_obj: str) -> bool:  # type: ignore[override]
        """Return *True* for files with ``.tgm`` extension."""
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except (TypeError, AttributeError):
            return False
        return ext.lower() == ".tgm"
