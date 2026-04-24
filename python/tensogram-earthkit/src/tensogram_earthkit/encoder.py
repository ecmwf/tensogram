# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""earthkit-data encoder plugin — ``data.to_target(…, encoder="tensogram")``.

The encoder produces a single tensogram message containing one data
object per input field / variable.  It is deliberately simple — no
compression, no filter, no encoding pipeline — so the encoded file is
a lossless mirror of the input values.  Callers who want a tuned
encoding pipeline can drop down to the ``tensogram`` Python API
directly; earthkit integration is about round-trip fidelity first.

The :class:`TensogramEncodedData` envelope is the interface
earthkit-data's :class:`Target` s consume — it exposes ``to_bytes``
and ``to_file``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from earthkit.data.encoders import EncodedData, Encoder

from tensogram_earthkit.mars import field_to_base_entry

__all__ = ["TensogramEncodedData", "TensogramEncoder", "encoder"]


# ---------------------------------------------------------------------------
# EncodedData envelope
# ---------------------------------------------------------------------------


_TENSOGRAM_DTYPES: dict[np.dtype, str] = {
    np.dtype("float16"): "float16",
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
    np.dtype("complex64"): "complex64",
    np.dtype("complex128"): "complex128",
    np.dtype("int8"): "int8",
    np.dtype("int16"): "int16",
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
    np.dtype("uint8"): "uint8",
    np.dtype("uint16"): "uint16",
    np.dtype("uint32"): "uint32",
    np.dtype("uint64"): "uint64",
}


def _numpy_to_tgm_dtype(arr: np.ndarray) -> str:
    """Map a numpy dtype to its tensogram dtype string."""
    name = _TENSOGRAM_DTYPES.get(arr.dtype)
    if name is None:
        raise ValueError(f"unsupported numpy dtype for tensogram encoding: {arr.dtype!r}")
    return name


def _compute_strides(shape: tuple[int, ...]) -> list[int]:
    """C-contiguous (row-major) strides in elements (not bytes)."""
    if not shape:
        return []
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _make_descriptor(arr: np.ndarray) -> dict[str, Any]:
    """Build a minimal tensogram DataObjectDescriptor dict for *arr*.

    No compression, no filter, no encoding — straight-through bytes.
    """
    shape = tuple(arr.shape)
    return {
        "type": "ntensor",
        "dtype": _numpy_to_tgm_dtype(arr),
        "ndim": len(shape),
        "shape": list(shape),
        "strides": _compute_strides(shape),
        "encoding": "none",
        "filter": "none",
        "compression": "none",
    }


class TensogramEncodedData(EncodedData):
    """Bytes of a single tensogram message, ready for any target sink."""

    prefer_file_path = False

    def __init__(self, payload: bytes, metadata: dict | None = None) -> None:
        self._payload = payload
        self._metadata = metadata or {}

    def to_bytes(self) -> bytes:
        return self._payload

    def to_file(self, file: Any) -> None:
        """Write bytes to *file* — path string or file-like object."""
        if isinstance(file, (str, bytes)):
            with open(file, "wb") as fh:
                fh.write(self._payload)
            return
        # file-like object
        file.write(self._payload)

    def metadata(self, key: Any = None) -> Any:
        if key is None:
            return dict(self._metadata)
        return self._metadata.get(key)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class TensogramEncoder(Encoder):
    """Encode earthkit data to a tensogram message."""

    def encode(self, data: Any = None, **kwargs: Any) -> EncodedData:
        if data is None:
            raise ValueError("tensogram encoder requires a data object to encode")

        # Fast-path xarray inputs — no need to round-trip through the
        # earthkit-data wrapper machinery.  xarray is always available
        # because tensogram-xarray is a hard dependency of this package.
        import xarray as xr

        if isinstance(data, (xr.Dataset, xr.DataArray)):
            return self._encode_xarray(data, **kwargs)

        # Double-dispatch through the wrapper machinery the same way
        # NetCDFEncoder does — this lets the encoder accept arbitrary
        # earthkit data types without an explicit dispatch table.
        from earthkit.data.wrappers import get_wrapper

        data = get_wrapper(data, fieldlist=False)
        return data._encode(self, **kwargs)

    def _encode(self, data: Any, **kwargs: Any) -> EncodedData:
        """Generic double-dispatch entry point.

        earthkit's data wrappers call this when they don't know the
        concrete type.  Route on what we can detect: FieldList-ish
        things (``__iter__`` + fields), xarray objects, single fields.
        """
        # Field-like?  Many earthkit Fields have ``metadata`` and
        # ``to_numpy``.
        if (
            hasattr(data, "metadata")
            and hasattr(data, "to_numpy")
            and not hasattr(data, "__len__")
        ):
            return self._encode_field(data, **kwargs)

        # FieldList-like?
        if hasattr(data, "__iter__") and hasattr(data, "__len__"):
            return self._encode_fieldlist(data, **kwargs)

        raise ValueError(
            f"cannot encode {type(data).__name__} as tensogram — "
            "supported inputs are earthkit FieldList / Field and xarray "
            "Dataset / DataArray"
        )

    def _encode_field(self, field: Any, **_kwargs: Any) -> EncodedData:
        return self._encode_fieldlist_like([field])

    def _encode_fieldlist(self, fieldlist: Any, **_kwargs: Any) -> EncodedData:
        return self._encode_fieldlist_like(list(fieldlist))

    def _encode_xarray(self, data: Any, **_kwargs: Any) -> EncodedData:
        import xarray as xr

        if isinstance(data, xr.DataArray):
            # DataArrays without a name would surface as a variable
            # keyed by ``None`` and serialise as the literal string
            # ``"None"`` in the base entry.  Synthesise a stable name
            # so the round-trip stays deterministic.
            if data.name is None:
                data = data.rename("data")
            datasets = [data.to_dataset()]
        elif isinstance(data, xr.Dataset):
            datasets = [data]
        else:  # pragma: no cover - guarded at call site
            raise TypeError(f"expected xarray.Dataset/DataArray, got {type(data)}")

        descriptors: list[dict[str, Any]] = []
        arrays: list[np.ndarray] = []
        base: list[dict[str, Any]] = []

        for ds in datasets:
            for name, var in ds.data_vars.items():
                # Force C-contiguous so the descriptor strides match
                # the raw bytes the encoder will write.  Non-contiguous
                # arrays (transposed / sliced views) would otherwise
                # round-trip to garbage.
                arr = np.ascontiguousarray(var.values)
                descriptors.append(_make_descriptor(arr))
                arrays.append(arr)
                # Stash the xarray variable name so it round-trips.
                entry: dict[str, Any] = {"name": str(name)}
                if var.attrs:
                    entry["_extra_"] = {k: str(v) for k, v in var.attrs.items()}
                base.append(entry)

        if not descriptors:
            raise ValueError("cannot encode an xarray object with no data variables")

        meta = {"base": base, "_extra_": {"encoder": "tensogram-earthkit"}}
        import tensogram

        buf = tensogram.encode(meta, list(zip(descriptors, arrays, strict=True)))
        return TensogramEncodedData(buf)

    # -- helpers ---------------------------------------------------------------

    def _encode_fieldlist_like(self, fields: list[Any]) -> EncodedData:
        if not fields:
            raise ValueError("cannot encode an empty FieldList")

        descriptors: list[dict[str, Any]] = []
        arrays: list[np.ndarray] = []
        base: list[dict[str, Any]] = []
        for field in fields:
            arr = np.asarray(field.to_numpy())
            # If to_numpy returned a flat vector, restore the 2-D shape
            # from the field's own shape attribute so consumers see the
            # grid properly.
            target_shape = tuple(getattr(field, "shape", arr.shape))
            if arr.shape != target_shape and arr.size == int(np.prod(target_shape)):
                arr = arr.reshape(target_shape)
            # Force C-contiguous so the descriptor strides match the
            # raw bytes the encoder will write (see _encode_xarray).
            arr = np.ascontiguousarray(arr)
            descriptors.append(_make_descriptor(arr))
            arrays.append(arr)
            base.append(field_to_base_entry(field))

        meta = {"base": base, "_extra_": {"encoder": "tensogram-earthkit"}}
        import tensogram

        buf = tensogram.encode(meta, list(zip(descriptors, arrays, strict=True)))
        return TensogramEncodedData(buf)


# Module-level attribute that earthkit-data's EncoderLoader picks up.
encoder = TensogramEncoder
