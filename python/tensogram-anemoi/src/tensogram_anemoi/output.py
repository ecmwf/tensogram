# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import logging
from functools import cached_property
from pathlib import Path

import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.decorators import main_argument
from anemoi.inference.output import Output
from anemoi.inference.types import ProcessorConfig
from anemoi.inference.types import State

LOG = logging.getLogger(__name__)

# Written into base[i]["name"] for lat/lon objects.  These names are
# deliberately NOT in tensogram-xarray's KNOWN_COORD_NAMES so that lat/lon
# objects share the same flat dimension as all field objects rather than each
# spawning its own dimension.  The canonical names are preserved in the "anemoi"
# namespace for downstream consumers.
_COORD_NAME_MAP = {
    "latitude": "grid_latitude",
    "longitude": "grid_longitude",
}


@main_argument("path")
class TensogramOutput(Output):
    """Tensogram output plugin for anemoi-inference.

    Writes each forecast step as one tensogram message appended to a .tgm file
    or streamed over a TCP socket.  Each message contains lat/lon coordinate
    objects followed by one object per field (or one stacked object per
    pressure-level parameter when ``stack_pressure_levels=True``).

    Per-object metadata is stored under the ``"anemoi"`` namespace in CBOR.
    Dimension-name hints are stored in ``_extra_["dim_names"]`` so the
    tensogram-xarray backend can resolve meaningful names without the reader
    having to pass ``dim_names`` explicitly.

    Supports local paths and remote URLs (s3://, gs://, az://, ...) via fsspec.
    Each step is encoded and written immediately -- no full-forecast buffering.

    Pressure-level stacking
    -----------------------
    When ``stack_pressure_levels=True``, all fields sharing the same ``param``
    are stacked into a single 2-D object of shape ``(n_grid, n_levels)``,
    sorted by level in ascending order.  The per-object metadata stores
    ``"levelist": [500, 850, ...]`` instead of the scalar ``"level"``
    key used for unstacked fields.

    Without stacking (default), every field is a separate 1-D object and the
    scalar ``"level"`` key is always stored when it is present in the
    checkpoint's GRIB keys.
    """

    def __init__(
        self,
        context: Context,
        path: str,
        *,
        encoding: str = "none",
        bits: int | None = None,
        compression: str = "zstd",
        dtype: str = "float32",
        storage_options: dict | None = None,
        stack_pressure_levels: bool = False,
        variables: list[str] | None = None,
        post_processors: list[ProcessorConfig] | None = None,
        output_frequency: int | None = None,
        write_initial_state: bool | None = None,
    ) -> None:
        super().__init__(
            context,
            variables=variables,
            post_processors=post_processors,
            output_frequency=output_frequency,
            write_initial_state=write_initial_state,
        )
        if encoding == "simple_packing" and bits is None:
            raise ValueError("bits must be set when encoding='simple_packing'")
        self.path = path
        self.encoding = encoding
        self.bits = bits
        self.compression = compression
        self.dtype = dtype
        self.storage_options = storage_options or {}
        self.stack_pressure_levels = stack_pressure_levels
        self._handle = None

    def __repr__(self) -> str:
        return f"TensogramOutput({self.path})"

    @cached_property
    def _numpy_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, state: State) -> None:
        import fsspec

        path_str = str(self.path)
        if "://" not in path_str:
            Path(path_str).parent.mkdir(parents=True, exist_ok=True)

        self._handle = fsspec.open(path_str, "wb", **self.storage_options).open()
        LOG.info("TensogramOutput: writing to %s", path_str)

    def write_initial_state(self, state: State) -> None:
        """Write the initial state, reducing multi-step fields to the last step."""
        from anemoi.inference.state import reduce_state

        state = reduce_state(state)
        return super().write_initial_state(state)

    def write_step(self, state: State) -> None:
        """Encode one forecast step as a tensogram message and write it immediately."""
        if self._handle is None:
            raise RuntimeError(f"{self!r}: write_step called before open() or after close()")

        global_meta = {
            "version": 2,
            "base": [],
            "_extra_": {},
        }
        descriptors_and_data = []

        step_seconds = state["step"].total_seconds()
        step_hours = int(step_seconds / 3600) if step_seconds % 3600 == 0 else step_seconds / 3600
        base_dt = state["date"] - state["step"]
        mars_extra = {
            "date": base_dt.strftime("%Y%m%d"),
            "time": base_dt.strftime("%H%M"),
            "step": step_hours,
        }

        for coord_name, coord_arr in [
            ("latitude", state["latitudes"]),
            ("longitude", state["longitudes"]),
        ]:
            arr = np.asarray(coord_arr, dtype=np.float64)
            global_meta["base"].append(
                {
                    "name": _COORD_NAME_MAP[coord_name],
                    "anemoi": {"variable": coord_name},
                }
            )
            descriptors_and_data.append(
                (
                    {"type": "ntensor", "shape": list(arr.shape), "dtype": "float64"},
                    arr,
                )
            )

        if self.stack_pressure_levels:
            self._add_fields_stacked(state, global_meta, descriptors_and_data, mars_extra)
        else:
            self._add_fields_flat(state, global_meta, descriptors_and_data, mars_extra)

        n_grid = len(state["latitudes"])
        dim_names_hint: dict[str, str] = {str(n_grid): "values"}
        if self.stack_pressure_levels:
            for _, arr in descriptors_and_data:
                if arr.ndim == 2:
                    level_size = str(arr.shape[1])
                    if level_size not in dim_names_hint:
                        dim_names_hint[level_size] = "level"
        global_meta["_extra_"]["dim_names"] = dim_names_hint

        import tensogram

        msg_bytes = tensogram.encode(global_meta, descriptors_and_data)
        self._handle.write(msg_bytes)

    def close(self) -> None:
        """Flush and close the output stream."""
        if self._handle is not None:
            try:
                self._handle.flush()
            except Exception:
                pass
            self._handle.close()
            self._handle = None

    # ------------------------------------------------------------------
    # Field object builders
    # ------------------------------------------------------------------

    def _add_fields_flat(
        self,
        state: State,
        global_meta: dict,
        descriptors_and_data: list,
        mars_extra: dict,
    ) -> None:
        """Add one object per field (default, no stacking)."""
        for name, values in state["fields"].items():
            if self.skip_variable(name):
                continue
            variable = self.typed_variables.get(name)
            if variable is None:
                LOG.warning("TensogramOutput: no typed variable for %r -- metadata will be incomplete", name)
            grib = getattr(variable, "grib_keys", {}) if variable else {}
            base_entry, descriptor, arr = self._build_field_object(name, grib, values, mars_extra)
            global_meta["base"].append(base_entry)
            descriptors_and_data.append((descriptor, arr))

    def _add_fields_stacked(
        self,
        state: State,
        global_meta: dict,
        descriptors_and_data: list,
        mars_extra: dict,
    ) -> None:
        """Group pressure-level fields by param and stack; write others flat."""
        pl_groups: dict[str, list[tuple[int, str, dict, np.ndarray]]] = {}
        non_pl: list[tuple[str, dict, np.ndarray]] = []

        for name, values in state["fields"].items():
            if self.skip_variable(name):
                continue
            variable = self.typed_variables.get(name)
            if variable is None:
                LOG.warning("TensogramOutput: no typed variable for %r -- metadata will be incomplete", name)
            grib = getattr(variable, "grib_keys", {}) if variable else {}
            if variable is not None and variable.is_pressure_level:
                param = variable.param
                level = variable.level
                pl_groups.setdefault(param, []).append((level, name, grib, values))
            else:
                non_pl.append((name, grib, values))

        if not pl_groups:
            LOG.warning("TensogramOutput: stack_pressure_levels=True but no pressure-level fields found")

        for name, grib, values in non_pl:
            base_entry, descriptor, arr = self._build_field_object(name, grib, values, mars_extra)
            global_meta["base"].append(base_entry)
            descriptors_and_data.append((descriptor, arr))

        for param in sorted(pl_groups):
            group = sorted(pl_groups[param], key=lambda x: x[0])
            base_entry, descriptor, arr = self._build_stacked_object(param, group, mars_extra)
            global_meta["base"].append(base_entry)
            descriptors_and_data.append((descriptor, arr))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_array(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=self._numpy_dtype)
        if self.encoding == "simple_packing":
            arr = arr.astype(np.float64)
        return arr

    def _build_descriptor(self, arr: np.ndarray) -> dict:
        descriptor = {
            "type": "ntensor",
            "shape": list(arr.shape),
            "dtype": arr.dtype.name,
            "encoding": self.encoding,
            "compression": self.compression,
        }
        if self.encoding == "simple_packing" and self.bits is not None:
            import tensogram

            sp_params = tensogram.compute_packing_params(arr.ravel(), self.bits, 0)
            descriptor.update(sp_params)
        return descriptor

    def _build_field_object(
        self,
        name: str,
        grib: dict,
        values: np.ndarray,
        mars_extra: dict,
    ) -> tuple[dict, dict, np.ndarray]:
        """Build (base_entry, descriptor, array) for a single flat field object."""
        mars = {**mars_extra, **grib}
        base_entry: dict = {"name": name, "anemoi": {"variable": name}}
        if mars:
            base_entry["mars"] = mars
        arr = self._prepare_array(values)
        return base_entry, self._build_descriptor(arr), arr

    def _build_stacked_object(
        self,
        param: str,
        group: list[tuple[int, str, dict, np.ndarray]],
        mars_extra: dict,
    ) -> tuple[dict, dict, np.ndarray]:
        """Build (base_entry, descriptor, array) for a stacked pressure-level object."""
        levels = [item[0] for item in group]
        first_grib = group[0][2]

        arrays = [self._prepare_array(item[3]) for item in group]
        stacked = np.column_stack(arrays)

        mars = {**mars_extra, **{k: v for k, v in first_grib.items() if k != "level"}}

        base_entry: dict = {"name": param, "anemoi": {"variable": param, "levelist": levels}}
        if mars:
            base_entry["mars"] = mars
        return base_entry, self._build_descriptor(stacked), stacked
