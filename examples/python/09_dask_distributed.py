# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""
Example 09 -- Dask distributed computing with tensogram

Demonstrates how tensogram's xarray backend supports Dask natively for
lazy-loading and distributed computation over 4-D tensors.

The example:
  1. Creates 4 .tgm files, each containing 10 data objects that form
     a 4-D tensor (time x level x latitude x longitude).
  2. Opens them lazily with xarray + dask chunking -- no data is loaded
     into memory until explicitly requested.
  3. Distributes statistical computations (mean, std, min, max) across
     the dask task graph.
  4. Shows selective lazy-loading: only the data needed for each
     computation is decoded from disk.

Prerequisites:
  pip install tensogram-xarray "dask[array]"
  # or: pip install -e tensogram-xarray/ && pip install "dask[array]"
"""

import tempfile
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import tensogram
import tensogram_xarray  # side-effect: registers engine="tensogram" with xarray
import xarray as xr

# ---------------------------------------------------------------------------
# Helper: build a descriptor dict
# ---------------------------------------------------------------------------


def _desc(shape, dtype="float32"):
    """Shorthand for a plain ntensor descriptor with no encoding pipeline.

    Application metadata (``name``, ``mars``, ...) is deliberately NOT
    accepted here — put those in ``metadata["base"][i]`` instead.
    """
    return {
        "type": "ntensor",
        "shape": list(shape),
        "dtype": dtype,
        "byte_order": "little",
        "encoding": "none",
        "filter": "none",
        "compression": "none",
    }


# ---------------------------------------------------------------------------
# 1. Create 4 tensogram files with 10 data objects each
# ---------------------------------------------------------------------------

# Dimensions for our 4-D tensor: (time, level, lat, lon)
NTIMES = 4  # one file per timestep
NLEVELS = 10  # 10 pressure levels → 10 data objects per file
NLAT = 36  # 36 latitude points (5° resolution)
NLON = 72  # 72 longitude points (5° resolution)

LEVEL_VALUES = [1000, 925, 850, 700, 500, 400, 300, 200, 100, 50]  # hPa
TIME_DATES = ["20260401", "20260402", "20260403", "20260404"]


def create_test_files(output_dir: Path) -> list[Path]:
    """Create 4 .tgm files, each with 10 data objects (one per level).

    Each data object is a 2-D field (lat x lon) representing temperature
    at a specific pressure level and forecast time step. The full 4-D
    tensor is: temperature(time, level, latitude, longitude).

    Returns the list of created file paths.
    """
    rng = np.random.default_rng(42)
    paths = []

    for t_idx, date in enumerate(TIME_DATES):
        path = output_dir / f"temperature_{date}.tgm"
        paths.append(path)

        # Build coordinate arrays and data objects
        lat = np.linspace(-87.5, 87.5, NLAT, dtype=np.float64)
        lon = np.linspace(0, 355, NLON, dtype=np.float64)

        # objects[i] is the (descriptor, data) pair for data object i.
        # base[i] carries the per-object application metadata that
        # tensogram-xarray and tensogram-zarr consume (name, mars, ...).
        objects = [
            (_desc([NLAT], dtype="float64"), lat),
            (_desc([NLON], dtype="float64"), lon),
        ]
        base = [
            {"name": "latitude"},
            {"name": "longitude"},
        ]

        for level_hpa in LEVEL_VALUES:
            # Simulate temperature: cooler at higher altitudes, slight
            # warming each day, latitude gradient, random weather noise
            base_temp = 288.0 - 6.5 * (1000 - level_hpa) / 100.0
            lat_effect = -20.0 * np.abs(np.sin(np.radians(lat)))[:, np.newaxis]
            time_trend = 0.5 * t_idx
            noise = rng.normal(0, 2.0, (NLAT, NLON)).astype(np.float32)
            field = (base_temp + lat_effect + time_trend + noise).astype(np.float32)

            objects.append((_desc([NLAT, NLON]), field))
            base.append(
                {
                    "name": f"temperature_{level_hpa}hPa",
                    "mars": {"param": "t", "levelist": str(level_hpa), "date": date},
                }
            )

        with tensogram.TensogramFile.create(str(path)) as f:
            f.append({"version": 3, "base": base}, objects)

        print(
            f"  Created {path.name}: {len(objects)} objects ({len(objects) - 2} levels + 2 coords)"
        )

    return paths


# ---------------------------------------------------------------------------
# 2. Open files lazily with Dask
# ---------------------------------------------------------------------------


def open_as_dask_datasets(paths: list[Path]) -> list[xr.Dataset]:
    """Open each .tgm file as a lazy xarray Dataset backed by dask.

    The key parameter is ``chunks={}`` which tells xarray to use dask
    for lazy loading.  No tensor data is decoded until .compute() or
    .values is called.
    """
    datasets = []
    for path in paths:
        # variable_key="name" names each variable from the descriptor's
        # ``name`` field (e.g. "temperature_1000hPa").
        # chunks={} enables dask lazy loading with automatic chunking.
        ds = xr.open_dataset(str(path), engine="tensogram", variable_key="name", chunks={})
        datasets.append(ds)
    return datasets


# ---------------------------------------------------------------------------
# 3. Compute distributed statistics over dask arrays
# ---------------------------------------------------------------------------


def compute_statistics(datasets: list[xr.Dataset]):
    """Compute statistics across all datasets using dask.

    Demonstrates:
      - Lazy array inspection (no data loaded)
      - Distributed mean, std, min, max
      - Selective computation via .compute()
    """
    # ── Inspect lazy arrays (no I/O yet) ──────────────────────────────────
    ds0 = datasets[0]
    print("\n── Lazy Dataset (no data loaded yet) ──")
    print(f"  Variables: {list(ds0.data_vars)}")
    print(f"  Coords:    {list(ds0.coords)}")
    print(f"  Dims:      {dict(ds0.sizes)}")

    # Pick a variable to inspect
    first_var = next(iter(ds0.data_vars))
    arr = ds0[first_var]
    print(f"\n  {first_var}:")
    print(f"    type:   {type(arr.data)}")  # should be dask.array
    print(f"    shape:  {arr.shape}")
    print(f"    dtype:  {arr.dtype}")
    print(f"    chunks: {arr.chunks}")
    assert isinstance(arr.data, da.Array), "Expected dask array for lazy loading!"

    # ── Gather all temperature variables across files ─────────────────────
    # Build the variable list in LEVEL_VALUES order (not alphabetical!)
    # so the stacked axis matches the level labels for statistics.
    temp_vars = [f"temperature_{lev}hPa" for lev in LEVEL_VALUES]
    missing = [v for v in temp_vars if v not in ds0.data_vars]
    if missing:
        raise KeyError(
            f"Expected variables not found in dataset: {missing}. Available: {list(ds0.data_vars)}"
        )
    print(f"\n── Temperature variables per file: {len(temp_vars)} ──")
    for v in temp_vars:
        print(f"    {v}: shape={ds0[v].shape}, dtype={ds0[v].dtype}")

    # Stack all levels from the first file into a single dask array
    level_arrays = [ds0[v].data for v in temp_vars]
    stacked = da.stack(level_arrays, axis=0)  # shape: (levels, lat, lon)
    print(f"  Stacked shape (one file): {stacked.shape}")
    print(f"  Stacked chunks: {stacked.chunks}")

    # Stack across all timesteps (files): shape (time, levels, lat, lon)
    # Verify all datasets have the same variables before stacking.
    for i, ds in enumerate(datasets):
        ds_vars = set(ds.data_vars)
        missing_in_ds = [v for v in temp_vars if v not in ds_vars]
        if missing_in_ds:
            raise KeyError(
                f"Dataset {i} is missing variables: {missing_in_ds}. Available: {sorted(ds_vars)}"
            )

    all_timesteps = []
    for ds in datasets:
        level_arrays = [ds[v].data for v in temp_vars]
        all_timesteps.append(da.stack(level_arrays, axis=0))

    full_4d = da.stack(all_timesteps, axis=0)
    print(f"\n── Full 4-D tensor (lazy) ──")
    print(f"  Shape: {full_4d.shape}  (time x level x lat x lon)")
    print(f"  Size:  {full_4d.nbytes / 1e6:.1f} MB (if fully loaded)")
    print(f"  Type:  {type(full_4d)}")

    # ── Schedule computations (lazy -- no data decoded yet) ──────────────
    print("\n── Scheduling lazy computations ──")
    global_mean = full_4d.mean()
    global_std = full_4d.std()
    global_min = full_4d.min()
    global_max = full_4d.max()
    print("  Scheduled: mean, std, min, max (no computation yet)")

    # ── Compute all at once (this is when data is actually loaded) ────────
    print("\n── Computing (data loaded from .tgm files now) ──")
    results = dask.compute(global_mean, global_std, global_min, global_max)
    mean_val, std_val, min_val, max_val = results

    print(f"  Global mean:   {mean_val:.2f} K")
    print(f"  Global std:    {std_val:.2f} K")
    print(f"  Global min:    {min_val:.2f} K")
    print(f"  Global max:    {max_val:.2f} K")

    # ── Per-level statistics (distributed) ────────────────────────────────
    print("\n── Per-level statistics (distributed over dask) ──")
    level_means = full_4d.mean(axis=(0, 2, 3))  # mean over time, lat, lon
    level_stds = full_4d.std(axis=(0, 2, 3))
    level_mins = full_4d.min(axis=(0, 2, 3))
    level_maxs = full_4d.max(axis=(0, 2, 3))

    # Single compute() call executes all pending operations
    lm, ls, lmn, lmx = dask.compute(level_means, level_stds, level_mins, level_maxs)

    print(f"  {'Level (hPa)':>12s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}")
    print(f"  {'─' * 12}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
    for i, level_hpa in enumerate(LEVEL_VALUES):
        print(f"  {level_hpa:>12d}  {lm[i]:8.2f}  {ls[i]:8.2f}  {lmn[i]:8.2f}  {lmx[i]:8.2f}")

    return full_4d


# ---------------------------------------------------------------------------
# 4. Demonstrate selective lazy-loading
# ---------------------------------------------------------------------------


def demonstrate_lazy_loading(full_4d):
    """Show that only the requested data is loaded from disk.

    When you slice a dask-backed xarray variable, tensogram decodes only
    the requested region (via decode_range for supported compressors, or
    decode_object + in-memory slice otherwise).
    """
    print("\n── Selective lazy-loading ──")

    # Slice: first timestep, first level, equator (lat=18), prime meridian (lon=0)
    point_value = full_4d[0, 0, NLAT // 2, 0].compute()
    print(f"  Single point (t=0, lev=0, equator, lon=0): {point_value:.2f} K")

    # Slice: all times, single level, all spatial points
    single_level = full_4d[:, 5, :, :]  # 400 hPa level
    single_level_mean = single_level.mean().compute()
    print(f"  400 hPa level mean (all times): {single_level_mean:.2f} K")

    # Slice: single file, single level, latitude band
    equatorial = full_4d[0, 0, NLAT // 4 : 3 * NLAT // 4, :]
    equatorial_mean = equatorial.mean().compute()
    print(f"  Equatorial band mean (t=0, 1000hPa): {equatorial_mean:.2f} K")

    # Demonstrate that xarray alignment works with dask
    anomaly = full_4d - full_4d.mean(axis=0)
    anomaly_std = anomaly.std().compute()
    print(f"  Temporal anomaly std: {anomaly_std:.2f} K")


# ---------------------------------------------------------------------------
# 5. Show dask task graph information
# ---------------------------------------------------------------------------


def show_task_graph_info(full_4d):
    """Display information about the dask task graph."""
    print("\n── Dask task graph info ──")

    # Per-level mean: demonstrates how dask decomposes the computation
    level_mean = full_4d.mean(axis=(0, 2, 3))
    graph = level_mean.__dask_graph__()
    print(f"  Tasks in graph for per-level mean: {len(graph)}")
    print(f"  Number of chunks: {full_4d.npartitions}")
    print(f"  Chunk sizes:      {full_4d.chunks}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 64)
    print("Tensogram + Dask: Distributed Computation over 4-D Tensors")
    print("=" * 64)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # 1. Create test files
        print(f"\n1. Creating {NTIMES} files x {NLEVELS} levels ({NLAT}x{NLON} grid each)...")
        paths = create_test_files(tmp_path)

        # 2. Open lazily with dask
        print("\n2. Opening files with dask lazy-loading (engine='tensogram', chunks={})...")
        datasets = open_as_dask_datasets(paths)
        print(f"  Opened {len(datasets)} datasets (all lazy -- no data loaded)")

        # 3. Compute statistics
        print("\n3. Computing statistics across 4-D tensor...")
        full_4d = compute_statistics(datasets)

        # 4. Selective loading
        print("\n4. Demonstrating selective lazy-loading...")
        demonstrate_lazy_loading(full_4d)

        # 5. Task graph info
        show_task_graph_info(full_4d)

    print("\n" + "=" * 64)
    print("All examples passed. Tensogram + Dask integration works!")
    print("=" * 64)


if __name__ == "__main__":
    main()
