# Jupyter Notebook Walk-through

The [`examples/jupyter/`](https://github.com/ecmwf/tensogram/tree/main/examples/jupyter)
directory carries a curated set of narrative notebooks that introduce
Tensogram interactively, with live visualisations. Unlike the flat
`.py` examples under [`examples/python/`](https://github.com/ecmwf/tensogram/tree/main/examples/python)
— which are minimal reference snippets for copy-paste — the notebooks
are for learning.

Every notebook is executed end-to-end on every PR by the
`notebooks` CI job, so they cannot rot.

## The five journeys

| # | Notebook | What you will learn |
|---|----------|---------------------|
| 1 | `01_quickstart_and_mars.ipynb` | Encode & decode a 2D tensor, visualise it, attach MARS metadata, walk the `base` / `_reserved_` / `_extra_` layout. |
| 2 | `02_encoding_and_fidelity.ipynb` | Sweep every encoding × filter × compression combination and plot ratio vs time vs fidelity. |
| 3 | `03_from_grib_to_tensogram.ipynb` | Convert a real ECMWF opendata GRIB2 file with the new Python API (`tensogram.convert_grib` + `tensogram.convert_grib_buffer`). |
| 4 | `04_from_netcdf_to_tensogram.ipynb` | Build a small CF-compliant NetCDF in-process, convert it with `tensogram.convert_netcdf`, and open the result as an xarray Dataset via `engine="tensogram"`. |
| 5 | `05_validation_and_parallelism.ipynb` | Run the four validation levels, inject corruption, sweep `threads=0…N` and plot the speedup. |

## Running the notebooks locally

### Option 1 — `uv pip install` (recommended)

```bash
# Build the Python bindings with GRIB and NetCDF support.
# Requires libeccodes + libnetcdf installed at the OS level.
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install maturin
cd python/bindings
maturin develop --features grib,netcdf
cd ../..

# Install notebook-only dependencies + the xarray backend.
uv pip install -e examples/jupyter

# Launch JupyterLab.
jupyter lab examples/jupyter/
```

### Option 2 — `conda env create`

```bash
conda env create -f examples/jupyter/environment.yml
conda activate tensogram-jupyter
jupyter lab examples/jupyter/
```

### Option 3 — Binder / Colab

Launch badges in the notebook directory's
[`README.md`](https://github.com/ecmwf/tensogram/blob/main/examples/jupyter/README.md)
— zero local install.

## OS-level dependencies

Notebooks 03 (GRIB) and 04 (NetCDF) need C libraries installed *at the
operating system level*. They are not Python packages.

| Library | Needed by | macOS (Homebrew) | Debian / Ubuntu |
|---------|-----------|------------------|-----------------|
| `libeccodes` | notebook 03 | `brew install eccodes` | `apt install libeccodes-dev` |
| `libnetcdf` + `libhdf5` | notebook 04 | `brew install netcdf hdf5` | `apt install libnetcdf-dev libhdf5-dev` |

The official PyPI wheels (`pip install tensogram`) **do not** ship
GRIB / NetCDF support: the `manylinux_2_28` base image lacks the C
libraries. If you try to call `tensogram.convert_grib(...)` on a
wheel without the feature, you get a clean
`RuntimeError("tensogram was built without GRIB support...")` that
points you at this page.

To enable the feature, rebuild from source:

```bash
git clone https://github.com/ecmwf/tensogram
cd tensogram/python/bindings
maturin develop --features grib,netcdf
```

## Running the notebooks in CI

The repository runs the notebooks end-to-end on every PR via a
dedicated `notebooks` job. The gate is:

```bash
pytest --nbval-lax examples/jupyter/ -v
```

`--nbval-lax` executes every cell in every notebook and fails the
build on any exception. Cell outputs are *not* compared — we commit
the notebooks with empty outputs (enforced by the
`python/tests/test_jupyter_structure.py` guard).

## Output hygiene

Committed notebooks must have **empty cell outputs**. Install the
`nbstripout` pre-commit hook once:

```bash
uv pip install nbstripout
nbstripout --install
```

With the hook installed, `git commit` automatically strips outputs.

## Adding a new notebook

1. Copy an existing `.ipynb` as a template.
2. First cell must be a markdown license banner mentioning
   "ECMWF" or "Apache".
3. Last cell must be a "Where to go next" markdown pointer.
4. If you import `matplotlib`, call `matplotlib.use("Agg")` before
   the first `import matplotlib.pyplot`.
5. Update `EXPECTED_NOTEBOOKS` in
   `python/tests/test_jupyter_structure.py`.
6. Link it from `examples/jupyter/README.md` and this guide page.
7. Run `pytest --nbval-lax examples/jupyter/` locally before
   committing.
