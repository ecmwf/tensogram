# Tensogram Jupyter Notebook Examples

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ecmwf/tensogram/main?urlpath=lab/tree/examples/jupyter)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ecmwf/tensogram/blob/main/examples/jupyter/01_quickstart_and_mars.ipynb)

A set of narrative notebooks that walk through Tensogram's main features
interactively, with live plots and decoded fields. These complement the
flat `.py` scripts under `examples/python/`: the scripts are minimal
reference snippets; these notebooks are for **learning**.

## The five journeys

| # | Notebook | What you will learn |
|---|----------|---------------------|
| 1 | [`01_quickstart_and_mars.ipynb`](01_quickstart_and_mars.ipynb) | Encode + decode a 2D temperature grid, visualise it, attach MARS metadata, see `base[i]` in action, use `compute_common` to find shared keys. |
| 2 | [`02_encoding_and_fidelity.ipynb`](02_encoding_and_fidelity.ipynb) | Sweep the pipeline stages (`none`, `simple_packing`, `shuffle`, `zstd`, `lz4`, `szip`, `zfp`, `sz3`), plot compression ratio × encode time, inspect fidelity on a lossy codec. |
| 3 | [`03_from_grib_to_tensogram.ipynb`](03_from_grib_to_tensogram.ipynb) | Open real ECMWF opendata GRIB, run `tensogram.convert_grib()`, demonstrate the in-memory `convert_grib_buffer()` entry point, validate the round-trip against raw GRIB values. |
| 4 | [`04_from_netcdf_to_tensogram.ipynb`](04_from_netcdf_to_tensogram.ipynb) | Build a CF-compliant NetCDF in-process, `tensogram.convert_netcdf()`, then open the `.tgm` with `xr.open_dataset(engine="tensogram")` and slice it like a native xarray Dataset. |
| 5 | [`05_validation_and_parallelism.ipynb`](05_validation_and_parallelism.ipynb) | Four validation levels, inject corruption and watch `IssueCode` appear, sweep `threads=0…N` and plot speedup. |

## Setup

### Option 1 — `uv pip install` (fastest)

```bash
# From repo root:
uv venv .venv --python 3.13
source .venv/bin/activate

# Build the Python bindings with GRIB and NetCDF support
# (requires libeccodes + libnetcdf installed at the OS level):
uv pip install maturin
cd python/bindings
maturin develop --features grib,netcdf
cd ../..

# Install notebook deps + the xarray backend extension
uv pip install -e examples/jupyter

# Launch JupyterLab
jupyter lab examples/jupyter/
```

### Option 2 — `conda env create` (batteries-included)

```bash
conda env create -f examples/jupyter/environment.yml
conda activate tensogram-jupyter
jupyter lab examples/jupyter/
```

### Option 3 — Binder / Colab

Click the badges at the top of this README. Binder launches a
fully-configured JupyterLab session with the exact environment.
Colab requires a one-line `!pip install tensogram` at the top of
each notebook.

## OS-level dependencies

Notebooks 03 (GRIB) and 04 (NetCDF) need C libraries installed at
the OS level. They are **not** Python packages.

| Library | macOS (Homebrew) | Debian / Ubuntu |
|---------|------------------|-----------------|
| `libeccodes` (GRIB) | `brew install eccodes` | `apt install libeccodes-dev` |
| `libnetcdf` + `libhdf5` (NetCDF) | `brew install netcdf hdf5` | `apt install libnetcdf-dev` |

If you installed tensogram from PyPI, the wheel does **not** include
GRIB/NetCDF support (the manylinux base image lacks the C libraries).
In that case, notebooks 03 and 04 will detect the missing feature via
`tensogram.__has_grib__` / `tensogram.__has_netcdf__` and raise a clear
`RuntimeError` with a pointer to this README. Rebuild from source
(`maturin develop --features grib,netcdf`) to enable them.

## Running the notebooks end-to-end

```bash
# Every notebook runs in CI via:
pytest --nbval-lax examples/jupyter/ -v
```

`nbval-lax` executes each notebook cell-by-cell and fails on any
exception. Cell outputs are not compared against what is committed in
the `.ipynb` files (those should be empty — see "Output hygiene"
below).

## Output hygiene

Committed notebooks must have **empty cell outputs** (the notebooks
are version-controlled and we do not want output diffs polluting PRs).
Set up the `nbstripout` pre-commit hook once:

```bash
uv pip install nbstripout
nbstripout --install
```

With that hook installed, every `git commit` automatically strips
outputs before staging.

## When something goes wrong

- `RuntimeError: tensogram was built without GRIB support` — rebuild
  with `maturin develop --features grib,netcdf`.
- `ImportError: libeccodes.so.0: cannot open shared object file` —
  install the OS-level C library (see table above).
- `ModuleNotFoundError: No module named 'matplotlib'` — run
  `uv pip install -e examples/jupyter` to pull in the notebook-only
  dependencies.
- Kernel dies on a large field — notebook 02 defaults to 4 M points;
  if your machine is memory-constrained, reduce `NPOINTS` in the
  setup cell.
