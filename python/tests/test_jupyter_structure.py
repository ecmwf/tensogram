# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Structural integrity checks for ``examples/jupyter/*.ipynb``.

These tests do **not** execute notebook cells — ``nbval`` does that in CI.
They run on every PR (via ``pytest python/tests/``) and guard the
*static* properties of each notebook:

- valid nbformat JSON that passes ``nbformat.validate``,
- cells carry ``id`` fields (no ``MissingIDFieldWarning``),
- no cell outputs committed (enforced so PR diffs stay clean),
- first cell is a markdown license header,
- last cell is a "where to go next" pointer.

The checks are intentionally lightweight so the bar for adding a new
notebook is low: follow the template, pass ``nbstripout``, done.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NB_DIR = Path(__file__).resolve().parents[2] / "examples" / "jupyter"

# Notebooks present in the repo right now.  We list them explicitly so
# the test fails loudly if someone renames / deletes one without updating
# the README + docs index.
EXPECTED_NOTEBOOKS = [
    "01_quickstart_and_mars.ipynb",
    "02_encoding_and_fidelity.ipynb",
    "03_from_grib_to_tensogram.ipynb",
    "04_from_netcdf_to_tensogram.ipynb",
    "05_validation_and_parallelism.ipynb",
]


def _load_notebook(name: str) -> dict:
    path = NB_DIR / name
    assert path.exists(), f"expected notebook {name!r} under examples/jupyter/"
    return json.loads(path.read_text())


def test_examples_jupyter_directory_exists() -> None:
    """The canonical notebook location exists and has a README."""
    assert NB_DIR.is_dir()
    assert (NB_DIR / "README.md").is_file(), "README.md is required next to the notebooks"
    assert (NB_DIR / "pyproject.toml").is_file(), (
        "pyproject.toml is required for `uv pip install -e examples/jupyter`"
    )


def test_expected_notebooks_present() -> None:
    """The five journey notebooks are all there and the set has not drifted."""
    present = sorted(p.name for p in NB_DIR.glob("*.ipynb"))
    assert present == sorted(EXPECTED_NOTEBOOKS), (
        f"notebook set drifted: got {present}, expected {sorted(EXPECTED_NOTEBOOKS)}"
    )


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_is_valid_nbformat(name: str) -> None:
    """Each notebook is valid nbformat v4."""
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(str(NB_DIR / name), as_version=4)
    nbformat.validate(nb)  # raises if invalid


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_cells_have_ids(name: str) -> None:
    """Every cell must carry an ``id`` field (nbformat 4.5+)."""
    nb = _load_notebook(name)
    for idx, cell in enumerate(nb["cells"]):
        assert cell.get("id"), f"{name!r} cell[{idx}] missing id"


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_no_committed_outputs(name: str) -> None:
    """Committed notebooks must have empty cell outputs (nbstripout-clean).

    This is what keeps PR diffs small and deterministic. If this fails,
    run ``nbstripout examples/jupyter/*.ipynb`` and re-commit.
    """
    nb = _load_notebook(name)
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        outputs = cell.get("outputs", [])
        exec_count = cell.get("execution_count")
        assert outputs == [], (
            f"{name!r} cell[{idx}] has committed output "
            f"(run `nbstripout examples/jupyter/*.ipynb` before committing)"
        )
        assert exec_count is None, (
            f"{name!r} cell[{idx}] has execution_count={exec_count!r} "
            f"(run `nbstripout examples/jupyter/*.ipynb` before committing)"
        )


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_first_cell_is_license_header(name: str) -> None:
    """First cell is a markdown cell carrying the Apache-2.0 attribution.

    We don't check the exact phrasing — just that it's a markdown cell
    containing either "ECMWF" or "Apache" so the license banner is
    always the first thing a reader sees.
    """
    nb = _load_notebook(name)
    assert nb["cells"], f"{name!r} has no cells"
    first = nb["cells"][0]
    assert first["cell_type"] == "markdown", (
        f"{name!r} first cell must be markdown, got {first['cell_type']!r}"
    )
    joined = "".join(first["source"])
    has_attribution = "ECMWF" in joined or "Apache" in joined
    assert has_attribution, f"{name!r} first cell must carry the ECMWF / Apache-2.0 license banner"


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_notebook_ends_with_next_steps(name: str) -> None:
    """Every notebook ends with a "where to go next" markdown pointer.

    Teaches readers how to walk the sequence and catches accidental
    truncation during editing.
    """
    nb = _load_notebook(name)
    trailing_md = [c for c in nb["cells"][-3:] if c["cell_type"] == "markdown"]
    joined = "\n".join("".join(c["source"]) for c in trailing_md).lower()
    assert "next" in joined or "further" in joined, (
        f"{name!r} should end with a 'where to go next' markdown section"
    )


@pytest.mark.parametrize("name", EXPECTED_NOTEBOOKS)
def test_uses_matplotlib_agg_if_plotting(name: str) -> None:
    """Notebooks that import matplotlib must call ``matplotlib.use('Agg')``.

    Keeps the same code running headlessly in CI without needing a GUI
    backend.
    """
    nb = _load_notebook(name)
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    any_matplotlib = any("import matplotlib" in s for s in sources)
    if not any_matplotlib:
        pytest.skip(f"{name!r} does not use matplotlib")
    uses_agg = any('matplotlib.use("Agg")' in s or "matplotlib.use('Agg')" in s for s in sources)
    assert uses_agg, (
        f"{name!r} uses matplotlib but does not select the Agg backend "
        f"— add `matplotlib.use('Agg')` before the first `import matplotlib.pyplot` call"
    )
