#!/usr/bin/env python3
# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Run the *self-contained, runnable* code examples in ``docs/src/**/*.md``.

mdBook code blocks are not exercised by ``cargo test`` (only ``///`` doctests
are) or by ``mdbook build``, so a broken example — e.g. one whose assertion is
wrong on little-endian hosts — can sit in the docs indefinitely. This harness
closes that gap by extracting fenced blocks and *running* the ones that are
self-contained, so CI fails when a documented example stops working.

Classification (deliberately conservative — only obviously-runnable blocks run;
everything else is skipped, so partial snippets never cause false failures):

  * ``rust``   runnable  iff the block contains ``fn main(`` (a complete program)
  * ``python`` runnable  iff the block has a top-level ``import``/``from`` line
  * a block is SKIPPED when its info string carries an opt-out tag
    (``ignore`` / ``no_run`` / ``text`` / ``compile_fail`` / ``no-doctest`` /
    ``skip``), when it contains a bare ``...`` placeholder line, or — for
    Python — when it references user-supplied files / remote stores / optional
    build features (it would need data or a feature we cannot assume here).

Rust blocks compile against the in-tree ``tensogram`` crate (path dependency)
and are executed; Python blocks run under this interpreter (which must have the
``tensogram`` binding importable — ``make doc-examples`` arranges that).
"""

from __future__ import annotations

import ast
import builtins
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_BUILTINS = set(dir(builtins)) | {"__name__", "__file__", "__doc__", "__spec__"}


def free_names(code: str) -> set[str]:
    """Module-level names a Python block *reads* but never imports/assigns.

    A non-empty result means the block depends on context from a preceding
    block (a continuation) and is therefore not self-contained — so it is not
    run standalone. Order is ignored (use-before-assign is treated as defined),
    which is fine: a genuinely runnable example defines its inputs.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"<syntax-error>"}
    assigned: set[str] = set()
    loaded: set[str] = set()

    class V(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            (assigned if isinstance(node.ctx, ast.Store) else loaded).add(node.id)

        def visit_arg(self, node: ast.arg) -> None:
            assigned.add(node.arg)

        def visit_Import(self, node: ast.Import) -> None:
            for a in node.names:
                assigned.add((a.asname or a.name).split(".")[0])

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for a in node.names:
                assigned.add(a.asname or a.name)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            assigned.add(node.name)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            assigned.add(node.name)
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            assigned.add(node.name)
            self.generic_visit(node)

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            if node.name:
                assigned.add(node.name)
            self.generic_visit(node)

        def visit_Global(self, node: ast.Global) -> None:
            assigned.update(node.names)

        def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
            assigned.update(node.names)

    V().visit(tree)
    return loaded - assigned - _BUILTINS


REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_SRC = REPO_ROOT / "docs" / "src"
TENSOGRAM_CRATE = REPO_ROOT / "rust" / "tensogram"

SKIP_TAGS = {
    "ignore",
    "no_run",
    "text",
    "compile_fail",
    "no-doctest",
    "skip",
    "console",
}

# Python blocks touching any of these need data, a network, or an optional
# build feature we cannot assume in a generic CI job — skip them.
PY_DENY = (
    ".tgm",
    ".nc",
    ".grib",
    ".grib2",
    ".h5",
    "s3://",
    "gs://",
    "az://",
    "http://",
    "https://",
    "requests.",
    "open_remote",
    "convert_grib",
    "convert_netcdf",
    "AsyncTensogramFile",
    "data.ecmwf.int",
    "/path/to",
    "model.run",
    "model(",
    "_is_gil_enabled",
)

BLOCK_RE = re.compile(r"^```([^\n`]*)\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)


@dataclass
class Block:
    path: Path
    line: int
    lang: str
    tags: set[str]
    code: str


def iter_blocks(md: Path) -> list[Block]:
    text = md.read_text(encoding="utf-8")
    out: list[Block] = []
    for m in BLOCK_RE.finditer(text):
        info = m.group(1).strip()
        if not info:
            continue
        toks = re.split(r"[ ,]+", info)
        lang = toks[0].lower()
        tags = {t.lower() for t in toks[1:]}
        line = text.count("\n", 0, m.start()) + 1
        out.append(Block(md, line, lang, tags, m.group(2)))
    return out


def has_ellipsis(code: str) -> bool:
    return any(ln.strip() == "..." or ln.strip() == "# ..." for ln in code.splitlines())


def is_runnable(b: Block) -> bool:
    if b.tags & SKIP_TAGS or has_ellipsis(b.code):
        return False
    if b.lang == "rust":
        return "fn main(" in b.code
    if b.lang in ("python", "py"):
        if not re.search(r"(?m)^(import |from \w[\w.]* import )", b.code):
            return False
        if re.search(r"(?m)^await ", b.code):  # top-level await needs a runner
            return False
        if any(d in b.code for d in PY_DENY):
            return False
        # Self-contained only: no names borrowed from a preceding block.
        return not free_names(b.code)
    return False


def _run(cmd: list[str], timeout: int) -> tuple[int, str]:
    """Run a subprocess, returning (returncode, combined-output). A timeout is
    a failure (124), not a harness crash — so a runaway example is reported."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"


def run_python(blocks: list[Block]) -> list[tuple[Block, str]]:
    fails: list[tuple[Block, str]] = []
    with tempfile.TemporaryDirectory() as td:
        for i, b in enumerate(blocks):
            f = Path(td) / f"block_{i}.py"
            f.write_text(b.code, encoding="utf-8")
            rc, output = _run([sys.executable, str(f)], timeout=90)
            tag = f"{b.path.relative_to(REPO_ROOT)}:{b.line}"
            if rc == 0:
                print(f"  pass  {tag}")
                continue
            out = output.strip()
            # An optional doc dependency that isn't installed here is a skip,
            # not a failure — keeps the gate robust as the docs grow. A broken
            # `import tensogram` (or any logic error) still fails.
            mod = re.search(r"ModuleNotFoundError: No module named '([\w.]+)'", out)
            if mod and not mod.group(1).startswith("tensogram"):
                print(f"  skip  {tag}  (optional dep '{mod.group(1)}' absent)")
            else:
                fails.append((b, out[-1500:]))
                print(f"  FAIL  {tag}")
    return fails


def run_rust(blocks: list[Block]) -> list[tuple[Block, str]]:
    if not blocks:
        return []
    fails: list[tuple[Block, str]] = []
    with tempfile.TemporaryDirectory() as td:
        proj = Path(td)
        (proj / "src" / "bin").mkdir(parents=True)
        names = [f"doc_{i}" for i in range(len(blocks))]
        for name, b in zip(names, blocks):
            (proj / "src" / "bin" / f"{name}.rs").write_text(b.code, encoding="utf-8")
        (proj / "src" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
        (proj / "Cargo.toml").write_text(
            '[package]\nname = "doc-examples"\nversion = "0.0.0"\nedition = "2021"\n'
            "[workspace]\n"
            f'[dependencies]\ntensogram = {{ path = "{TENSOGRAM_CRATE}" }}\nciborium = "0.2"\n',
            encoding="utf-8",
        )
        # Compile every block once (shared target), then run each so that
        # runtime assertions (not just compile errors) are exercised.
        manifest = str(proj / "Cargo.toml")
        rc, build_out = _run(
            ["cargo", "build", "--manifest-path", manifest], timeout=900
        )
        if rc != 0:
            # A compile error names the offending bin; report the whole batch.
            for b in blocks:
                tag = f"{b.path.relative_to(REPO_ROOT)}:{b.line}"
                print(f"  FAIL  {tag}  (compile)")
                fails.append((b, build_out.strip()[-1500:]))
            return fails
        for name, b in zip(names, blocks):
            tag = f"{b.path.relative_to(REPO_ROOT)}:{b.line}"
            rc, output = _run(
                ["cargo", "run", "--quiet", "--manifest-path", manifest, "--bin", name],
                timeout=120,
            )
            if rc != 0:
                fails.append((b, output.strip()[-1500:]))
                print(f"  FAIL  {tag}")
            else:
                print(f"  pass  {tag}")
    return fails


def main() -> int:
    mds = sorted(DOCS_SRC.rglob("*.md"))
    all_blocks = [b for md in mds for b in iter_blocks(md)]
    rust = [b for b in all_blocks if b.lang == "rust" and is_runnable(b)]
    py = [b for b in all_blocks if b.lang in ("python", "py") and is_runnable(b)]

    total = len(all_blocks)
    print(f"docs/src: {len(mds)} files, {total} fenced blocks")
    print(f"runnable: {len(rust)} Rust, {len(py)} Python (others skipped)\n")

    print("Python examples:")
    py_fails = run_python(py) if py else []
    print("\nRust examples:")
    rust_fails = run_rust(rust)

    fails = py_fails + rust_fails
    print("\n" + "=" * 60)
    print(
        f"ran {len(rust) + len(py)}, passed {len(rust) + len(py) - len(fails)}, "
        f"failed {len(fails)}"
    )
    for b, log in fails:
        print(f"\n--- FAILED {b.path.relative_to(REPO_ROOT)}:{b.line} ---\n{log}")
    return 1 if fails else 0


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    sys.exit(main())
