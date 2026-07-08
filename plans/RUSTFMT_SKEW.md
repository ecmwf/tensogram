# Rustfmt version skew — diagnosis and proposals

Status: **proposals for review** (no repo-wide change applied yet).

## Symptom

`cargo fmt --check` results depend on the rustfmt version:

- The environment's `rustfmt 1.96.0` wants to reorder imports and explode some
  call chains in files that are already committed and (were) CI-clean.
- Different files disagree in *opposite* directions — some want uppercase-first
  imports, some want lowercase-first — so no single `cargo fmt` run leaves the
  tree unchanged.

## Root cause

1. **Crates are `edition = "2024"`** (workspace `Cargo.toml` and each member).
2. rustfmt defaults **`style_edition` to the crate `edition`**. A rustfmt new
   enough to implement the 2024 style (≈ the environment's `1.96.0`) therefore
   applies 2024 formatting rules — which changed import sorting (case ordering)
   and chain-width behaviour relative to the 2021 style.
3. The repo's code was formatted over time by **different rustfmt versions**
   (dev machines, CI `dtolnay/rust-toolchain@stable` floating over time, and —
   during this PR — a mix of `cargo fmt` runs and hand-edits). The result is a
   tree where **some files are in 2021 import style and some in 2024**.

CI uses `dtolnay/rust-toolchain@stable` with no pin, so its rustfmt version
floats — the check is **non-deterministic across time**.

## Evidence

`rustfmt --check` exit codes (0 = clean, 1 = would reformat), forcing
`--edition 2024`:

| file | default (`style_edition=2024`) | `style_edition = "2021"` |
|------|:--:|:--:|
| `tensogram-netcdf/tests/roundtrip.rs` | 1 | **0** |
| `tensogram-netcdf/src/export.rs` | 1 | **0** |
| `tensogram-netcdf/src/converter.rs` | 1 | 1 |
| `tensogram-grib/src/metadata.rs` | 1 | 1 |
| `python/bindings/src/lib.rs` | 1 | 1 |

Two independent layers of divergence:

- **Import ordering (style-edition driven).** `roundtrip.rs` is committed as
  `{convert_netcdf_file, to_netcdf, ConvertOptions}` (lowercase-first = 2021),
  while `converter.rs` is committed as `{DataPipeline, Dtype, encode}`
  (uppercase-first = 2024). `style_edition = "2021"` fixes the first group but
  reports the second as dirty (it wants lowercase-first), and vice-versa.
- **Chain width (version driven, not fixed by `style_edition`).** e.g.
  `metadata.rs` has `a.into_iter().map(..).collect()` on one line (~73 cols)
  that current rustfmt explodes regardless of `style_edition`, because the
  nested-call/`chain_width` heuristic changed between the version that wrote the
  file and `1.96`.

Conclusion: **no `rustfmt.toml` setting alone makes the current tree clean** —
the files are genuinely formatted under different rules. The tree must be
normalized once, under a single pinned toolchain.

## Proposals

### A. `rustfmt.toml` with `style_edition = "2021"` (config only, no reformat)
- **Pro:** Freezes the style-edition-driven divergence (import ordering) so any
  future rustfmt applies 2021 rules; zero code churn.
- **Con:** Does **not** fix the chain-width layer, and does not fix the files
  already committed in 2024 order — so `cargo fmt --check` still fails on a
  subset. Partial only. Also locks the repo to 2021 style indefinitely.

### B. Pin the toolchain via `rust-toolchain.toml` (no reformat)
- Pin `channel = "1.87.0"` (matches the declared `rust-version`) so dev + CI use
  the exact rustfmt that produced most of the tree.
- **Pro:** Deterministic; likely zero reformat if that version matches the tree.
- **Con:** Freezes the **whole** toolchain (compiler + clippy), forgoing newer
  diagnostics; CI must read the pin instead of `@stable`. Heavy-handed, and the
  tree is *already mixed*, so even a pinned old rustfmt may not be fully clean.

### C. One-time repo-wide normalize under a pinned toolchain (recommended)
1. Add `rust-toolchain.toml` pinning a specific stable (e.g. the current CI
   stable) — used for **fmt/clippy determinism**, not necessarily the build MSRV.
2. Add `rustfmt.toml` with an explicit `style_edition` — recommend **`"2024"`**
   to match the crate edition and the future default (or `"2021"` to minimise
   churn; pick one deliberately).
3. Run `cargo fmt --all` **once** with that toolchain; commit as a standalone
   `chore(fmt): normalize formatting under pinned toolchain <ver>` PR (its own
   PR so the large diff never mixes with feature work).
4. Update CI to install the pinned toolchain (or let `dtolnay/rust-toolchain`
   read `rust-toolchain.toml`).
- **Pro:** Tree becomes deterministically clean; future skew impossible; blame
  churn is isolated to one clearly-labelled commit.
- **Con:** One large diff to review; must land on `main` (rebase feature
  branches after).

### D. Do nothing (status quo)
- CI `cargo fmt --check` stays non-deterministic and will re-break whenever the
  floating stable rustfmt changes formatting. **Not recommended.**

## Recommendation

**Proposal C**, as a dedicated PR on `main`, choosing `style_edition = "2024"`
(the crate edition) and pinning the toolchain used for the normalize. Concretely:

```toml
# rust-toolchain.toml
[toolchain]
channel = "1.90.0"        # pick the CI-blessed stable; used for fmt/clippy determinism
components = ["clippy", "rustfmt"]
```

```toml
# rustfmt.toml
style_edition = "2024"
```

Then `cargo fmt --all` once, commit, and switch CI to the pinned toolchain.

## Interaction with this PR

This feature PR does **not** apply any of the above (a repo-wide reformat must
not ride on a feature branch). All **new** code added here is fmt-clean under
`style_edition = "2021"` (the style most of the touched files already use); the
`cargo fmt --check` failures observed on this branch are pre-existing skew in
lines this PR did not author. After Proposal C lands on `main`, rebasing this
branch will normalize everything together.
