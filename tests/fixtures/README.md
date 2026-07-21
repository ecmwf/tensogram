# Shared test fixtures

## `metadata_parity.tgm`

The cross-language metadata-access parity fixture — a deterministic two-object
message whose CBOR metadata frame exercises the full capability matrix in
`plans/METADATA_ACCESS_PARITY.md` §7 (existence vs. a real `0` / `""`, precise
typed access, nested maps, arrays, per-object scoping, `_extra_` fallback, and
`_reserved_` visibility rules).

Regenerate with the `tensogram` Python extension installed:

```
python tests/fixtures/gen_metadata_parity.py
```

The **same bytes** are decoded and asserted against by:

- Rust — `rust/tensogram/tests/metadata_parity.rs` (the oracle)
- Python — `python/tests/test_metadata_parity.py`
- TypeScript — `typescript/tests/metadataParity.test.ts`

The C, C++, and Fortran bindings assert the same contract against an equivalent
in-memory message in their own metadata tests (`cpp/tests/test_meta_value.cpp`,
`fortran/test/test_metadata_value.f90`, and the `tensogram-ffi` unit tests).
