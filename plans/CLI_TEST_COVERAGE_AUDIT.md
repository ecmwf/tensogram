# Tensogram CLI Test Coverage Audit

**Date:** April 6, 2026  
**Scope:** Complete flag and command coverage analysis  
**Conclusion:** Significant test gaps across multiple commands

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Commands** | 11 (10 + ConvertGrib with feature gate) |
| **Total Command Flags** | 28 |
| **Commands With Any Tests** | 5 of 11 (45%) |
| **Commands With ZERO Tests** | 6 of 11 (55%) |
| **Flags Tested** | ~12 of 28 (43%) |
| **Flags NOT Tested** | ~16 of 28 (57%) |
| **Test Count** | 22 unit tests |

---

## Detailed Command Audit

### 1. **`info`** ✅ HAS TESTS (3 tests)

**Flags/Arguments:**
- `files` (positional, required) — Path list

**Tests Present:**
- ✅ `info_single_file()` — Multiple messages
- ✅ `info_empty_file()` — Zero messages
- ✅ `info_missing_file()` — Error handling (nonexistent file)

**Coverage Status:** **GOOD** (100%)
- Covers basic flow, edge case (empty file), error case

**Test File:** `src/commands/info.rs` (lines 26–78)

---

### 2. **`ls`** ❌ NO TESTS

**Flags/Arguments:**
- `-w, --where-clause` `[String]` — Optional filter
- `-p, --keys` `[String]` — Comma-separated keys (default: "version")
- `-j, --json` `bool` — JSON output flag
- `files` (positional, required) — Path list

**Flags Tested:**
- ❌ `-w` (where-clause)
- ❌ `-p` (keys)
- ❌ `-j` (json output)
- ❌ `files` argument

**Coverage Status:** **CRITICAL GAP**
- Zero integration tests for the command
- Filter logic (`filter::parse_where`, `filter::matches`) IS tested in `filter.rs` module
- Output formatting (`output::format_table_row`, `output::format_json`) is NOT tested

**Recommended Tests:**
1. Test `-j` flag outputs valid JSON
2. Test `-p` with multiple keys (comma-separated)
3. Test `-p` with nested keys (e.g., `mars.param`)
4. Test `-w` filtering with `=` operator
5. Test `-w` filtering with `!=` operator
6. Test combination: `-w` + `-p` + `-j`
7. Test multiple files
8. Test missing file error

---

### 3. **`dump`** ❌ NO TESTS

**Flags/Arguments:**
- `-w, --where-clause` `[String]` — Optional filter
- `-p, --keys` `[String]` — Comma-separated keys (default: all)
- `-j, --json` `bool` — JSON output flag
- `files` (positional, required) — Path list

**Flags Tested:**
- ❌ `-w` (where-clause)
- ❌ `-p` (keys)
- ❌ `-j` (json output)
- ❌ `files` argument

**Coverage Status:** **CRITICAL GAP**
- Zero integration tests
- Depends on filter + output modules (partially tested)
- Per-object descriptor decoding logic NOT tested

**Recommended Tests:**
1. Test `-j` outputs JSON with `objects` array
2. Test `-p` filters keys in JSON output
3. Test full dump without flags
4. Test `-w` + `-j` combination
5. Test multi-object messages
6. Test object descriptor summaries in text format

---

### 4. **`get`** ❌ NO TESTS

**Flags/Arguments:**
- `-w, --where-clause` `[String]` — Optional filter
- `-p, --keys` `String` — Comma-separated keys (REQUIRED, strict)
- `files` (positional, required) — Path list

**Flags Tested:**
- ❌ `-w` (where-clause)
- ❌ `-p` (keys)
- ❌ `files` argument

**Coverage Status:** **CRITICAL GAP**
- Zero tests
- Uses `lookup_key()` (tested in filter module)
- Missing key error handling is NOT tested in this context

**Recommended Tests:**
1. Test basic key extraction
2. Test multiple keys (comma-separated)
3. Test missing key (should error)
4. Test `-w` filtering + key extraction
5. Test key not found error message quality
6. Test nested key access (e.g., `mars.param`)
7. Test multiple files with `-w`

---

### 5. **`set`** ✅ HAS TESTS (1 test)

**Flags/Arguments:**
- `-s, --set-values` `String` — Key=value pairs (REQUIRED, comma-separated)
- `-w, --where-clause` `[String]` — Optional filter
- `input` (positional, required) — Input file path
- `output` (positional, required) — Output file path

**Flags Tested:**
- ✅ `-s` with metadata mutation
- ❌ `-w` (where-clause filtering)
- ❌ Immutable key validation

**Coverage Status:** **PARTIAL** (~30%)
- ✅ `test_set_preserves_hash()` — Validates hash preservation and basic set operation
- Missing tests for:
  - Where-clause filtering before mutation
  - Immutable key rejection (CRITICAL: no test for this validation!)
  - Multiple key mutations
  - Nested key paths (e.g., `objects.0.param`)
  - Global metadata vs. per-object params

**Test File:** `src/commands/set.rs` (lines 200–282)

**Recommended Tests:**
1. Test immutable key rejection (e.g., `set_values="dtype=float64"`)
2. Test multiple mutations in single `-s` value
3. Test nested object parameter mutation (`objects.0.key=value`)
4. Test `-w` filtering + mutation
5. Test error on missing input file
6. Test invalid set-values format (e.g., no `=`)
7. Test unicode/special characters in values

---

### 6. **`copy`** ❌ NO TESTS (1 test, placeholder only)

**Flags/Arguments:**
- `-w, --where-clause` `[String]` — Optional filter
- `input` (positional, required) — Input file path
- `output` (positional, required) — Output template (supports `[key]` placeholders)

**Tests Present:**
- ✅ `expand_placeholders_uses_unknown_for_missing_keys()` — Unit test of helper function only
- ❌ NO integration tests of `run()`

**Coverage Status:** **CRITICAL GAP**
- Helper function tested, but main command flow NOT tested
- No tests for:
  - Simple copy (no placeholders)
  - Splitting with `[key]` placeholders
  - Where-clause filtering

**Recommended Tests:**
1. Test simple copy (all messages to one file)
2. Test copy with `-w` filter
3. Test placeholder expansion with `[version]`
4. Test placeholder expansion with nested keys `[mars.param]`
5. Test multiple placeholders in template
6. Test missing placeholder key (defaults to "unknown")
7. Test file creation/append behavior with placeholders
8. Test invalid placeholder syntax

---

### 7. **`merge`** ✅ HAS TESTS (7 tests)

**Flags/Arguments:**
- `inputs` (positional, required) — Input file list
- `-o, --output` `PathBuf` — Output file (REQUIRED)
- `-s, --strategy` `String` — Merge strategy (default: "first", values: "first"/"last"/"error")

**Flags Tested:**
- ✅ `-s first` strategy (keep first value)
- ✅ `-s last` strategy (overwrite with last)
- ✅ `-s error` strategy (fail on conflict)
- ✅ Strategy parsing validation
- ❌ `-o` output flag (implicit in tests)
- ❌ Multiple input files (integration)

**Coverage Status:** **GOOD for unit logic, GAPS for integration** (~60%)
- ✅ Tests focus on `merge_key()` internal function
- ✅ Strategy parsing validated
- Missing tests:
  - Full end-to-end merge with real files
  - Multiple input files
  - Object concatenation behavior
  - Metadata map merging

**Test File:** `src/commands/merge.rs` (lines 130–215)

**Recommended Tests:**
1. Test full merge workflow (create 2+ input files, verify merged output)
2. Test conflict handling across 3+ files
3. Test common/reserved/extra/payload map merging
4. Test error handling for empty input list
5. Test output file creation
6. Test invalid strategy string

---

### 8. **`split`** ❌ MINIMAL TESTS (2 tests, helper only)

**Flags/Arguments:**
- `input` (positional, required) — Input file path
- `-o, --output` `String` — Output template (default behavior: `filename_NNNN.ext`, explicit: `[index]` placeholder)

**Tests Present:**
- ✅ `expand_index_with_placeholder()` — Unit test of helper
- ✅ `expand_index_without_placeholder()` — Unit test of helper
- ❌ NO integration tests of `run()`

**Coverage Status:** **CRITICAL GAP**
- Only helper functions tested
- No end-to-end split tests
- Template expansion is tested, but actual file splitting is NOT

**Recommended Tests:**
1. Test single-object message (write as-is)
2. Test multi-object message (split into separate files)
3. Test payload preservation per object
4. Test `[index]` placeholder expansion
5. Test default filename generation
6. Test output directory creation
7. Test message counting output message
8. Test per-object metadata preservation (mars keys, etc.)

---

### 9. **`reshuffle`** ❌ NO TESTS

**Flags/Arguments:**
- `input` (positional, required) — Input file path
- `-o, --output` `PathBuf` — Output file (REQUIRED)

**Flags Tested:**
- ❌ No tests at all

**Coverage Status:** **CRITICAL GAP**
- Zero tests
- Straightforward decode → re-encode flow, but untested

**Recommended Tests:**
1. Test message re-encoding (streaming → header mode)
2. Test metadata preservation
3. Test multi-message files
4. Test output file creation
5. Test error handling (invalid input)
6. Verify header/footer frame reordering in binary output

---

### 10. **`convert_grib`** ❌ NO TESTS

**Flags/Arguments:**
- `inputs` (positional, required) — GRIB file list
- `-o, --output` `[String]` — Output file (optional, defaults to stdout)
- `--split` `bool` — Each GRIB message → separate Tensogram message
- `--all_keys` `bool` — Preserve all GRIB namespace keys under "grib" sub-object

**Flags Tested:**
- ❌ `-o` (output file)
- ❌ `--split` flag
- ❌ `--all_keys` flag
- ❌ stdin output path

**Coverage Status:** **CRITICAL GAP** (Feature-gated, but NO tests)
- Zero tests despite complex logic
- Delegated to `tensogram_grib` library, but CLI orchestration NOT tested

**Recommended Tests:**
1. Test `--split` flag (one GRIB → multiple Tensogram messages)
2. Test default behavior (merge all)
3. Test `-o` output file creation
4. Test stdout output (no `-o`)
5. Test `--all_keys` preservation of GRIB metadata
6. Test multiple input files
7. Test error on missing input files

---

### 11. **Supporting Modules**

#### **`filter.rs`** — Where-clause parser ✅ WELL TESTED (9 tests)

**Tested Functions:**
- ✅ `parse_where()` — Eq and NotEq operators
- ✅ `lookup_key()` — Depth 1, 2, 3+ nested lookups
- ✅ `lookup_key()` — Payload fallback
- ✅ Invalid where-clause syntax
- ✅ Missing path handling

**NOT Tested:**
- ❌ `matches()` filter logic (only internal functions tested)
- ❌ CBOR value edge cases (Float, Bool, Array, nested Map)
- ❌ Integer formatting edge cases (large i128 values)

**Recommended Additional Tests:**
1. Test `matches()` function directly with Eq/NotEq
2. Test Float CBOR value stringification
3. Test Bool/Null value conversion
4. Test deeply nested CBOR structures

#### **`output.rs`** — JSON/table formatting ❌ NO TESTS

**Functions:**
- `format_table_row()` — Outputs tab-separated values
- `format_json()` — Full metadata to JSON, with optional key filtering
- `format_json_value()` — Single CBOR value to JSON
- `cbor_to_json()` — CBOR → serde_json recursion

**Coverage Status:** **ZERO**
- No tests for any output formatting
- CBOR → JSON conversion untested
- Key filtering logic untested

**Recommended Tests:**
1. Test `format_table_row()` with missing keys (should show "N/A")
2. Test `format_json()` with key filtering
3. Test `format_json()` with objects array
4. Test `cbor_to_json()` for all CBOR types (Integer, Float, Text, Bool, Array, Map, Bytes, Tag)
5. Test hex encoding of binary data

---

## Summary Table: Commands and Test Status

| Command | Flags | Tested Flags | Test Count | Status | Priority |
|---------|-------|--------------|-----------|--------|----------|
| `info` | 1 | 1 | 3 | ✅ | — |
| `ls` | 4 | 0 | 0 | ❌ | 🔴 CRITICAL |
| `dump` | 4 | 0 | 0 | ❌ | 🔴 CRITICAL |
| `get` | 3 | 0 | 0 | ❌ | 🔴 CRITICAL |
| `set` | 4 | 1 | 1 | ⚠️ | 🟠 HIGH |
| `copy` | 3 | 0 (helper only) | 1 | ❌ | 🔴 CRITICAL |
| `merge` | 3 | 2 | 7 | ✅ | — |
| `split` | 2 | 0 (helper only) | 2 | ❌ | 🔴 CRITICAL |
| `reshuffle` | 2 | 0 | 0 | ❌ | 🟠 HIGH |
| `convert_grib` | 4 | 0 | 0 | ❌ | 🟠 HIGH |
| **TOTAL** | **28** | **~5** | **22** | — | — |

---

## Test Coverage by Category

### ✅ Well-Covered
- `info` (100%)
- `merge` strategies (100% of unit logic)
- `filter.rs` (9 tests)

### ⚠️ Partially Covered
- `set` (hash preservation tested, immutable keys NOT tested)
- `split` & `copy` (helpers tested, main logic NOT tested)

### 🔴 CRITICAL GAPS
- `ls` — Core command, NO tests
- `dump` — Core command, NO tests
- `get` — Core command, NO tests
- `reshuffle` — Simple but untested
- `convert_grib` — Complex, untested
- `output.rs` — All formatting functions untested

---

## Recommended Testing Strategy (Priority Order)

### Phase 1 (Critical) — Core Query Commands
**Target:** `ls`, `dump`, `get`, and `output.rs`

```
Commands/Ls Tests (8 tests):
  ✓ Basic list (all messages)
  ✓ With -w filter (equals)
  ✓ With -w filter (not-equals)
  ✓ With -p keys (custom keys)
  ✓ With -j (JSON output)
  ✓ Combination: -w + -p + -j
  ✓ Multiple input files
  ✓ Missing file error

Commands/Dump Tests (6 tests):
  ✓ Full dump (text format)
  ✓ Full dump (JSON format)
  ✓ With -w filtering
  ✓ With -p keys
  ✓ Multi-object messages
  ✓ Object descriptor summaries

Commands/Get Tests (7 tests):
  ✓ Basic key extraction
  ✓ Multiple keys
  ✓ Missing key error (strict)
  ✓ With -w filtering
  ✓ Nested key lookup
  ✓ Multiple files with -w
  ✓ Output format (space-separated)

Output Tests (5 tests):
  ✓ format_table_row with N/A fallback
  ✓ format_json with key filtering
  ✓ format_json with objects
  ✓ cbor_to_json all types
  ✓ Hex encoding of bytes
```

### Phase 2 (High) — Mutation & Reshuffle
**Target:** `set` immutable key validation, `reshuffle`

```
Set Tests (5 tests):
  ✓ Reject immutable keys (dtype, shape, etc.)
  ✓ Multiple mutations in -s
  ✓ Nested object paths (objects.0.key)
  ✓ -w filtering + mutation
  ✓ Invalid set-values format

Reshuffle Tests (4 tests):
  ✓ Decode → re-encode (streaming → header)
  ✓ Metadata preservation
  ✓ Multi-message files
  ✓ Binary output validation
```

### Phase 3 (Medium) — File Operations
**Target:** `split`, `copy`, `convert_grib`

```
Split Tests (6 tests):
  ✓ Single-object passthrough
  ✓ Multi-object splitting
  ✓ Payload per-object preservation
  ✓ [index] template expansion
  ✓ Default filename generation
  ✓ Output message counting

Copy Tests (6 tests):
  ✓ Simple copy (all to one)
  ✓ With -w filtering
  ✓ [key] placeholder expansion
  ✓ [nested.key] placeholders
  ✓ Missing key → "unknown"
  ✓ File append behavior

ConvertGrib Tests (4 tests):
  ✓ --split flag behavior
  ✓ Default merge behavior
  ✓ -o output file creation
  ✓ --all_keys metadata preservation
```

---

## Edge Cases Not Yet Covered

1. **Unicode & special characters** in key values and filenames
2. **Large files** (> 1GB) — streaming behavior
3. **Concurrent file access** (multiple threads reading same file)
4. **CBOR type coercion** (Integer as Float, etc.)
5. **Deeply nested metadata** (5+ levels deep)
6. **Empty metadata maps** (common = {}, extra = {})
7. **Binary data in CBOR** (bytes → hex encoding)
8. **Invalid UTF-8** in file paths
9. **Permission errors** (read-only files)
10. **Disk full** scenarios
11. **Circular placeholder references** (edge case in copy)
12. **Very long template names** (filesystem limits)

---

## Verification Checklist for Test Implementation

- [ ] All new tests follow existing patterns in `info.rs`, `merge.rs`, `set.rs`
- [ ] Use `tempfile::tempdir()` for test file creation
- [ ] All tests clean up temporary files
- [ ] Test names describe exact scenario (e.g., `test_ls_with_json_and_filter`)
- [ ] Each test is independent (no shared state)
- [ ] Error cases include `.is_err()` and `.unwrap_err()` assertions
- [ ] Integration tests create real `.tgm` files via `encode()`
- [ ] JSON output tests validate via `serde_json::Value` parsing
- [ ] Filter tests use both `=` and `!=` operators
- [ ] Run full test suite: `cargo test --package tensogram-cli`
- [ ] Verify coverage with: `cargo tarpaulin --package tensogram-cli`
