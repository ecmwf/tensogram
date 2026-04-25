---
description: Systematic edge-case audit and patching across the codebase
agent: build
---

# Improve Edge Case Handling

Perform a systematic edge case audit across the codebase.

## What to Look For

### Data Edge Cases
- Empty inputs: zero-length arrays, empty messages, empty files, empty metadata
- Boundary values: max/min integers, NaN, Inf, -Inf, subnormals, negative zero
- Single-element arrays, scalar (0-D) arrays, very large arrays
- All supported dtypes including bitmask, bfloat16, complex types

### API Edge Cases
- Index out of range (negative, zero, beyond length)
- Null/None arguments where non-null expected
- Concurrent access patterns (multi-threaded reads)
- File system edge cases: read-only, full disk, symlinks, permissions

### Wire Format Edge Cases
- Streaming vs buffered mode messages
- Footer-indexed vs header-indexed files
- Messages with 0 objects (metadata-only)
- Messages with many objects (100+)
- Corrupt magic bytes, truncated messages, garbage between messages
- Very large CBOR metadata, deeply nested metadata

### Cross-Language Edge Cases
- Unicode in metadata keys/values
- Very long strings
- Binary data in metadata

## Process

1. Scan each module and identify unhandled or under-tested edge cases
2. For ambiguous behavior, ask the user to clarify the intended semantics
3. Add handling code for each edge case (fail gracefully, not silently)
4. Write tests for each new edge case
5. Document all edge case behavior in docs/
6. Run full test suite to verify no regressions
7. Summarize findings and changes
