# tensogram validate

Check whether `.tgm` files are well-formed and intact. Analogous to `grib_check` or `h5check`.

## Usage

```bash
tensogram validate [OPTIONS] <FILES>...
```

## Validation Levels

The command runs up to three validation levels:

| Level | Name | What it checks |
|-------|------|---------------|
| 1 | **Structure** | Magic bytes, frame headers, ENDF markers, total_length, postamble, frame ordering, preceder legality, preamble flags vs observed frames |
| 2 | **Metadata** | CBOR parses correctly, required keys present (`_reserved_.tensor`, dtype, shape, strides), encoding/filter/compression types recognized, object count consistency, shape/strides/ndim consistency |
| 3 | **Integrity** | xxh3 hash in descriptor/hash-frame matches recomputed hash, compressed payloads decompress without error |

## Modes

| Mode | Levels | Description |
|------|--------|-------------|
| default | 1–3 | Structure + metadata + integrity |
| quick | 1 | Structure only, no payloads |
| checksum | 3 | Hash check (structural errors still reported) |
| canonical | 1–3 | Default + RFC 8949 CBOR key ordering |

The mode flags (`--quick`, `--checksum`, `--canonical`) are mutually exclusive.

## Options

| Flag | Description |
|------|-------------|
| `--json` | Machine-parseable JSON output |

## Output

### Human-readable (default)

```
file.tgm: OK (3 messages, 47 objects, hash verified)
```

On failure:

```
bad.tgm: FAILED — message 2, object 5: hash mismatch (expected a3f7..., got 91c2...) (at byte 4096)
bad.tgm: FAILED (1 errors, 3 messages, 47 objects)
```

### JSON (`--json`)

```json
[
  {
    "file": "file.tgm",
    "status": "ok",
    "messages": 1,
    "objects": 3,
    "hash_verified": true,
    "file_issues": [],
    "message_reports": [
      {
        "issues": [],
        "object_count": 3,
        "hash_verified": true
      }
    ]
  }
]
```

On failure, issues within `message_reports[i].issues` contain:

```json
{
  "code": "hash_mismatch",
  "level": "integrity",
  "severity": "error",
  "object_index": 5,
  "byte_offset": null,
  "description": "object 5: hash mismatch (expected a3f7..., got 91c2...)"
}
```

Issue codes are stable snake_case strings (e.g. `hash_mismatch`, `invalid_magic`, `buffer_too_short`) suitable for machine parsing.

## Exit Code

- `0` — all files pass validation
- `1` — one or more files have errors

## Batch Mode

```bash
tensogram validate data/*.tgm
```

Validates all files. Reports per-file. Exits 1 if any file fails.

## File-level Checks

When validating a file with multiple messages, the command also detects:

- Unrecognized bytes between messages (garbage or padding)
- Truncated messages at end of file
- Trailing bytes after the last valid message

These are reported as file-level warnings.

## Library API

The same validation is available programmatically:

```rust
use std::path::Path;
use tensogram_core::{validate_message, validate_file, ValidateOptions};

// Validate a single message buffer
let report = validate_message(&bytes, &ValidateOptions::default());
assert!(report.is_ok());

// Validate a file
let file_report = validate_file(Path::new("data.tgm"), &ValidateOptions::default())?;
println!("{} messages, {} objects", file_report.messages.len(), file_report.total_objects());
```

## Examples

```bash
# Default validation (levels 1-3)
tensogram validate measurements.tgm

# Quick structural check
tensogram validate --quick *.tgm

# Verify checksums only
tensogram validate --checksum archive/*.tgm

# Check canonical CBOR encoding
tensogram validate --canonical output.tgm

# JSON output for CI pipelines
tensogram validate --json data/*.tgm
```
