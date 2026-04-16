# tensogram info

Displays a summary of a Tensogram file: number of messages, total file size, and format version.

## Usage

```
tensogram info [FILES]...
```

## Options

| Option | Description |
|---|---|
| `-h, --help` | Print help |

## Example

```bash
$ tensogram info forecast.tgm
Messages : 48
File size: 1.2 GB
Version  : 1
```

## What it Shows

| Field | Description |
|---|---|
| Messages | Total number of valid messages found by scanning the file |
| File size | Raw byte count of the file on disk |
| Version | Format version from the first message's metadata |

## Notes

- The scan counts only **valid** messages (those with a matching `TENSOGRM` header and `39277777` terminator). Corrupted regions are skipped.
- If the file is empty, `Messages: 0` is shown.
- Version is read from the first message. If messages have different versions, only the first is shown.
