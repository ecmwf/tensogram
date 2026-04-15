# tensogram reshuffle

Reshuffle frames: move footer frames to header position.

## Usage

```bash
tensogram reshuffle --output <OUTPUT> <INPUT>
```

## Options

| Option | Description |
|---|---|
| `-o, --output <OUTPUT>` | Output file |
| `-h, --help` | Print help |

## Description

Converts streaming-mode messages (footer-based index and hash frames) into random-access-mode messages (header-based index and hash frames).

This is a decode → re-encode operation. The data is not modified; only the frame layout changes so that index and hash information appears before the data objects, enabling efficient random access.

## Examples

```bash
tensogram reshuffle streamed.tgm -o random_access.tgm
```
