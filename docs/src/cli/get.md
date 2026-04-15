# tensogram get

Extracts a single metadata value from messages in a file. Returns an error if the key is missing.

## Usage

```
tensogram get [OPTIONS] -p <KEYS> [FILES]...
```

## Options

| Option | Description |
|---|---|
| `-w <WHERE_CLAUSE>` | Filter messages (e.g. `mars.param=2t`, same syntax as `ls`) |
| `-p <KEYS>` | Comma-separated keys to extract (required) |
| `-h, --help` | Print help |

## Examples

```bash
# Get the mars.param value from all messages
tensogram get -p mars.param forecast.tgm

# Get the date from messages where param is 2t
tensogram get -p mars.date -w "mars.param=2t" forecast.tgm

# Get the shape of object 0
tensogram get -p shape forecast.tgm
```

## Strict Key Lookup

Unlike `ls` which shows a blank for missing keys, `get` exits with a non-zero status if any matching message does not have the requested key:

```bash
$ tensogram get -p mars.nonexistent forecast.tgm
Error: key "mars.nonexistent" not found in message 0
```

This makes `get` safe to use in shell scripts where missing data should fail fast.

## Multi-Object Messages

For messages with multiple objects, `get` returns the first matching value found. Lookup checks top-level metadata first and then scans objects in order until it finds a match.


