# tensogram get

Extracts a single metadata value from messages in a file. Returns an error if the key is missing.

## Usage

```
tensogram get [OPTIONS] <FILE> <KEY>
```

## Options

| Option | Description |
|---|---|
| `-w, --where <EXPR>` | Filter messages before extracting |
| `-j, --json` | Output JSON |

## Examples

```bash
# Get the mars.param value from all messages
tensogram get forecast.tgm mars.param

# Get the date from messages where param is 2t
tensogram get forecast.tgm mars.date -w "mars.param=2t"

# Get the shape of object 0
tensogram get forecast.tgm shape
```

## Strict Key Lookup

Unlike `ls` which shows a blank for missing keys, `get` exits with a non-zero status if any matching message does not have the requested key:

```bash
$ tensogram get forecast.tgm mars.nonexistent
Error: key "mars.nonexistent" not found in message 0
```

This makes `get` safe to use in shell scripts where missing data should fail fast.

## Multi-Object Messages

For messages with multiple objects, `get` returns the first matching value found. Lookup checks top-level metadata first and then scans objects in order until it finds a match.

## JSON Output

```bash
$ tensogram get forecast.tgm mars.param -j
"2t"
"10u"
"10v"
```

One JSON value per line, one per matching message.
