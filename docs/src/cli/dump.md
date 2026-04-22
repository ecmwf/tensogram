# tensogram dump

Prints the full contents of every message in a Tensogram file — metadata keys and optionally the raw data values.

## Usage

```
tensogram dump [OPTIONS] [FILES]...
```

## Options

| Option | Description |
|---|---|
| `-w <WHERE_CLAUSE>` | Filter messages (e.g. `mars.param=2t`, same syntax as `ls`) |
| `-p <KEYS>` | Comma-separated keys to display |
| `-j` | JSON output |
| `-h, --help` | Print help |

## Example

```bash
$ tensogram dump forecast.tgm
─── Message 0 ───
version    : 3
mars.class : od
mars.type  : fc
mars.date  : 20260401
mars.step  : 0

  Object 0
  type     : ntensor
  ndim     : 2
  shape    : [721, 1440]
  strides  : [1440, 1]
  dtype    : float32
  mars.param: 2t
  encoding : none
  filter   : none
  compression: none

─── Message 1 ───
...
```

## Filtering

Use `-w` to limit the dump to specific messages:

```bash
# Dump only wave spectra
tensogram dump forecast.tgm -w "mars.param=wave_spectra"
```

## JSON Output

With `-j`, each message is a JSON object:

```json
{
  "message": 0,
  "metadata": {
    "version": 3,
    "base": [
      {
        "mars": {"class": "od", "type": "fc", "date": "20260401", "step": 0, "param": "2t"},
        "_reserved_": {"tensor": {"ndim": 2, "shape": [721, 1440], "strides": [1440, 1], "dtype": "float32"}}
      }
    ]
  },
  "objects": [
    {"type": "ntensor", "ndim": 2, "shape": [721, 1440], "dtype": "float32",
     "encoding": "none"}
  ]
}
```

> **Note.** Per-object integrity hashes live in the frame's inline
> hash slot (see [wire-format.md §Common Frame Footer](../format/wire-format.md))
> and are not surfaced as a `hash` field on the decoded descriptor.

## When to Use dump vs ls

- Use `ls` for a quick overview of many messages (one line per message)
- Use `dump` when you need to see all keys for a specific message, or check encoding parameters
