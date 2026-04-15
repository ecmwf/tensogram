# tensogram ls

Lists messages in a Tensogram file, showing metadata in tabular or JSON format.

## Usage

```
tensogram ls [OPTIONS] [FILES]...
```

## Options

| Option | Description |
|---|---|
| `-w <WHERE_CLAUSE>` | Where-clause filter (e.g., mars.param=2t/10u) |
| `-p <KEYS>` | Comma-separated keys to display |
| `-j` | JSON output |
| `-h, --help` | Print help |

## Examples

```bash
# List all messages with default columns
tensogram ls forecast.tgm

# Only temperature fields
tensogram ls forecast.tgm -w "mars.param=2t"

# Temperature or wind
tensogram ls forecast.tgm -w "mars.param=2t/10u/10v"

# Exclude ensemble members
tensogram ls forecast.tgm -w "mars.type!=em"

# Show only date and step columns
tensogram ls forecast.tgm -p "mars.date,mars.step"

# JSON output (one object per line, good for jq)
tensogram ls forecast.tgm -j | jq '.["mars.param"]'
```

## Where Clause Syntax

The `-w` flag accepts a single expression:

```
key=value           # exact match
key=v1/v2/v3        # OR — matches any of v1, v2, v3
key!=value          # not equal
key!=v1/v2          # not any of v1, v2
```

**Key format:** `namespace.field` for namespaced keys (e.g. `mars.param`) or just `field` for top-level keys (e.g. `version`).

**Missing key:** For `key=value`, a missing key is treated as non-matching. For `key!=value`, a missing key passes the filter.

Only one `-w` expression can be specified per command. To apply multiple filters, pipe commands:

```bash
tensogram ls forecast.tgm -w "mars.type=fc" | grep "2t"
```

## Pick Keys

The `-p` flag selects which metadata columns to display. Keys use the same dot-notation as `-w`:

```bash
tensogram ls forecast.tgm -p "mars.date,mars.step,mars.param"
```

Without `-p`, all available metadata keys are shown.

## Default Table Output

```
mars.date   mars.step  mars.param  mars.type  shape
20260401    0          2t          fc         [721, 1440]
20260401    0          10u         fc         [721, 1440]
20260401    0          10v         fc         [721, 1440]
20260401    6          2t          fc         [721, 1440]
...
```

## JSON Output

With `-j`, each matching message is printed as a JSON object on its own line:

```json
{"mars.date": "20260401", "mars.step": "0", "mars.param": "2t", "shape": "[721, 1440]"}
{"mars.date": "20260401", "mars.step": "0", "mars.param": "10u", "shape": "[721, 1440]"}
```

This is compatible with `jq`, `grep`, and any tool that processes newline-delimited JSON.
