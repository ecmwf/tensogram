# tensogram copy

Copies messages from one file to one or more output files. The output filename can include **placeholders** that expand to metadata values, allowing a single file to be split by parameter, date, step, or any other key.

## Usage

```
tensogram copy [OPTIONS] <INPUT> <OUTPUT_PATTERN>
```

## Options

| Option | Description |
|---|---|
| `-w, --where <EXPR>` | Only copy messages that match this filter |

## Basic Copy

```bash
# Copy all messages from one file to another
tensogram copy input.tgm output.tgm
```

## Filename Placeholders

Wrap any metadata key in square brackets to expand it in the output filename:

```bash
# One file per parameter
tensogram copy forecast.tgm "by_param/[mars.param].tgm"
# Produces: by_param/2t.tgm, by_param/10u.tgm, by_param/msl.tgm, ...

# One file per date+step combination
tensogram copy forecast.tgm "archive/[mars.date]_[mars.step].tgm"
# Produces: archive/20260401_0.tgm, archive/20260401_6.tgm, ...

# Split by type and param
tensogram copy forecast.tgm "split/[mars.type]/[mars.param].tgm"
# Produces: split/fc/2t.tgm, split/an/2t.tgm, etc.
```

Multiple messages with the same expanded filename are **appended** to the same output file. This is how you split-then-concatenate: a 1000-message file with 4 unique `mars.param` values produces 4 output files with ~250 messages each.

## Filtering During Copy

Combine `-w` with placeholders for targeted extraction:

```bash
# Copy only forecasts, split by step
tensogram copy forecast.tgm "steps/[mars.step].tgm" -w "mars.type=fc"
```

## Edge Cases

### Missing Placeholder Key

If a message does not have the key referenced by a placeholder, that placeholder expands to `unknown`:

```bash
# If mars.param is missing, the message is written to by_param/unknown.tgm
tensogram copy forecast.tgm "by_param/[mars.param].tgm"
```

### Output Directory

The output directory must exist before running `copy`. The command does not create directories. Use `mkdir -p` beforehand:

```bash
mkdir -p by_param
tensogram copy forecast.tgm "by_param/[mars.param].tgm"
```

### Overwriting

If the expanded output filename already exists before the copy starts, it is **truncated once** and matching messages are then appended in order. This means running `copy` twice will duplicate messages. To avoid this, delete or rename existing outputs first.

### Placeholder Syntax Conflicts

If a metadata value contains `/`, `\`, or other characters that are invalid in filenames on your OS, the resulting filename will be invalid. Choose placeholder keys whose values are filesystem-safe (e.g. dates, step numbers, short codes).
