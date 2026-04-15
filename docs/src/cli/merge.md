# tensogram merge

Merge messages from one or more files into a single message.

## Usage

```bash
tensogram merge [OPTIONS] --output <OUTPUT> [INPUTS]...
```

## Options

| Option | Description |
|---|---|
| `-o, --output <OUTPUT>` | Output file |
| `-s, --strategy <STRATEGY>` | Merge strategy for conflicting metadata keys: first (default) — first value wins, last — last value wins, error — fail on conflict [default: first] |
| `-h, --help` | Print help |

## Description

All data objects from all input messages are collected into a single Tensogram message. Global metadata is merged — keys from the first message take precedence on conflict.

## Examples

```bash
# Merge two files into one
tensogram merge file1.tgm file2.tgm -o merged.tgm

# Merge all messages in a single multi-message file
tensogram merge multi.tgm -o single.tgm
```
