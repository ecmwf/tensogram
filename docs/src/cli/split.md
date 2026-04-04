# tensogram split

Split multi-object messages into separate single-object files.

## Usage

```bash
tensogram split <INPUT> -o <OUTPUT_TEMPLATE>
```

## Description

Each data object from each message in the input file becomes its own Tensogram message, inheriting the global metadata.

Output files are named using the template:
- Use `[index]` for zero-padded numbering: `split_[index].tgm` → `split_0000.tgm`, `split_0001.tgm`, ...
- Without `[index]`: the index is appended before the extension: `out.tgm` → `out_0000.tgm`, `out_0001.tgm`, ...

## Examples

```bash
# Split with index template
tensogram split multi_object.tgm -o 'field_[index].tgm'

# Split with auto-numbered names
tensogram split multi_object.tgm -o output.tgm
```
