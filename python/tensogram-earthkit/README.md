# tensogram-earthkit

[earthkit-data](https://github.com/ecmwf/earthkit-data) source and encoder
plugins for [Tensogram](https://github.com/ecmwf/tensogram) `.tgm` files.

## Install

```bash
pip install tensogram-earthkit
```

## Read

```python
import earthkit.data as ekd

# Local file, remote URL, in-memory bytes, or byte stream
data = ekd.from_source("tensogram", "file.tgm")

ds = data.to_xarray()              # xarray.Dataset (always)
fl = data.to_fieldlist()           # FieldList — only when MARS keys are present
```

`.to_fieldlist()` raises `NotImplementedError` when the tensogram file has no
`base[i]["mars"]` metadata. Use `.to_xarray()` instead for generic N-D tensors.

## Write

```python
fl.to_target("file", "out.tgm", encoder="tensogram")
```

Writes every Field in the FieldList as one Tensogram data object in a single
message, preserving MARS metadata under `base[i]["mars"]`.

## See also

- [Tensogram documentation](https://sites.ecmwf.int/docs/tensogram/main)
- [earthkit-data documentation](https://earthkit-data.readthedocs.io)
