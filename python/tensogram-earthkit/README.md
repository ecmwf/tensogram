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

Fields follow the earthkit-data 1.x component model — metadata keys are
namespaced, and the full flat MARS request is preserved under the `labels`
component:

```python
field = fl[0]
field.get("labels.mars")                      # the full MARS dict, as stored
field.get("parameter.variable")               # "2t"
field.get("time.step")                        # datetime.timedelta
fl.sel(**{"parameter.variable": "2t"})        # component-key selection
fl.order_by("time.step")
```

## Write

```python
fl.to_target("file", "out.tgm", encoder="tensogram")
```

Writes every Field in the FieldList as one Tensogram data object in a single
message, preserving MARS metadata under `base[i]["mars"]`.

## See also

- [Tensogram documentation](https://sites.ecmwf.int/docs/tensogram/main)
- [earthkit-data documentation](https://earthkit-data.readthedocs.io)
