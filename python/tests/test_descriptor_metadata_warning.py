# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Write-side warning when application-metadata keys appear in a descriptor.

See issue #67 and the ``METADATA_LIKE_DESCRIPTOR_KEYS`` list in
``python/bindings/src/lib.rs``.  The warning fires once per descriptor
(aggregated over multiple keys) to avoid spam on multi-object messages.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import tensogram

METADATA_LIKE_KEYS = [
    "name",
    "param",
    "shortName",
    "long_name",
    "description",
    "units",
    "dim_names",
    "mars",
    "cf",
    "product",
    "instrument",
]


def _zero(shape=(4,)):
    return np.zeros(shape, dtype=np.float32)


@pytest.mark.parametrize("key", METADATA_LIKE_KEYS)
def test_warning_fires_for_each_metadata_like_key(key: str):
    desc = {"type": "ntensor", "shape": [4], "dtype": "float32", key: "some-value"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tensogram.encode({"version": 2}, [(desc, _zero())])
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    assert f"'{key}'" in str(user_warnings[0].message)
    assert "meta['base'][i]" in str(user_warnings[0].message)


def test_single_aggregated_warning_for_multiple_keys():
    desc = {
        "type": "ntensor",
        "shape": [4],
        "dtype": "float32",
        "name": "temp",
        "units": "K",
        "description": "two-metre temperature",
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tensogram.encode({"version": 2}, [(desc, _zero())])
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    msg = str(user_warnings[0].message)
    for key in ("name", "units", "description"):
        assert f"'{key}'" in msg


def test_no_warning_for_clean_descriptor():
    desc = {"type": "ntensor", "shape": [4], "dtype": "float32"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tensogram.encode({"version": 2}, [(desc, _zero())])
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


def test_no_warning_for_known_encoding_param_names():
    desc = {
        "type": "ntensor",
        "shape": [4],
        "dtype": "float32",
        "reference_value": 0.0,
        "bits_per_value": 16,
        "decimal_scale_factor": 2,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tensogram.encode({"version": 2}, [(desc, _zero())])
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


def test_no_warning_for_unknown_non_metadata_key():
    desc = {
        "type": "ntensor",
        "shape": [4],
        "dtype": "float32",
        "custom_encoding_knob": 42,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tensogram.encode({"version": 2}, [(desc, _zero())])
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


def test_one_warning_per_descriptor_in_multi_object_message():
    descs_and_data = [
        (
            {"type": "ntensor", "shape": [4], "dtype": "float32", "name": "a"},
            _zero(),
        ),
        (
            {"type": "ntensor", "shape": [4], "dtype": "float32", "name": "b"},
            _zero(),
        ),
        (
            {"type": "ntensor", "shape": [4], "dtype": "float32", "name": "c"},
            _zero(),
        ),
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tensogram.encode({"version": 2}, descs_and_data)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 3


def test_warning_fires_on_tensogram_file_append(tmp_path: Path):
    path = str(tmp_path / "warn.tgm")
    desc = {"type": "ntensor", "shape": [4], "dtype": "float32", "name": "x"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with tensogram.TensogramFile.create(path) as f:
            f.append({"version": 2}, [(desc, _zero())])
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    assert "'name'" in str(user_warnings[0].message)


def test_warning_is_suppressible_by_filterwarnings():
    desc = {"type": "ntensor", "shape": [4], "dtype": "float32", "name": "x"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore", UserWarning)
        tensogram.encode({"version": 2}, [(desc, _zero())])
    assert not caught


def test_key_preserved_in_params_despite_warning():
    desc = {"type": "ntensor", "shape": [4], "dtype": "float32", "name": "kept"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        msg = tensogram.encode({"version": 2}, [(desc, _zero())])
    decoded = tensogram.decode(msg)
    decoded_desc, _ = decoded.objects[0]
    assert decoded_desc.params["name"] == "kept"
