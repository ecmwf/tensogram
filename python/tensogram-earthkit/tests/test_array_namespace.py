# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Array namespace interop for tensogram-backed fields.

Since :class:`TensogramField` is built from :class:`ArrayField`, it
inherits the full :meth:`Field.to_array` machinery backed by
`earthkit-utils`' array namespace abstraction.  This test file pins
the contract:

* ``to_array(array_namespace="numpy")`` → :class:`numpy.ndarray`
* ``to_array(array_namespace="torch")`` → :class:`torch.Tensor`
  (skipped when torch is not installed)
* ``dtype=`` converts across backends
* ``flatten=True`` returns a flat 1-D array
* ``device=`` works for namespaces that support it (torch)
* The FieldList exposes the same conversions across all fields
"""

from __future__ import annotations

import earthkit.data as ekd
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed — skipping torch interop")


class TestFieldToArrayNumpy:
    def test_numpy_default(self, mars_tensogram_file) -> None:
        fl = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()
        arr = fl[0].to_array(array_namespace="numpy")
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (4, 6)

    def test_numpy_dtype_conversion(self, mars_tensogram_file) -> None:
        fl = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()
        arr = fl[0].to_array(dtype="float64", array_namespace="numpy")
        assert arr.dtype == np.dtype("float64")

    def test_numpy_flatten(self, mars_tensogram_file) -> None:
        fl = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()
        arr = fl[0].to_array(flatten=True, array_namespace="numpy")
        assert arr.shape == (24,)


class TestFieldToArrayTorch:
    def test_torch_namespace(self, mars_tensogram_file) -> None:
        fl = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()
        arr = fl[0].to_array(array_namespace="torch")
        assert isinstance(arr, torch.Tensor)
        assert arr.shape == (4, 6)

    def test_torch_values_match_numpy(self, mars_tensogram_file) -> None:
        fl = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()
        np_arr = fl[0].to_array(array_namespace="numpy")
        torch_arr = fl[0].to_array(array_namespace="torch")
        np.testing.assert_array_equal(torch_arr.numpy(), np_arr)

    def test_torch_device_cpu(self, mars_tensogram_file) -> None:
        fl = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()
        arr = fl[0].to_array(array_namespace="torch", device="cpu")
        assert isinstance(arr, torch.Tensor)
        assert str(arr.device) == "cpu"


class TestFieldListValues:
    """FieldList-level :attr:`values` and :meth:`to_array` also work."""

    def test_fieldlist_values(self, mars_tensogram_file) -> None:
        fl = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()
        vals = fl.values
        assert isinstance(vals, np.ndarray)
        # 2 fields of flattened (4 by 6 = 24) = (2, 24)
        assert vals.shape == (2, 24)

    def test_fieldlist_to_numpy_per_field(self, mars_tensogram_file) -> None:
        fl = ekd.from_source("tensogram", str(mars_tensogram_file)).to_fieldlist()
        per_field = [f.to_numpy() for f in fl]
        assert len(per_field) == 2
        for arr in per_field:
            assert isinstance(arr, np.ndarray)
