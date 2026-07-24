# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tensogram Python bindings."""

from collections import namedtuple
from collections.abc import Mapping

from .tensogram import *  # noqa: F403

# `from .tensogram import *` skips dunder names, so re-export the package
# version explicitly. It is stamped into the compiled extension from the
# crate's CARGO_PKG_VERSION, so it always matches the installed binary.
from .tensogram import Metadata as Metadata
from .tensogram import __version__ as __version__

# Explicit re-exports for the metadata/wire helpers so static analysers and
# ``from tensogram import compute_common`` see them (``import *`` already
# exposes them at runtime). ``AsyncStreamingEncoder`` is feature-gated on the
# ``async`` build (default-on, like ``AsyncTensogramFile``) and is left to the
# star import so a no-async build still imports cleanly.
from .tensogram import compute_common as compute_common
from .tensogram import object_inline_hashes as object_inline_hashes

# `Metadata` implements the full read-only Mapping protocol at the Rust level
# (``__getitem__``/``__iter__``/``__len__``/``__contains__``/``get``/``keys``/
# ``values``/``items``).  Register it as a virtual subclass so
# ``isinstance(meta, collections.abc.Mapping)`` is True and generic code that
# dispatches on Mapping accepts it. Duck-typing already works; this makes the
# relationship explicit.
Mapping.register(Metadata)

Message = namedtuple("Message", ["metadata", "objects"])
"""Decoded message with named fields.

Returned by ``decode()``, ``decode_with_masks()``, ``decode_message()``,
file iteration, and ``iter_messages()``::


    msg = tensogram.decode(buf)
    msg.metadata           # Metadata object
    msg.objects            # list[(DataObjectDescriptor, ndarray)]
    meta, objects = msg    # tuple unpacking also works

``decode_with_masks()`` returns one extra trailing field per object —
a ``dict`` of per-kind bitmasks with any subset of ``"nan"``,
``"inf+"``, ``"inf-"`` keys, each mapped to a ``bool`` numpy array::

    for desc, payload, masks in tensogram.decode_with_masks(buf).objects:
        n_missing = masks.get("nan", np.zeros(0, dtype=bool)).sum()
"""
