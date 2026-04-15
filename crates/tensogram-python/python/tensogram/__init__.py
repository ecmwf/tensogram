# (C) Copyright 2024- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tensogram Python bindings."""

from collections import namedtuple

from .tensogram import *  # noqa: F401, F403

Message = namedtuple("Message", ["metadata", "objects"])
"""Decoded message with named fields.

Returned by ``decode()``, ``decode_message()``, file iteration, and
``iter_messages()``::


    msg = tensogram.decode(buf)
    msg.metadata          # Metadata object
    msg.objects            # list[(DataObjectDescriptor, ndarray)]
    meta, objects = msg    # tuple unpacking also works
"""
