"""Tensogram Python bindings."""

from collections import namedtuple

from .tensogram import *  # noqa: F401, F403

Message = namedtuple("Message", ["metadata", "objects"])
"""Decoded message with named fields.

All decode functions and iterators return ``Message`` namedtuples::

    msg = tensogram.decode(buf)
    msg.metadata          # Metadata object
    msg.objects            # list[(DataObjectDescriptor, ndarray)]
    meta, objects = msg    # tuple unpacking also works
"""
