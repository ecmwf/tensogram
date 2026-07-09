# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Helpers bridging tensogram ``base[i]["mars"]`` entries to earthkit fields.

earthkit-data 1.0 models a :class:`Field` as a set of typed components
(parameter / time / vertical / ensemble / geography), not a flat metadata
dict.  The bridge therefore has two directions:

* :func:`base_entry_to_field` flattens a ``base[i]`` entry into a MARS
  request dict (:func:`extract_mars_keys`) and builds the components from
  it — mirroring :func:`earthkit.data.field.mars.create.create_mars_field`
  but tolerant of partial requests (missing ``date`` / ``levtype`` /
  unknown values simply omit that component instead of raising).  The
  full flat request is preserved under the field's ``labels`` component
  as ``labels.mars``, so nothing is lost.
* :func:`field_to_base_entry` reads ``labels.mars`` back (or falls back
  to the field's raw ``metadata.*`` keys, e.g. for GRIB-backed fields)
  and routes canonical MARS keys into the ``mars`` sub-map, everything
  else into ``_extra_``.

Keeping this isolated from :mod:`.fieldlist` means the encoder in
:mod:`.encoder` can reuse the reverse direction without importing the
reader side.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "base_entry_to_field",
    "extract_mars_keys",
    "field_to_base_entry",
    "has_mars_namespace",
]


def extract_mars_keys(base_entry: dict[str, Any]) -> dict[str, Any]:
    """Flatten a tensogram ``base[i]`` entry into a plain MARS request dict.

    Rules:

    * The ``"_reserved_"`` key (library-managed tensor info) is dropped.
    * Every non-MARS sibling key is copied first (e.g. ``name``,
      ``cf``, ``grib``); dict values stay nested for the caller's
      inspection.
    * The ``"mars"`` sub-map is then merged on top, flattening its
      entries into the top-level dict.  MARS keys therefore win on
      conflict, which matches how GRIB-converted tensograms are
      typically consumed.

    An entry that is not a dict yields an empty dict — the caller decides
    whether such objects count as MARS fields or coordinate/auxiliary
    objects.
    """
    if not isinstance(base_entry, dict):
        return {}

    out: dict[str, Any] = {}
    for k, v in base_entry.items():
        if k in ("_reserved_", "mars"):
            continue
        out[k] = v

    mars = base_entry.get("mars")
    if isinstance(mars, dict):
        out.update(mars)

    return out


def has_mars_namespace(base_entry: Any) -> bool:
    """Return ``True`` if ``base_entry`` carries a non-empty MARS sub-map.

    Mirrors the per-object discriminator used by
    :func:`tensogram_earthkit.detection.is_mars_tensogram`.
    """
    if not isinstance(base_entry, dict):
        return False
    mars = base_entry.get("mars")
    return isinstance(mars, dict) and bool(mars)


# ---------------------------------------------------------------------------
# request dict → Field (reader direction)
# ---------------------------------------------------------------------------


def _time_component(request: dict[str, Any]) -> Any:
    """Build the time component, or ``None`` when no usable date is present.

    Mirrors :class:`earthkit.data.field.mars.time.MarsTimeBuilder` (including
    the ``hdate`` precedence) but accepts both GRIB-style (``20250101`` /
    ``"0000"``) and ISO (``"2025-01-01"``) dates, and returns ``None``
    instead of raising for entries without one.
    """
    date = request.get("hdate", request.get("date"))
    if date is None:
        return None

    from earthkit.data.utils.dates import datetime_from_date_and_time, datetime_from_grib

    time = request.get("time")
    try:
        base = datetime_from_grib(date, 0 if time is None else time)
    except (TypeError, ValueError):
        try:
            base = datetime_from_date_and_time(date, time)
        except (TypeError, ValueError):
            return None
    if base is None:
        return None

    from earthkit.data.field.component.time import create_time
    from earthkit.data.field.handler.time import TimeFieldComponentHandler

    component = create_time({"base_datetime": base, "step": request.get("step", 0)})
    return TimeFieldComponentHandler.from_component(component)


def _parameter_component(request: dict[str, Any]) -> Any:
    """Build the parameter component, or ``None`` when absent/unusable."""
    from earthkit.data.field.mars.parameter import MarsParameterBuilder

    try:
        return MarsParameterBuilder.build(request)
    except (KeyError, TypeError, ValueError):
        return None


def _vertical_component(request: dict[str, Any]) -> Any:
    """Build the vertical component, or ``None`` when absent/unsupported.

    ``MarsVerticalBuilder`` raises for a missing or unknown ``levtype``;
    tensogram MARS maps are not guaranteed to carry one.
    """
    if request.get("levtype") is None:
        return None
    from earthkit.data.field.mars.vertical import MarsVerticalBuilder

    try:
        return MarsVerticalBuilder.build(request)
    except (KeyError, TypeError, ValueError):
        return None


def _ensemble_component(request: dict[str, Any]) -> Any:
    """Build the ensemble component, or ``None`` when absent."""
    from earthkit.data.field.mars.ensemble import MarsEnsembleBuilder

    try:
        return MarsEnsembleBuilder.build(request)
    except (KeyError, TypeError, ValueError):
        return None


def base_entry_to_field(
    base_entry: dict[str, Any],
    values: Any,
    shape: tuple[int, ...],
) -> Any:
    """Build an earthkit :class:`Field` from a tensogram base entry.

    Parameters
    ----------
    base_entry
        One element of ``tensogram.Metadata.base``.
    values
        The decoded tensor values (any array-like accepted by
        :class:`~earthkit.data.field.handler.data.ArrayDataFieldComponentHandler`).
    shape
        The tensor shape — carried by an
        :class:`~earthkit.data.field.component.geography.EmptyGeography` so
        ``field.shape`` / ``field.to_numpy()`` report the grid extent.

    Returns
    -------
    An :class:`earthkit.data.core.field.Field` whose typed components are
    derived from the flattened MARS request, and whose full flat request is
    retrievable via ``field.get("labels.mars")``.
    """
    from earthkit.data.core.field import Field
    from earthkit.data.field.component.geography import EmptyGeography
    from earthkit.data.field.handler.data import ArrayDataFieldComponentHandler
    from earthkit.data.field.handler.geography import GeographyFieldComponentHandler
    from earthkit.data.field.handler.labels import SimpleLabels

    request = extract_mars_keys(base_entry)

    geography = None
    if shape is not None:
        geography = GeographyFieldComponentHandler.from_any(EmptyGeography(shape=tuple(shape)))

    return Field(
        data=ArrayDataFieldComponentHandler(values),
        parameter=_parameter_component(request),
        time=_time_component(request),
        geography=geography,
        vertical=_vertical_component(request),
        ensemble=_ensemble_component(request),
        labels=SimpleLabels({"mars": request}),
    )


# ---------------------------------------------------------------------------
# Field → base entry (encoder direction)
# ---------------------------------------------------------------------------

# Canonical MARS key set — everything not in this set falls to ``_extra_``
# when an earthkit Field is serialised back to a tensogram base entry.
# Kept conservative: adding more keys only moves them out of `_extra_`
# into `mars`, which is always the safer default for MARS consumers.
_MARS_CANONICAL_KEYS: frozenset[str] = frozenset(
    {
        "class",
        "stream",
        "expver",
        "type",
        "param",
        "shortName",
        "date",
        "time",
        "step",
        "levtype",
        "levelist",
        "number",
        "ensemble",
        "domain",
        "grid",
        "area",
    }
)


def field_to_base_entry(field: Any) -> dict[str, Any]:
    """Build a tensogram ``base[i]`` entry from an earthkit :class:`Field`.

    Collects the field's flat MARS metadata, routes known MARS keys into
    the ``mars`` sub-map, and stashes the rest in ``_extra_`` so nothing
    is dropped on round-trips.
    """
    items = _collect_field_metadata(field)

    mars: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    for k, v in items.items():
        if k in _MARS_CANONICAL_KEYS:
            mars[k] = v
        else:
            extras[k] = v

    entry: dict[str, Any] = {}
    if mars:
        entry["mars"] = mars
    if extras:
        entry["_extra_"] = extras
    return entry


def _collect_field_metadata(field: Any) -> dict[str, Any]:
    """Return a flat metadata dict for *field* — robust to Field variants.

    Two strategies, in order:

    1. ``field.get("labels.mars")`` — fields built by this plugin (and by
       earthkit's own MARS machinery, e.g. gribjump) carry the full flat
       request under the ``labels`` component.
    2. The field's raw metadata (``metadata.<key>``) for each canonical
       MARS key — covers e.g. GRIB-backed fields, whose ecCodes keys are
       exposed under the ``metadata`` prefix.
    """
    try:
        mars = field.get("labels.mars", None)
    except (AttributeError, TypeError):
        mars = None
    if isinstance(mars, dict) and mars:
        return dict(mars)

    out: dict[str, Any] = {}
    for k in _MARS_CANONICAL_KEYS:
        try:
            v = field.get(f"metadata.{k}", None)
        except (AttributeError, TypeError, KeyError, ValueError):
            continue
        if v is not None:
            out[k] = v
    return out
