// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Grouping logic for the field-selector sidebar.
 *
 * Lives alongside `FieldSelector.tsx` but in its own module so the
 * component file can export only React components (keeps
 * `react-refresh/only-export-components` happy) and so the grouping
 * logic is straightforward to unit-test against synthetic
 * `ObjectInfo` inputs.
 */

import type { ObjectInfo } from '../../tensogram';

export interface MarsMetadata {
  step?: number;
  param?: string | number;
  levtype?: string;
  levelist?: number | string;
  level?: number | string;
  [key: string]: unknown;
}

export type LevelStrategy =
  | { kind: 'none' }
  | { kind: 'packed'; levels: number[]; dim: number }
  | { kind: 'per-object'; levels: number[] };

export interface ParamEntry {
  step: number;
  marsLevel: number | null;
  variable: ObjectInfo;
}

export interface ParamGroup {
  param: string;
  levtype: string;
  units: string;
  levelStrategy: LevelStrategy;
  entries: ParamEntry[];
}

export function getMars(v: ObjectInfo): MarsMetadata {
  return (v.metadata?.mars as MarsMetadata) ?? {};
}

export function getMarsLevel(mars: MarsMetadata): number | null {
  const raw = mars.levelist ?? mars.level;
  if (raw == null) return null;
  if (mars.levtype === 'sfc') return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

export function getPackedLevels(v: ObjectInfo, coordLength: number): number[] | null {
  if (v.shape.length <= 1) return null;
  const levelDimSize = v.shape.find((s) => s !== coordLength);
  if (!levelDimSize || levelDimSize <= 1) return null;

  const anemoi = v.metadata?.anemoi as Record<string, unknown> | undefined;
  const levels = anemoi?.levels as number[] | undefined;
  if (levels && levels.length === levelDimSize) return levels;

  return Array.from({ length: levelDimSize }, (_, i) => i);
}

export function groupByParam(variables: ObjectInfo[], coordLength: number): ParamGroup[] {
  const map = new Map<string, ParamGroup>();

  for (const v of variables) {
    const mars = getMars(v);
    // mars.param may be a GRIB integer code (133, 167, ...) or a
    // short string ("t", "2t", ...); coerce to string so the Map
    // keying and the localeCompare sort below both work.
    const paramRaw = mars.param ?? v.name;
    const param = typeof paramRaw === 'string' ? paramRaw : String(paramRaw ?? '');
    const step = Number(mars.step ?? 0);
    const marsLevel = getMarsLevel(mars);

    let group = map.get(param);
    if (!group) {
      const packed = getPackedLevels(v, coordLength);
      let levelStrategy: LevelStrategy;
      if (packed && packed.length > 1) {
        const dim = v.shape.findIndex((s) => s !== coordLength);
        levelStrategy = { kind: 'packed', levels: packed, dim: dim >= 0 ? dim : 1 };
      } else {
        levelStrategy = { kind: 'none' };
      }
      group = {
        param,
        levtype: mars.levtype ?? '',
        units: (v.metadata?.units as string) ?? '',
        levelStrategy,
        entries: [],
      };
      map.set(param, group);
    }
    group.entries.push({ step, marsLevel, variable: v });
  }

  for (const group of map.values()) {
    if (group.levelStrategy.kind === 'none') {
      const levelSet = new Set<number>();
      for (const e of group.entries) {
        if (e.marsLevel != null) levelSet.add(e.marsLevel);
      }
      if (levelSet.size > 1) {
        group.levelStrategy = {
          kind: 'per-object',
          levels: [...levelSet].sort((a, b) => a - b),
        };
      }
    }
    group.entries.sort((a, b) => {
      const la = a.marsLevel ?? -Infinity;
      const lb = b.marsLevel ?? -Infinity;
      if (la !== lb) return la - lb;
      return a.step - b.step;
    });
  }

  return Array.from(map.values()).sort((a, b) => a.param.localeCompare(b.param));
}

export function getLevels(group: ParamGroup): number[] {
  if (group.levelStrategy.kind === 'none') return [];
  return group.levelStrategy.levels;
}

export function getEntriesForLevel(group: ParamGroup, level: number | null): ParamEntry[] {
  if (group.levelStrategy.kind === 'per-object' && level != null) {
    return group.entries.filter((e) => e.marsLevel === level);
  }
  return group.entries;
}
