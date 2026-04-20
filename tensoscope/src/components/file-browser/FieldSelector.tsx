/** Field selector: tree view grouped by param, then level, then step. */

import { useState, useMemo } from 'react';
import { useAppStore } from '../../store/useAppStore';
import type { ObjectInfo } from '../../tensogram';

// ── Metadata helpers ────────────────────────────────────────────────

interface MarsMetadata {
  step?: number;
  param?: string;
  levtype?: string;
  levelist?: number | string;
  level?: number | string;
  [key: string]: unknown;
}

function getMars(v: ObjectInfo): MarsMetadata {
  return (v.metadata?.mars as MarsMetadata) ?? {};
}

function getMarsLevel(mars: MarsMetadata): number | null {
  const raw = mars.levelist ?? mars.level;
  if (raw == null) return null;
  if (mars.levtype === 'sfc') return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

function getPackedLevels(v: ObjectInfo, coordLength: number): number[] | null {
  if (v.shape.length <= 1) return null;
  const levelDimSize = v.shape.find((s) => s !== coordLength);
  if (!levelDimSize || levelDimSize <= 1) return null;

  const anemoi = v.metadata?.anemoi as Record<string, unknown> | undefined;
  const levels = anemoi?.levels as number[] | undefined;
  if (levels && levels.length === levelDimSize) return levels;

  return Array.from({ length: levelDimSize }, (_, i) => i);
}

// ── Grouping types ──────────────────────────────────────────────────

type LevelStrategy =
  | { kind: 'none' }
  | { kind: 'packed'; levels: number[]; dim: number }
  | { kind: 'per-object'; levels: number[] };

interface ParamEntry {
  step: number;
  marsLevel: number | null;
  variable: ObjectInfo;
}

interface ParamGroup {
  param: string;
  levtype: string;
  units: string;
  levelStrategy: LevelStrategy;
  entries: ParamEntry[];
}

function groupByParam(variables: ObjectInfo[], coordLength: number): ParamGroup[] {
  const map = new Map<string, ParamGroup>();

  for (const v of variables) {
    const mars = getMars(v);
    const param = mars.param ?? v.name;
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

// ── Helpers ─────────────────────────────────────────────────────────

function getLevels(group: ParamGroup): number[] {
  if (group.levelStrategy.kind === 'none') return [];
  return group.levelStrategy.levels;
}

function getEntriesForLevel(group: ParamGroup, level: number | null): ParamEntry[] {
  if (group.levelStrategy.kind === 'per-object' && level != null) {
    return group.entries.filter((e) => e.marsLevel === level);
  }
  return group.entries;
}

function levelUnit(levtype: string): string {
  if (levtype === 'pl') return ' hPa';
  return '';
}

// ── Component ───────────────────────────────────────────────────────

export function FieldSelector() {
  const { fileIndex, selectedObject, coordinates, selectField, fetchSlice } = useAppStore();
  const selectedLevel = useAppStore((s) => s.selectedLevel);
  const setSelectedLevel = useAppStore((s) => s.setSelectedLevel);
  const [filter, setFilter] = useState('');
  const [expandedParam, setExpandedParam] = useState<string | null>(null);
  const [expandedLevel, setExpandedLevel] = useState<number | null>(null);

  const coordLength = coordinates?.lat.length ?? 0;

  const filtered = useMemo(() => {
    if (!fileIndex) return [];
    return fileIndex.variables.filter((v) => {
      if (!filter) return true;
      const term = filter.toLowerCase();
      return (
        v.name.toLowerCase().includes(term) ||
        JSON.stringify(v.metadata).toLowerCase().includes(term)
      );
    });
  }, [fileIndex, filter]);

  const groups = useMemo(() => groupByParam(filtered, coordLength), [filtered, coordLength]);

  if (!fileIndex) return null;

  const effectiveExpanded = groups.length === 1 ? groups[0].param : expandedParam;

  const toggleParam = (param: string) => {
    if (expandedParam === param) {
      setExpandedParam(null);
      setExpandedLevel(null);
    } else {
      setExpandedParam(param);
      setExpandedLevel(null);
    }
  };

  /** Select a level (updates map) without expanding steps. */
  const handleSelectLevel = (group: ParamGroup, level: number) => {
    setSelectedLevel(level);

    // Select first step at this level
    if (group.levelStrategy.kind === 'packed') {
      const entry = group.entries[0];
      if (entry) selectField(entry.variable.msgIndex, entry.variable.objIndex);
    } else if (group.levelStrategy.kind === 'per-object') {
      const entry = group.entries.find((e) => e.marsLevel === level);
      if (entry) selectField(entry.variable.msgIndex, entry.variable.objIndex);
    }
  };

  /** Toggle expand/collapse of step list under a level. */
  const toggleLevel = (level: number) => {
    setExpandedLevel(expandedLevel === level ? null : level);
  };

  const handleClickStep = (_group: ParamGroup, entry: ParamEntry) => {
    selectField(entry.variable.msgIndex, entry.variable.objIndex);
  };

  /** When clicking the param header directly, select the first entry. */
  const handleClickParam = (group: ParamGroup) => {
    const levels = getLevels(group);
    if (levels.length > 0) {
      // Has levels -- just toggle expand, don't select
      toggleParam(group.param);
    } else {
      // No levels -- select first step
      const entry = group.entries[0];
      if (entry) selectField(entry.variable.msgIndex, entry.variable.objIndex);
    }
  };

  /** Handle changing the level for packed-levels data via fetchSlice. */
  const handleLevelSlice = (group: ParamGroup, level: number) => {
    if (group.levelStrategy.kind === 'packed') {
      const idx = group.levelStrategy.levels.indexOf(level);
      if (idx >= 0) fetchSlice(group.levelStrategy.dim, idx);
    }
  };

  return (
    <div>
      <h2>Fields</h2>
      <input
        type="text"
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        placeholder="Filter fields..."
        className="filter-input"
        aria-label="Filter fields"
      />
      <div className="step-groups">
        {groups.map((group) => {
          const isExpanded = effectiveExpanded === group.param;
          const levels = getLevels(group);
          const hasLevels = levels.length > 0;
          const isGroupSelected = group.entries.some(
            ({ variable: v }) =>
              selectedObject?.msgIdx === v.msgIndex &&
              selectedObject?.objIdx === v.objIndex,
          );

          const stepCount = hasLevels
            ? new Set(group.entries.map((e) => e.step)).size
            : group.entries.length;

          return (
            <div key={group.param} className="step-group">
              {/* Param header */}
              <div
                className={`step-header ${isExpanded ? 'step-header-expanded' : ''} ${isGroupSelected ? 'step-header-selected' : ''}`}
              >
                <button
                  className="step-header-main"
                  onClick={() => handleClickParam(group)}
                >
                  <span className="step-label">{group.param}</span>
                  <span className="step-count">
                    {group.levtype}
                    {hasLevels ? ` / ${levels.length} levels` : ''}
                    {group.units ? ` / ${group.units}` : ''}
                    {' '}({stepCount} steps)
                  </span>
                </button>
                <button
                  className="step-chevron-btn"
                  onClick={(e) => { e.stopPropagation(); toggleParam(group.param); }}
                  aria-label={isExpanded ? 'Collapse' : 'Expand'}
                >
                  <span className="step-chevron">{isExpanded ? '▲' : '▼'}</span>
                </button>
              </div>

              {/* Expanded content */}
              {isExpanded && hasLevels && (
                <div className="step-fields">
                  {levels.map((lvl) => {
                    const isLevelExpanded = expandedLevel === lvl;
                    const isLevelSelected = selectedLevel === lvl && isGroupSelected;
                    const entries = getEntriesForLevel(group, lvl);

                    return (
                      <div key={lvl} className="level-group">
                        <div
                          className={`level-row ${isLevelSelected ? 'level-row-selected' : ''} ${isLevelExpanded ? 'level-row-expanded' : ''}`}
                        >
                          <button
                            className="level-main"
                            onClick={() => handleSelectLevel(group, lvl)}
                          >
                            <span className="level-label">{lvl}{levelUnit(group.levtype)}</span>
                            <span className="level-count">{entries.length} steps</span>
                          </button>
                          <button
                            className="level-chevron-btn"
                            onClick={() => toggleLevel(lvl)}
                            aria-label={isLevelExpanded ? 'Collapse' : 'Expand'}
                          >
                            <span className="level-chevron">{isLevelExpanded ? '▼' : '▶'}</span>
                          </button>
                        </div>
                        {isLevelExpanded && entries.map(({ step, variable: v }) => {
                          const isSelected =
                            selectedObject?.msgIdx === v.msgIndex &&
                            selectedObject?.objIdx === v.objIndex;

                          return (
                            <button
                              key={`${v.msgIndex}-${v.objIndex}`}
                              className={`field-row field-row-nested ${isSelected ? 'field-row-selected' : ''}`}
                              onClick={() => {
                                handleClickStep(group, { step, marsLevel: lvl, variable: v });
                                handleLevelSlice(group, lvl);
                              }}
                            >
                              <span className="field-row-name">T+{step}h</span>
                              <span className="field-row-meta">
                                msg {v.msgIndex} / obj {v.objIndex}
                              </span>
                              <span className="field-row-shape">{v.shape.join('x')}</span>
                            </button>
                          );
                        })}
                      </div>
                    );
                  })}
                </div>
              )}

              {isExpanded && !hasLevels && (
                <div className="step-fields">
                  {group.entries.map(({ step, variable: v }) => {
                    const isSelected =
                      selectedObject?.msgIdx === v.msgIndex &&
                      selectedObject?.objIdx === v.objIndex;

                    return (
                      <button
                        key={`${v.msgIndex}-${v.objIndex}`}
                        className={`field-row ${isSelected ? 'field-row-selected' : ''}`}
                        onClick={() => handleClickStep(group, { step, marsLevel: null, variable: v })}
                      >
                        <span className="field-row-name">T+{step}h</span>
                        <span className="field-row-meta">
                          msg {v.msgIndex} / obj {v.objIndex}
                        </span>
                        <span className="field-row-shape">{v.shape.join('x')}</span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
        {groups.length === 0 && (
          <p className="no-results">No matching fields</p>
        )}
      </div>
    </div>
  );
}
