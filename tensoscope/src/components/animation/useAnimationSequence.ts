import { useMemo } from 'react';
import type { FileIndex } from '../../tensogram';

export interface AnimationFrame {
  msgIdx: number;
  objIdx: number;
  step: number;
  label: string;
}

/** Extract numeric level from per-object MARS metadata. */
function getMarsLevel(mars: Record<string, unknown>): number | null {
  const raw = mars.levelist ?? mars.level;
  if (raw == null) return null;
  if (mars.levtype === 'sfc') return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

/**
 * Builds an ordered list of animation frames from a FileIndex for a given
 * parameter name, optionally filtered by level.
 *
 * Handles two data layouts:
 * - **Per-object levels**: separate objects per level with mars.level/levelist.
 *   Filters to `selectedLevel` so animation only cycles through steps at one level.
 * - **Packed levels**: levels baked into the object shape. All objects match
 *   (level slicing is handled by fetchSlice, not by object selection).
 *
 * Frames are sorted by mars.step, deduplicated.
 */
export function useAnimationSequence(
  fileIndex: FileIndex | null,
  paramName: string | number | null,
  selectedLevel: number | null,
): AnimationFrame[] {
  return useMemo(() => {
    if (!fileIndex || paramName == null) return [];

    const matched = fileIndex.variables.filter((v) => {
      const mars = v.metadata?.mars as Record<string, unknown> | undefined;
      return mars?.param === paramName;
    });

    if (matched.length === 0) return [];

    // Check if objects span multiple distinct per-object levels.
    // Only filter when there are genuinely different level values across
    // objects -- if all objects share the same mars.level (common for packed-
    // level data) that's just a metadata annotation, not a real split.
    const withLevels = matched.map((v) => {
      const mars = (v.metadata?.mars as Record<string, unknown>) ?? {};
      return { v, mars, level: getMarsLevel(mars), step: Number(mars.step ?? 0) };
    });

    const uniqueLevels = new Set(withLevels.filter((e) => e.level != null).map((e) => e.level!));
    const hasPerObjectLevels = uniqueLevels.size > 1;

    let filtered = withLevels;
    if (hasPerObjectLevels && selectedLevel != null) {
      filtered = withLevels.filter((e) => e.level === selectedLevel);
    } else if (hasPerObjectLevels) {
      const levels = [...uniqueLevels].sort((a, b) => a - b);
      const defaultLevel = levels[0];
      filtered = withLevels.filter((e) => e.level === defaultLevel);
    }

    // Sort by step, deduplicate
    filtered.sort((a, b) => a.step - b.step);

    const seen = new Set<number>();
    const frames: AnimationFrame[] = [];
    for (const e of filtered) {
      if (seen.has(e.step)) continue;
      seen.add(e.step);
      frames.push({
        msgIdx: e.v.msgIndex,
        objIdx: e.v.objIndex,
        step: e.step,
        label: `T+${e.step}h`,
      });
    }

    return frames;
  }, [fileIndex, paramName, selectedLevel]);
}
