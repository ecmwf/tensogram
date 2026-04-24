// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

export type SheetState = 'collapsed' | 'half' | 'full';

const THRESHOLD = 60;

/**
 * Given the current sheet state and the vertical drag delta in pixels
 * (negative = upward / opening, positive = downward / closing), returns
 * the next snap state. Deltas below THRESHOLD leave the state unchanged.
 * Deltas beyond 2×THRESHOLD skip one level entirely.
 */
export function snapSheet(current: SheetState, deltaY: number): SheetState {
  if (Math.abs(deltaY) < THRESHOLD) return current;

  if (deltaY < 0) {
    if (current === 'collapsed') return Math.abs(deltaY) > THRESHOLD * 2 ? 'full' : 'half';
    if (current === 'half') return 'full';
    return 'full'; // current === 'full', already at top
  }

  if (current === 'full') return deltaY > THRESHOLD * 2 ? 'collapsed' : 'half';
  if (current === 'half') return 'collapsed';
  return 'collapsed';
}

/** Returns the CSS height value for a given sheet snap state. */
export function sheetHeightCss(state: SheetState): string {
  if (state === 'collapsed') return '28px';
  if (state === 'half') return '50svh';
  return '90svh';
}
