// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import { describe, expect, it } from 'vitest';
import { snapSheet, sheetHeightCss } from '../sheetSnap';

describe('snapSheet', () => {
  it('swipe up from collapsed past threshold snaps to half', () => {
    expect(snapSheet('collapsed', -70)).toBe('half');
  });

  it('swipe up from half past threshold snaps to full', () => {
    expect(snapSheet('half', -70)).toBe('full');
  });

  it('swipe down from full past threshold snaps to half', () => {
    expect(snapSheet('full', 70)).toBe('half');
  });

  it('swipe down from half past threshold snaps to collapsed', () => {
    expect(snapSheet('half', 70)).toBe('collapsed');
  });

  it('small upward delta below threshold returns same state', () => {
    expect(snapSheet('half', -30)).toBe('half');
  });

  it('small downward delta below threshold returns same state', () => {
    expect(snapSheet('half', 30)).toBe('half');
  });

  it('large upward delta from collapsed skips directly to full', () => {
    expect(snapSheet('collapsed', -130)).toBe('full');
  });

  it('large downward delta from full skips directly to collapsed', () => {
    expect(snapSheet('full', 130)).toBe('collapsed');
  });

  it('swipe up from full stays full', () => {
    expect(snapSheet('full', -70)).toBe('full');
  });

  it('swipe down from collapsed stays collapsed', () => {
    expect(snapSheet('collapsed', 70)).toBe('collapsed');
  });
});

describe('sheetHeightCss', () => {
  it('returns 28px for collapsed', () => {
    expect(sheetHeightCss('collapsed')).toBe('28px');
  });

  it('returns 50svh for half', () => {
    expect(sheetHeightCss('half')).toBe('50svh');
  });

  it('returns 90svh for full', () => {
    expect(sheetHeightCss('full')).toBe('90svh');
  });
});
