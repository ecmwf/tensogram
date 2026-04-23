import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { PALETTE_NAMES, getPaletteCSS, getPaletteLUT, LUT_SIZE } from './colormaps';
import type { CustomStop, PaletteOptions } from './colormaps';
import { getUnitGroup } from './units';

export type { CustomStop };

// ── Public interface ─────────────────────────────────────────────────────────

export interface ColorScaleControlsProps {
  palette: string;
  colorMin: number;
  colorMax: number;
  dataMin: number;
  dataMax: number;
  paletteReversed: boolean;
  customStops: CustomStop[];
  nativeUnits: string;
  displayUnit: string;
  onPaletteChange: (palette: string) => void;
  onColorMinChange: (v: number) => void;
  onColorMaxChange: (v: number) => void;
  onPaletteReversedChange: (v: boolean) => void;
  onCustomStopsChange: (stops: CustomStop[]) => void;
  onDisplayUnitChange: (unit: string) => void;
}

// ── Shared styles ────────────────────────────────────────────────────────────

const PANEL: React.CSSProperties = {
  background: 'rgba(30,30,30,0.92)',
  borderRadius: 8,
  padding: '10px 12px',
  display: 'flex',
  flexDirection: 'column',
  gap: 8,
  width: 220,
  color: '#ddd',
  fontSize: 12,
  boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
};

const SECTION_LABEL: React.CSSProperties = {
  fontSize: 10,
  color: '#888',
  textTransform: 'uppercase',
  letterSpacing: '0.6px',
};

const SMALL_LABEL: React.CSSProperties = {
  fontSize: 10,
  color: '#888',
};

const HINT: React.CSSProperties = {
  fontSize: 10,
  color: '#555',
  fontStyle: 'italic',
};

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatValue(v: number): string {
  if (Math.abs(v) >= 10000 || (Math.abs(v) < 0.01 && v !== 0)) {
    return v.toExponential(2);
  }
  return parseFloat(v.toPrecision(4)).toString();
}

function interpolateStopColor(sorted: CustomStop[], pos: number): string {
  if (sorted.length === 0) return '#888888';
  if (pos <= sorted[0].pos) return sorted[0].color;
  if (pos >= sorted[sorted.length - 1].pos) return sorted[sorted.length - 1].color;
  for (let i = 0; i < sorted.length - 1; i++) {
    const lo = sorted[i];
    const hi = sorted[i + 1];
    if (lo.pos <= pos && hi.pos >= pos) {
      const span = hi.pos - lo.pos;
      const f = span === 0 ? 0 : (pos - lo.pos) / span;
      const parse = (hex: string): [number, number, number] => [
        parseInt(hex.slice(1, 3), 16),
        parseInt(hex.slice(3, 5), 16),
        parseInt(hex.slice(5, 7), 16),
      ];
      const [r1, g1, b1] = parse(lo.color.length === 7 ? lo.color : '#000000');
      const [r2, g2, b2] = parse(hi.color.length === 7 ? hi.color : '#000000');
      const hex = (n: number) => Math.round(n).toString(16).padStart(2, '0');
      return `#${hex(r1 + (r2 - r1) * f)}${hex(g1 + (g2 - g1) * f)}${hex(b1 + (b2 - b1) * f)}`;
    }
  }
  return '#888888';
}

/** Sample 5 colours from a named palette LUT to seed the custom editor. */
function seedFromPalette(palette: string, reversed: boolean): CustomStop[] {
  const lut = getPaletteLUT(palette, { reversed });
  const N = 5;
  return Array.from({ length: N }, (_, i) => {
    const pos = i / (N - 1);
    const idx = Math.round(pos * (LUT_SIZE - 1));
    const r = lut[idx * 3];
    const g = lut[idx * 3 + 1];
    const b = lut[idx * 3 + 2];
    const h = (n: number) => n.toString(16).padStart(2, '0');
    return { pos, color: `#${h(r)}${h(g)}${h(b)}` };
  });
}

// ── SpinboxInput ─────────────────────────────────────────────────────────────

interface SpinboxInputProps {
  label: string;
  value: number;
  step: number;
  onChange: (v: number) => void;
  onDoubleClick: () => void;
  toDisplay?: (v: number) => number;
  toNative?: (v: number) => number;
}

function SpinboxInput({ label, value, step, onChange, onDoubleClick, toDisplay, toNative }: SpinboxInputProps) {
  const [draft, setDraft] = useState(formatValue(toDisplay ? toDisplay(value) : value));
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (document.activeElement !== inputRef.current) {
      setDraft(formatValue(toDisplay ? toDisplay(value) : value));
    }
  }, [value, toDisplay]);

  const commit = useCallback(() => {
    const v = parseFloat(draft);
    if (!isNaN(v)) onChange(toNative ? toNative(v) : v);
    else setDraft(formatValue(toDisplay ? toDisplay(value) : value));
  }, [draft, onChange, value, toDisplay, toNative]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 3, flex: 1 }}>
      <span style={SMALL_LABEL}>{label}</span>
      <div
        style={{
          display: 'flex',
          background: '#111',
          border: '1px solid #444',
          borderRadius: 4,
          overflow: 'hidden',
          cursor: 'default',
        }}
        onDoubleClick={onDoubleClick}
        title="Double-click to reset to data range"
      >
        <input
          ref={inputRef}
          value={draft}
          onChange={e => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={e => { if (e.key === 'Enter') commit(); }}
          onDoubleClick={e => e.stopPropagation()}
          style={{
            flex: 1,
            background: 'transparent',
            border: 'none',
            color: '#ddd',
            padding: '4px 6px',
            fontSize: 12,
            width: 0,
            outline: 'none',
          }}
        />
        <div style={{ display: 'flex', flexDirection: 'column', borderLeft: '1px solid #333' }}>
          <div
            style={{
              flex: 1,
              padding: '1px 5px',
              cursor: 'pointer',
              color: '#666',
              fontSize: 9,
              lineHeight: 1,
              display: 'flex',
              alignItems: 'center',
              userSelect: 'none',
            }}
            onMouseDown={e => { e.preventDefault(); onChange(toNative ? toNative((toDisplay ? toDisplay(value) : value) + step) : value + step); }}
          >▲</div>
          <div
            style={{
              flex: 1,
              padding: '1px 5px',
              cursor: 'pointer',
              color: '#666',
              fontSize: 9,
              lineHeight: 1,
              display: 'flex',
              alignItems: 'center',
              borderTop: '1px solid #333',
              userSelect: 'none',
            }}
            onMouseDown={e => { e.preventDefault(); onChange(toNative ? toNative((toDisplay ? toDisplay(value) : value) - step) : value - step); }}
          >▼</div>
        </div>
      </div>
    </div>
  );
}

// ── UnitToggle ───────────────────────────────────────────────────────────────

interface UnitToggleProps {
  nativeUnits: string;
  displayUnit: string;
  onDisplayUnitChange: (u: string) => void;
}

function UnitToggle({ nativeUnits, displayUnit, onDisplayUnitChange }: UnitToggleProps) {
  const group = getUnitGroup(nativeUnits);
  if (!group) return null;

  return (
    <div style={{ display: 'flex', border: '1px solid #444', borderRadius: 3, overflow: 'hidden' }}>
      {group.units.map(u => (
        <div
          key={u}
          onClick={() => onDisplayUnitChange(u)}
          style={{
            fontSize: 10,
            padding: '1px 6px',
            cursor: 'pointer',
            userSelect: 'none',
            lineHeight: 1.6,
            background: u === displayUnit ? '#4a9' : 'transparent',
            color: u === displayUnit ? '#000' : '#888',
          }}
        >
          {u}
        </div>
      ))}
    </div>
  );
}

// ── GradientEditor ────────────────────────────────────────────────────────────

interface GradientEditorProps {
  stops: CustomStop[];
  onChange: (stops: CustomStop[]) => void;
}

function GradientEditor({ stops, onChange }: GradientEditorProps) {
  const barRef = useRef<HTMLDivElement>(null);
  const dragging = useRef<{ sortedIdx: number; startClientX: number; startPos: number } | null>(null);
  const colorInputRef = useRef<HTMLInputElement>(null);
  const editingIdx = useRef<number>(-1);

  const sorted = useMemo(() => [...stops].sort((a, b) => a.pos - b.pos), [stops]);
  const sortedRef = useRef(sorted);
  useEffect(() => { sortedRef.current = sorted; }, [sorted]);

  const getBarFraction = (clientX: number): number => {
    if (!barRef.current) return 0;
    const rect = barRef.current.getBoundingClientRect();
    return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  };

  const handleBarClick = (e: React.MouseEvent) => {
    const pos = getBarFraction(e.clientX);
    const color = interpolateStopColor(sortedRef.current, pos);
    onChange([...sortedRef.current, { pos, color }]);
  };

  const handleHandleMouseDown = (e: React.MouseEvent, sortedIdx: number) => {
    e.stopPropagation();
    e.preventDefault();
    dragging.current = { sortedIdx, startClientX: e.clientX, startPos: sorted[sortedIdx].pos };
  };

  const handleHandleClick = (e: React.MouseEvent, sortedIdx: number) => {
    e.stopPropagation();
    editingIdx.current = sortedIdx;
    if (colorInputRef.current) {
      colorInputRef.current.value = sorted[sortedIdx].color;
      colorInputRef.current.click();
    }
  };

  const handleHandleContextMenu = (e: React.MouseEvent, sortedIdx: number) => {
    e.preventDefault();
    if (sortedRef.current.length <= 2) return;
    onChange(sortedRef.current.filter((_, i) => i !== sortedIdx));
  };

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragging.current || !barRef.current) return;
      const { sortedIdx, startClientX, startPos } = dragging.current;
      const current = sortedRef.current;
      const rect = barRef.current.getBoundingClientRect();
      const delta = (e.clientX - startClientX) / rect.width;
      let newPos = Math.max(0, Math.min(1, startPos + delta));
      const prev = sortedIdx > 0 ? current[sortedIdx - 1].pos + 0.001 : 0;
      const next = sortedIdx < current.length - 1 ? current[sortedIdx + 1].pos - 0.001 : 1;
      newPos = Math.max(prev, Math.min(next, newPos));
      onChange(current.map((s, i) => i === sortedIdx ? { ...s, pos: newPos } : s));
    };
    const onUp = () => { dragging.current = null; };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [onChange]);

  const gradientCss = sorted.map(s => `${s.color} ${(s.pos * 100).toFixed(1)}%`).join(', ');

  return (
    <div style={{
      border: '1px solid #444',
      borderRadius: 5,
      padding: '8px',
      background: '#0e0e0e',
      display: 'flex',
      flexDirection: 'column',
      gap: 8,
    }}>
      <input
        ref={colorInputRef}
        type="color"
        style={{ position: 'absolute', opacity: 0, pointerEvents: 'none', width: 0, height: 0 }}
        onChange={e => {
          const idx = editingIdx.current;
          if (idx < 0) return;
          onChange(sortedRef.current.map((s, i) => i === idx ? { ...s, color: e.target.value } : s));
        }}
      />

      <div style={{ position: 'relative', height: 28 }}>
        <div
          ref={barRef}
          style={{
            position: 'absolute',
            top: 12,
            left: 6,
            right: 6,
            height: 16,
            borderRadius: 4,
            background: `linear-gradient(to right, ${gradientCss})`,
            cursor: 'crosshair',
          }}
          onClick={handleBarClick}
        />
        {sorted.map((stop, idx) => (
          <div
            key={idx}
            style={{
              position: 'absolute',
              top: 4,
              left: `calc(6px + ${stop.pos} * (100% - 12px))`,
              transform: 'translateX(-50%)',
              cursor: 'grab',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
            }}
            onMouseDown={e => handleHandleMouseDown(e, idx)}
            onClick={e => handleHandleClick(e, idx)}
            onContextMenu={e => handleHandleContextMenu(e, idx)}
          >
            <div style={{
              width: 0,
              height: 0,
              borderLeft: '5px solid transparent',
              borderRight: '5px solid transparent',
              borderTop: `8px solid ${stop.color}`,
              filter: 'drop-shadow(0 1px 1px rgba(0,0,0,0.8))',
            }} />
          </div>
        ))}
      </div>

      <span style={HINT}>Drag · click to recolour · right-click to remove</span>
    </div>
  );
}

// ── PalettePicker ─────────────────────────────────────────────────────────────

interface PalettePickerProps {
  palette: string;
  paletteReversed: boolean;
  customStops: CustomStop[];
  onPaletteChange: (p: string) => void;
  onPaletteReversedChange: (r: boolean) => void;
  onCustomStopsChange: (stops: CustomStop[]) => void;
}

function PalettePicker({
  palette, paletteReversed, customStops,
  onPaletteChange, onPaletteReversedChange, onCustomStopsChange,
}: PalettePickerProps) {
  const [open, setOpen] = useState(false);

  const wrapperRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  const handleSelect = (name: string) => {
    if (name === 'custom' && palette !== 'custom') {
      onCustomStopsChange(seedFromPalette(palette, paletteReversed));
    }
    onPaletteChange(name);
    setOpen(false);
  };

  const options: PaletteOptions = { reversed: paletteReversed, customStops };
  const activeCss = getPaletteCSS(palette, options);

  return (
    <div ref={wrapperRef} style={{ position: 'relative' }}>
      <div style={{ display: 'flex', gap: 6, alignItems: 'stretch' }}>
        <div
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            background: '#111',
            border: `1px solid ${open ? '#4a9' : '#444'}`,
            borderRadius: 5,
            padding: '5px 8px',
            cursor: 'pointer',
          }}
          onClick={() => setOpen(o => !o)}
        >
          <div style={{ height: 16, flex: 1, borderRadius: 3, background: activeCss }} />
          <span style={{ fontSize: 11, color: '#bbb', whiteSpace: 'nowrap' }}>
            {palette === 'custom' ? 'Custom' : palette}
          </span>
          <span style={{ fontSize: 10, color: open ? '#4a9' : '#666' }}>{open ? '⌃' : '⌄'}</span>
        </div>
        <div
          style={{
            background: '#111',
            border: '1px solid #444',
            borderRadius: 5,
            padding: '5px 8px',
            cursor: 'pointer',
            color: paletteReversed ? '#4a9' : '#888',
            fontSize: 14,
            display: 'flex',
            alignItems: 'center',
            userSelect: 'none',
          }}
          onClick={() => onPaletteReversedChange(!paletteReversed)}
          title="Reverse palette"
        >⇄</div>
      </div>

      {open && (
        <div style={{
          position: 'absolute',
          top: '100%',
          left: 0,
          right: 0,
          marginTop: 4,
          background: '#111',
          border: '1px solid #444',
          borderRadius: 5,
          overflow: 'hidden',
          zIndex: 20,
        }}>
          {PALETTE_NAMES.map(name => (
            <div
              key={name}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '5px 8px',
                cursor: 'pointer',
                background: name === palette ? '#1a3a2a' : 'transparent',
                borderLeft: name === palette ? '2px solid #4a9' : '2px solid transparent',
              }}
              onClick={() => handleSelect(name)}
            >
              <div style={{
                height: 12,
                width: 80,
                flexShrink: 0,
                borderRadius: 2,
                background: getPaletteCSS(name, { reversed: paletteReversed }),
              }} />
              <span style={{ fontSize: 11, color: '#ccc' }}>{name}</span>
            </div>
          ))}
          <div style={{ borderTop: '1px solid #333' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '5px 8px',
                cursor: 'pointer',
                background: palette === 'custom' ? '#1a3a2a' : 'transparent',
                borderLeft: palette === 'custom' ? '2px solid #4a9' : '2px solid transparent',
              }}
              onClick={() => handleSelect('custom')}
            >
              <div style={{
                height: 12,
                width: 80,
                flexShrink: 0,
                borderRadius: 2,
                background: getPaletteCSS('custom', { customStops }),
                border: '1px dashed #555',
              }} />
              <span style={{ fontSize: 11, color: '#ccc' }}>Custom...</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── ColorScaleControls (public) ───────────────────────────────────────────────

const MINIMISE_BTN: React.CSSProperties = {
  background: 'none',
  border: 'none',
  color: '#666',
  cursor: 'pointer',
  fontSize: 13,
  padding: '0 2px',
  lineHeight: 1,
  display: 'flex',
  alignItems: 'center',
};

export function ColorScaleControls({
  palette,
  colorMin,
  colorMax,
  dataMin,
  dataMax,
  paletteReversed,
  customStops,
  nativeUnits,
  displayUnit,
  onPaletteChange,
  onColorMinChange,
  onColorMaxChange,
  onPaletteReversedChange,
  onCustomStopsChange,
  onDisplayUnitChange,
}: ColorScaleControlsProps) {
  const [minimised, setMinimised] = useState(false);

  const group = getUnitGroup(nativeUnits);
  const toDisplay = useMemo(
    () => group ? (v: number) => group.toDisplay(v, displayUnit) : undefined,
    [group, displayUnit],
  );
  const toNative = useMemo(
    () => group ? (v: number) => group.toNative(v, displayUnit) : undefined,
    [group, displayUnit],
  );
  const nativeStep = Math.max(0.001, (dataMax - dataMin) * 0.01);
  const step = group && toDisplay
    ? Math.abs(group.toDisplay(nativeStep, displayUnit) - group.toDisplay(0, displayUnit))
    : nativeStep;

  const resetRange = () => {
    onColorMinChange(dataMin);
    onColorMaxChange(dataMax);
  };

  return (
    <div style={PANEL}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={SECTION_LABEL}>Colour Palette</div>
        <button
          style={MINIMISE_BTN}
          onClick={() => setMinimised(m => !m)}
          title={minimised ? 'Expand' : 'Minimise'}
        >
          {minimised ? '▴' : '▾'}
        </button>
      </div>
      {!minimised && (
        <>
          <PalettePicker
            palette={palette}
            paletteReversed={paletteReversed}
            customStops={customStops}
            onPaletteChange={onPaletteChange}
            onPaletteReversedChange={onPaletteReversedChange}
            onCustomStopsChange={onCustomStopsChange}
          />

          {palette === 'custom' && (
            <GradientEditor stops={customStops} onChange={onCustomStopsChange} />
          )}

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={SECTION_LABEL}>Range</div>
            <UnitToggle
              nativeUnits={nativeUnits}
              displayUnit={displayUnit}
              onDisplayUnitChange={onDisplayUnitChange}
            />
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <SpinboxInput
              label="Min"
              value={colorMin}
              step={step}
              onChange={onColorMinChange}
              onDoubleClick={resetRange}
              toDisplay={toDisplay}
              toNative={toNative}
            />
            <SpinboxInput
              label="Max"
              value={colorMax}
              step={step}
              onChange={onColorMaxChange}
              onDoubleClick={resetRange}
              toDisplay={toDisplay}
              toNative={toNative}
            />
          </div>
          <span style={HINT}>Double-click to reset to data range</span>
        </>
      )}
    </div>
  );
}
