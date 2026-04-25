import { useEffect, useRef } from 'react';
import type { InspectionData, TimeSeriesEntry } from './usePointInspection';
import { UnitToggle } from './ColorScaleControls';
import { getUnitGroup } from './units';

// ── Chart constants ──────────────────────────────────────────────────────────

const SVG_W = 248;
const SVG_H = 110;
const Y_AXIS_X = 36;
const X_AXIS_Y = 78;
const DATA_W = SVG_W - Y_AXIS_X;  // 212
const DATA_H = X_AXIS_Y - 4;      // 74
const PADDING_TOP = 4;

const MAX_FULL_VALUES = 8;

// ── Helpers ──────────────────────────────────────────────────────────────────

function fmt(v: number): string {
  if (!Number.isFinite(v)) return 'N/A';
  const abs = Math.abs(v);
  if (abs === 0) return '0';
  if (abs >= 10000 || (abs < 0.01)) return v.toExponential(3);
  return parseFloat(v.toPrecision(4)).toString();
}

/** Format a tick label with precision driven by the chart's value range. */
function fmtTick(v: number, range: number): string {
  if (!Number.isFinite(v)) return 'N/A';
  if (range === 0) return fmt(v);
  const tickInterval = range / 3;
  if (tickInterval >= 10000 || tickInterval < 0.001) return v.toExponential(2);
  const decimals = Math.max(0, Math.ceil(-Math.log10(tickInterval)) + 1);
  return v.toFixed(Math.min(decimals, 6));
}

function yForValue(v: number, min: number, max: number): number {
  if (max === min) return PADDING_TOP + DATA_H / 2;
  return PADDING_TOP + (1 - (v - min) / (max - min)) * DATA_H;
}

function xForIndex(i: number, n: number): number {
  if (n <= 1) return Y_AXIS_X + DATA_W / 2;
  return Y_AXIS_X + (i / (n - 1)) * DATA_W;
}

function tickIndices(n: number, maxTicks: number): number[] {
  if (n <= maxTicks) return Array.from({ length: n }, (_, i) => i);
  const result: number[] = [0];
  const step = (n - 1) / (maxTicks - 1);
  for (let t = 1; t < maxTicks - 1; t++) result.push(Math.round(t * step));
  result.push(n - 1);
  return result;
}

function yTicks(min: number, max: number): number[] {
  if (min === max) return [min];
  return Array.from({ length: 4 }, (_, i) => min + (i / 3) * (max - min));
}

// ── TimeSeriesChart ──────────────────────────────────────────────────────────

interface TimeSeriesChartProps {
  entries: TimeSeriesEntry[];
  toDisplay: (v: number) => number;
}

function TimeSeriesChart({ entries, toDisplay }: TimeSeriesChartProps) {
  const displayed = entries.map((e) => toDisplay(e.value));
  const min = Math.min(...displayed);
  const max = Math.max(...displayed);
  const range = max - min;
  const n = entries.length;

  const points = displayed
    .map((v, i) => `${xForIndex(i, n).toFixed(1)},${yForValue(v, min, max).toFixed(1)}`)
    .join(' ');

  const xTicks = tickIndices(n, 5);
  const yTickValues = yTicks(min, max);

  return (
    <svg width={SVG_W} height={SVG_H} style={{ display: 'block', overflow: 'visible' }}>
      {/* Axes */}
      <line x1={Y_AXIS_X} y1={PADDING_TOP} x2={Y_AXIS_X} y2={X_AXIS_Y} stroke="#444" strokeWidth={1} />
      <line x1={Y_AXIS_X} y1={X_AXIS_Y} x2={SVG_W} y2={X_AXIS_Y} stroke="#444" strokeWidth={1} />

      {/* Y gridlines + ticks + labels */}
      {yTickValues.map((v, i) => {
        const y = yForValue(v, min, max);
        return (
          <g key={i}>
            <line x1={Y_AXIS_X - 3} y1={y} x2={Y_AXIS_X} y2={y} stroke="#555" strokeWidth={1} />
            <line x1={Y_AXIS_X} y1={y} x2={SVG_W} y2={y} stroke="#2a2a3e" strokeWidth={1} />
            <text x={Y_AXIS_X - 5} y={y + 3} fontSize={9} fill="#666" textAnchor="end">
              {fmtTick(v, range)}
            </text>
          </g>
        );
      })}

      {/* X ticks + labels */}
      {xTicks.map((idx) => {
        const x = xForIndex(idx, n);
        return (
          <g key={idx}>
            <line x1={x} y1={X_AXIS_Y} x2={x} y2={X_AXIS_Y + 3} stroke="#555" strokeWidth={1} />
            <text x={x} y={X_AXIS_Y + 13} fontSize={9} fill="#666" textAnchor="middle">
              {entries[idx].label}
            </text>
          </g>
        );
      })}

      {/* Data line */}
      {n > 1 && (
        <polyline
          points={points}
          fill="none"
          stroke="#4a9eff"
          strokeWidth={1.8}
          strokeLinejoin="round"
        />
      )}

      {/* Data points */}
      {displayed.map((v, i) => (
        <circle
          key={i}
          cx={xForIndex(i, n)}
          cy={yForValue(v, min, max)}
          r={3}
          fill="#1e1e2e"
          stroke="#4a9eff"
          strokeWidth={1.5}
        />
      ))}
    </svg>
  );
}

// ── ValueGrid (≤ MAX_FULL_VALUES steps) ──────────────────────────────────────

interface ValueGridProps {
  entries: TimeSeriesEntry[];
  toDisplay: (v: number) => number;
  unitLabel: string;
}

function ValueGrid({ entries, toDisplay, unitLabel }: ValueGridProps) {
  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gap: '2px 12px',
      fontSize: 10,
      color: '#888',
    }}>
      {entries.map((e) => (
        <span key={e.step}>
          {e.label}:{' '}
          <span style={{ color: '#4a9eff' }}>
            {fmt(toDisplay(e.value))}{unitLabel ? ` ${unitLabel}` : ''}
          </span>
        </span>
      ))}
    </div>
  );
}

// ── SummaryStats (> MAX_FULL_VALUES steps) ────────────────────────────────────

interface SummaryStatsProps {
  entries: TimeSeriesEntry[];
  toDisplay: (v: number) => number;
  unitLabel: string;
}

function SummaryStats({ entries, toDisplay, unitLabel }: SummaryStatsProps) {
  const vals = entries.map((e) => toDisplay(e.value));
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const std = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length);
  const u = unitLabel ? ` ${unitLabel}` : '';
  const first = entries[0];
  const last = entries[entries.length - 1];

  return (
    <div>
      <div style={{ fontSize: 10, color: '#666', marginBottom: 5 }}>
        {entries.length} time steps · {first.label} to {last.label}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 12px', fontSize: 11 }}>
        {([['min', min], ['max', max], ['mean', mean], ['std', std]] as [string, number][]).map(([label, val]) => (
          <div key={label} style={{ color: '#888' }}>
            {label}{' '}
            <span style={{ color: '#4a9eff', float: 'right' }}>{fmt(val)}{u}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── PointInspector (public) ──────────────────────────────────────────────────

export interface PointInspectorProps {
  result: InspectionData;
  screenX: number;
  screenY: number;
  paramName: string;
  levelLabel: string;
  nativeUnits: string;
  displayUnit: string;
  onDisplayUnitChange: (u: string) => void;
  onClose: () => void;
}

export function PointInspector({
  result,
  screenX,
  screenY,
  paramName,
  levelLabel,
  nativeUnits,
  displayUnit,
  onDisplayUnitChange,
  onClose,
}: PointInspectorProps) {
  const popupRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (popupRef.current && !popupRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [onClose]);

  const group = getUnitGroup(nativeUnits);
  const toDisplay = group ? (v: number) => group.toDisplay(v, displayUnit) : (v: number) => v;
  const unitLabel = displayUnit || nativeUnits;

  const hasTimeSeries = result.entries.length > 0;

  const popupStyle: React.CSSProperties = {
    position: 'absolute',
    top: Math.max(8, screenY - 10),
    left: screenX + 14,
    zIndex: 20,
    background: 'rgba(30,30,30,0.95)',
    border: '1px solid #555',
    borderRadius: 8,
    padding: '12px 14px',
    width: 280,
    boxShadow: '0 6px 24px rgba(0,0,0,0.7)',
    fontFamily: 'monospace',
    fontSize: 12,
    color: '#ddd',
    pointerEvents: 'auto',
  };

  return (
    <div ref={popupRef} style={popupStyle}>
      {/* Row 1: coords + close */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <span style={{ color: '#aaa', fontSize: 11 }}>
          Lat: {result.pointLat.toFixed(3)} &nbsp; Lon: {result.pointLon.toFixed(3)}
        </span>
        <span
          style={{ color: '#666', fontSize: 15, cursor: 'pointer', lineHeight: 1 }}
          onMouseDown={(e) => { e.stopPropagation(); onClose(); }}
        >
          ✕
        </span>
      </div>

      {/* Row 2: param + level + unit toggle */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
          <span style={{ fontWeight: 600, fontSize: 12 }}>{paramName}</span>
          {levelLabel && <span style={{ color: '#888', fontSize: 10 }}>{levelLabel}</span>}
        </div>
        <UnitToggle
          nativeUnits={nativeUnits}
          displayUnit={displayUnit}
          onDisplayUnitChange={onDisplayUnitChange}
        />
      </div>

      {/* Loading */}
      {result.loading && (
        <div style={{ color: '#666', fontSize: 11, padding: '8px 0' }}>Loading...</div>
      )}

      {/* Time series */}
      {!result.loading && hasTimeSeries && (
        <>
          <TimeSeriesChart entries={result.entries} toDisplay={toDisplay} />
          <div style={{ borderTop: '1px solid #333', marginTop: 10, paddingTop: 8 }}>
            {result.entries.length <= MAX_FULL_VALUES ? (
              <ValueGrid entries={result.entries} toDisplay={toDisplay} unitLabel={unitLabel} />
            ) : (
              <SummaryStats entries={result.entries} toDisplay={toDisplay} unitLabel={unitLabel} />
            )}
          </div>
        </>
      )}

      {/* Single value */}
      {!result.loading && !hasTimeSeries && (
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, padding: '8px 0' }}>
          <span style={{ color: '#4a9eff', fontSize: 22, fontWeight: 600 }}>
            {result.singleValue !== null ? fmt(toDisplay(result.singleValue)) : 'N/A'}
          </span>
          {unitLabel && <span style={{ color: '#888', fontSize: 12 }}>{unitLabel}</span>}
        </div>
      )}
    </div>
  );
}
