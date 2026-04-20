/**
 * ColorBar -- renders a vertical colour scale legend using a canvas gradient.
 */

import { useEffect, useRef } from 'react';
import { applyColormap } from './colormaps';
import type { CustomStop } from './colormaps';
import { getUnitGroup } from './units';

export interface ColorBarProps {
  min: number;
  max: number;
  palette: string;
  units: string;
  paletteReversed?: boolean;
  customStops?: CustomStop[];
  nativeUnits?: string;
  displayUnit?: string;
}

const BAR_WIDTH = 24;
const BAR_HEIGHT = 200;
const STEPS = 256;
const TICK_COUNT = 5;

export function ColorBar({ min, max, palette, units, paletteReversed, customStops, nativeUnits, displayUnit }: ColorBarProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const ramp = new Float32Array(STEPS);
    for (let i = 0; i < STEPS; i++) {
      ramp[i] = max - (i / (STEPS - 1)) * (max - min);
    }

    const rgba = applyColormap(ramp, min, max, palette, {
      reversed: paletteReversed,
      customStops,
    });

    for (let i = 0; i < STEPS; i++) {
      const base = i * 4;
      ctx.fillStyle = `rgb(${rgba[base]},${rgba[base + 1]},${rgba[base + 2]})`;
      const y = Math.round((i / STEPS) * BAR_HEIGHT);
      const h = Math.round(((i + 1) / STEPS) * BAR_HEIGHT) - y;
      ctx.fillRect(0, y, BAR_WIDTH, h);
    }
  }, [min, max, palette, paletteReversed, customStops]);

  const fmt = (v: number) =>
    Math.abs(v) < 10000 && Math.abs(v) >= 0.01
      ? v.toPrecision(3)
      : v.toExponential(2);

  const group = nativeUnits ? getUnitGroup(nativeUnits) : undefined;
  const unitLabel = displayUnit || nativeUnits || units;

  const ticks = Array.from({ length: TICK_COUNT }, (_, i) => {
    const frac = i / (TICK_COUNT - 1);
    const nativeVal = min + frac * (max - min);
    const displayVal = group && displayUnit ? group.toDisplay(nativeVal, displayUnit) : nativeVal;
    return { frac, displayVal };
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 2 }}>
      <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'stretch', gap: 4 }}>
        <div style={{ position: 'relative', width: 44, height: BAR_HEIGHT }}>
          {ticks.map(({ frac, displayVal }, i) => (
            <span
              key={i}
              style={{
                position: 'absolute',
                right: 0,
                top: `${(1 - frac) * 100}%`,
                transform: `translateY(${(frac - 1) * 100}%)`,
                fontSize: 11,
                color: '#ddd',
                whiteSpace: 'nowrap',
              }}
            >
              {fmt(displayVal)}
            </span>
          ))}
        </div>
        <canvas
          ref={canvasRef}
          width={BAR_WIDTH}
          height={BAR_HEIGHT}
          style={{ border: '1px solid rgba(255,255,255,0.15)', borderRadius: 3 }}
        />
      </div>
      {unitLabel && (
        <span style={{ fontSize: 10, color: '#aaa', textAlign: 'right' }}>{unitLabel}</span>
      )}
    </div>
  );
}
