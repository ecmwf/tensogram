import type { AnimationFrame } from './useAnimationSequence';

export interface StepSliderProps {
  frames: AnimationFrame[];
  currentIndex: number;
  onChange: (index: number) => void;
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '4px',
    padding: '4px 10px',
    background: '#1a1a2e',
  },
  row: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  slider: {
    flex: 1,
    accentColor: '#4a90d9',
    cursor: 'pointer',
    height: '4px',
  },
  label: {
    color: '#a0c4ff',
    fontFamily: 'monospace',
    fontSize: '12px',
    minWidth: '52px',
    textAlign: 'right' as const,
  },
  ticks: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '0 2px',
    overflow: 'hidden',
  },
  tick: {
    width: '1px',
    height: '4px',
    background: '#3a3a6a',
    flexShrink: 0,
  },
};

/** Maximum number of ticks to render -- avoids crowding on large frame sets. */
const MAX_VISIBLE_TICKS = 60;

export function StepSlider({ frames, currentIndex, onChange }: StepSliderProps) {
  const total = frames.length;

  if (total === 0) {
    return (
      <div style={styles.container}>
        <div style={styles.row}>
          <input type="range" min={0} max={0} value={0} style={styles.slider} disabled />
          <span style={styles.label}>--</span>
        </div>
      </div>
    );
  }

  const currentFrame = frames[currentIndex];

  // Only render tick marks when the frame count is reasonable
  const showTicks = total <= MAX_VISIBLE_TICKS;

  return (
    <div style={styles.container}>
      <div style={styles.row}>
        <input
          type="range"
          min={0}
          max={total - 1}
          step={1}
          value={currentIndex}
          style={styles.slider}
          onChange={(e) => onChange(Number(e.target.value))}
          title={currentFrame?.label}
        />
        <span style={styles.label}>{currentFrame?.label ?? '--'}</span>
      </div>

      {showTicks && (
        <div style={styles.ticks} aria-hidden>
          {frames.map((_, i) => (
            <div key={i} style={styles.tick} />
          ))}
        </div>
      )}
    </div>
  );
}
