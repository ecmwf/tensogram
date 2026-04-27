import type { AnimationFrame } from './useAnimationSequence';

export interface AnimationControlsProps {
  frames: AnimationFrame[];
  currentFrameIndex: number;
  isPlaying: boolean;
  speed: number; // frames per second
  onPlay: () => void;
  onPause: () => void;
  onStop: () => void;
  onFrameChange: (index: number) => void;
  onSpeedChange: (fps: number) => void;
}

const styles = {
  container: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '6px 10px',
    background: '#1a1a2e',
    borderTop: '1px solid #2a2a4a',
    color: '#e0e0e0',
    fontFamily: 'monospace',
    fontSize: '13px',
    flexWrap: 'wrap' as const,
  },
  button: {
    background: '#2a2a4a',
    border: '1px solid #3a3a6a',
    color: '#e0e0e0',
    borderRadius: '4px',
    padding: '4px 10px',
    cursor: 'pointer',
    fontSize: '13px',
  },
  label: {
    minWidth: '60px',
    textAlign: 'center' as const,
    color: '#a0c4ff',
    fontWeight: 'bold' as const,
  },
  counter: {
    color: '#888',
  },
  sliderGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  },
  slider: {
    accentColor: '#4a90d9',
    cursor: 'pointer',
  },
};

export function AnimationControls({
  frames,
  currentFrameIndex,
  isPlaying,
  speed,
  onPlay,
  onPause,
  onStop,
  onSpeedChange,
}: AnimationControlsProps) {
  const currentFrame = frames[currentFrameIndex];
  const total = frames.length;

  return (
    <div style={styles.container}>
      {/* Play/Pause toggle */}
      <button
        style={styles.button}
        onClick={isPlaying ? onPause : onPlay}
        disabled={total === 0}
        title={isPlaying ? 'Pause' : 'Play'}
      >
        {isPlaying ? '⏸' : '▶'}
      </button>

      {/* Stop -- resets to frame 0 */}
      <button
        style={styles.button}
        onClick={onStop}
        disabled={total === 0}
        title="Stop"
      >
        ⏹
      </button>

      {/* Current step label */}
      <span style={styles.label}>
        {currentFrame ? currentFrame.label : '--'}
      </span>

      {/* Frame counter */}
      <span style={styles.counter}>
        {total > 0 ? `${currentFrameIndex + 1} / ${total}` : '0 / 0'}
      </span>

      {/* Speed slider */}
      <div style={styles.sliderGroup}>
        <span>Speed:</span>
        <input
          type="range"
          min={0.5}
          max={60}
          step={0.5}
          value={speed}
          style={styles.slider}
          onChange={(e) => onSpeedChange(Number(e.target.value))}
          title={`${speed} fps`}
        />
        <span style={styles.counter}>{speed} fps</span>
      </div>
    </div>
  );
}
