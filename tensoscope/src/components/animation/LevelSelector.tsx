export interface LevelSelectorProps {
  levels: number[];
  currentLevel: number;
  onChange: (level: number) => void;
}

const styles = {
  container: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '4px 10px',
    background: '#1a1a2e',
    color: '#e0e0e0',
    fontFamily: 'monospace',
    fontSize: '13px',
  },
  select: {
    background: '#2a2a4a',
    border: '1px solid #3a3a6a',
    color: '#e0e0e0',
    borderRadius: '4px',
    padding: '3px 6px',
    cursor: 'pointer',
    fontSize: '13px',
  },
  label: {
    color: '#888',
  },
};

/**
 * Dropdown selector for pressure levels on 3D fields.
 * Levels are expected to be in hPa, displayed in the order provided.
 */
export function LevelSelector({ levels, currentLevel, onChange }: LevelSelectorProps) {
  if (levels.length === 0) return null;

  return (
    <div style={styles.container}>
      <span style={styles.label}>Level:</span>
      <select
        value={currentLevel}
        style={styles.select}
        onChange={(e) => onChange(Number(e.target.value))}
      >
        {levels.map((lvl) => (
          <option key={lvl} value={lvl}>
            {lvl} hPa
          </option>
        ))}
      </select>
    </div>
  );
}
