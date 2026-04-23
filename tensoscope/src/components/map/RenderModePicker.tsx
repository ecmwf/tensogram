const MODES: { id: 'heatmap' | 'contours'; label: string }[] = [
  { id: 'heatmap', label: 'Heatmap' },
  { id: 'contours', label: 'Contours' },
];

interface RenderModePickerProps {
  mode: 'heatmap' | 'contours';
  onChange: (mode: 'heatmap' | 'contours') => void;
}

export function RenderModePicker({ mode, onChange }: RenderModePickerProps) {
  return (
    <div className="map-picker-pill">
      {MODES.map((m) => (
        <button
          key={m.id}
          className={`map-picker-btn${m.id === mode ? ' map-picker-btn-active' : ''}`}
          onClick={() => onChange(m.id)}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}
