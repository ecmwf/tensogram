interface RenderModePickerProps {
  mode: 'heatmap' | 'contours';
  onChange: (mode: 'heatmap' | 'contours') => void;
}

export function RenderModePicker({ mode, onChange }: RenderModePickerProps) {
  return (
    <div className="render-mode-picker">
      <button
        className={`render-mode-btn${mode === 'heatmap' ? ' render-mode-btn-active' : ''}`}
        onClick={() => onChange('heatmap')}
        title="Heatmap"
      >
        Heatmap
      </button>
      <button
        className={`render-mode-btn${mode === 'contours' ? ' render-mode-btn-active' : ''}`}
        onClick={() => onChange('contours')}
        title="Contours"
      >
        Contours
      </button>
    </div>
  );
}
