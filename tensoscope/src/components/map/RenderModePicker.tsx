import { useState, useEffect, useRef } from 'react';

const MODES: { id: 'heatmap' | 'contours'; label: string }[] = [
  { id: 'heatmap', label: 'Heatmap' },
  { id: 'contours', label: 'Contours' },
];

interface RenderModePickerProps {
  mode: 'heatmap' | 'contours';
  onChange: (mode: 'heatmap' | 'contours') => void;
}

export function RenderModePicker({ mode, onChange }: RenderModePickerProps) {
  const [open, setOpen] = useState(false);
  const [everOpened, setEverOpened] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  const currentLabel = MODES.find((m) => m.id === mode)?.label ?? mode;

  useEffect(() => {
    if (!open) return;
    const handle = (e: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handle);
    return () => document.removeEventListener('mousedown', handle);
  }, [open]);

  const handleToggle = () => {
    if (!everOpened) setEverOpened(true);
    setOpen((o) => !o);
  };

  const handleSelect = (id: 'heatmap' | 'contours') => {
    onChange(id);
    setOpen(false);
  };

  return (
    <div ref={wrapperRef} className="render-mode-picker">
      <button className="render-mode-toggle" onClick={handleToggle}>
        <span className="render-mode-toggle-label">{currentLabel}</span>
        <span className={`render-mode-arrow${open ? ' render-mode-arrow-open' : ''}`}>▾</span>
      </button>

      {everOpened && (
        <div className={`render-mode-panel${open ? ' render-mode-panel-open' : ''}`}>
          {MODES.map((m) => (
            <button
              key={m.id}
              className={`render-mode-option${m.id === mode ? ' render-mode-option-active' : ''}`}
              onClick={() => handleSelect(m.id)}
            >
              {m.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
