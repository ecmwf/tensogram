import { useState, useEffect, useRef } from 'react';
import globeImg from '../../assets/globe.png';
import flatImg from '../../assets/flat.png';

export interface ProjectionPreset {
  id: string;
  label: string;
  type: 'mercator' | 'globe';
  center: [number, number];
  zoom: number;
  image: string;
}

export const PROJECTION_PRESETS: ProjectionPreset[] = [
  { id: 'globe', label: 'Globe', type: 'globe',    center: [0, 20], zoom: 1.5, image: globeImg },
  { id: 'flat',  label: 'Flat',  type: 'mercator', center: [0, 20], zoom: 1.5, image: flatImg  },
];

interface ProjectionPickerProps {
  current: string;
  onSelect: (preset: ProjectionPreset) => void;
}

export function ProjectionPicker({ current, onSelect }: ProjectionPickerProps) {
  const [open, setOpen] = useState(false);
  const [everOpened, setEverOpened] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  const activePreset = PROJECTION_PRESETS.find((p) => p.id === current) ?? PROJECTION_PRESETS[0];

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

  const handleSelect = (preset: ProjectionPreset) => {
    onSelect(preset);
    setOpen(false);
  };

  return (
    <div ref={wrapperRef} className="projection-picker">
      <button className="projection-toggle" onClick={handleToggle}>
        <img src={activePreset.image} alt={activePreset.label} className="projection-toggle-img" />
      </button>

      {everOpened && (
        <div className={`projection-panel${open ? ' projection-panel-open' : ''}`}>
          {PROJECTION_PRESETS.map((preset) => (
            <button
              key={preset.id}
              className={`projection-thumb${preset.id === current ? ' projection-thumb-active' : ''}`}
              onClick={() => handleSelect(preset)}
            >
              <div className="projection-thumb-map">
                <img src={preset.image} alt={preset.label} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              </div>
              <span className="projection-thumb-label">{preset.label}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
