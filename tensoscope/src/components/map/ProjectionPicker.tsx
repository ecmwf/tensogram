export interface ProjectionPreset {
  id: string;
  label: string;
  type: 'mercator' | 'globe';
  center: [number, number];
  zoom: number;
}

export const PROJECTION_PRESETS: ProjectionPreset[] = [
  { id: 'globe', label: 'Globe', type: 'globe',    center: [0, 20], zoom: 1.5 },
  { id: 'flat',  label: 'Flat',  type: 'mercator', center: [0, 20], zoom: 1.5 },
];

interface ProjectionPickerProps {
  current: string;
  onSelect: (preset: ProjectionPreset) => void;
}

export function ProjectionPicker({ current, onSelect }: ProjectionPickerProps) {
  return (
    <div className="map-picker-pill">
      {PROJECTION_PRESETS.map((preset) => (
        <button
          key={preset.id}
          className={`map-picker-btn${preset.id === current ? ' map-picker-btn-active' : ''}`}
          onClick={() => onSelect(preset)}
        >
          {preset.label}
        </button>
      ))}
    </div>
  );
}
