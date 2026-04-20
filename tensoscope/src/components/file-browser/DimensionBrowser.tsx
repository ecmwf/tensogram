/** Sliders for selecting a slice index along each extra dimension (ndim > 1).
 *
 * For unstructured grids the spatial dimension is a single 1D array of grid
 * points, so any dimension beyond the spatial one is "extra" (e.g. pressure
 * levels).  The spatial dimension is identified by matching its size against
 * the coordinate array length.
 */

import { useAppStore } from '../../store/useAppStore';

export function DimensionBrowser() {
  const { fileIndex, selectedObject, fieldShape, coordinates, fetchSlice } = useAppStore();

  if (!fileIndex || !selectedObject) return null;
  if (fieldShape.length <= 1) return null;

  const coordLength = coordinates?.lat.length ?? 0;

  // Find browsable (non-spatial) dimensions
  const browsableDims = fieldShape
    .map((size, idx) => ({ size, idx }))
    .filter((d) => d.size !== coordLength);

  if (browsableDims.length === 0) return null;

  // Try to get dimension names from the selected variable's metadata
  const varInfo = fileIndex.variables.find(
    (v) => v.msgIndex === selectedObject.msgIdx && v.objIndex === selectedObject.objIdx,
  );
  const dimNames = (varInfo?.metadata?.dim_names as Record<string, string>) ?? {};

  return (
    <div className="dimension-browser">
      <h2>Dimensions</h2>
      {browsableDims.map(({ size, idx }) => {
        const label = dimNames[String(size)] ?? `Dim ${idx}`;
        return (
          <DimSlider
            key={idx}
            dim={idx}
            size={size}
            label={label}
            onSelect={(sliceIdx) => fetchSlice(idx, sliceIdx)}
          />
        );
      })}
    </div>
  );
}

interface DimSliderProps {
  dim: number;
  size: number;
  label: string;
  onSelect: (idx: number) => void;
}

function DimSlider({ dim, size, label, onSelect }: DimSliderProps) {
  return (
    <div className="dim-slider">
      <label htmlFor={`dim-${dim}`}>
        {label} <span className="dim-size">(0 -- {size - 1})</span>
      </label>
      <input
        id={`dim-${dim}`}
        type="range"
        min={0}
        max={size - 1}
        defaultValue={0}
        onChange={(e) => onSelect(Number(e.target.value))}
        className="slider"
      />
    </div>
  );
}
