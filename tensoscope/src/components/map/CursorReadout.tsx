/**
 * CursorReadout -- displays the lat, lon, and interpolated data value at the
 * current cursor position.
 */

export interface CursorReadoutProps {
  data: Float32Array;
  shape: [number, number]; // [nlat, nlon]
  latBounds: [number, number]; // [south, north]
  lonBounds: [number, number]; // [west, east]
  lat: number;
  lon: number;
}

/**
 * Given geographic coordinates, return the nearest grid-cell value by
 * snapping to the nearest index in lat/lon.
 */
function lookupValue(
  data: Float32Array,
  shape: [number, number],
  latBounds: [number, number],
  lonBounds: [number, number],
  lat: number,
  lon: number,
): number | null {
  const [nlat, nlon] = shape;
  const [south, north] = latBounds;
  const [west, east] = lonBounds;

  // Normalise to [0, 1] then map to grid indices
  const tLat = (lat - south) / (north - south);
  const tLon = (lon - west) / (east - west);

  if (tLat < 0 || tLat > 1 || tLon < 0 || tLon > 1) return null;

  const iLat = Math.round(tLat * (nlat - 1));
  const iLon = Math.round(tLon * (nlon - 1));

  const idx = iLat * nlon + iLon;
  if (idx < 0 || idx >= data.length) return null;

  const v = data[idx];
  return isNaN(v) ? null : v;
}

const containerStyle: React.CSSProperties = {
  background: 'rgba(30,30,30,0.85)',
  borderRadius: 4,
  padding: '6px 10px',
  color: '#ddd',
  fontSize: 12,
  fontFamily: 'monospace',
  display: 'flex',
  flexDirection: 'column',
  gap: 2,
  minWidth: 150,
};

function fmt(v: number, decimals = 4): string {
  return v.toFixed(decimals);
}

export function CursorReadout({
  data,
  shape,
  latBounds,
  lonBounds,
  lat,
  lon,
}: CursorReadoutProps) {
  const value = lookupValue(data, shape, latBounds, lonBounds, lat, lon);

  return (
    <div style={containerStyle}>
      <span>Lat: {fmt(lat, 3)}</span>
      <span>Lon: {fmt(lon, 3)}</span>
      <span>
        Val:{' '}
        {value !== null
          ? Math.abs(value) < 1e4 && Math.abs(value) >= 1e-3
            ? fmt(value)
            : value.toExponential(3)
          : 'N/A'}
      </span>
    </div>
  );
}
