/**
 * Web Worker for field regridding and colormapping.
 *
 * Receives source data, coordinates, colour parameters, and grid dimensions.
 * Returns an ImageBitmap (or raw RGBA + dimensions as fallback) ready for
 * the main thread to use as a map overlay.
 */

// ── Types ────────────────────────────────────────────────────────────────

interface RegridRequest {
  id: number;
  srcLat: Float32Array;
  srcLon: Float32Array;
  srcData: Float32Array;
  lut: Uint8Array;        // 256*3 RGB lookup table
  colorMin: number;
  colorMax: number;
  width: number;
  height: number;
  binDeg: number;
  rowLats: Float64Array;  // pre-computed latitudes per row (Mercator or geographic)
  lonMin: number;         // western edge of the rendered window (degrees)
  lonMax: number;         // eastern edge of the rendered window (degrees)
  renderMode: 'heatmap' | 'contours';
  numBands: number;
}

interface RegridResponse {
  id: number;
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
}

// ── Regrid + colormap ────────────────────────────────────────────────────

function regridAndColormap(req: RegridRequest): RegridResponse {
  const { srcLat, srcLon, srcData, lut, colorMin, colorMax, width, height, binDeg, rowLats, lonMin, lonMax, renderMode, numBands } = req;
  const lonSpan = lonMax - lonMin;
  const n = srcData.length;

  const binsLon = Math.ceil(360 / binDeg);
  const binsLat = Math.ceil(180 / binDeg);

  // Normalise longitudes to [-180, 180]
  const normLon = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    normLon[i] = srcLon[i] > 180 ? srcLon[i] - 360 : srcLon[i];
  }

  // Build spatial hash
  const bins: Int32Array[] = new Array(binsLon * binsLat);
  // Use typed array builder for better memory locality
  const binCounts = new Int32Array(binsLon * binsLat);

  // Count pass
  for (let i = 0; i < n; i++) {
    const bi = Math.max(0, Math.min(binsLon - 1, (normLon[i] + 180) / binDeg | 0));
    const bj = Math.max(0, Math.min(binsLat - 1, (srcLat[i] + 90) / binDeg | 0));
    binCounts[bj * binsLon + bi]++;
  }

  // Allocate bins
  for (let i = 0; i < bins.length; i++) {
    bins[i] = new Int32Array(binCounts[i]);
    binCounts[i] = 0; // reset for fill pass
  }

  // Fill pass
  for (let i = 0; i < n; i++) {
    const bi = Math.max(0, Math.min(binsLon - 1, (normLon[i] + 180) / binDeg | 0));
    const bj = Math.max(0, Math.min(binsLat - 1, (srcLat[i] + 90) / binDeg | 0));
    const key = bj * binsLon + bi;
    bins[key][binCounts[key]++] = i;
  }

  // Regrid + colormap in a single pass (avoids intermediate Float32Array)
  const range = colorMax - colorMin;
  const invRange = range === 0 ? 0 : 255 / range;
  const rgba = new Uint8ClampedArray(width * height * 4);

  for (let row = 0; row < height; row++) {
    const targetLat = rowLats[row];
    const bj = Math.max(0, Math.min(binsLat - 1, (targetLat + 90) / binDeg | 0));
    const rowOffset = row * width * 4;

    for (let col = 0; col < width; col++) {
      const targetLon = lonMin + (col + 0.5) * (lonSpan / width);
      const bi = Math.max(0, Math.min(binsLon - 1, (targetLon + 180) / binDeg | 0));

      let bestDist = Infinity;
      let bestIdx = -1;

      for (let dj = -1; dj <= 1; dj++) {
        const nj = bj + dj;
        if (nj < 0 || nj >= binsLat) continue;
        for (let di = -1; di <= 1; di++) {
          const ni = (bi + di + binsLon) % binsLon; // wrap longitude
          const cell = bins[nj * binsLon + ni];
          for (let k = 0; k < cell.length; k++) {
            const idx = cell[k];
            const dLat = srcLat[idx] - targetLat;
            let dLon = normLon[idx] - targetLon;
            if (dLon > 180) dLon -= 360;
            else if (dLon < -180) dLon += 360;
            const dist = dLat * dLat + dLon * dLon;
            if (dist < bestDist) {
              bestDist = dist;
              bestIdx = idx;
            }
          }
        }
      }

      const base = rowOffset + col * 4;
      if (bestIdx >= 0) {
        const v = srcData[bestIdx];
        if (v !== v) continue; // NaN -- leave transparent (already 0)

        let lutIdx: number;
        if (renderMode === 'contours') {
          const t = range === 0 ? 0 : Math.max(0, Math.min(1, (v - colorMin) / range));
          const bandIdx = Math.min(numBands - 1, Math.floor(t * numBands));
          lutIdx = numBands <= 1 ? 0 : Math.round((bandIdx * 255) / (numBands - 1));
        } else {
          lutIdx = Math.max(0, Math.min(255, ((v - colorMin) * invRange) | 0));
        }

        const lutBase = lutIdx * 3;
        rgba[base] = lut[lutBase];
        rgba[base + 1] = lut[lutBase + 1];
        rgba[base + 2] = lut[lutBase + 2];
        rgba[base + 3] = 255;
      }
      // else: leave transparent (already 0)
    }
  }

  return { id: req.id, rgba, width, height };
}

// ── Message handler ──────────────────────────────────────────────────────

self.onmessage = (e: MessageEvent<RegridRequest>) => {
  const result = regridAndColormap(e.data);
  // Transfer the RGBA buffer back to avoid copying
  (self as unknown as Worker).postMessage(result, [result.rgba.buffer]);
};
