import { describe, it, expect, vi, beforeEach } from 'vitest';

// ── Shared mock fns (must be hoisted alongside vi.mock) ──────────────────────

const mocks = vi.hoisted(() => ({
  buildIndex: vi.fn(),
  fetchCoordinates: vi.fn(),
  decodeField: vi.fn(),
  decodeFieldSlice: vi.fn(),
  close: vi.fn(),
  fromFile: vi.fn(),
  fromUrl: vi.fn(),
}));

vi.mock('../../tensogram', () => {
  const makeViewer = () => ({
    buildIndex: mocks.buildIndex,
    fetchCoordinates: mocks.fetchCoordinates,
    decodeField: mocks.decodeField,
    decodeFieldSlice: mocks.decodeFieldSlice,
    close: mocks.close,
  });
  mocks.fromFile.mockImplementation(async () => makeViewer());
  mocks.fromUrl.mockImplementation(async () => makeViewer());
  return {
    Tensoscope: {
      fromFile: mocks.fromFile,
      fromUrl: mocks.fromUrl,
    },
  };
});

// ── Store import (after mock registration) ───────────────────────────────────

import { useAppStore } from '../useAppStore';

// ── Fixtures ─────────────────────────────────────────────────────────────────

const DEFAULT_SCALE = {
  min: 0, max: 1, palette: 'viridis', logScale: false,
  paletteReversed: false,
  customStops: [{ pos: 0, color: '#000000' }, { pos: 1, color: '#ffffff' }],
  nativeUnits: '', displayUnit: '',
};

const FILE_INDEX = {
  source: 'test.tgm',
  messageCount: 2,
  variables: [
    { msgIndex: 0, objIndex: 0, name: 'xtest', shape: [4], dtype: 'f32', encoding: '', compression: '', metadata: {} },
    { msgIndex: 1, objIndex: 0, name: 'ytest', shape: [4], dtype: 'f32', encoding: '', compression: '', metadata: {} },
  ],
  coordinates: [],
};

// 4-point coordinates to match shape [4] without triggering multi-dim slice logic
const COORDS = {
  lat: new Float32Array([10, 20, 30, 40]),
  lon: new Float32Array([0, 10, 20, 30]),
};

const DECODED_FIELD = {
  data: new Float32Array([1, 2, 3, 4]),
  shape: [4],
  stats: { min: 1, max: 4, mean: 2.5, std: 1.12 },
};

// ── Helpers ───────────────────────────────────────────────────────────────────

function resetStore() {
  useAppStore.setState({
    viewer: null,
    fileIndex: null,
    selectedObject: null,
    selectedLevel: null,
    fieldData: null,
    fieldShape: [],
    fieldStats: null,
    coordinates: null,
    colorScale: { ...DEFAULT_SCALE },
    loading: false,
    error: null,
  });
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('useAppStore — openLocalFile', () => {
  beforeEach(() => {
    resetStore();
    vi.clearAllMocks();
    mocks.buildIndex.mockResolvedValue(FILE_INDEX);
    mocks.fetchCoordinates.mockResolvedValue(COORDS);
    mocks.decodeField.mockResolvedValue(DECODED_FIELD);
    mocks.fromFile.mockImplementation(async () => ({
      buildIndex: mocks.buildIndex,
      fetchCoordinates: mocks.fetchCoordinates,
      decodeField: mocks.decodeField,
      decodeFieldSlice: mocks.decodeFieldSlice,
      close: mocks.close,
    }));
  });

  it('auto-selects the first variable so the map is never blank', async () => {
    await useAppStore.getState().openLocalFile(new File([], 'test.tgm'));

    const { selectedObject, fieldData } = useAppStore.getState();
    expect(selectedObject).toEqual({ msgIdx: 0, objIdx: 0 });
    expect(fieldData).not.toBeNull();
  });

  it('selects the first variable by its msgIndex and objIndex', async () => {
    const shifted = {
      ...FILE_INDEX,
      variables: [
        { msgIndex: 3, objIndex: 1, name: 'xtest', shape: [4], dtype: 'f32', encoding: '', compression: '', metadata: {} },
      ],
    };
    mocks.buildIndex.mockResolvedValue(shifted);

    await useAppStore.getState().openLocalFile(new File([], 'test.tgm'));

    expect(useAppStore.getState().selectedObject).toEqual({ msgIdx: 3, objIdx: 1 });
  });

  it('preserves the user palette when the variable has no auto-style', async () => {
    useAppStore.setState({ colorScale: { ...DEFAULT_SCALE, palette: 'plasma' } });

    await useAppStore.getState().openLocalFile(new File([], 'test.tgm'));

    expect(useAppStore.getState().colorScale.palette).toBe('plasma');
  });

  it('preserves paletteReversed across file opens', async () => {
    useAppStore.setState({ colorScale: { ...DEFAULT_SCALE, paletteReversed: true } });

    await useAppStore.getState().openLocalFile(new File([], 'test.tgm'));

    expect(useAppStore.getState().colorScale.paletteReversed).toBe(true);
  });

  it('updates colorScale min/max from the decoded field stats', async () => {
    await useAppStore.getState().openLocalFile(new File([], 'test.tgm'));

    const { colorScale } = useAppStore.getState();
    expect(colorScale.min).toBe(DECODED_FIELD.stats.min);
    expect(colorScale.max).toBe(DECODED_FIELD.stats.max);
  });

  it('clears loading and error on success', async () => {
    await useAppStore.getState().openLocalFile(new File([], 'test.tgm'));

    const { loading, error } = useAppStore.getState();
    expect(loading).toBe(false);
    expect(error).toBeNull();
  });

  it('sets error and clears loading when the file cannot be opened', async () => {
    mocks.fromFile.mockRejectedValue(new Error('corrupt file'));

    await useAppStore.getState().openLocalFile(new File([], 'bad.tgm'));

    const { loading, error } = useAppStore.getState();
    expect(loading).toBe(false);
    expect(error).toContain('corrupt file');
  });
});

describe('useAppStore — openUrl', () => {
  beforeEach(() => {
    resetStore();
    vi.clearAllMocks();
    mocks.buildIndex.mockResolvedValue(FILE_INDEX);
    mocks.fetchCoordinates.mockResolvedValue(COORDS);
    mocks.decodeField.mockResolvedValue(DECODED_FIELD);
    mocks.fromUrl.mockImplementation(async () => ({
      buildIndex: mocks.buildIndex,
      fetchCoordinates: mocks.fetchCoordinates,
      decodeField: mocks.decodeField,
      decodeFieldSlice: mocks.decodeFieldSlice,
      close: mocks.close,
    }));
  });

  it('auto-selects the first variable after loading from a URL', async () => {
    await useAppStore.getState().openUrl('https://example.com/data.tgm');

    const { selectedObject, fieldData } = useAppStore.getState();
    expect(selectedObject).toEqual({ msgIdx: 0, objIdx: 0 });
    expect(fieldData).not.toBeNull();
  });

  it('preserves the user palette when opening from a URL', async () => {
    useAppStore.setState({ colorScale: { ...DEFAULT_SCALE, palette: 'magma' } });

    await useAppStore.getState().openUrl('https://example.com/data.tgm');

    expect(useAppStore.getState().colorScale.palette).toBe('magma');
  });

  it('sets error and clears loading when the URL fetch fails', async () => {
    mocks.fromUrl.mockRejectedValue(new Error('network error'));

    await useAppStore.getState().openUrl('https://bad.example.com/data.tgm');

    expect(useAppStore.getState().loading).toBe(false);
    expect(useAppStore.getState().error).toContain('network error');
  });
});

describe('useAppStore — initViewer does not reset colorScale', () => {
  beforeEach(() => {
    resetStore();
    vi.clearAllMocks();
    mocks.buildIndex.mockResolvedValue(FILE_INDEX);
    mocks.fetchCoordinates.mockResolvedValue(COORDS);
    mocks.decodeField.mockResolvedValue(DECODED_FIELD);
    mocks.fromFile.mockImplementation(async () => ({
      buildIndex: mocks.buildIndex,
      fetchCoordinates: mocks.fetchCoordinates,
      decodeField: mocks.decodeField,
      decodeFieldSlice: mocks.decodeFieldSlice,
      close: mocks.close,
    }));
  });

  it('opening a second file keeps the custom palette from the first session', async () => {
    // Simulate a user who has opened a file, customised their palette, then opens another.
    useAppStore.setState({
      colorScale: { ...DEFAULT_SCALE, palette: 'coolwarm', paletteReversed: true },
    });

    await useAppStore.getState().openLocalFile(new File([], 'second.tgm'));

    const { colorScale } = useAppStore.getState();
    expect(colorScale.palette).toBe('coolwarm');
    expect(colorScale.paletteReversed).toBe(true);
  });
});

// Heterogeneous multi-message files (different grids per message) must
// have their coordinates fetched per-message, not once at file open.
// Regression guard for Copilot's comment on PR #85: the per-msgIdx
// cache at the Tensoscope wrapper layer is only useful if the store
// also fetches per-message rather than reusing msg-0 coords forever.
describe('useAppStore — selectField fetches coords per message', () => {
  const COORDS_MSG0 = {
    lat: new Float32Array([10, 20, 30, 40]),
    lon: new Float32Array([0, 10, 20, 30]),
  };
  const COORDS_MSG1 = {
    lat: new Float32Array([50, 60]),
    lon: new Float32Array([100, 110]),
  };

  beforeEach(() => {
    resetStore();
    vi.clearAllMocks();
    mocks.buildIndex.mockResolvedValue({
      source: 'multi.tgm',
      messageCount: 2,
      variables: [
        { msgIndex: 0, objIndex: 0, name: 'a', shape: [4], dtype: 'f32', encoding: '', compression: '', metadata: {} },
        { msgIndex: 1, objIndex: 0, name: 'b', shape: [4], dtype: 'f32', encoding: '', compression: '', metadata: {} },
      ],
      coordinates: [],
    });
    mocks.fetchCoordinates.mockImplementation(async (msgIdx: number) =>
      msgIdx === 0 ? COORDS_MSG0 : COORDS_MSG1,
    );
    mocks.decodeField.mockResolvedValue(DECODED_FIELD);
    // Both decode paths must succeed so selectField reaches the
    // atomic final set; msg-1's shorter coords force the slice path.
    mocks.decodeFieldSlice.mockResolvedValue({
      data: new Float32Array([5, 6]),
      shape: [2],
      stats: { min: 5, max: 6, mean: 5.5, std: 0.5 },
    });
    mocks.fromFile.mockImplementation(async () => ({
      buildIndex: mocks.buildIndex,
      fetchCoordinates: mocks.fetchCoordinates,
      decodeField: mocks.decodeField,
      decodeFieldSlice: mocks.decodeFieldSlice,
      close: mocks.close,
    }));
  });

  it('calls fetchCoordinates with the selected msgIdx, not always 0', async () => {
    await useAppStore.getState().openLocalFile(new File([], 'multi.tgm'));
    mocks.fetchCoordinates.mockClear();

    await useAppStore.getState().selectField(1, 0);

    expect(mocks.fetchCoordinates).toHaveBeenCalledWith(1);
    expect(mocks.fetchCoordinates).not.toHaveBeenCalledWith(0);
  });

  it('stores the selected message coords, not msg-0 stale coords', async () => {
    await useAppStore.getState().openLocalFile(new File([], 'multi.tgm'));
    // Auto-select took msg 0; coords should reflect msg 0.
    expect(useAppStore.getState().coordinates?.lat).toEqual(COORDS_MSG0.lat);

    await useAppStore.getState().selectField(1, 0);

    expect(useAppStore.getState().coordinates?.lat).toEqual(COORDS_MSG1.lat);
    expect(useAppStore.getState().coordinates?.lon).toEqual(COORDS_MSG1.lon);
  });

  it('initViewer does not eagerly fetch coords for msg 0', async () => {
    // Shifted index: first variable lives on msg 3, not msg 0.  The
    // old code pre-fetched msg 0 unconditionally, wasting work and
    // seeding the store with wrong coords; after the fix selectField
    // is the only caller.
    mocks.buildIndex.mockResolvedValue({
      source: 'shifted.tgm',
      messageCount: 5,
      variables: [
        { msgIndex: 3, objIndex: 1, name: 'x', shape: [4], dtype: 'f32', encoding: '', compression: '', metadata: {} },
      ],
      coordinates: [],
    });

    await useAppStore.getState().openLocalFile(new File([], 'shifted.tgm'));

    const calls = mocks.fetchCoordinates.mock.calls.map((c) => c[0]);
    expect(calls).not.toContain(0);
    expect(calls).toContain(3);
  });

  it('decideSliceDim uses the selected message coord length', async () => {
    // msg 0 ships 4-point coords, msg 1 ships 2-point coords.  Both
    // variables have shape [4].  On msg 1, total (4) is a multiple
    // of coordLength (2) → decideSliceDim picks dim 0 → decodeFieldSlice
    // is called; on msg 0 with coord length 4, total === coord length
    // → no slice → decodeField is called.
    await useAppStore.getState().openLocalFile(new File([], 'multi.tgm'));
    expect(mocks.decodeField).toHaveBeenCalledTimes(1);
    expect(mocks.decodeFieldSlice).not.toHaveBeenCalled();

    await useAppStore.getState().selectField(1, 0);

    expect(mocks.decodeFieldSlice).toHaveBeenCalledTimes(1);
    expect(mocks.decodeFieldSlice).toHaveBeenLastCalledWith(1, 0, 0, 0);
  });
});
