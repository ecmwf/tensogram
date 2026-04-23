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
