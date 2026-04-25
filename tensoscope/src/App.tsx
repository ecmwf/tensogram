/** Root application layout: sidebar with file browser, main area for the map. */

import './App.css';
import { useState, useCallback, useRef, useEffect } from 'react';
import { FileOpenDialog } from './components/file-browser/FileOpenDialog';
import { FieldSelector } from './components/file-browser/FieldSelector';
import { MetadataPanel } from './components/file-browser/MetadataPanel';
import { WelcomeModal } from './components/file-browser/WelcomeModal';
import { MapView } from './components/map/MapView';
import logo from './assets/tensogram-logo.png';
import { AnimationControls } from './components/animation/AnimationControls';
import { StepSlider } from './components/animation/StepSlider';
import { useAnimationSequence } from './components/animation/useAnimationSequence';
import { useAnimationPlayer } from './components/animation/useAnimationPlayer';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { useAppStore, decideSliceDim } from './store/useAppStore';
import type { ClickPoint } from './components/map/usePointInspection';
import { usePointInspection } from './components/map/usePointInspection';
import { PointInspector } from './components/map/PointInspector';
import { useIsMobile } from './hooks/useIsMobile';
import { snapSheet, sheetHeightCss } from './components/mobile/sheetSnap';
import type { SheetState } from './components/mobile/sheetSnap';
import { getFrameCache } from './tensogram/frameCache';
import type { CachedFrame } from './tensogram/frameCache';

async function decodeFrameForCache(msgIdx: number, objIdx: number): Promise<CachedFrame> {
  const { viewer, fileIndex, selectedLevel } = useAppStore.getState();
  if (!viewer || !fileIndex) throw new Error('viewer not ready');
  const coords = await viewer.fetchCoordinates(msgIdx);
  const varInfo = fileIndex.variables.find((v) => v.msgIndex === msgIdx && v.objIndex === objIdx);
  const originalShape = varInfo?.shape ?? [];
  const coordLength = coords?.lat.length ?? 0;
  const sliceDim = decideSliceDim(originalShape, coordLength);
  let result;
  if (sliceDim >= 0) {
    let sliceIdx = 0;
    if (selectedLevel != null) {
      const anemoi = varInfo?.metadata?.anemoi as Record<string, unknown> | undefined;
      const levels = anemoi?.levels as number[] | undefined;
      if (levels) {
        const idx = levels.indexOf(selectedLevel);
        if (idx >= 0) sliceIdx = idx;
      }
    }
    result = await viewer.decodeFieldSlice(msgIdx, objIdx, sliceDim, sliceIdx);
  } else {
    result = await viewer.decodeField(msgIdx, objIdx);
  }
  return { data: result.data, coordinates: coords, stats: result.stats, shape: originalShape };
}

function App() {
  const {
    viewer,
    fileIndex,
    selectedObject,
    selectedLevel,
    fieldData,
    fieldStats,
    coordinates,
    colorScale,
    setColorScale,
    selectField,
    loading,
    error,
  } = useAppStore();

  // Determine the selected parameter identifier for animation.
  // `mars.param` may be a short string ("2t", "t") or a GRIB integer
  // code (167, 130); useAnimationSequence does a `===` comparison so
  // we keep the value in its original type (no coercion) — string
  // matches string, integer matches integer.
  const selectedParam = selectedObject && fileIndex
    ? ((fileIndex.variables.find(
        (v) => v.msgIndex === selectedObject.msgIdx && v.objIndex === selectedObject.objIdx,
      )?.metadata?.mars as Record<string, unknown> | undefined)?.param as
        | string
        | number
        | undefined)
    : undefined;

  const frames = useAnimationSequence(fileIndex, selectedParam ?? null, selectedLevel);

  const handleFrameChange = (index: number) => {
    const frame = frames[index];
    if (frame) {
      selectField(frame.msgIdx, frame.objIdx);
      getFrameCache()?.prefetch(frames, index, decodeFrameForCache);
    }
  };

  const player = useAnimationPlayer(frames.length, handleFrameChange);

  // Flush the frame cache when level changes — cached frames encode the
  // selected level at decode time, so stale entries must be discarded.
  useEffect(() => {
    getFrameCache()?.flush();
  }, [selectedLevel]);

  useKeyboardShortcuts({
    enabled: frames.length > 1,
    onPlayPause: player.isPlaying ? player.pause : player.play,
    onStepBack: () => player.setFrame(Math.max(0, player.currentFrameIndex - 1)),
    onStepForward: () => player.setFrame(Math.min(frames.length - 1, player.currentFrameIndex + 1)),
  });

  const isMobile = useIsMobile();
  const [sheetState, setSheetState] = useState<SheetState>('collapsed');

  const [sidebarWidth, setSidebarWidth] = useState(380);
  const dragging = useRef(false);
  const startX = useRef(0);
  const startWidth = useRef(0);
  const sheetDragging = useRef(false);
  const sheetStartY = useRef(0);
  const sheetStartState = useRef<SheetState>('collapsed');

  const onDragStart = useCallback((e: React.MouseEvent) => {
    dragging.current = true;
    startX.current = e.clientX;
    startWidth.current = sidebarWidth;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, [sidebarWidth]);

  const onDragMove = useCallback((e: MouseEvent) => {
    if (!dragging.current) return;
    const delta = e.clientX - startX.current;
    const next = Math.max(280, Math.min(600, startWidth.current + delta));
    setSidebarWidth(next);
  }, []);

  const onDragEnd = useCallback(() => {
    if (!dragging.current) return;
    dragging.current = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  }, []);

  useEffect(() => {
    window.addEventListener('mousemove', onDragMove);
    window.addEventListener('mouseup', onDragEnd);
    return () => {
      window.removeEventListener('mousemove', onDragMove);
      window.removeEventListener('mouseup', onDragEnd);
    };
  }, [onDragMove, onDragEnd]);

  const prevFileIndex = useRef<typeof fileIndex>(null);
  useEffect(() => {
    if (isMobile && prevFileIndex.current === null && fileIndex !== null) {
      setSheetState(s => s === 'collapsed' ? 'half' : s);
    }
    prevFileIndex.current = fileIndex;
  }, [fileIndex, isMobile]);

  const onSheetDragStart = useCallback((e: React.MouseEvent) => {
    sheetDragging.current = true;
    sheetStartY.current = e.clientY;
    sheetStartState.current = sheetState;
    document.body.style.userSelect = 'none';
  }, [sheetState]);

  const onSheetTouchStart = useCallback((e: React.TouchEvent) => {
    sheetDragging.current = true;
    sheetStartY.current = e.touches[0].clientY;
    sheetStartState.current = sheetState;
  }, [sheetState]);

  useEffect(() => {
    const onMouseUp = (e: MouseEvent) => {
      if (!sheetDragging.current) return;
      sheetDragging.current = false;
      document.body.style.userSelect = '';
      setSheetState(snapSheet(sheetStartState.current, e.clientY - sheetStartY.current));
    };
    const onTouchEnd = (e: TouchEvent) => {
      if (!sheetDragging.current) return;
      sheetDragging.current = false;
      const touch = e.changedTouches[0];
      setSheetState(snapSheet(sheetStartState.current, touch.clientY - sheetStartY.current));
    };
    const onTouchCancel = () => {
      sheetDragging.current = false;
    };
    window.addEventListener('mouseup', onMouseUp);
    window.addEventListener('touchend', onTouchEnd);
    window.addEventListener('touchcancel', onTouchCancel);
    return () => {
      window.removeEventListener('mouseup', onMouseUp);
      window.removeEventListener('touchend', onTouchEnd);
      window.removeEventListener('touchcancel', onTouchCancel);
    };
  }, []);

  // Find units for the selected variable -- prefer auto-style units, then metadata
  const selectedVar = selectedObject && fileIndex
    ? fileIndex.variables.find(
        (v) => v.msgIndex === selectedObject.msgIdx && v.objIndex === selectedObject.objIdx,
      )
    : undefined;
  // Units: prefer an explicit `units` field on the object; fall back
  // to the param identifier (coerced to string) so the colorbar has
  // *something* to display even for raw GRIB files where the code is
  // the only identifier available.
  const marsParamRaw = (selectedVar?.metadata?.mars as Record<string, unknown>)?.param;
  const marsParam =
    marsParamRaw == null
      ? ''
      : typeof marsParamRaw === 'string'
        ? marsParamRaw
        : String(marsParamRaw);
  const units = (selectedVar?.metadata?.units as string) ?? marsParam;

  const [clickPoint, setClickPoint] = useState<ClickPoint | null>(null);
  const [inspectedScreenPos, setInspectedScreenPos] = useState<{ x: number; y: number } | null>(null);

  const gridSpacing = coordinates
    ? Math.sqrt(360 * 170 / Math.max(1, coordinates.lat.length))
    : null;

  const paramName = selectedVar?.name ?? marsParam ?? '';

  const levelLabel = (() => {
    if (selectedLevel == null) return '';
    const mars = selectedVar?.metadata?.mars as Record<string, unknown> | undefined;
    const levtype = mars?.levtype as string | undefined;
    if (levtype === 'pl') return `${selectedLevel} hPa`;
    if (levtype === 'ml') return `L${selectedLevel}`;
    return `${selectedLevel}`;
  })();

  const inspectionResult = usePointInspection({
    point: clickPoint,
    coordinates,
    fieldData,
    viewer,
    fileIndex,
    frames,
    selectedLevel,
  });

  const selectedPoint = inspectionResult
    ? { lat: inspectionResult.pointLat, lon: inspectionResult.pointLon }
    : null;

  return (
    <div
      className="app-layout"
      style={isMobile ? { '--sheet-height': sheetHeightCss(sheetState) } as React.CSSProperties : undefined}
    >
      {!fileIndex && <WelcomeModal />}
      <aside className="sidebar" style={isMobile ? {} : { width: sidebarWidth }}>
        <div
          className="sheet-drag-handle"
          onMouseDown={onSheetDragStart}
          onTouchStart={onSheetTouchStart}
        />
        <div className="sidebar-brand">
          <img src={logo} alt="" className="sidebar-brand-logo" />
          <span className="sidebar-brand-name">Tensoscope</span>
          <span className="sidebar-brand-version" title={`@ecmwf.int/tensogram ${__TG_VERSION__}`}>
            v{__APP_VERSION__}
          </span>
        </div>
        <FileOpenDialog />
        <FieldSelector />
        <MetadataPanel />
      </aside>
      <div className="sidebar-resize-handle" onMouseDown={onDragStart} />
      <main className="map-area" style={{ position: 'relative' }}>
        {loading && (
          <div className="map-loading-overlay">
            <div className="map-loading-spinner" />
          </div>
        )}
        {error && !loading && (
          <div className="map-error-overlay">
            <span className="map-error-text">{error}</span>
          </div>
        )}
        <MapView
          colorScaleInitialMinimised={isMobile}
          data={fieldData}
          lat={coordinates?.lat ?? null}
          lon={coordinates?.lon ?? null}
          colorMin={colorScale.min}
          colorMax={colorScale.max}
          palette={colorScale.palette}
          units={units}
          paletteReversed={colorScale.paletteReversed}
          customStops={colorScale.customStops}
          dataMin={fieldStats?.min ?? 0}
          dataMax={fieldStats?.max ?? 1}
          onColorMinChange={(v) => setColorScale({ min: v })}
          onColorMaxChange={(v) => setColorScale({ max: v })}
          onPaletteChange={(v) => setColorScale({ palette: v })}
          onPaletteReversedChange={(v) => setColorScale({ paletteReversed: v })}
          onCustomStopsChange={(stops) => setColorScale({ customStops: stops })}
          nativeUnits={colorScale.nativeUnits}
          displayUnit={colorScale.displayUnit}
          onDisplayUnitChange={(u) => setColorScale({ displayUnit: u })}
          onMapClick={setClickPoint}
          selectedPoint={selectedPoint}
          selectedPointGridSpacing={gridSpacing}
          onSelectedPointScreen={(x, y) => setInspectedScreenPos({ x, y })}
          onSelectedPointOutOfView={() => { setClickPoint(null); setInspectedScreenPos(null); }}
        />
        {frames.length > 1 && (
          <div className="animation-bar">
            <StepSlider
              frames={frames}
              currentIndex={player.currentFrameIndex}
              onChange={player.setFrame}
            />
            <AnimationControls
              frames={frames}
              currentFrameIndex={player.currentFrameIndex}
              isPlaying={player.isPlaying}
              speed={player.speed}
              onPlay={player.play}
              onPause={player.pause}
              onStop={player.stop}
              onFrameChange={player.setFrame}
              onSpeedChange={player.setSpeed}
            />
          </div>
        )}
        {clickPoint !== null && inspectionResult !== null && inspectedScreenPos !== null && (
          <PointInspector
            result={inspectionResult}
            screenX={inspectedScreenPos.x}
            screenY={inspectedScreenPos.y}
            paramName={paramName}
            levelLabel={levelLabel}
            nativeUnits={colorScale.nativeUnits}
            displayUnit={colorScale.displayUnit}
            onDisplayUnitChange={(u) => setColorScale({ displayUnit: u })}
            onClose={() => { setClickPoint(null); setInspectedScreenPos(null); }}
          />
        )}
      </main>
    </div>
  );
}

export default App;
