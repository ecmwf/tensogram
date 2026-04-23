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
import { useAppStore } from './store/useAppStore';

function App() {
  const {
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
    }
  };

  const player = useAnimationPlayer(frames.length, handleFrameChange);

  useKeyboardShortcuts({
    enabled: frames.length > 1,
    onPlayPause: player.isPlaying ? player.pause : player.play,
    onStepBack: () => player.setFrame(Math.max(0, player.currentFrameIndex - 1)),
    onStepForward: () => player.setFrame(Math.min(frames.length - 1, player.currentFrameIndex + 1)),
  });

  const [sidebarWidth, setSidebarWidth] = useState(380);
  const dragging = useRef(false);
  const startX = useRef(0);
  const startWidth = useRef(0);

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

  return (
    <div className="app-layout">
      {!fileIndex && <WelcomeModal />}
      <aside className="sidebar" style={{ width: sidebarWidth }}>
        <div className="sidebar-brand">
          <img src={logo} alt="" className="sidebar-brand-logo" />
          <span className="sidebar-brand-name">Tensoscope</span>
        </div>
        <FileOpenDialog />
        <FieldSelector />
        <MetadataPanel />
      </aside>
      <div className="sidebar-resize-handle" onMouseDown={onDragStart} />
      <main className="map-area">
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
      </main>
    </div>
  );
}

export default App;
