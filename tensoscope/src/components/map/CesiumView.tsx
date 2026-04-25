import { useEffect, useRef } from 'react';
import {
  Viewer,
  ImageryLayer,
  UrlTemplateImageryProvider,
  Rectangle,
  ImageMaterialProperty,
  Color,
  Cartesian3,
  Math as CesiumMath,
  Credit,
  Entity,
  ConstantProperty,
  ConstantPositionProperty,
  ScreenSpaceEventHandler,
  ScreenSpaceEventType,
  Cartographic,
  SceneTransforms,
} from 'cesium';

function markerPixelSize(gridSpacing: number | null | undefined): number {
  const s = gridSpacing ?? 2;
  return Math.min(Math.max(s * 4, 8), 28);
}
import 'cesium/Build/Cesium/Widgets/widgets.css';
import './CesiumView.css';
import type { FieldImage, ViewBounds } from './FieldOverlay';
import type { ClickPoint } from './usePointInspection';

interface CesiumViewProps {
  fieldImage: FieldImage | null;
  initialCenter: { lat: number; lon: number };
  onUnmount: (lat: number, lon: number) => void;
  onViewChange?: (bounds: ViewBounds | null, width: number, height: number) => void;
  onMapClick?: (point: ClickPoint) => void;
  selectedPoint?: { lat: number; lon: number } | null;
  selectedPointGridSpacing?: number | null;
  onSelectedPointScreen?: (x: number, y: number) => void;
  onSelectedPointOutOfView?: () => void;
}

export function CesiumView({ fieldImage, initialCenter, onUnmount, onViewChange, onMapClick, selectedPoint, selectedPointGridSpacing, onSelectedPointScreen, onSelectedPointOutOfView }: CesiumViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Viewer | undefined>(undefined);
  const overlayRef = useRef<Entity | undefined>(undefined);
  const markerRef = useRef<Entity | undefined>(undefined);
  const selectedPointRef = useRef(selectedPoint);
  const onUnmountRef = useRef(onUnmount);
  const onViewChangeRef = useRef(onViewChange);
  const initialCenterRef = useRef(initialCenter);
  const onMapClickRef = useRef(onMapClick);
  const onSelectedPointScreenRef = useRef(onSelectedPointScreen);
  const onSelectedPointOutOfViewRef = useRef(onSelectedPointOutOfView);

  useEffect(() => { onUnmountRef.current = onUnmount; }, [onUnmount]);
  useEffect(() => { onViewChangeRef.current = onViewChange; }, [onViewChange]);
  useEffect(() => { initialCenterRef.current = initialCenter; }, [initialCenter]);
  useEffect(() => { onMapClickRef.current = onMapClick; }, [onMapClick]);
  useEffect(() => { onSelectedPointScreenRef.current = onSelectedPointScreen; }, [onSelectedPointScreen]);
  useEffect(() => { onSelectedPointOutOfViewRef.current = onSelectedPointOutOfView; }, [onSelectedPointOutOfView]);
  useEffect(() => { selectedPointRef.current = selectedPoint; }, [selectedPoint]);

  // Mount Cesium viewer once
  useEffect(() => {
    if (!containerRef.current) return;

    const osm = new UrlTemplateImageryProvider({
      url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
      subdomains: ['a', 'b', 'c', 'd'],
      credit: new Credit('© OpenStreetMap contributors © CARTO'),
      maximumLevel: 19,
    });

    const viewer = new Viewer(containerRef.current, {
      baseLayer: ImageryLayer.fromProviderAsync(Promise.resolve(osm)),
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      projectionPicker: false,
      navigationHelpButton: false,
      animation: false,
      timeline: false,
      fullscreenButton: false,
      vrButton: false,
      infoBox: false,
      selectionIndicator: false,
      creditContainer: document.createElement('div'),
    });

    viewer.scene.globe.enableLighting = false;
    viewer.scene.screenSpaceCameraController.enableCollisionDetection = false;

    viewer.camera.flyTo({
      destination: Cartesian3.fromDegrees(
        initialCenterRef.current.lon,
        initialCenterRef.current.lat,
        20000000,
      ),
      duration: 0,
    });

    viewerRef.current = viewer;

    // Handle selected point marker position on globe
    const updateSelectedPointScreen = () => {
      const pt = selectedPointRef.current;
      if (!pt || !onSelectedPointScreenRef.current) return;
      const worldPos = Cartesian3.fromDegrees(pt.lon, pt.lat);
      const windowPos = SceneTransforms.worldToWindowCoordinates(viewer.scene, worldPos);
      if (windowPos) {
        const canvas = viewer.canvas;
        const x = windowPos.x;
        const y = windowPos.y;
        onSelectedPointScreenRef.current(x, y);
        const margin = 80;
        if (x < -margin || x > canvas.clientWidth + margin || y < -margin || y > canvas.clientHeight + margin) {
          onSelectedPointOutOfViewRef.current?.();
        }
      }
    };

    // Update marker position on camera changes
    const cameraChangedHandler = () => {
      updateSelectedPointScreen();
    };
    viewer.camera.changed.addEventListener(cameraChangedHandler);
    // Emit once on mount
    updateSelectedPointScreen();

    const clickHandler = new ScreenSpaceEventHandler(viewer.scene.canvas);
    clickHandler.setInputAction((event: { position: { x: number; y: number } }) => {
      const pt = onMapClickRef.current;
      if (!pt) return;
      const cartesian = viewer.camera.pickEllipsoid(
        event.position,
        viewer.scene.globe.ellipsoid,
      );
      if (!cartesian) return;
      const carto = Cartographic.fromCartesian(cartesian);
      pt({
        lat: CesiumMath.toDegrees(carto.latitude),
        lon: CesiumMath.toDegrees(carto.longitude),
        screenX: event.position.x,
        screenY: event.position.y,
      });
    }, ScreenSpaceEventType.LEFT_CLICK);

    // Debounced camera-change listener: computes the visible rectangle and
    // passes it back so the caller can request a viewport-resolution render.
    let boundsTimer: ReturnType<typeof setTimeout> | null = null;
    const emitBounds = () => {
      if (boundsTimer !== null) clearTimeout(boundsTimer);
      boundsTimer = setTimeout(() => {
        const rect = viewer.camera.computeViewRectangle();
        const w = containerRef.current?.clientWidth ?? 1024;
        const h = containerRef.current?.clientHeight ?? 512;
        if (rect) {
          onViewChangeRef.current?.({
            west: CesiumMath.toDegrees(rect.west),
            east: CesiumMath.toDegrees(rect.east),
            south: CesiumMath.toDegrees(rect.south),
            north: CesiumMath.toDegrees(rect.north),
          }, w, h);
        } else {
          // Camera looking into space or very zoomed out -- use full extent
          onViewChangeRef.current?.(null, w, h);
        }
      }, 200);
    };

    viewer.camera.changed.addEventListener(emitBounds);
    // Emit once on mount so we have initial bounds
    emitBounds();

    return () => {
      if (boundsTimer !== null) clearTimeout(boundsTimer);
      viewer.camera.changed.removeEventListener(emitBounds);
      viewer.camera.changed.removeEventListener(cameraChangedHandler);
      clickHandler.destroy();
      const cart = viewer.camera.positionCartographic;
      onUnmountRef.current(
        CesiumMath.toDegrees(cart.latitude),
        CesiumMath.toDegrees(cart.longitude),
      );
      viewer.destroy();
      viewerRef.current = undefined;
      overlayRef.current = undefined;
      markerRef.current = undefined;
    };
  }, []); // mount-once: intentional empty deps

  // Update field overlay when fieldImage changes
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    if (!fieldImage) {
      if (overlayRef.current) {
        overlayRef.current.show = false;
      }
      return;
    }

    const [lonMin, latMax] = fieldImage.coordinates[0];
    const [lonMax, latMin] = fieldImage.coordinates[2];
    const rect = Rectangle.fromDegrees(lonMin, latMin, lonMax, latMax);

    if (overlayRef.current) {
      // Update image and bounds in-place to avoid a blank frame between renders.
      overlayRef.current.show = true;
      const rectProp = overlayRef.current.rectangle!;
      rectProp.coordinates = new ConstantProperty(rect);
      const mat = rectProp.material as ImageMaterialProperty;
      mat.image = new ConstantProperty(fieldImage.dataUrl);
    } else {
      overlayRef.current = viewer.entities.add({
        rectangle: {
          coordinates: rect,
          material: new ImageMaterialProperty({
            image: fieldImage.dataUrl,
            transparent: true,
            color: new Color(1, 1, 1, 0.7),
          }),
          fill: true,
        },
      });
    }
  }, [fieldImage]);

  // Update selected-point marker when selectedPoint or grid spacing changes
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    if (!selectedPoint) {
      if (markerRef.current) markerRef.current.show = false;
      return;
    }

    const position = Cartesian3.fromDegrees(selectedPoint.lon, selectedPoint.lat);
    const pixelSize = markerPixelSize(selectedPointGridSpacing);

    if (markerRef.current) {
      markerRef.current.show = true;
      markerRef.current.position = new ConstantPositionProperty(position);
      markerRef.current.point!.pixelSize = new ConstantProperty(pixelSize);
    } else {
      markerRef.current = viewer.entities.add({
        position,
        point: {
          pixelSize,
          color: Color.TRANSPARENT,
          outlineColor: Color.WHITE,
          outlineWidth: 2,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
    }
  }, [selectedPoint, selectedPointGridSpacing]);

  return <div ref={containerRef} className="cesium-container" />;
}
