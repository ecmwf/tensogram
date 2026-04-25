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
} from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';
import './CesiumView.css';
import type { FieldImage, ViewBounds } from './FieldOverlay';

interface CesiumViewProps {
  fieldImage: FieldImage | null;
  initialCenter: { lat: number; lon: number };
  onUnmount: (lat: number, lon: number) => void;
  onViewChange?: (bounds: ViewBounds | null, width: number, height: number) => void;
}

export function CesiumView({ fieldImage, initialCenter, onUnmount, onViewChange }: CesiumViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Viewer | undefined>(undefined);
  const overlayRef = useRef<Entity | undefined>(undefined);
  const onUnmountRef = useRef(onUnmount);
  const onViewChangeRef = useRef(onViewChange);
  const initialCenterRef = useRef(initialCenter);

  useEffect(() => { onUnmountRef.current = onUnmount; }, [onUnmount]);
  useEffect(() => { onViewChangeRef.current = onViewChange; }, [onViewChange]);

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
      const cart = viewer.camera.positionCartographic;
      onUnmountRef.current(
        CesiumMath.toDegrees(cart.latitude),
        CesiumMath.toDegrees(cart.longitude),
      );
      viewer.destroy();
      viewerRef.current = undefined;
      overlayRef.current = undefined;
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

  return <div ref={containerRef} className="cesium-container" />;
}
