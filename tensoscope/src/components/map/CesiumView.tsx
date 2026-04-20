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
} from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';
import './CesiumView.css';
import type { FieldImage } from './FieldOverlay';

interface CesiumViewProps {
  fieldImage: FieldImage | null;
  initialCenter: { lat: number; lon: number };
  onUnmount: (lat: number, lon: number) => void;
}

export function CesiumView({ fieldImage, initialCenter, onUnmount }: CesiumViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Viewer | undefined>(undefined);
  const overlayRef = useRef<Entity | undefined>(undefined);
  const onUnmountRef = useRef(onUnmount);
  const initialCenterRef = useRef(initialCenter);

  // Keep onUnmount ref current so the cleanup closure always has the latest callback
  useEffect(() => {
    onUnmountRef.current = onUnmount;
  }, [onUnmount]);

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
      navigationHelpButton: false,
      animation: false,
      timeline: false,
      fullscreenButton: false,
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

    return () => {
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

    if (overlayRef.current) {
      viewer.entities.remove(overlayRef.current);
      overlayRef.current = undefined;
    }

    if (!fieldImage) return;

    overlayRef.current = viewer.entities.add({
      rectangle: {
        coordinates: Rectangle.fromDegrees(-180, -85, 180, 85),
        material: new ImageMaterialProperty({
          image: fieldImage.dataUrl,
          transparent: true,
          color: new Color(1, 1, 1, 0.7),
        }),
        fill: true,
      },
    });
  }, [fieldImage]);

  return <div ref={containerRef} className="cesium-container" />;
}
