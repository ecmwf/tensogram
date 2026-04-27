import { useEffect, useRef } from 'react';
import {
  Viewer,
  ImageryLayer,
  UrlTemplateImageryProvider,
  SingleTileImageryProvider,
  GeoJsonDataSource,
  Rectangle,
  Color,
  Cartesian2,
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

// CARTO publishes the dark-matter tiles split into separate sub-products:
// `dark_nolabels` keeps the basemap tint and (some) borders; `dark_only_labels`
// renders just place names on transparent tiles. We use nolabels at the bottom
// and stack labels on top of the field so the user's data is never obscured.
const BASEMAP_URL = 'https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}.png';
const LABELS_URL = 'https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}.png';
const BASEMAP_SUBDOMAINS = ['a', 'b', 'c', 'd'];
const BASEMAP_CREDIT = '© OpenStreetMap contributors © CARTO';

// Natural Earth vector data via JSDelivr's GitHub CDN. 50 m is the middle
// resolution — crisper than 110 m (which looked blocky on close zoom) but
// without the rendering cost of 10 m at zoom-out (~500 KB coastlines,
// ~150 KB borders, browser-cached after the first load).
const COASTLINES_URL = 'https://cdn.jsdelivr.net/gh/nvkelso/natural-earth-vector/geojson/ne_50m_coastline.geojson';
const BORDERS_URL = 'https://cdn.jsdelivr.net/gh/nvkelso/natural-earth-vector/geojson/ne_50m_admin_0_boundary_lines_land.geojson';

interface CesiumViewProps {
  fieldImage: FieldImage | null;
  backFieldImage?: FieldImage | null;
  showLabels?: boolean;
  showLines?: boolean;
  initialCenter: { lat: number; lon: number };
  onUnmount: (lat: number, lon: number) => void;
  onViewChange?: (bounds: ViewBounds | null, width: number, height: number) => void;
  onMapClick?: (point: ClickPoint) => void;
  selectedPoint?: { lat: number; lon: number } | null;
  selectedPointGridSpacing?: number | null;
  onSelectedPointScreen?: (x: number, y: number) => void;
  onSelectedPointOutOfView?: () => void;
}

export function CesiumView({
  fieldImage, backFieldImage,
  showLabels = false, showLines = true,
  initialCenter, onUnmount, onViewChange, onMapClick,
  selectedPoint, selectedPointGridSpacing,
  onSelectedPointScreen, onSelectedPointOutOfView,
}: CesiumViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Viewer | undefined>(undefined);
  const backLayerRef = useRef<ImageryLayer | null>(null);
  const frontLayerRef = useRef<ImageryLayer | null>(null);
  const labelsLayerRef = useRef<ImageryLayer | null>(null);
  const bordersDsRef = useRef<GeoJsonDataSource | null>(null);
  const coastlinesDsRef = useRef<GeoJsonDataSource | null>(null);
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

  // Latest toggle values mirrored into refs so async data-source loads can
  // honour the current value when they finish (otherwise a load that
  // completes after a toggle ignores the toggle).
  const showLabelsRef = useRef(showLabels);
  const showLinesRef = useRef(showLines);
  useEffect(() => { showLabelsRef.current = showLabels; }, [showLabels]);
  useEffect(() => { showLinesRef.current = showLines; }, [showLines]);

  // Mount Cesium viewer once
  useEffect(() => {
    if (!containerRef.current) return;

    const basemap = new UrlTemplateImageryProvider({
      url: BASEMAP_URL,
      subdomains: BASEMAP_SUBDOMAINS,
      credit: new Credit(BASEMAP_CREDIT),
      maximumLevel: 19,
    });

    const viewer = new Viewer(containerRef.current, {
      baseLayer: ImageryLayer.fromProviderAsync(Promise.resolve(basemap)),
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
    // Cesium applies a hazy "ground atmosphere" pass to the globe surface
    // and a sky-atmosphere halo around it. Both wash the field with a
    // white-blue sheen at zoom-out — disable them so the field reads at
    // its true colour.
    viewer.scene.globe.showGroundAtmosphere = false;
    if (viewer.scene.skyAtmosphere) viewer.scene.skyAtmosphere.show = false;
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

    // Labels imagery layer — renders on top of the field. Toggled by the
    // showLabels prop.
    const labelsProvider = new UrlTemplateImageryProvider({
      url: LABELS_URL,
      subdomains: BASEMAP_SUBDOMAINS,
      credit: new Credit(BASEMAP_CREDIT),
      maximumLevel: 19,
    });
    labelsLayerRef.current = viewer.imageryLayers.addImageryProvider(labelsProvider);
    labelsLayerRef.current.show = showLabelsRef.current;

    // Borders and coastlines as ground-clamped GeoJSON polylines. They sit
    // in `dataSources`, which Cesium renders above all imagery layers.
    GeoJsonDataSource.load(COASTLINES_URL, {
      stroke: Color.WHITE.withAlpha(0.55),
      strokeWidth: 1,
      clampToGround: true,
    }).then((ds) => {
      coastlinesDsRef.current = ds;
      ds.show = showLinesRef.current;
      viewer.dataSources.add(ds);
      viewer.scene.requestRender();
    }).catch(() => { /* network error — degrade silently */ });

    GeoJsonDataSource.load(BORDERS_URL, {
      stroke: Color.WHITE.withAlpha(0.55),
      strokeWidth: 1,
      clampToGround: true,
    }).then((ds) => {
      bordersDsRef.current = ds;
      ds.show = showLinesRef.current;
      viewer.dataSources.add(ds);
      viewer.scene.requestRender();
    }).catch(() => { /* network error — degrade silently */ });

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

    const cameraChangedHandler = () => {
      updateSelectedPointScreen();
    };
    viewer.camera.changed.addEventListener(cameraChangedHandler);
    updateSelectedPointScreen();

    const clickHandler = new ScreenSpaceEventHandler(viewer.scene.canvas);
    clickHandler.setInputAction((event: { position: { x: number; y: number } }) => {
      const pt = onMapClickRef.current;
      if (!pt) return;
      const cartesian = viewer.camera.pickEllipsoid(
        new Cartesian2(event.position.x, event.position.y),
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
          onViewChangeRef.current?.(null, w, h);
        }
      }, 200);
    };

    viewer.camera.changed.addEventListener(emitBounds);
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
      backLayerRef.current = null;
      frontLayerRef.current = null;
      labelsLayerRef.current = null;
      bordersDsRef.current = null;
      coastlinesDsRef.current = null;
      markerRef.current = undefined;
    };
  }, []); // mount-once: intentional empty deps

  // Swap a field-imagery layer in place. Cesium has no `setProvider`, so we
  // remove the old layer and add a fresh one in the same z-order slot
  // (between basemap and labels). `raiseToTop` on the labels layer afterwards
  // keeps the labels above the field even though `addImageryProvider`
  // appends to the end of the collection.
  const swapFieldImagery = (
    layerRef: React.MutableRefObject<ImageryLayer | null>,
    image: FieldImage | null,
  ) => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    if (layerRef.current) {
      viewer.imageryLayers.remove(layerRef.current, true);
      layerRef.current = null;
    }
    if (!image) return;
    const [lonMin, latMax] = image.coordinates[0];
    const [lonMax, latMin] = image.coordinates[2];
    const rect = Rectangle.fromDegrees(lonMin, latMin, lonMax, latMax);
    const provider = new SingleTileImageryProvider({
      url: image.dataUrl,
      rectangle: rect,
      tileWidth: image.width,
      tileHeight: image.height,
    });
    layerRef.current = viewer.imageryLayers.addImageryProvider(provider);
    if (labelsLayerRef.current) {
      viewer.imageryLayers.raiseToTop(labelsLayerRef.current);
    }
    viewer.scene.requestRender();
  };

  // Back layer — full-globe low-resolution render.
  useEffect(() => {
    swapFieldImagery(backLayerRef, backFieldImage ?? null);
    // swapFieldImagery is stable; no deps needed.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backFieldImage]);

  // Front layer — viewport at screen resolution.
  useEffect(() => {
    swapFieldImagery(frontLayerRef, fieldImage);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fieldImage]);

  // Toggle labels visibility.
  useEffect(() => {
    if (labelsLayerRef.current) {
      labelsLayerRef.current.show = showLabels;
      viewerRef.current?.scene.requestRender();
    }
  }, [showLabels]);

  // Toggle borders + coastlines visibility. Sets `show` on each data
  // source; Cesium hides the contained entities and ground primitives
  // accordingly. Falls through cleanly if the async load hasn't completed
  // yet — the load callback applies showLinesRef.current at that point.
  useEffect(() => {
    if (bordersDsRef.current) bordersDsRef.current.show = showLines;
    if (coastlinesDsRef.current) coastlinesDsRef.current.show = showLines;
    viewerRef.current?.scene.requestRender();
  }, [showLines]);

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
