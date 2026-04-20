# Tensoscope

Interactive web viewer for [Tensogram](https://github.com/ecmwf/tensogram) `.tgm` files.

Decodes data directly in the browser via the `@ecmwf/tensogram` WebAssembly package and
renders geospatial fields on an interactive map using deck.gl + MapLibre GL JS.

## Features

- Drag-and-drop or URL-based `.tgm` file loading
- Field browser with metadata panel
- Interactive map with deck.gl bitmap layers
- Colour scale controls and colour bar
- Multi-step animation with play/pause
- Multiple map projections (equirectangular, globe)
- Regridding worker to avoid blocking the main thread

## Prerequisites

- Node ≥ 20
- The WASM package built first: `cd typescript && make ts-build`

## Dev server

```bash
cd tensoscope
npm install
npm run dev
```

Starts at http://localhost:5173.

## Production build

```bash
cd tensoscope
npm run build
```

Output goes to `tensoscope/dist/`.

## Docker

```bash
cd tensoscope
make build   # builds podman/docker image
make run     # serves at http://localhost:8000
```

Set `BASE_PATH=/subpath` to deploy under a subpath.

## Architecture

Tensoscope is a thin UI layer. All heavy lifting is done by the
`@ecmwf/tensogram` WASM package (`typescript/`):

- `src/tensogram/index.ts` — wraps the WASM API into a `TensoscopeViewer` class
- `src/store/useAppStore.ts` — Zustand state for selected file, field, and step
- `src/components/map/` — deck.gl + MapLibre rendering, regrid worker, colour maps
- `src/components/file-browser/` — file open dialog, field selector, metadata panel
- `src/components/animation/` — step slider and animation controls

## Licence

Apache 2.0 — see [LICENSE](../LICENSE).
