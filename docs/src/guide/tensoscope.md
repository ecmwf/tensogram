# Tensoscope

Tensoscope is an interactive web viewer for `.tgm` files. It runs entirely in the
browser — no server-side component — by decoding data via the `@ecmwf/tensogram`
WebAssembly package.

## Quick start

Build the WASM package first, then start the dev server:

```bash
cd typescript && make ts-build
cd tensoscope && npm install && npm run dev
```

Open http://localhost:5173 in your browser, then drag-and-drop a `.tgm` file onto
the page or paste a URL into the file open dialog.

## Loading a file

Two modes are supported:

- **Local file** — drag the `.tgm` file onto the drop zone, or click *Open file*.
- **Remote URL** — paste an HTTP/HTTPS URL. The file is fetched in full before
  scanning. (HTTP Range support for lazy loading is planned.)

Once loaded, Tensoscope scans all messages and builds a field index without decoding
any payloads.

## Field browser

The left sidebar lists every decodable field in the file. Each entry shows:

- Variable name (resolved from `mars.param`, `name`, or `param` metadata keys)
- Shape and dtype

Click a field to decode it and render it on the map.

## Map view

Fields with two spatial dimensions (latitude × longitude) are rendered as a
`BitmapLayer` over a MapLibre GL JS base map.

Regridding onto the map pixel grid runs in a web worker so the UI stays responsive
while large arrays are processed.

### Projections

Switch between equirectangular (flat) and globe projections using the projection
picker in the toolbar.

## Colour scale

The colour bar at the bottom of the map shows the current field range. Use the
colour scale controls to:

- Change the colour map (perceptually uniform maps from d3-scale-chromatic)
- Lock or reset the min/max range

## Animation

For files with a time or step dimension, the step slider appears below the map.
Use play/pause to animate through steps at a fixed frame rate.

## Docker deployment

```bash
cd tensoscope
make build          # build the container image
make run            # serve at http://localhost:8000
BASE_PATH=/scope make run   # serve under a subpath
```

The image uses nginx and accepts a `BASE_PATH` environment variable for subpath
deployments behind a reverse proxy.

## Known limitations

- Only lat/lon grids are currently regridded; polar stereographic and other
  projections are not yet handled.
- 3D fields (pressure levels) cannot yet be sliced via the level selector
  (the UI component exists but is not yet wired up).
- HTTP Range-based lazy loading is not yet implemented; the full file is fetched
  before any field can be displayed.
