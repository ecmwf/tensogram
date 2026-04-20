import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import wasm from 'vite-plugin-wasm'
import path from 'path'

const tensogramPkg = path.resolve(
  __dirname, 'node_modules/@ecmwf/tensogram/dist/index.js',
)

/** Vite plugin: proxy /api/proxy?url=... to bypass CORS for remote .tgm files.
 *  Forwards the request method and Range header so TensogramFile.fromUrl can
 *  use lazy per-message Range requests in dev mode, matching nginx behaviour. */
function corsProxy(): Plugin {
  return {
    name: 'cors-proxy',
    configureServer(server) {
      server.middlewares.use('/api/proxy', async (req, res) => {
        const parsed = new URL(req.url ?? '', 'http://localhost');
        const target = parsed.searchParams.get('url');
        if (!target) {
          res.statusCode = 400;
          res.end('Missing url parameter');
          return;
        }
        try {
          const init: RequestInit = { method: req.method };
          const range = req.headers['range'];
          if (range) init.headers = { Range: range };

          const upstream = await fetch(target, init);
          res.statusCode = upstream.status;

          for (const key of ['content-type', 'content-length', 'content-range', 'accept-ranges']) {
            const val = upstream.headers.get(key);
            if (val) res.setHeader(key, val);
          }

          if (req.method === 'HEAD') {
            res.end();
            return;
          }

          const buf = Buffer.from(await upstream.arrayBuffer());
          res.end(buf);
        } catch (err) {
          res.statusCode = 502;
          res.end(`Proxy error: ${err}`);
        }
      });
    },
  };
}

// https://vite.dev/config/
export default defineConfig({
  base: './',
  plugins: [react(), wasm(), corsProxy()],
  resolve: {
    alias: {
      '@ecmwf/tensogram': tensogramPkg,
    },
    preserveSymlinks: true,
  },
  optimizeDeps: {
    exclude: ['@ecmwf/tensogram'],
  },
  server: {
    fs: {
      allow: ['..'],
    },
  },
})
