import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import wasm from 'vite-plugin-wasm'
import path from 'path'

const tensogramPkg = path.resolve(
  __dirname, 'node_modules/@ecmwf/tensogram/dist/index.js',
)

/** Vite plugin: proxy /api/proxy?url=... to bypass CORS for remote .tgm files. */
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
          const upstream = await fetch(target);
          if (!upstream.ok) {
            res.statusCode = upstream.status;
            res.end(`Upstream error: ${upstream.status} ${upstream.statusText}`);
            return;
          }
          res.setHeader('Content-Type', upstream.headers.get('content-type') ?? 'application/octet-stream');
          const contentLength = upstream.headers.get('content-length');
          if (contentLength) res.setHeader('Content-Length', contentLength);
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
