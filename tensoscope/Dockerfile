# syntax=docker/dockerfile:1
# Build context must be the repo root:
#   podman build -f tensoscope/Dockerfile -t tensoscope:latest .
# Or from within tensoscope/:
#   podman build -f Dockerfile -t tensoscope:latest ..

# ── Stage 1: build frontend ──────────────────────────────────────────────────
FROM node:22-alpine AS builder

WORKDIR /workspace

# Copy only the pre-built @ecmwf/tensogram artifacts (not the whole tree —
# tensoscope/node_modules/@ecmwf/tensogram is a symlink back to typescript/ which
# causes a cycle if we copy the directory recursively).
COPY typescript/package.json typescript/package.json
COPY typescript/dist/ typescript/dist/
COPY typescript/wasm/ typescript/wasm/

# Copy only package files first for layer caching
COPY tensoscope/package.json tensoscope/package-lock.json tensoscope/

WORKDIR /workspace/tensoscope
RUN npm ci

# Copy the rest of the frontend source
COPY tensoscope/ .

# Replace the host symlink with the real pre-built package
RUN rm -rf node_modules/@ecmwf/tensogram && \
    mkdir -p node_modules/@ecmwf/tensogram && \
    cp /workspace/typescript/package.json node_modules/@ecmwf/tensogram/ && \
    cp -r /workspace/typescript/dist node_modules/@ecmwf/tensogram/ && \
    cp -r /workspace/typescript/wasm node_modules/@ecmwf/tensogram/

RUN npm run build

# ── Stage 2: serve with nginx ─────────────────────────────────────────────────
FROM nginx:1.27-alpine

# Replace the default nginx.conf — routes all temp dirs to /tmp so the
# container works rootless (podman, non-root Docker, Kubernetes)
COPY tensoscope/docker/nginx.conf /etc/nginx/nginx.conf

# Keep server-block templates outside /etc/nginx/templates/ so nginx's own
# 20-envsubst-on-templates.sh does not try to process them
COPY tensoscope/docker/nginx.conf.template /etc/tensogram/nginx.conf.template
COPY tensoscope/docker/nginx.root.template /etc/tensogram/nginx.root.template

# Entrypoint writes the final server-block to /tmp/nginx/conf.d/ at runtime
COPY tensoscope/docker/entrypoint.sh /docker-entrypoint.d/40-base-path.sh
RUN chmod +x /docker-entrypoint.d/40-base-path.sh

# Static assets from the build stage
COPY --from=builder /workspace/tensoscope/dist/ /usr/share/nginx/html

EXPOSE 8000
