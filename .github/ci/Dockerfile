# ────────────────────────────────────────────────────────────────
# tensogram CI image
#
# Pre-bakes Rust toolchain + system libs. Python and Node.js are
# installed at CI runtime by uv / curl (adds ~10 s per job that
# needs them, but shrinks the image by ~430 MB).
#
#   docker build -t tensogram-ci:test .github/ci/
# ────────────────────────────────────────────────────────────────

# ── Pinned versions (edit here to bump) ─────────────────────────
ARG UV_VERSION=0.11.6
ARG SCCACHE_VERSION=0.10.0

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM ubuntu:24.04 AS builder

ARG RUST_VERSION=1.94.0
ARG WASM_PACK_VERSION=0.14.0

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN set -eux \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential pkg-config curl ca-certificates \
 && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain "${RUST_VERSION}" \
                 --profile minimal -c clippy -c rustfmt \
 && rustup target add wasm32-unknown-unknown \
 && cargo install wasm-pack --version "${WASM_PACK_VERSION}" --locked \
 && rm -rf "${CARGO_HOME}"/registry "${CARGO_HOME}"/git /tmp/* \
           "${RUSTUP_HOME}"/toolchains/*/share/doc \
           "${RUSTUP_HOME}"/toolchains/*/share/man

# ── Final image ─────────────────────────────────────────────────
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    LIBCLANG_PATH=/usr/lib/x86_64-linux-gnu \
    PATH=/usr/local/cargo/bin:$PATH

# ── System packages ─────────────────────────────────────────────
# Ubuntu 24.04 ships cmake 3.28 and LLVM 18 — zero PPAs needed.
RUN set -eux \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential pkg-config git curl ca-certificates \
      cmake \
      libclang1-18 libclang-common-18-dev \
 && rm -f /usr/lib/x86_64-linux-gnu/libicu*.a \
         /usr/lib/x86_64-linux-gnu/libssl.a \
         /usr/lib/x86_64-linux-gnu/libcrypto.a \
         /usr/lib/x86_64-linux-gnu/libxml2.a \
         /usr/lib/x86_64-linux-gnu/libcurl*.a \
 && rm -rf /usr/lib/llvm-18/include \
 && strip --strip-unneeded /usr/lib/x86_64-linux-gnu/lib*.so* 2>/dev/null || true \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
           /usr/share/doc /usr/share/man /usr/share/info \
           /usr/share/lintian /usr/share/cmake-3.28/Help

# ── Rust toolchain + wasm-pack (from builder) ──────────────────
COPY --from=builder /usr/local/rustup /usr/local/rustup
COPY --from=builder /usr/local/cargo  /usr/local/cargo

# ── uv (Python + Node installed at CI runtime) ─────────────────
COPY --from=uv /uv /uvx /usr/local/bin/

# ── sccache (S3-backed compilation cache) ──────────────────────
ARG SCCACHE_VERSION
RUN set -eux \
 && curl -fsSL "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
    | tar xz --strip-components=1 -C /usr/local/bin \
        "sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl/sccache"

# ── Smoke test ──────────────────────────────────────────────────
RUN set -eux \
 && rustc --version && cargo --version \
 && cmake --version \
 && wasm-pack --version \
 && uv --version \
 && sccache --version

WORKDIR /workspace
