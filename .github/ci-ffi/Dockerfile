# ────────────────────────────────────────────────────────────────
# tensogram FFI publish image
#
# AlmaLinux 8 ships glibc 2.28 — the same ABI floor the manylinux_2_28
# image guarantees — at ~80 MB instead of ~1.5 GB.  The manylinux image
# adds Python interpreters + PyPy + a manylinux-audited build-tool
# stack none of which a pure Rust C-API .so consumes; libtensogram
# only links against glibc / libstdc++ / libgcc.  The
# manylinux_2_28 ABI contract reduces here to "glibc <= 2.28", which
# AlmaLinux 8 satisfies natively.  Python wheel auditwheel runs
# against publish-pypi.yml's own image, not this one.
#
# Pre-bakes everything publish-ffi.yml's Linux leg used to install on
# every run:
#
#   - Rust toolchain pinned to the same version as .github/ci/
#   - cargo-c with --features vendored-openssl (~4 min cold install
#     compiling openssl + ~50 transitive crates), so the publish job
#     no longer rebuilds it on every tag push
#   - sccache (S3-backed, same bucket as .github/ci/), so the actual
#     `cargo cinstall -p tensogram-ffi` step is incrementally cached
#     across runs
#   - perl-core + clang-libs + clang-resource-filesystem — needed for
#     openssl-sys (vendored perl scripts) and bindgen-driven *-sys
#     crates.  `clang-libs` ships the runtime libclang.so that bindgen
#     dlopens; `clang-resource-filesystem` ships clang's standard C
#     headers (stddef.h, stdarg.h, ...) which bindgen needs at parse
#     time when libclang traverses `#include <stddef.h>` in headers
#     like zstd.h.  AlmaLinux's LLVM 20 module installs the resource
#     dir at `/usr/lib/clang/20/include/` but libclang's resource-dir
#     auto-discovery (compile-time-baked-in path + relative-path-from-
#     libclang.so heuristic) doesn't find it on this packaging.  Two
#     belt-and-suspenders fixes below:
#     (a) symlink /usr/lib64/clang -> /usr/lib/clang so the relative-
#         path heuristic resolves;
#     (b) ENV BINDGEN_EXTRA_CLANG_ARGS=\"-isystem /usr/lib/clang/20/
#         include\" so bindgen's libclang invocation includes the path
#         explicitly regardless of discovery.
#     `clang-devel` would also drag in the full Clang dev environment
#     (~250 MB) we never use.
#   - gcc + gcc-c++ + make + cmake + pkg-config: cargo cinstall's
#     transitive C/C++ compilation (zstd-sys, blosc2-rs-sys, libaec-sys
#     for szip, sz3-sys, openssl-sys vendored build, final cdylib link).
#     Provided implicitly by manylinux; on AlmaLinux they are explicit
#     dnf installs.  cmake covers libaec/blosc2/sz3; gcc-c++ covers
#     anything pulling in C++ (libaec, sz3, blosc2 internals).
#   - which: AlmaLinux 8 base ships without /usr/bin/which; some
#     autoconf-style configure scripts in transitive C deps shell out
#     to it.  ~80 KB defensive insurance.
#
#   docker build -t tensogram-ci-ffi:test .github/ci-ffi/
# ────────────────────────────────────────────────────────────────

ARG BASE=almalinux:8
FROM ${BASE}

ARG RUST_VERSION=1.94.0
ARG CARGO_C_VERSION=0.10.21
ARG SCCACHE_VERSION=0.10.0

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    BINDGEN_EXTRA_CLANG_ARGS="-isystem /usr/lib/clang/20/include"

RUN set -eux \
 && dnf install -y --setopt=install_weak_deps=False \
      gcc gcc-c++ make cmake pkg-config which \
      curl ca-certificates tar gzip \
      perl-core clang-libs clang-resource-filesystem \
 && dnf clean all \
 && rm -rf /var/cache/dnf \
 && ln -sfn /usr/lib/clang /usr/lib64/clang

RUN set -eux \
 && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain "${RUST_VERSION}" --profile minimal \
 && rustup target add x86_64-unknown-linux-gnu \
 && rm -rf "${RUSTUP_HOME}"/toolchains/*/share/doc \
           "${RUSTUP_HOME}"/toolchains/*/share/man

RUN set -eux \
 && cargo install cargo-c --version "${CARGO_C_VERSION}" --locked --features vendored-openssl \
 && strip "${CARGO_HOME}"/bin/cargo-cbuild \
          "${CARGO_HOME}"/bin/cargo-capi \
          "${CARGO_HOME}"/bin/cargo-cinstall \
 && rm -rf "${CARGO_HOME}"/registry "${CARGO_HOME}"/git /tmp/*

RUN set -eux \
 && curl -fsSL "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
    | tar xz --strip-components=1 -C /usr/local/bin \
        "sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl/sccache"

RUN set -eux \
 && rustc --version && cargo --version \
 && cargo cbuild --help >/dev/null \
 && sccache --version \
 && ldd --version | head -1

WORKDIR /workspace
