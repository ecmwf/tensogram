# ────────────────────────────────────────────────────────────────
# tensogram FFI publish image
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
#   - perl-core + clang-devel — the manylinux gaps documented in
#     publish-ffi.yml ("Install build dependencies"), needed for
#     openssl-sys (perl) and bindgen-driven *-sys crates (libclang)
#
# Builds on quay.io/pypa/manylinux_2_28_x86_64 so the produced
# libtensogram.so links against glibc 2.28, matching the manylinux_2_28
# wheel ABI.  Anything newer (e.g. the ubuntu:24.04 base used by
# .github/ci/Dockerfile) would link against glibc 2.39 and break every
# distro older than Ubuntu 24.04 / RHEL 9.4 — the whole point of using
# the manylinux base in publish-ffi is glibc compatibility.
#
#   docker build -t tensogram-ci-ffi:test .github/ci-ffi/
# ────────────────────────────────────────────────────────────────

ARG BASE=quay.io/pypa/manylinux_2_28_x86_64
FROM ${BASE}

ARG RUST_VERSION=1.94.0
ARG CARGO_C_VERSION=0.10.21
ARG SCCACHE_VERSION=0.10.0

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN set -eux \
 && dnf install -y perl-core clang-devel \
 && dnf clean all \
 && rm -rf /var/cache/dnf

RUN set -eux \
 && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain "${RUST_VERSION}" --profile minimal \
 && rustup target add x86_64-unknown-linux-gnu \
 && rm -rf "${RUSTUP_HOME}"/toolchains/*/share/doc \
           "${RUSTUP_HOME}"/toolchains/*/share/man

RUN set -eux \
 && cargo install cargo-c --version "${CARGO_C_VERSION}" --locked --features vendored-openssl \
 && rm -rf "${CARGO_HOME}"/registry "${CARGO_HOME}"/git /tmp/*

RUN set -eux \
 && curl -fsSL "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
    | tar xz --strip-components=1 -C /usr/local/bin \
        "sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl/sccache"

RUN set -eux \
 && rustc --version && cargo --version \
 && cargo cbuild --help >/dev/null \
 && sccache --version

WORKDIR /workspace
