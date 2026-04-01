# Tensogram

[![Static Badge](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity/emerging_badge.svg)](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity#emerging)

> [!IMPORTANT]
> This software is **Emerging** and subject to ECMWF's guidelines on [Software Maturity](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity).

A library for encoding and decoding binary scientific data with embedded
semantic metadata in a network-transmissible binary message format.

Tensogram defines a network-transmissible binary message format, not a file format.
Multiple messages can be appended to a file, and that remains valid since each
message carries its own begin/terminator codes.

