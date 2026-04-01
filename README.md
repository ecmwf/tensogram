# Tensogram

[![Static Badge](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity/emerging_badge.svg)](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity#emerging)

> [!IMPORTANT]
> This software is **Emerging** and subject to ECMWF's guidelines on [Software Maturity](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity).

A library to encode and decode binary N-Tensor scientific data with semantic metadata close to the data, in a serialisable format that can be sent over the network, encoded into in-memory buffers and decoded with zero-copy. It is geared to a lightweight implementation, self-description of data and high-performance with limited dependencies.

Tensogram defines a network-transmissible binary message format, not a file format.
Multiple messages can be appended to a file, and that remains valid since each
message carries its own begin/terminator codes.
