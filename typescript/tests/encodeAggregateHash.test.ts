// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
//
// Tests for the `aggregateHash` and `compressionBackend` encode knobs.
// Mirror the Rust core `EncodeOptions { aggregate_hash, compression_backend }`
// (`rust/tensogram/src/encode.rs`).  The closed loop is verified through the
// frame walker: the requested policy must be observable on the wire.

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  InvalidArgumentError,
  TensogramFile,
  decode,
  encode,
  frames,
  init,
  messageHeader,
} from '../src/index.js';
import type {
  AggregateHashPolicy,
  CompressionBackend,
  EncodeOptions,
  FrameTypeName,
} from '../src/index.js';
import { defaultMeta, makeDescriptor } from './helpers.js';

function message(opts?: EncodeOptions): Uint8Array {
  return encode(
    defaultMeta(),
    [
      { descriptor: makeDescriptor([4], 'float32'), data: new Float32Array([1, 2, 3, 4]) },
      { descriptor: makeDescriptor([4], 'float32'), data: new Float32Array([5, 6, 7, 8]) },
    ],
    opts,
  );
}

function frameTypes(msg: Uint8Array): FrameTypeName[] {
  return frames(msg).map((f) => f.frameType);
}

describe('EncodeOptions.aggregateHash', () => {
  it('defaults to the buffered-mode header placement', async () => {
    await init();
    const types = frameTypes(message());
    expect(types.filter((t) => t === 'HeaderHash')).toHaveLength(1);
    expect(types).not.toContain('FooterHash');
  });

  it("'auto' matches the default", async () => {
    await init();
    expect(frameTypes(message({ aggregateHash: 'auto' }))).toEqual(frameTypes(message()));
  });

  it("'none' emits no aggregate hash frame at all", async () => {
    await init();
    const msg = message({ aggregateHash: 'none' });
    const types = frameTypes(msg);
    expect(types).not.toContain('HeaderHash');
    expect(types).not.toContain('FooterHash');
    const h = messageHeader(msg);
    expect(h.hasHeaderHashes).toBe(false);
    expect(h.hasFooterHashes).toBe(false);
    // Per-frame inline hashes are unaffected.
    expect(h.hasHashesPresent).toBe(true);
  });

  it("'header' emits a HeaderHash frame only", async () => {
    await init();
    const msg = message({ aggregateHash: 'header' });
    const types = frameTypes(msg);
    expect(types.filter((t) => t === 'HeaderHash')).toHaveLength(1);
    expect(types).not.toContain('FooterHash');
    const h = messageHeader(msg);
    expect(h.hasHeaderHashes).toBe(true);
    expect(h.hasFooterHashes).toBe(false);
  });

  it("'footer' emits a FooterHash frame only", async () => {
    await init();
    const msg = message({ aggregateHash: 'footer' });
    const types = frameTypes(msg);
    expect(types).not.toContain('HeaderHash');
    expect(types.filter((t) => t === 'FooterHash')).toHaveLength(1);
    const h = messageHeader(msg);
    expect(h.hasHeaderHashes).toBe(false);
    expect(h.hasFooterHashes).toBe(true);
  });

  it("'both' puts a hash frame in BOTH the header and the footer", async () => {
    await init();
    const msg = message({ aggregateHash: 'both' });
    const types = frameTypes(msg);
    expect(types.filter((t) => t === 'HeaderHash')).toHaveLength(1);
    expect(types.filter((t) => t === 'FooterHash')).toHaveLength(1);
    // The header aggregate precedes the data objects; the footer
    // aggregate follows them.
    expect(types.indexOf('HeaderHash')).toBeLessThan(types.indexOf('NTensorFrame'));
    expect(types.indexOf('FooterHash')).toBeGreaterThan(types.lastIndexOf('NTensorFrame'));
    const h = messageHeader(msg);
    expect(h.hasHeaderHashes).toBe(true);
    expect(h.hasFooterHashes).toBe(true);
  });

  it('carries identical aggregate payloads in header and footer', async () => {
    await init();
    const fs = frames(message({ aggregateHash: 'both' }));
    const header = fs.find((f) => f.frameType === 'HeaderHash');
    const footer = fs.find((f) => f.frameType === 'FooterHash');
    expect(header).toBeDefined();
    expect(footer).toBeDefined();
    expect(Array.from(header!.payload)).toEqual(Array.from(footer!.payload));
  });

  it('round-trips through decode under every policy', async () => {
    await init();
    const policies: AggregateHashPolicy[] = ['auto', 'none', 'header', 'footer', 'both'];
    for (const aggregateHash of policies) {
      const msg = decode(message({ aggregateHash }));
      try {
        expect(msg.objects).toHaveLength(2);
        expect(Array.from(msg.objects[0].data() as Float32Array)).toEqual([1, 2, 3, 4]);
        expect(Array.from(msg.objects[1].data() as Float32Array)).toEqual([5, 6, 7, 8]);
      } finally {
        msg.close();
      }
    }
  });

  it('is ignored when hashing is disabled (nothing to aggregate)', async () => {
    await init();
    const types = frameTypes(message({ hash: false, aggregateHash: 'both' }));
    expect(types).not.toContain('HeaderHash');
    expect(types).not.toContain('FooterHash');
  });

  it('rejects an unknown policy name', async () => {
    await init();
    // @ts-expect-error deliberate invalid policy
    expect(() => message({ aggregateHash: 'sideways' })).toThrow(InvalidArgumentError);
    // @ts-expect-error deliberate invalid policy
    expect(() => message({ aggregateHash: 'sideways' })).toThrow(
      /unknown aggregateHash 'sideways', expected one of: auto, none, header, footer, both/,
    );
  });
});

describe('AppendOptions mirror the encode knobs', () => {
  let tmp: string;
  beforeEach(() => {
    tmp = mkdtempSync(join(tmpdir(), 'tensogram-ts-aggregate-'));
  });
  afterEach(() => {
    rmSync(tmp, { recursive: true, force: true });
  });

  it('forwards aggregateHash through TensogramFile#append', async () => {
    await init();
    const path = join(tmp, 'append.tgm');
    writeFileSync(path, message());

    const file = await TensogramFile.open(path);
    try {
      await file.append(
        defaultMeta(),
        [{ descriptor: makeDescriptor([4], 'float32'), data: new Float32Array([9, 9, 9, 9]) }],
        { aggregateHash: 'both' },
      );
      expect(file.messageCount).toBe(2);
      // The seed message used the default policy; only the appended one
      // must carry the footer aggregate.
      const seed = await file.rawMessage(0);
      const appended = await file.rawMessage(1);
      expect(frameTypes(seed)).not.toContain('FooterHash');
      expect(frameTypes(appended)).toContain('HeaderHash');
      expect(frameTypes(appended)).toContain('FooterHash');
    } finally {
      file.close();
    }
  });

  it('rejects an unknown appended aggregateHash policy', async () => {
    await init();
    const path = join(tmp, 'reject.tgm');
    writeFileSync(path, message());
    const file = await TensogramFile.open(path);
    try {
      await expect(
        file.append(
          defaultMeta(),
          [{ descriptor: makeDescriptor([4], 'float32'), data: new Float32Array([1, 2, 3, 4]) }],
          // @ts-expect-error deliberate invalid policy
          { aggregateHash: 'sideways' },
        ),
      ).rejects.toThrow(InvalidArgumentError);
    } finally {
      file.close();
    }
  });
});

describe('EncodeOptions.compressionBackend', () => {
  /**
   * A zstd-compressed message — the only kind whose bytes a real backend
   * switch could change.  Returns the compressed data-object payloads so
   * the comparison ignores the per-message provenance (UUID / timestamp)
   * that lives in the metadata frame.
   */
  function dataObjectPayloads(
    compression: 'none' | 'zstd',
    compressionBackend?: CompressionBackend,
  ): number[][] {
    const descriptor = { ...makeDescriptor([64], 'float32'), compression };
    const data = new Float32Array(64).map((_, i) => i % 4);
    const opts: EncodeOptions = compressionBackend ? { compressionBackend } : {};
    const msg = encode(defaultMeta(), [{ descriptor, data }], opts);
    return frames(msg)
      .filter((f) => f.frameType === 'NTensorFrame')
      .map((f) => Array.from(f.payload));
  }

  it('accepts every backend name and is a no-op on WASM', async () => {
    await init();
    // WASM ships pure-Rust codecs only (lz4 / szip-pure / zstd-pure), so
    // no value can change the codec that actually runs.  Compare the
    // data-object payload BYTES — the message as a whole also carries
    // per-encode provenance (uuid / timestamp) that always differs.
    const auto = dataObjectPayloads('zstd', 'auto');
    // Guard against a vacuous comparison: the codec really did run.
    expect(auto[0].length).toBeLessThan(dataObjectPayloads('none')[0].length);
    expect(dataObjectPayloads('zstd', 'pure')).toEqual(auto);
    expect(dataObjectPayloads('zstd', 'ffi')).toEqual(auto);
    expect(dataObjectPayloads('zstd')).toEqual(auto);
  });

  it('rejects an unknown backend name', async () => {
    await init();
    // @ts-expect-error deliberate invalid backend
    expect(() => message({ compressionBackend: 'cuda' })).toThrow(InvalidArgumentError);
    // @ts-expect-error deliberate invalid backend
    expect(() => message({ compressionBackend: 'cuda' })).toThrow(
      /unknown compressionBackend 'cuda', expected one of: auto, ffi, pure/,
    );
  });
});
