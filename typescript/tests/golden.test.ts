// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Cross-language parity tests. Decode the same golden `.tgm` files used
 * by the Rust, Python, and C++ suites from TypeScript, and assert
 * identical observable behaviour.
 *
 * The golden files live under `rust/tensogram-core/tests/golden/*.tgm`
 * and are committed to the repo. They are regenerated from Rust only
 * when explicitly requested via the `regenerate_golden_files` Rust
 * test — so running this file verifies that the TS wrapper agrees
 * with the Rust core on every byte-level semantic.
 */

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  decode,
  decodeMetadata,
  init,
  scan,
} from '../src/index.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const GOLDEN_DIR = join(__dirname, '../../rust/tensogram-core/tests/golden');

function loadGolden(name: string): Uint8Array {
  const buf = readFileSync(join(GOLDEN_DIR, name));
  return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
}

describe('Cross-language golden-file parity', () => {
  it('simple_f32.tgm decodes to [1.0, 2.0, 3.0, 4.0]', async () => {
    await init();
    const bytes = loadGolden('simple_f32.tgm');
    const decoded = decode(bytes);
    try {
      expect(decoded.metadata.version).toBe(2);
      expect(decoded.objects).toHaveLength(1);

      const obj = decoded.objects[0];
      expect(obj.descriptor.shape).toEqual([4]);
      expect(obj.descriptor.dtype).toBe('float32');

      const values = obj.data() as Float32Array;
      expect(values).toBeInstanceOf(Float32Array);
      expect(values.length).toBe(4);
      expect(Array.from(values)).toEqual([1.0, 2.0, 3.0, 4.0]);
    } finally {
      decoded.close();
    }
  });

  it('multi_object.tgm decodes 3 mixed-dtype objects', async () => {
    await init();
    const bytes = loadGolden('multi_object.tgm');
    const decoded = decode(bytes);
    try {
      expect(decoded.metadata.version).toBe(2);
      expect(decoded.objects).toHaveLength(3);

      // object[0] — float32 [2]
      expect(decoded.objects[0].descriptor.dtype).toBe('float32');
      expect(decoded.objects[0].descriptor.shape).toEqual([2]);

      // object[1] — int64 [3] with values [100, -200, 300]
      expect(decoded.objects[1].descriptor.dtype).toBe('int64');
      expect(decoded.objects[1].descriptor.shape).toEqual([3]);
      const i64 = decoded.objects[1].data() as BigInt64Array;
      expect(Array.from(i64)).toEqual([100n, -200n, 300n]);

      // object[2] — uint8 [5] with values [10, 20, 30, 40, 50]
      expect(decoded.objects[2].descriptor.dtype).toBe('uint8');
      expect(decoded.objects[2].descriptor.shape).toEqual([5]);
      const u8 = decoded.objects[2].data() as Uint8Array;
      expect(Array.from(u8)).toEqual([10, 20, 30, 40, 50]);
    } finally {
      decoded.close();
    }
  });

  it('mars_metadata.tgm exposes MARS keys under base[0].mars', async () => {
    await init();
    const bytes = loadGolden('mars_metadata.tgm');
    const decoded = decode(bytes);
    try {
      expect(decoded.metadata.version).toBe(2);
      expect(decoded.metadata.base).toBeDefined();
      expect(decoded.metadata.base!.length).toBeGreaterThan(0);

      const mars = decoded.metadata.base![0].mars as Record<string, unknown> | undefined;
      expect(mars).toBeDefined();
      // Rust suite asserts these three keys are present.
      expect(mars).toHaveProperty('class');
      expect(mars).toHaveProperty('type');
      expect(mars).toHaveProperty('step');

      expect(decoded.objects).toHaveLength(1);
      expect(decoded.objects[0].descriptor.shape).toEqual([2, 3]);
      expect(decoded.objects[0].descriptor.dtype).toBe('float64');
    } finally {
      decoded.close();
    }
  });

  it('multi_message.tgm contains two messages that scan + decode independently', async () => {
    await init();
    const bytes = loadGolden('multi_message.tgm');
    const positions = scan(bytes);
    expect(positions).toHaveLength(2);

    const m1 = decode(bytes.subarray(positions[0].offset, positions[0].offset + positions[0].length));
    try {
      const vals1 = m1.objects[0].data() as Float32Array;
      expect(Array.from(vals1)).toEqual([1.0, 2.0]);
    } finally {
      m1.close();
    }

    const m2 = decode(bytes.subarray(positions[1].offset, positions[1].offset + positions[1].length));
    try {
      const vals2 = m2.objects[0].data() as Float32Array;
      expect(Array.from(vals2)).toEqual([3.0, 4.0]);
    } finally {
      m2.close();
    }
  });

  it('hash_xxh3.tgm decodes with verifyHash=true and carries a hash descriptor', async () => {
    await init();
    const bytes = loadGolden('hash_xxh3.tgm');
    const decoded = decode(bytes, { verifyHash: true });
    try {
      expect(decoded.metadata.version).toBe(2);
      expect(decoded.objects).toHaveLength(1);
      const hash = decoded.objects[0].descriptor.hash;
      expect(hash).toBeDefined();
      expect(hash?.type).toBe('xxh3');
      expect(typeof hash?.value).toBe('string');
      expect(hash!.value.length).toBeGreaterThan(0);
    } finally {
      decoded.close();
    }
  });

  it('hash_xxh3.tgm flags corruption under verifyHash', async () => {
    await init();
    const bytes = loadGolden('hash_xxh3.tgm');
    // Corrupt a byte in the last 32 bytes of the file. For this small
    // golden file the payload sits near the end, so a late-byte flip
    // is guaranteed to land in either the payload itself (triggers
    // HashMismatchError) or the hash/postamble bookkeeping (triggers
    // a FramingError). Both are valid "tamper detected" outcomes.
    const tampered = new Uint8Array(bytes);
    const target = tampered.length - 24;
    tampered[target] ^= 0xff;
    expect(() => decode(tampered, { verifyHash: true })).toThrow();
  });

  it('decodeMetadata on every golden file yields version 2', async () => {
    await init();
    for (const name of [
      'simple_f32.tgm',
      'multi_object.tgm',
      'mars_metadata.tgm',
      'hash_xxh3.tgm',
    ]) {
      const bytes = loadGolden(name);
      const meta = decodeMetadata(bytes);
      expect(meta.version, `version in ${name}`).toBe(2);
    }
  });
});
