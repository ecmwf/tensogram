// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
//
// Tests for the lazy per-object iterators `objects` / `objectsMetadata`
// (Wave D / O-TS-1).  Mirror the Rust core `iter::objects` /
// `iter::objects_metadata` and the Python `objects` / `objects_metadata`
// generators.  See `plans/INTERFACE_SYMMETRY.md` §5.4 / §8.

import { describe, expect, it } from 'vitest';
import {
  decode,
  encode,
  init,
  objects,
  objectsMetadata,
  type DataObjectDescriptor,
} from '../src/index.js';
import { defaultMeta, makeDescriptor } from './helpers.js';

/** Encode a message with three objects of distinct shapes / dtypes. */
function threeObjectMessage(): Uint8Array {
  return encode(defaultMeta(), [
    { descriptor: makeDescriptor([4], 'float32'), data: new Float32Array([1, 2, 3, 4]) },
    { descriptor: makeDescriptor([2, 2], 'int32'), data: new Int32Array([5, 6, 7, 8]) },
    { descriptor: makeDescriptor([3], 'float64'), data: new Float64Array([9, 10, 11]) },
  ]);
}

describe('objects (lazy per-object iterator)', () => {
  it('yields each decoded object in wire order', async () => {
    await init();
    const msg = threeObjectMessage();
    const collected = [...objects(msg)];
    expect(collected).toHaveLength(3);

    expect(collected[0].descriptor.dtype).toBe('float32');
    expect(Array.from(collected[0].data() as Float32Array)).toEqual([1, 2, 3, 4]);

    expect(collected[1].descriptor.dtype).toBe('int32');
    expect(collected[1].descriptor.shape).toEqual([2, 2]);
    expect(Array.from(collected[1].data() as Int32Array)).toEqual([5, 6, 7, 8]);

    expect(collected[2].descriptor.dtype).toBe('float64');
    expect(Array.from(collected[2].data() as Float64Array)).toEqual([9, 10, 11]);
  });

  it('is a real generator (lazy — supports early break)', async () => {
    await init();
    const msg = threeObjectMessage();
    const seen: string[] = [];
    for (const obj of objects(msg)) {
      seen.push(obj.descriptor.dtype);
      if (seen.length === 1) break; // stop after the first
    }
    expect(seen).toEqual(['float32']);
  });

  it('yielded objects stay valid after the iterator is exhausted (self-contained)', async () => {
    await init();
    const msg = threeObjectMessage();
    const all = [...objects(msg)];
    // No handle to close; data() is still readable after iteration ends.
    expect(Array.from(all[0].data() as Float32Array)).toEqual([1, 2, 3, 4]);
    expect(Array.from(all[2].data() as Float64Array)).toEqual([9, 10, 11]);
  });

  it('matches eager decode().objects descriptors and byte lengths', async () => {
    await init();
    const msg = threeObjectMessage();
    const eager = decode(msg);
    try {
      const lazy = [...objects(msg)];
      expect(lazy).toHaveLength(eager.objects.length);
      for (let i = 0; i < lazy.length; i++) {
        expect(lazy[i].descriptor.dtype).toBe(eager.objects[i].descriptor.dtype);
        expect(lazy[i].descriptor.shape).toEqual(eager.objects[i].descriptor.shape);
        expect(lazy[i].byteLength).toBe(eager.objects[i].byteLength);
      }
      // Content parity for each known dtype.
      expect(Array.from(lazy[0].data() as Float32Array)).toEqual(
        Array.from(eager.objects[0].data() as Float32Array),
      );
      expect(Array.from(lazy[1].data() as Int32Array)).toEqual(
        Array.from(eager.objects[1].data() as Int32Array),
      );
      expect(Array.from(lazy[2].data() as Float64Array)).toEqual(
        Array.from(eager.objects[2].data() as Float64Array),
      );
    } finally {
      eager.close();
    }
  });

  it('honours decode options (restoreNonFinite=false)', async () => {
    await init();
    const msg = encode(
      defaultMeta(),
      [{ descriptor: makeDescriptor([3], 'float64'), data: new Float64Array([1, NaN, 3]) }],
      { allowNan: true, smallMaskThresholdBytes: 0 },
    );
    // Default: NaN restored.
    const restored = [...objects(msg)][0].data() as Float64Array;
    expect(Number.isNaN(restored[1])).toBe(true);
    // Opt out: 0.0 at masked position.
    const raw = [...objects(msg, { restoreNonFinite: false })][0].data() as Float64Array;
    expect(raw[1]).toBe(0);
  });

  it('yields nothing for a zero-object message', async () => {
    await init();
    const msg = encode(defaultMeta(), []);
    expect([...objects(msg)]).toHaveLength(0);
  });

  it('throws on non-Uint8Array input', async () => {
    await init();
    expect(() => {
      // @ts-expect-error deliberate wrong type
      [...objects('nope')];
    }).toThrow(/Uint8Array/);
  });
});

describe('objectsMetadata (descriptors without payload decode)', () => {
  it('yields each descriptor in wire order', async () => {
    await init();
    const msg = threeObjectMessage();
    const descs: DataObjectDescriptor[] = [...objectsMetadata(msg)];
    expect(descs).toHaveLength(3);
    expect(descs.map((d) => d.dtype)).toEqual(['float32', 'int32', 'float64']);
    expect(descs[1].shape).toEqual([2, 2]);
    // Descriptors carry no decoded payload — just structural metadata.
    expect(descs[0].encoding).toBe('none');
  });

  it('matches decode().objects descriptors', async () => {
    await init();
    const msg = threeObjectMessage();
    const eager = decode(msg);
    try {
      const lazy = [...objectsMetadata(msg)];
      expect(lazy.map((d) => d.dtype)).toEqual(
        eager.objects.map((o) => o.descriptor.dtype),
      );
      expect(lazy.map((d) => d.shape)).toEqual(
        eager.objects.map((o) => o.descriptor.shape),
      );
    } finally {
      eager.close();
    }
  });

  it('is lazy — supports early break', async () => {
    await init();
    const msg = threeObjectMessage();
    const seen: string[] = [];
    for (const d of objectsMetadata(msg)) {
      seen.push(d.dtype);
      if (seen.length === 2) break;
    }
    expect(seen).toEqual(['float32', 'int32']);
  });

  it('yields nothing for a zero-object message', async () => {
    await init();
    const msg = encode(defaultMeta(), []);
    expect([...objectsMetadata(msg)]).toHaveLength(0);
  });

  it('throws on a structurally invalid buffer', async () => {
    await init();
    expect(() => [...objectsMetadata(new Uint8Array([1, 2, 3, 4]))]).toThrow();
  });
});
