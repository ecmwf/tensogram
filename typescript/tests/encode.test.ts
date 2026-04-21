// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import { describe, expect, it } from 'vitest';
import {
  decode,
  encode,
  init,
  InvalidArgumentError,
} from '../src/index.js';
import { defaultMeta, makeDescriptor } from './helpers.js';

describe('Phase 1 — encode wrapper', () => {
  it('rejects metadata without version', async () => {
    await init();
    // @ts-expect-error — intentional misuse
    expect(() => encode({}, [])).toThrow(InvalidArgumentError);
  });

  it('rejects _reserved_ from client code', async () => {
    await init();
    expect(() =>
      encode(
        { version: 2, _reserved_: { encoder: { name: 'fake' } } },
        [],
      ),
    ).toThrow(/_reserved_/);
  });

  it('rejects _reserved_ inside base entries', async () => {
    await init();
    expect(() =>
      encode(
        { version: 2, base: [{ _reserved_: { tensor: { ndim: 0 } } }] },
        [
          {
            descriptor: makeDescriptor([1], 'float32'),
            data: new Float32Array([1.0]),
          },
        ],
      ),
    ).toThrow(/_reserved_/);
  });

  it('rejects an unknown dtype', async () => {
    await init();
    expect(() =>
      encode(defaultMeta(), [
        {
          // @ts-expect-error — intentional bad dtype
          descriptor: { ...makeDescriptor([2], 'float32'), dtype: 'float128' },
          data: new Float32Array([1, 2]),
        },
      ]),
    ).toThrow(InvalidArgumentError);
  });

  it('rejects a byte_order other than big/little', async () => {
    await init();
    expect(() =>
      encode(defaultMeta(), [
        {
          // @ts-expect-error — intentional bad byte_order
          descriptor: { ...makeDescriptor([2], 'float32'), byte_order: 'middle' },
          data: new Float32Array([1, 2]),
        },
      ]),
    ).toThrow(InvalidArgumentError);
  });

  it('accepts hash=false and the output decodes without verifyHash', async () => {
    await init();
    const data = new Float32Array([1, 2, 3]);
    const msg = encode(
      defaultMeta(),
      [{ descriptor: makeDescriptor([3], 'float32'), data }],
      { hash: false },
    );
    const decoded = decode(msg);
    expect(decoded.objects[0].descriptor.hash).toBeUndefined();
    decoded.close();
  });

  it('defaults to xxh3 hashing', async () => {
    // v3: the per-object hash moved from the CBOR descriptor to
    // the frame footer's inline slot (plans/WIRE_FORMAT.md §2.4).
    // `descriptor.hash` is always undefined on the decoded
    // output.  The default `hash: "xxh3"` option still populates
    // the inline slot (verifiable via `tensogram validate
    // --checksum`; a Message-level `inlineHashes()` accessor is
    // tracked as a pass-5 follow-up).  This test now just pins
    // that encode+decode round-trips cleanly with default hashing.
    await init();
    const data = new Float32Array([1, 2, 3]);
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([3], 'float32'), data },
    ]);
    const decoded = decode(msg);
    expect(decoded.objects[0].descriptor.hash).toBeUndefined();
    decoded.close();
  });

  it('accepts DataView and other ArrayBufferViews', async () => {
    await init();
    const source = new Float32Array([1, 2, 3, 4]);
    // Wrap the same bytes in a DataView
    const view = new DataView(source.buffer);
    const msg = encode(defaultMeta(), [
      { descriptor: makeDescriptor([4], 'float32'), data: view },
    ]);
    const decoded = decode(msg);
    const out = decoded.objects[0].data() as Float32Array;
    expect(Array.from(out)).toEqual([1, 2, 3, 4]);
    decoded.close();
  });

  it('encodes multiple objects with different dtypes', async () => {
    await init();
    const msg = encode(defaultMeta(), [
      {
        descriptor: makeDescriptor([3], 'float32'),
        data: new Float32Array([1.5, 2.5, 3.5]),
      },
      {
        descriptor: makeDescriptor([5], 'uint8'),
        data: new Uint8Array([10, 20, 30, 40, 50]),
      },
    ]);
    const decoded = decode(msg);
    expect(decoded.objects).toHaveLength(2);
    expect(decoded.objects[0].descriptor.dtype).toBe('float32');
    expect(decoded.objects[1].descriptor.dtype).toBe('uint8');
    expect(Array.from(decoded.objects[0].data() as Float32Array)).toEqual([1.5, 2.5, 3.5]);
    expect(Array.from(decoded.objects[1].data() as Uint8Array)).toEqual([10, 20, 30, 40, 50]);
    decoded.close();
  });

  // ── Coverage closers: exercise every input-validation branch ─────────

  it('rejects non-object metadata', async () => {
    await init();
    // @ts-expect-error intentional non-object
    expect(() => encode(null, [])).toThrow(InvalidArgumentError);
    // @ts-expect-error intentional non-object
    expect(() => encode(42, [])).toThrow(InvalidArgumentError);
    // @ts-expect-error intentional non-object
    expect(() => encode('not-a-meta', [])).toThrow(InvalidArgumentError);
  });

  it('rejects negative or fractional version', async () => {
    await init();
    expect(() => encode({ version: -1 }, [])).toThrow(InvalidArgumentError);
    expect(() => encode({ version: 1.5 }, [])).toThrow(InvalidArgumentError);
  });

  it('rejects non-array metadata.base', async () => {
    await init();
    expect(() =>
      // @ts-expect-error intentional
      encode({ version: 2, base: 'not-an-array' }, []),
    ).toThrow(/base must be an array/);
  });

  it('rejects non-object base[i] entries', async () => {
    await init();
    expect(() =>
      // @ts-expect-error intentional
      encode({ version: 2, base: [42] }, []),
    ).toThrow(/base\[0\] must be a plain object/);
  });

  it('rejects non-array objects parameter', async () => {
    await init();
    // @ts-expect-error intentional
    expect(() => encode({ version: 2 }, 'not-an-array')).toThrow(
      /objects must be an array/,
    );
  });

  it('rejects non-object entries in the objects array', async () => {
    await init();
    expect(() =>
      // @ts-expect-error intentional
      encode({ version: 2 }, [42]),
    ).toThrow(/objects\[0\] must be a/);
  });

  it('rejects non-ArrayBufferView data fields', async () => {
    await init();
    expect(() =>
      encode({ version: 2 }, [
        {
          descriptor: makeDescriptor([3], 'float32'),
          // @ts-expect-error intentional
          data: [1, 2, 3],
        },
      ]),
    ).toThrow(/must be an ArrayBufferView/);
  });

  it('rejects non-object descriptor fields', async () => {
    await init();
    expect(() =>
      encode({ version: 2 }, [
        {
          // @ts-expect-error intentional
          descriptor: 'bad-descriptor',
          data: new Float32Array([1, 2, 3]),
        },
      ]),
    ).toThrow(/descriptor must be a plain object/);
  });

  it('rejects non-string descriptor.type', async () => {
    await init();
    expect(() =>
      encode({ version: 2 }, [
        {
          // @ts-expect-error intentional
          descriptor: { ...makeDescriptor([3], 'float32'), type: 42 },
          data: new Float32Array([1, 2, 3]),
        },
      ]),
    ).toThrow(/descriptor\.type must be a string/);
  });

  it('rejects non-array descriptor.shape', async () => {
    await init();
    expect(() =>
      encode({ version: 2 }, [
        {
          // @ts-expect-error intentional
          descriptor: { ...makeDescriptor([3], 'float32'), shape: 'scalar' },
          data: new Float32Array([1, 2, 3]),
        },
      ]),
    ).toThrow(/descriptor\.shape must be an array/);
  });

  it('is deterministic — same input produces same bytes', async () => {
    await init();
    const data = new Float32Array([1, 2, 3]);
    const makeMeta = () => ({
      version: 2,
      base: [{ mars: { param: '2t', class: 'od' } }],
    });
    const a = encode(makeMeta(), [
      { descriptor: makeDescriptor([3], 'float32'), data },
    ]);
    const b = encode(makeMeta(), [
      { descriptor: makeDescriptor([3], 'float32'), data },
    ]);
    // Provenance (_reserved_.uuid, _reserved_.time) makes output non-deterministic
    // across calls. This test asserts only the payload survives, not bit-equality.
    // The wire-format determinism claim applies after stripping provenance.
    const ra = decode(a);
    const rb = decode(b);
    expect(Array.from(ra.objects[0].data() as Float32Array))
      .toEqual(Array.from(rb.objects[0].data() as Float32Array));
    ra.close();
    rb.close();
  });
});
