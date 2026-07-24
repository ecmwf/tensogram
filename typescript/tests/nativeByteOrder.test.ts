// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * `DecodeOptions.nativeByteOrder` — surfaces the Rust core
 * `DecodeOptions.native_byte_order` through the wasm `decode` /
 * `decode_object` widen.
 *
 * When `true` (the default) the decoded payload is converted to the
 * host's native byte order; when `false` it is left in the message's
 * declared wire `byte_order`.  We stage a payload declared big-endian
 * on the wire and observe the two behaviours.
 *
 * See `plans/INTERFACE_SYMMETRY.md` §5.4 / §8.2 (O-TS-1).
 */

import { describe, expect, it } from 'vitest';
import { decode, decodeObject, encode, init, TensogramFile } from '../src/index.js';
import type { DataObjectDescriptor, DecodedObject } from '../src/index.js';
import { defaultMeta } from './helpers.js';

/** Host endianness — `true` on x86 / ARM (typical CI). */
const HOST_LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

/** A `uint32` descriptor whose payload is declared **big-endian** on the wire. */
function bigU32Descriptor(n: number): DataObjectDescriptor {
  return {
    type: 'ntensor',
    ndim: 1,
    shape: [n],
    strides: [1],
    dtype: 'uint32',
    byte_order: 'big',
    encoding: 'none',
    filter: 'none',
    compression: 'none',
  };
}

/** Copy of the raw payload bytes behind a decoded object's typed view. */
function rawBytes(obj: DecodedObject): Uint8Array {
  // `data()` returns the dtype-aware `TypedArray` union; for the numeric
  // dtypes used here it is a concrete `Uint32Array`, so read its backing
  // bytes through the structural `ArrayBufferView` shape.
  const view = obj.data() as unknown as ArrayBufferView;
  return new Uint8Array(view.buffer, view.byteOffset, view.byteLength).slice();
}

/** Reverse each `width`-byte element (an endianness swap). */
function swapPerElement(bytes: Uint8Array, width: number): Uint8Array {
  const out = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i += width) {
    for (let j = 0; j < width; j++) out[i + j] = bytes[i + width - 1 - j];
  }
  return out;
}

describe('O-TS-1 — DecodeOptions.nativeByteOrder', () => {
  // Two uint32 stored big-endian on the wire: 0x01020304 and 0x05060708.
  const wire = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
  const BIG_ENDIAN_VALUES = [0x01020304, 0x05060708]; // 16909060, 84281096

  function encoded(): Uint8Array {
    return encode(defaultMeta(), [{ descriptor: bigU32Descriptor(2), data: wire }]);
  }

  it('default decode converts wire bytes to host order (semantically-correct values)', async () => {
    await init();
    // native_byte_order defaults to true: the Uint32Array reads the
    // declared big-endian values back correctly on any host.
    const d = decode(encoded());
    const arr = d.objects[0].data() as Uint32Array;
    d.close();
    expect(Array.from(arr)).toEqual(BIG_ENDIAN_VALUES);
  });

  it('nativeByteOrder=false leaves the payload in the declared wire byte order', async () => {
    await init();
    const d = decode(encoded(), { nativeByteOrder: false });
    const raw = rawBytes(d.objects[0]);
    d.close();
    // Bytes are returned exactly as declared on the wire (big-endian),
    // independent of host endianness.
    expect(Array.from(raw)).toEqual(Array.from(wire));
  });

  it('the option has an observable effect on a byte-swapping host', async () => {
    await init();
    const nat = decode(encoded(), { nativeByteOrder: true });
    const wireOrder = decode(encoded(), { nativeByteOrder: false });
    const natBytes = rawBytes(nat.objects[0]);
    const wireBytes = rawBytes(wireOrder.objects[0]);
    nat.close();
    wireOrder.close();
    if (HOST_LITTLE_ENDIAN) {
      expect(Array.from(natBytes)).not.toEqual(Array.from(wireBytes));
      expect(Array.from(natBytes)).toEqual(Array.from(swapPerElement(wire, 4)));
    } else {
      // On a big-endian host native == wire, so both paths agree.
      expect(Array.from(natBytes)).toEqual(Array.from(wireBytes));
    }
  });

  it('omitting nativeByteOrder is identical to passing true (backwards-compatible default)', async () => {
    await init();
    const a = decode(encoded());
    const b = decode(encoded(), { nativeByteOrder: true });
    const av = Array.from(a.objects[0].data() as Uint32Array);
    const bv = Array.from(b.objects[0].data() as Uint32Array);
    a.close();
    b.close();
    expect(av).toEqual(bv);
    expect(av).toEqual(BIG_ENDIAN_VALUES);
  });

  it('decodeObject honours nativeByteOrder', async () => {
    await init();
    const o = decodeObject(encoded(), 0, { nativeByteOrder: false });
    const raw = rawBytes(o.objects[0]);
    o.close();
    expect(Array.from(raw)).toEqual(Array.from(wire));
  });

  it('TensogramFile.message forwards nativeByteOrder through to decode', async () => {
    await init();
    const file = TensogramFile.fromBytes(encoded());
    try {
      const m = await file.message(0, { nativeByteOrder: false });
      const raw = rawBytes(m.objects[0]);
      m.close();
      expect(Array.from(raw)).toEqual(Array.from(wire));
    } finally {
      file.close();
    }
  });
});
