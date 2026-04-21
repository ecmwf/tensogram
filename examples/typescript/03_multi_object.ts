// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Example 03 — Multiple objects with different dtypes (TypeScript)
 *
 * Mirrors `examples/python/04_multi_object.py`. Shows that each object
 * in a message has its own descriptor and dtype, so `object.data()`
 * returns the correct `TypedArray` for each.
 */

import {
  decode,
  decodeObject,
  encode,
  init,
  type DataObjectDescriptor,
} from '@ecmwf.int/tensogram';

function desc(shape: number[], dtype: DataObjectDescriptor['dtype']): DataObjectDescriptor {
  const strides = new Array<number>(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) strides[i] = strides[i + 1] * shape[i + 1];
  return {
    type: 'ntensor',
    ndim: shape.length,
    shape,
    strides,
    dtype,
    byte_order: 'little',
    encoding: 'none',
    filter: 'none',
    compression: 'none',
  };
}

async function main(): Promise<void> {
  await init();

  const spectrum = new Float64Array(256);
  for (let i = 0; i < spectrum.length; i++) spectrum[i] = Math.sin(i * 0.1);

  const landmask = new Uint8Array(16);
  for (let i = 0; i < landmask.length; i++) landmask[i] = i % 2;

  const counts = new Int32Array([10, 20, 30, 40, 50]);

  const msg = encode({ version: 2 }, [
    { descriptor: desc([256], 'float64'), data: spectrum },
    { descriptor: desc([4, 4], 'uint8'), data: landmask },
    { descriptor: desc([5], 'int32'), data: counts },
  ]);
  console.log(`Message size: ${msg.byteLength} bytes, ${3} objects\n`);

  // Full decode — all three objects come back with their correct dtype.
  const full = decode(msg);
  try {
    for (let i = 0; i < full.objects.length; i++) {
      const o = full.objects[i];
      const arr = o.data();
      console.log(
        `  object[${i}]  dtype=${o.descriptor.dtype}  shape=${JSON.stringify(o.descriptor.shape)}  ` +
        `array=${arr.constructor.name}  length=${arr.length}`,
      );
    }
  } finally {
    full.close();
  }

  // Selective decode — grab object 1 without touching the others.
  console.log(`\nSelective decode of object[1]:`);
  const only = decodeObject(msg, 1);
  try {
    const mask = only.objects[0].data() as Uint8Array;
    console.log(`  mask elements: ${Array.from(mask).join(', ')}`);
  } finally {
    only.close();
  }
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});
