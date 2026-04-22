// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Example 02b — Per-object metadata with a generic application namespace (TypeScript)
 *
 * Shows that the metadata mechanism in example 02 is not specific to the
 * MARS vocabulary. Any application namespace works the same way. Here we use
 * a made-up `"product"` namespace plus an `"instrument"` namespace to tag a
 * 2-D field with semantic context.
 *
 * The same pattern applies to any domain vocabulary: CF conventions (`"cf"`),
 * BIDS (`"bids"`), DICOM (`"dicom"`), or anything your application defines.
 * The library never interprets any of these — it simply stores and returns
 * the keys you supply.
 */

import {
  decode,
  encode,
  getMetaKey,
  init,
  type DataObjectDescriptor,
  type GlobalMetadata,
} from '@ecmwf.int/tensogram';

async function main(): Promise<void> {
  await init();

  const shape = [512, 512] as const;
  const strides = [512, 1] as const;
  const descriptor: DataObjectDescriptor = {
    type: 'ntensor',
    ndim: 2,
    shape: [...shape],
    strides: [...strides],
    dtype: 'float32',
    byte_order: 'little',
    encoding: 'none',
    filter: 'none',
    compression: 'none',
  };

  const data = new Float32Array(shape[0] * shape[1]);

  // Two parallel namespaces coexist freely in the same base[i] entry.
  const metadata: GlobalMetadata = {
    version: 3,
    base: [
      {
        product: {
          name: 'intensity',
          units: 'counts',
          device: 'detector_A',
          run_id: 42,
          acquired_at: '2026-04-18T10:30:00Z',
        },
        instrument: {
          serial: 'XYZ-001',
          firmware: 'v3.1.2',
        },
      },
    ],
  };

  const msg = encode(metadata, [{ descriptor, data }]);
  console.log(`Message size: ${msg.byteLength} bytes`);

  const result = decode(msg);
  try {
    const meta = result.metadata;

    // Dotted-path lookup works across any namespace.
    console.log(`\nproduct.name       : ${getMetaKey(meta, 'product.name') as string}`);
    console.log(`product.device     : ${getMetaKey(meta, 'product.device') as string}`);
    console.log(`product.run_id     : ${getMetaKey(meta, 'product.run_id') as number}`);
    console.log(`instrument.serial  : ${getMetaKey(meta, 'instrument.serial') as string}`);
    console.log(`instrument.firmware: ${getMetaKey(meta, 'instrument.firmware') as string}`);

    const obj = result.objects[0];
    console.log(`\nDecoded shape: ${JSON.stringify(obj.descriptor.shape)}`);
    console.log(`Decoded dtype: ${obj.descriptor.dtype}`);
    console.log('\nGeneric-namespace round-trip OK.');
  } finally {
    result.close();
  }
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});
