// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Example 02 — Per-object metadata with the MARS vocabulary (TypeScript)
 *
 * Mirrors `examples/python/02_mars_metadata.py`.
 *
 * Each data object gets its own `base[i]` entry with application metadata.
 * This example uses the ECMWF MARS vocabulary for concreteness; the same
 * pattern works with any namespace (CF, BIDS, DICOM, custom). See
 * `02b_generic_metadata.ts` for a non-MARS example.
 *
 * `getMetaKey` looks up dotted paths using first-match semantics;
 * `computeCommon` extracts keys that are shared across every object's
 * metadata.
 */

import {
  computeCommon,
  decode,
  encode,
  getMetaKey,
  init,
  type DataObjectDescriptor,
  type GlobalMetadata,
} from '@ecmwf/tensogram';

async function main(): Promise<void> {
  await init();

  const shape = [2, 3] as const;
  const strides = [3, 1] as const;
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

  const temperature = new Float32Array([273.15, 274.0, 275.0, 276.0, 277.0, 278.0]);
  const wind = new Float32Array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

  const metadata: GlobalMetadata = {
    version: 2,
    base: [
      {
        mars: {
          class: 'od', type: 'fc', stream: 'oper',
          date: '20260401', time: '1200',
          param: '2t', levtype: 'sfc',
        },
      },
      {
        mars: {
          class: 'od', type: 'fc', stream: 'oper',
          date: '20260401', time: '1200',
          param: '10u', levtype: 'sfc',
        },
      },
    ],
    _extra_: { source: 'ifs-cycle49r2' },
  };

  const msg = encode(metadata, [
    { descriptor, data: temperature },
    { descriptor, data: wind },
  ]);
  console.log(`Message size: ${msg.byteLength} bytes`);

  const result = decode(msg);
  try {
    const meta = result.metadata;

    // Dotted-path lookup across base[i] with first-match semantics.
    console.log(`\nmars.param (first match): ${getMetaKey(meta, 'mars.param') as string}`);
    console.log(`mars.class (first match): ${getMetaKey(meta, 'mars.class') as string}`);
    console.log(`_extra_.source:           ${getMetaKey(meta, '_extra_.source') as string}`);
    console.log(`missing key:              ${String(getMetaKey(meta, 'does.not.exist'))}`);

    // computeCommon() extracts top-level keys whose value is identical across
    // every base entry. The `mars` sub-objects here differ by `param`, so at
    // the top level they are NOT equal — nothing comes out as common. This is
    // by design: commonality is a whole-value check, not a recursive merge.
    const common = computeCommon(meta);
    console.log(`\ncomputeCommon() (no shared top-level key expected here):`);
    console.log(`  keys: [${Object.keys(common).join(', ')}]`);

    // To see computeCommon() actually return a shared key, flatten the MARS
    // sub-object into top-level keys at the base level.
    const flatMeta = {
      version: 2,
      base: [
        { class: 'od', type: 'fc', param: '2t' },
        { class: 'od', type: 'fc', param: '10u' },
      ],
    };
    console.log(`  on a flat metadata: keys = [${Object.keys(computeCommon(flatMeta)).join(', ')}]`);

    console.log(`\nPer-object param values:`);
    for (let i = 0; i < result.objects.length; i++) {
      const entry = meta.base?.[i];
      const param = (entry?.mars as Record<string, unknown> | undefined)?.param;
      console.log(`  object[${i}] mars.param = ${String(param)}`);
    }
  } finally {
    result.close();
  }
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});
