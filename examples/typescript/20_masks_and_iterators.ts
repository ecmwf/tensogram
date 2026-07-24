// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Example 20 — Advanced decode masks + lazy per-object iterators (TypeScript)
 *
 * Four Wave-D capabilities that mirror the Rust core and complete the
 * read surface:
 *
 * 1. {@link decodeWithMasks} — decode returning the raw NaN / +Inf / -Inf
 *    bitmasks alongside the `0.0`-substituted payloads.  Instead of the
 *    canonical NaN / ±Inf restoration {@link decode} performs, you get
 *    one byte-per-element mask per kind (`1` = masked) and can apply your
 *    own missing-value representation.
 *
 * 2. {@link objects} — a lazy generator over the decoded objects of one
 *    message; each object is decoded on demand.
 *
 * 3. {@link objectsMetadata} — a lazy generator over the per-object
 *    descriptors WITHOUT decoding any payload.
 *
 * 4. {@link scanWithOptions} + {@link dataObjectInlineHashes} — explicit
 *    scan options and per-object inline integrity digests.
 */

import {
  dataObjectInlineHashes,
  decodeWithMasks,
  encode,
  init,
  objects,
  objectsMetadata,
  scanWithOptions,
  type DataObjectDescriptor,
} from '@ecmwf.int/tensogram';

function describe(shape: number[], dtype: DataObjectDescriptor['dtype']): DataObjectDescriptor {
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

  // ── A message with three objects; the first carries NaN / ±Inf holes ──
  const field = new Float64Array(8).map((_, i) => i + 1);
  field[2] = NaN;
  field[5] = Infinity;
  field[6] = -Infinity;

  const msg = encode(
    { base: [{ note: 'gappy field' }, { note: 'temps' }, { note: 'counts' }] },
    [
      { descriptor: describe([8], 'float64'), data: field },
      { descriptor: describe([4], 'float32'), data: new Float32Array([10, 11, 12, 13]) },
      { descriptor: describe([2, 2], 'int32'), data: new Int32Array([1, 2, 3, 4]) },
    ],
    { allowNan: true, allowInf: true, smallMaskThresholdBytes: 0 },
  );

  // ── 1. decodeWithMasks ──────────────────────────────────────────────
  console.log('─── 1. decodeWithMasks (raw NaN / Inf masks) ────────────');
  const decoded = decodeWithMasks(msg);
  try {
    const obj = decoded.objects[0];
    const payload = obj.data() as Float64Array;
    const { nan, posInf, negInf } = obj.masks;

    // Payload holds 0.0 at every non-finite position (no restoration).
    console.log(`  payload:  [${Array.from(payload).join(', ')}]`);
    // Each mask is one byte per element: 1 = that kind was present.
    const idx = (m: Uint8Array | null): number[] =>
      m ? [...m].flatMap((v, i) => (v ? [i] : [])) : [];
    console.log(`  nan  at:  [${idx(nan).join(', ')}]`);
    console.log(`  +Inf at:  [${idx(posInf).join(', ')}]`);
    console.log(`  -Inf at:  [${idx(negInf).join(', ')}]`);

    // Apply our own missing-value representation: rebuild the field with
    // JS `null` for every masked position, `NaN`/`±Inf` restored otherwise.
    const withNulls = Array.from(payload).map((v, i) => {
      if (nan?.[i]) return null;
      if (posInf?.[i]) return '+Inf';
      if (negInf?.[i]) return '-Inf';
      return v;
    });
    console.log(`  masked→:  [${withNulls.map(String).join(', ')}]`);

    // Object 2 had no non-finite values → all-null mask set.
    console.log(`  object[2] masks all null: ${decoded.objects[2].masks.nan === null}`);
  } finally {
    decoded.close();
  }

  // ── 2. objects — lazy per-object decode ─────────────────────────────
  console.log('\n─── 2. objects(buf) — lazy per-object decode ────────────');
  let n = 0;
  for (const obj of objects(msg)) {
    // Each object is decoded only when pulled from the generator.
    console.log(`  [${n++}] dtype=${obj.descriptor.dtype}  shape=[${obj.descriptor.shape.join(', ')}]  bytes=${obj.byteLength}`);
  }

  // ── 3. objectsMetadata — descriptors without payload decode ─────────
  console.log('\n─── 3. objectsMetadata(buf) — descriptors only ──────────');
  for (const [i, d] of [...objectsMetadata(msg)].entries()) {
    console.log(`  [${i}] dtype=${d.dtype}  ndim=${d.ndim}  encoding=${d.encoding}`);
  }

  // ── 4. scanWithOptions + dataObjectInlineHashes ─────────────────────
  console.log('\n─── 4. scanWithOptions + dataObjectInlineHashes ─────────');
  const positions = scanWithOptions(msg, { bidirectional: true });
  console.log(`  scanWithOptions found ${positions.length} message(s)`);
  const hashes = dataObjectInlineHashes(msg);
  console.log(`  inline hashes: [${hashes.join(', ')}]`);

  console.log('\nMasks + iterators OK.');
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});
