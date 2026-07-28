// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Example 21 — Frame walker + message header (TypeScript)
 *
 * Structural introspection: look at what a `.tgm` message *is* made of,
 * without decoding a single payload or parsing a byte of CBOR.
 *
 * 1. {@link messageHeader} — the 24-byte preamble as typed values.  Tells
 *    you whether a message is random-access (metadata / index / hashes in
 *    the *header*) or streaming (in the *footer*) before you read anything
 *    else.
 *
 * 2. {@link frames} — the frame chain of ONE message: type, offset,
 *    length, flags, and the frame's content with the 16-byte frame header
 *    and the type-specific footer already stripped.  The preamble and the
 *    postamble are NOT frames and never appear.
 *
 * 3. Multi-message files — `frames` and `messageHeader` are per-message.
 *    Use {@link scan} to find the boundaries, then slice.
 *
 * 4. {@link EncodeOptions.aggregateHash} — choose where the aggregate hash
 *    frame goes, and see the choice on the wire through the walker.
 *
 * Run with:  npx tsx 21_frame_walker.ts
 */

import {
  encode,
  frames,
  init,
  messageHeader,
  scan,
  type AggregateHashPolicy,
  type DataObjectDescriptor,
  type Frame,
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

/** One tidy line per frame. */
function render(f: Frame): string {
  const span = `${String(f.offset).padStart(5)}..${String(f.offset + f.length).padEnd(5)}`;
  return (
    `  ${f.frameType.padEnd(16)} type=${String(f.frameTypeCode).padStart(2)}  ` +
    `bytes ${span} (${String(f.length).padStart(4)} B)  ` +
    `payload=${String(f.payload.byteLength).padStart(4)} B  ` +
    `hash=${f.hasHash ? 'yes' : 'no '}`
  );
}

function twoObjectMessage(options?: { aggregateHash?: AggregateHashPolicy }): Uint8Array {
  return encode(
    { base: [{ note: 'temperature' }, { note: 'humidity' }] },
    [
      { descriptor: describe([2, 2], 'float32'), data: new Float32Array([1, 2, 3, 4]) },
      { descriptor: describe([3], 'float64'), data: new Float64Array([10, 20, 30]) },
    ],
    options,
  );
}

async function main(): Promise<void> {
  await init();

  const msg = twoObjectMessage();

  // ── 1. messageHeader — the envelope, from 24 bytes ───────────────────
  console.log('─── 1. messageHeader(msg) — the message envelope ────────');
  const header = messageHeader(msg);
  console.log(`  wire version:        ${header.version}`);
  console.log(`  total length:        ${header.totalLength} B (buffer is ${msg.byteLength} B)`);
  console.log(`  header metadata:     ${header.hasHeaderMetadata}`);
  console.log(`  header index:        ${header.hasHeaderIndex}`);
  console.log(`  header hashes:       ${header.hasHeaderHashes}`);
  console.log(`  footer metadata:     ${header.hasFooterMetadata}`);
  console.log(`  footer index:        ${header.hasFooterIndex}`);
  console.log(`  footer hashes:       ${header.hasFooterHashes}`);
  console.log(`  preceder metadata:   ${header.hasPrecederMetadata}`);
  console.log(`  all frames hashed:   ${header.hasHashesPresent}`);
  // Header-side metadata + index ⇒ this message supports random access.
  const randomAccess = header.hasHeaderMetadata && header.hasHeaderIndex;
  console.log(`  ⇒ layout:            ${randomAccess ? 'random-access' : 'streaming'}`);

  // ── 2. frames — the frame chain ──────────────────────────────────────
  console.log('\n─── 2. frames(msg) — the frame chain ────────────────────');
  const chain = frames(msg);
  for (const f of chain) console.log(render(f));
  console.log(
    `  ${chain.length} frames; ` +
      `${chain.filter((f) => f.frameType === 'NTensorFrame').length} data object(s).`,
  );
  // The preamble (24 B) and postamble (24 B) are not frames.
  console.log(`  first frame starts at byte ${chain[0].offset} — after the preamble.`);

  // `payload` is the frame CONTENT: the 16-byte frame header and the
  // type-specific footer (20 B for NTensorFrame, 12 B otherwise) are
  // stripped.  It is a COPY on the JS heap, so it stays valid across
  // later WASM calls and can be mutated without touching `msg`.
  const meta = chain.find((f) => f.frameType === 'HeaderMetadata');
  if (meta) {
    const overhead = meta.length - meta.payload.byteLength;
    console.log(
      `  HeaderMetadata: ${meta.payload.byteLength} B of CBOR + ${overhead} B of framing.`,
    );
  }

  // ── 3. Per-message: scan first, then slice ───────────────────────────
  console.log('\n─── 3. Multi-message file — scan() then slice ───────────');
  const other = twoObjectMessage();
  const file = new Uint8Array(msg.byteLength + other.byteLength);
  file.set(msg, 0);
  file.set(other, msg.byteLength);

  for (const [i, pos] of scan(file).entries()) {
    // frames()/messageHeader() take ONE message — slice it out first.
    const message = file.subarray(pos.offset, pos.offset + pos.length);
    const types = frames(message).map((f) => f.frameType);
    console.log(
      `  message ${i} @ ${pos.offset} (${pos.length} B): ` +
        `v${messageHeader(message).version}  [${types.join(' → ')}]`,
    );
  }
  // Frame offsets are relative to the message, so add the message offset
  // to turn one into a file offset.
  const second = scan(file)[1];
  const firstFrame = frames(file.subarray(second.offset, second.offset + second.length))[0];
  console.log(
    `  message 1's first frame: offset ${firstFrame.offset} in the message, ` +
      `${second.offset + firstFrame.offset} in the file.`,
  );

  // ── 4. aggregateHash — placement, observed on the wire ───────────────
  console.log('\n─── 4. aggregateHash — where the digest list lands ──────');
  const policies: AggregateHashPolicy[] = ['auto', 'none', 'header', 'footer', 'both'];
  for (const policy of policies) {
    const types = frames(twoObjectMessage({ aggregateHash: policy })).map((f) => f.frameType);
    const where = [
      types.includes('HeaderHash') ? 'HeaderHash' : null,
      types.includes('FooterHash') ? 'FooterHash' : null,
    ].filter(Boolean);
    console.log(
      `  ${policy.padEnd(7)} → ${(where.length ? where.join(' + ') : '(no aggregate frame)').padEnd(24)} ` +
        `[${types.join(' → ')}]`,
    );
  }
  console.log('  ("auto" resolves to a header frame in this buffered encoder.)');

  console.log('\nFrame walker + message header OK.');
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});
