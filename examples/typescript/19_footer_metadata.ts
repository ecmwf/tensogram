// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

/**
 * Example 19 — Message metadata known only at the end of a stream
 *
 * A streaming producer often cannot describe its own message up front:
 * which of the requested parameters actually had data, the observed
 * value range, how many objects ended up written.
 * `StreamingEncoder.setFooterMetadata` records those facts while
 * encoding; they are merged into the footer metadata frame, which
 * supersedes the header frame on decode.
 *
 * The header can therefore carry a provisional value and the footer the
 * final one — footer `_extra_` keys win on conflict.
 *
 * Run:
 *   cd typescript && npm run build
 *   cd ../examples/typescript
 *   npm install
 *   npx tsx 19_footer_metadata.ts
 */

import {
	decode,
	init,
	StreamingEncoder,
	type DataObjectDescriptor,
} from "@ecmwf.int/tensogram";

/** The parameters a consumer asked for.  Only some have data. */
const REQUESTED = ["2t", "10u", "tp"];

function describe(
	shape: number[],
	dtype: DataObjectDescriptor["dtype"],
): DataObjectDescriptor {
	const strides = new Array<number>(shape.length).fill(1);
	for (let i = shape.length - 2; i >= 0; i--)
		strides[i] = strides[i + 1] * shape[i + 1];
	return {
		type: "ntensor",
		ndim: shape.length,
		shape,
		strides,
		dtype,
		byte_order: "little",
		encoding: "none",
		filter: "none",
		compression: "none",
	};
}

async function main(): Promise<void> {
	await init();

	// The header metadata states the request and marks the result
	// provisional: at this point the producer does not yet know what it
	// will be able to write.
	const enc = new StreamingEncoder({
		_extra_: { requestedParams: REQUESTED, status: "provisional" },
	});

	const written: string[] = [];
	let maximum = Number.NEGATIVE_INFINITY;
	try {
		for (const [i, param] of REQUESTED.entries()) {
			if (param === "tp") {
				console.log(`  ${param}: no data — skipped`);
				continue;
			}
			const values = Float32Array.from([0, 1, 2, 3], (v) => 250 + v + 4 * i);
			enc.writePreceder({ param });
			enc.writeObject(describe([values.length], "float32"), values);
			written.push(param);
			maximum = Math.max(maximum, ...values);
			console.log(`  ${param}: wrote ${values.length} values`);
		}

		// Now the message-level facts are known.  Record them for the footer
		// frame; `status` replaces the provisional header value.
		enc.setFooterMetadata({
			status: "complete",
			writtenParams: written,
			valueMaximum: maximum,
		});
		// Repeated calls accumulate, so late additions need no bookkeeping
		// of their own.
		enc.setFooterMetadata({ producer: "footer-metadata-example" });

		const bytes = enc.finish();
		console.log(`\nStreamed ${bytes.byteLength} bytes`);

		const decoded = decode(bytes);
		try {
			const extra = decoded.metadata._extra_ ?? {};
			console.log(`Decoded ${decoded.objects.length} object(s)`);
			console.log(`  status:           ${String(extra["status"])}`);
			console.log(
				`  writtenParams:    ${JSON.stringify(extra["writtenParams"])}`,
			);
			console.log(`  valueMaximum:     ${String(extra["valueMaximum"])}`);
			console.log(
				`  requestedParams:  ${JSON.stringify(extra["requestedParams"])}`,
			);
		} finally {
			decoded.close();
		}
	} finally {
		enc.close();
	}
}

main().catch((err: unknown) => {
	console.error(err);
	process.exit(1);
});
