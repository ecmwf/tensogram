// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Remote-parity harness driver (TypeScript).
 *
 * Usage: `npx tsx ts_driver.ts --url <URL> --op <OP> [--bidirectional]`
 *
 * Operations — see drivers/rust_driver/src/main.rs for the canonical
 * list. The TS driver mirrors the Rust one exactly so the server-side
 * request logs can be compared round-for-round.
 *
 * The `--bidirectional` flag opens the file with
 * `{ bidirectional: true }`; without it the forward-only walker is
 * used (the default).  The `dump-layout` op emits a JSON
 * `[{"offset", "length"}, ...]` array to stdout that the orchestrator
 * compares against the Rust driver's output to assert walker-mode
 * equivalence.
 *
 * Like the Rust driver, this emits no logs of its own. The mock
 * server captures every HTTP request, tagged by the run_id in the
 * URL path; the orchestrator collects that captured request log from
 * the in-process server after the driver exits.
 */

import { init, TensogramFile } from '@ecmwf.int/tensogram';

type Op = 'open' | 'message-count' | 'read-first' | 'read-last' | 'dump-layout';

interface Args {
  url: string;
  op: Op;
  bidirectional: boolean;
}

function parseArgs(argv: string[]): Args {
  let url: string | undefined;
  let op: Op | undefined;
  let bidirectional = false;
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--url') {
      if (i + 1 >= argv.length) throw new Error('--url requires a value');
      url = argv[++i];
    } else if (arg === '--op') {
      if (i + 1 >= argv.length) throw new Error('--op requires a value');
      const value = argv[++i];
      if (!isOp(value)) throw new Error(`unknown --op '${value}'`);
      op = value;
    } else if (arg === '--bidirectional') {
      bidirectional = true;
    } else if (arg === '--help' || arg === '-h') {
      printUsage();
      process.exit(0);
    } else {
      throw new Error(`unknown argument '${arg}'`);
    }
  }
  if (url === undefined) throw new Error('missing --url');
  if (op === undefined) throw new Error('missing --op');
  return { url, op, bidirectional };
}

function isOp(value: string | undefined): value is Op {
  return (
    value === 'open' ||
    value === 'message-count' ||
    value === 'read-first' ||
    value === 'read-last' ||
    value === 'dump-layout'
  );
}

function printUsage(): void {
  console.error(
    'usage: npx tsx ts_driver.ts --url <URL> --op <open|message-count|read-first|read-last|dump-layout> [--bidirectional]',
  );
}

async function run(args: Args): Promise<void> {
  await init();
  const file = await TensogramFile.fromUrl(args.url, {
    bidirectional: args.bidirectional,
  });
  try {
    switch (args.op) {
      case 'open':
        return;
      case 'message-count':
        console.log(file.messageCount);
        return;
      case 'read-first':
        await file.rawMessage(0);
        return;
      case 'read-last': {
        const n = file.messageCount;
        if (n === 0) throw new Error('file contains 0 messages');
        await file.rawMessage(n - 1);
        return;
      }
      case 'dump-layout': {
        const layouts = file.messageLayouts.map((l) => ({
          offset: l.offset,
          length: l.length,
        }));
        process.stdout.write(JSON.stringify(layouts) + '\n');
        return;
      }
    }
  } finally {
    file.close();
  }
}

async function main(): Promise<number> {
  let args: Args;
  try {
    args = parseArgs(process.argv.slice(2));
  } catch (err) {
    console.error(`error: ${(err as Error).message}`);
    printUsage();
    return 2;
  }
  try {
    await run(args);
    return 0;
  } catch (err) {
    console.error(`error: ${(err as Error).message}`);
    return 1;
  }
}

main().then((code) => process.exit(code));
