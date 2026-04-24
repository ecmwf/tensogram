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
 * Usage: `npx tsx ts_driver.ts --url <URL> --op <OP>`
 *
 * Operations — see drivers/rust_driver/src/main.rs for the canonical
 * list. The TS driver mirrors the Rust one exactly so the server-side
 * request logs can be compared round-for-round.
 *
 * Like the Rust driver, this emits no logs of its own. The mock server
 * captures every HTTP request on its side, tagged by the run_id in the
 * URL path; the orchestrator fetches those logs after the driver exits.
 */

import { init, TensogramFile } from '@ecmwf.int/tensogram';

type Op = 'open' | 'message-count' | 'read-first' | 'read-last';

function parseArgs(argv: string[]): { url: string; op: Op } {
  let url: string | undefined;
  let op: Op | undefined;
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--url') {
      url = argv[++i];
    } else if (arg === '--op') {
      const value = argv[++i];
      if (!isOp(value)) throw new Error(`unknown --op '${value}'`);
      op = value;
    } else if (arg === '--help' || arg === '-h') {
      printUsage();
      process.exit(0);
    } else {
      throw new Error(`unknown argument '${arg}'`);
    }
  }
  if (url === undefined) throw new Error('missing --url');
  if (op === undefined) throw new Error('missing --op');
  return { url, op };
}

function isOp(value: string | undefined): value is Op {
  return value === 'open' || value === 'message-count' || value === 'read-first' || value === 'read-last';
}

function printUsage(): void {
  console.error(
    'usage: npx tsx ts_driver.ts --url <URL> --op <open|message-count|read-first|read-last>',
  );
}

async function run(url: string, op: Op): Promise<void> {
  await init();
  const file = await TensogramFile.fromUrl(url);
  try {
    switch (op) {
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
    }
  } finally {
    file.close();
  }
}

async function main(): Promise<number> {
  let args: { url: string; op: Op };
  try {
    args = parseArgs(process.argv.slice(2));
  } catch (err) {
    console.error(`error: ${(err as Error).message}`);
    printUsage();
    return 2;
  }
  try {
    await run(args.url, args.op);
    return 0;
  } catch (err) {
    console.error(`error: ${(err as Error).message}`);
    return 1;
  }
}

main().then((code) => process.exit(code));
