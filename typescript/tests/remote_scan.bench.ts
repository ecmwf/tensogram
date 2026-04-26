// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Vitest wall-clock for the remote-scan walker headline cells.
 *
 * Wall-clock only: HTTP request counters live in the NDJSON sidecar
 * emitted by `tests/run_remote_bench.ts` (deterministic single-pass).
 * Vitest's iteration count would smear the counters; here we only
 * care about timing distributions.
 */

import { spawn, type ChildProcessByStdio } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import type { Readable } from 'node:stream';

import { afterAll, beforeAll, bench, describe } from 'vitest';

import { init, TensogramFile } from '@ecmwf.int/tensogram';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, '../..');
const PARITY_DIR = resolve(REPO_ROOT, 'tests/remote-parity');
const FIXTURES_DIR = resolve(PARITY_DIR, 'fixtures');
const MOCK_SERVER_PY = resolve(PARITY_DIR, 'mock_server.py');

let mockProc: ChildProcessByStdio<null, Readable, Readable> | undefined;
let baseUrl = '';

async function startMockServer(): Promise<void> {
  const proc = spawn(
    'python',
    [
      MOCK_SERVER_PY,
      '--port',
      '0',
      '--fixtures-dir',
      FIXTURES_DIR,
      // The wall-clock bench never reads the per-request log; disable
      // logging server-side so the per-run_id buffer can't accumulate
      // across Vitest's many statistical iterations.
      '--no-log-requests',
    ],
    {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    },
  );
  proc.stderr.on('data', () => {});
  await new Promise<void>((resolveStart, reject) => {
    let buffered = '';
    let timeoutHandle: NodeJS.Timeout | undefined;
    const cleanup = (): void => {
      proc.stdout.off('data', onData);
      if (timeoutHandle !== undefined) clearTimeout(timeoutHandle);
    };
    const onData = (chunk: Buffer): void => {
      buffered += chunk.toString('utf-8');
      const match = buffered.match(/on\s+(http:\/\/127\.0\.0\.1:\d+)/);
      if (match) {
        cleanup();
        baseUrl = match[1];
        mockProc = proc;
        resolveStart();
      }
    };
    proc.stdout.on('data', onData);
    proc.once('error', (err) => {
      cleanup();
      reject(err);
    });
    proc.once('exit', (code) => {
      cleanup();
      reject(new Error(`mock_server.py exited with code ${code} before announcing URL`));
    });
    timeoutHandle = setTimeout(() => {
      cleanup();
      // Stuck startup: the child is alive but hasn't printed its URL.
      // SIGTERM it so Vitest doesn't leak an orphaned python process
      // when the bench fails; Python's http.server handles SIGTERM cleanly.
      proc.kill('SIGTERM');
      reject(new Error('timed out waiting for mock_server.py to announce its URL'));
    }, 10_000);
  });
}

interface Cell {
  fixture: string;
  n: number;
  scenario: 'iter' | 'read_message_last';
}

const HEADLINE_CELLS: Cell[] = [
  { fixture: 'hundred-msg.tgm', n: 100, scenario: 'iter' },
  { fixture: 'hundred-msg-footer.tgm', n: 100, scenario: 'iter' },
  { fixture: 'thousand-msg.tgm', n: 1000, scenario: 'read_message_last' },
  { fixture: 'streaming-tail.tgm', n: 10, scenario: 'iter' },
];

async function runCell(cell: Cell, bidirectional: boolean, runId: string): Promise<void> {
  const url = `${baseUrl}/${runId}/${cell.fixture}`;
  const file = await TensogramFile.fromUrl(url, { bidirectional });
  try {
    if (cell.scenario === 'iter') {
      const count = file.messageCount;
      for (let i = 0; i < count; i++) {
        await file.rawMessage(i);
      }
    } else {
      await file.rawMessage(cell.n - 1);
    }
  } finally {
    file.close();
  }
}

describe('remote_scan walker (wall-clock only)', () => {
  beforeAll(async () => {
    await init();
    await startMockServer();
  });

  afterAll(() => {
    if (mockProc !== undefined) {
      mockProc.kill('SIGTERM');
    }
  });

  const SHARED_RUN_ID = 'vitest-bench';
  for (const cell of HEADLINE_CELLS) {
    for (const bidirectional of [false, true]) {
      const walkerLabel = bidirectional ? 'bidir' : 'forward';
      const id = `${cell.fixture}/${cell.scenario}/${walkerLabel}`;
      bench(id, async () => {
        await runCell(cell, bidirectional, SHARED_RUN_ID);
      });
    }
  }
});
