// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * Remote-scan walker microbench (TypeScript) — metrics runner.
 *
 * Spawns `tests/remote-parity/mock_server.py` once via subprocess and
 * iterates the full (fixture × tier × scenario × walker) matrix
 * deterministically: each cell creates a fresh `TensogramFile` via
 * `fromUrl`, runs the scenario, closes, and queries the server's
 * per-`run_id` log to extract `total_requests`, `range_get_requests`,
 * `head_requests`, and `response_body_bytes`.
 *
 * Wall-clock samples are emitted alongside the request counters so the
 * NDJSON sidecar matches Rust and Python cell-for-cell.  The Vitest
 * bench harness (`tests/remote_scan.bench.ts`) covers headline cells
 * with multi-sample statistical wall-clock; this runner is the
 * deterministic single-pass companion that feeds the decision artifact.
 *
 * Usage:
 *   npx tsx tests/run_remote_bench.ts            # full matrix
 *   npx tsx tests/run_remote_bench.ts --quick    # N=10 only
 *   npx tsx tests/run_remote_bench.ts --json out.ndjson
 */

import { spawn, type ChildProcessByStdio } from 'node:child_process';
import type { Readable } from 'node:stream';
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

import { init, TensogramFile } from '@ecmwf.int/tensogram';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, '../..');
const PARITY_DIR = resolve(REPO_ROOT, 'tests/remote-parity');
const FIXTURES_DIR = resolve(PARITY_DIR, 'fixtures');
const MOCK_SERVER_PY = resolve(PARITY_DIR, 'mock_server.py');
const DEFAULT_OUT = resolve(REPO_ROOT, 'target/remote-scan-bench/typescript.ndjson');

interface CellSpec {
  fixtureKind: 'header-indexed' | 'footer-indexed' | 'streaming-tail';
  fixtureName: string;
  tier: number;
}

const HEADER_FIXTURES: Array<[number, string]> = [
  [1, 'single-msg.tgm'],
  [10, 'ten-msg.tgm'],
  [100, 'hundred-msg.tgm'],
  [1000, 'thousand-msg.tgm'],
];
const FOOTER_FIXTURES: Array<[number, string]> = [
  [1, 'single-msg-footer.tgm'],
  [10, 'ten-msg-footer.tgm'],
  [100, 'hundred-msg-footer.tgm'],
  [1000, 'thousand-msg-footer.tgm'],
];
const STREAMING_TAIL: [number, string] = [10, 'streaming-tail.tgm'];

const SCENARIOS = [
  'message_count',
  'read_message(0)',
  'read_message(N-1)',
  'read_message(N/2)',
  'iter',
] as const;
type Scenario = (typeof SCENARIOS)[number];

function fullMatrix(): CellSpec[] {
  const out: CellSpec[] = [];
  for (const [tier, fixtureName] of HEADER_FIXTURES) {
    out.push({ fixtureKind: 'header-indexed', fixtureName, tier });
  }
  for (const [tier, fixtureName] of FOOTER_FIXTURES) {
    out.push({ fixtureKind: 'footer-indexed', fixtureName, tier });
  }
  out.push({
    fixtureKind: 'streaming-tail',
    fixtureName: STREAMING_TAIL[1],
    tier: STREAMING_TAIL[0],
  });
  return out;
}

function quickMatrix(): CellSpec[] {
  return fullMatrix().filter((c) => c.tier === 10);
}

interface RequestRecord {
  method: string;
  path: string;
  range_header: string | null;
  status: number;
  response_bytes: number;
}

async function fetchLog(baseUrl: string, runId: string): Promise<RequestRecord[]> {
  const resp = await fetch(`${baseUrl}/_log/${runId}`);
  if (!resp.ok) {
    throw new Error(`/_log/${runId} returned HTTP ${resp.status}`);
  }
  const text = await resp.text();
  const lines = text.split('\n').filter((l) => l.length > 0);
  return lines.map((l) => JSON.parse(l) as RequestRecord);
}

function classify(records: RequestRecord[]): {
  total_requests: number;
  range_get_requests: number;
  head_requests: number;
  response_body_bytes: number;
} {
  return {
    total_requests: records.length,
    range_get_requests: records.filter((r) => r.method === 'GET' && r.range_header !== null).length,
    head_requests: records.filter((r) => r.method === 'HEAD').length,
    response_body_bytes: records.reduce((acc, r) => acc + r.response_bytes, 0),
  };
}

async function runScenario(
  url: string,
  n: number,
  scenario: Scenario,
  bidirectional: boolean,
): Promise<void> {
  const file = await TensogramFile.fromUrl(url, { bidirectional });
  try {
    switch (scenario) {
      case 'message_count':
        void file.messageCount;
        break;
      case 'read_message(0)':
        await file.rawMessage(0);
        break;
      case 'read_message(N-1)':
        await file.rawMessage(n - 1);
        break;
      case 'read_message(N/2)':
        await file.rawMessage(Math.floor(n / 2));
        break;
      case 'iter': {
        const count = file.messageCount;
        for (let i = 0; i < count; i++) {
          await file.rawMessage(i);
        }
        break;
      }
    }
  } finally {
    file.close();
  }
}

interface MockServerHandle {
  baseUrl: string;
  proc: ChildProcessByStdio<null, Readable, Readable>;
}

async function startMockServer(): Promise<MockServerHandle> {
  const proc = spawn('python', [MOCK_SERVER_PY, '--port', '0', '--fixtures-dir', FIXTURES_DIR], {
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
  });
  proc.stderr.on('data', () => {});
  return new Promise<MockServerHandle>((resolveStart, reject) => {
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
        resolveStart({ baseUrl: match[1], proc });
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
      reject(new Error('timed out waiting for mock_server.py to announce its URL'));
    }, 10_000);
  });
}

function stopMockServer(handle: MockServerHandle): void {
  handle.proc.kill('SIGTERM');
}

async function resetMockServer(baseUrl: string): Promise<void> {
  const resp = await fetch(`${baseUrl}/_reset`, { method: 'POST' });
  if (!resp.ok && resp.status !== 204) {
    throw new Error(`POST /_reset returned HTTP ${resp.status}`);
  }
}

interface CellRecord {
  language: 'typescript';
  mode: 'sync';
  fixture_kind: string;
  fixture_name: string;
  tier: number;
  scenario: string;
  walker: 'forward-only' | 'bidirectional';
  total_requests: number;
  range_get_requests: number;
  head_requests: number;
  response_body_bytes: number;
  wall_ms: number;
  semantics: 'cold_open_plus_operation_plus_close';
}

async function runOneCell(
  baseUrl: string,
  spec: CellSpec,
  scenario: Scenario,
  bidirectional: boolean,
  cellId: number,
): Promise<CellRecord> {
  const runId = `ts-${cellId}`;
  const url = `${baseUrl}/${runId}/${spec.fixtureName}`;
  const started = process.hrtime.bigint();
  await runScenario(url, spec.tier, scenario, bidirectional);
  const elapsed = process.hrtime.bigint() - started;
  const wallMs = Number(elapsed) / 1e6;

  const records = await fetchLog(baseUrl, runId);
  const counters = classify(records);
  return {
    language: 'typescript',
    mode: 'sync',
    fixture_kind: spec.fixtureKind,
    fixture_name: spec.fixtureName,
    tier: spec.tier,
    scenario,
    walker: bidirectional ? 'bidirectional' : 'forward-only',
    ...counters,
    wall_ms: wallMs,
    semantics: 'cold_open_plus_operation_plus_close',
  };
}

interface CliArgs {
  matrix: CellSpec[];
  out: string;
}

function parseArgs(argv: string[]): CliArgs {
  let quick = false;
  let out = DEFAULT_OUT;
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--quick') {
      quick = true;
    } else if (arg === '--json') {
      if (i + 1 >= argv.length) throw new Error('--json requires a path argument');
      out = resolve(argv[++i]);
    } else if (arg === '--help' || arg === '-h') {
      console.log('usage: npx tsx tests/run_remote_bench.ts [--quick] [--json <path>]');
      process.exit(0);
    } else {
      throw new Error(`unknown argument '${arg}'`);
    }
  }
  return { matrix: quick ? quickMatrix() : fullMatrix(), out };
}

async function main(): Promise<number> {
  const args = parseArgs(process.argv.slice(2));
  await init();
  const server = await startMockServer();
  const cells: CellRecord[] = [];
  let cellId = 0;
  try {
    for (const spec of args.matrix) {
      for (const scenario of SCENARIOS) {
        for (const bidirectional of [false, true]) {
          cellId += 1;
          await resetMockServer(server.baseUrl);
          const record = await runOneCell(server.baseUrl, spec, scenario, bidirectional, cellId);
          cells.push(record);
        }
      }
    }
  } finally {
    stopMockServer(server);
  }

  await mkdir(dirname(args.out), { recursive: true });
  const ndjson = cells.map((c) => JSON.stringify(c)).join('\n') + '\n';
  await writeFile(args.out, ndjson);
  console.error(`wrote ${cells.length} cells to ${args.out}`);
  return 0;
}

main().then(
  (code) => process.exit(code),
  (err) => {
    console.error(`error: ${(err as Error).message}`);
    process.exit(1);
  },
);
