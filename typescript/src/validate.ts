// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * `validate` / `validateFile` — structural and integrity validation.
 *
 * `validate(buf)` works everywhere (browser, Node, any runtime).
 * `validateFile(path)` is a Node convenience that reads the file, then
 * runs `validate_buffer` in WASM on the contents — mirroring the
 * buffer/file split present in Rust core and every other language
 * binding.
 *
 * The WASM side returns a JSON string; we parse it once here so callers
 * see typed {@link ValidationReport} / {@link FileValidationReport}
 * objects.  JSON is the neutral bridge because it preserves large
 * integers unambiguously (issue `byte_offset` can exceed 2^32 in
 * pathological file-level reports).
 */

import { getWbg } from './init.js';
import { rethrowTyped, InvalidArgumentError, IoError } from './errors.js';
import { scan } from './decode.js';
import { errorMessage, withCause } from './internal/io.js';
import type {
  FileIssue,
  FileValidationReport,
  ValidateOptions,
  ValidateMode,
  ValidationReport,
} from './types.js';

/**
 * Validate a single Tensogram message buffer (not a file).  The
 * returned report carries any discovered issues; a "valid" message
 * is one with no `error`-severity entries.
 *
 * Structural / integrity / fidelity problems are reported in the
 * returned `issues` list — this function does NOT throw on malformed
 * input.  It throws only on programmer error (wrong argument type or
 * unknown validation mode).
 *
 * @throws {InvalidArgumentError} when `buf` is not a `Uint8Array` or
 *   `opts.mode` is not one of the documented modes.
 */
export function validate(buf: Uint8Array, opts?: ValidateOptions): ValidationReport {
  if (!(buf instanceof Uint8Array)) {
    throw new InvalidArgumentError(`buf must be a Uint8Array, got ${typeof buf}`);
  }
  const { level, canonical } = resolveOptions(opts);
  const wbg = getWbg();
  const json = rethrowTyped(() => wbg.validate_buffer(buf, level, canonical));
  return parseMessageReport(json);
}

/**
 * Validate every message in a multi-message buffer, reporting gaps and
 * trailing garbage between messages.
 *
 * Runtime-agnostic — does not require Node.  For Node callers wanting
 * to validate a `.tgm` file on disk, use {@link validateFile}.
 *
 * @throws {InvalidArgumentError} when `buf` is not a `Uint8Array` or
 *   `opts.mode` is not one of the documented modes.
 */
export function validateBuffer(
  buf: Uint8Array,
  opts?: ValidateOptions,
): FileValidationReport {
  if (!(buf instanceof Uint8Array)) {
    throw new InvalidArgumentError(`buf must be a Uint8Array, got ${typeof buf}`);
  }
  const positions = scan(buf);
  const file_issues: FileIssue[] = [];
  const messages: ValidationReport[] = [];

  let expected = 0;
  for (const { offset, length } of positions) {
    if (offset > expected) {
      file_issues.push({
        byte_offset: expected,
        length: offset - expected,
        description: `${offset - expected} unrecognized bytes at offset ${expected}`,
      });
    }
    // subarray shares the underlying `ArrayBuffer` — the WASM
    // boundary reads through the view's `byteOffset` / `byteLength`,
    // so no copy is needed to isolate one message's bytes.
    messages.push(validate(buf.subarray(offset, offset + length), opts));
    expected = offset + length;
  }

  if (expected < buf.byteLength) {
    const trailing = buf.byteLength - expected;
    const desc =
      messages.length === 0
        ? `${trailing} bytes with no valid messages`
        : `${trailing} trailing bytes after last message at offset ${expected}`;
    file_issues.push({ byte_offset: expected, length: trailing, description: desc });
  }

  return { file_issues, messages };
}

/**
 * Node-only: validate every message in a local `.tgm` file.  Reads the
 * entire file into memory once, then delegates to {@link validateBuffer}.
 *
 * For very large files (several GiB) consider
 * {@link TensogramFile.fromUrl} with `validate(rawMessage(i))` in a
 * loop to keep memory bounded — see `docs/src/guide/typescript-api.md`.
 *
 * @throws {IoError} when the path cannot be read or `node:fs/promises`
 *   is unavailable (non-Node runtime).
 */
export async function validateFile(
  path: string | URL,
  opts?: ValidateOptions,
): Promise<FileValidationReport> {
  if (typeof path !== 'string' && !(path instanceof URL)) {
    throw new InvalidArgumentError(
      'validateFile: path must be a string or file:// URL',
    );
  }
  let readFile: typeof import('node:fs/promises').readFile;
  try {
    ({ readFile } = await import('node:fs/promises'));
  } catch (cause) {
    throw withCause(
      new IoError('validateFile requires Node; use validate(buf) in browsers'),
      cause,
    );
  }

  let bytes: Uint8Array;
  try {
    const buf = await readFile(path);
    bytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  } catch (err) {
    throw withCause(new IoError(`validateFile: ${errorMessage(err)}`), err);
  }
  return validateBuffer(bytes, opts);
}

// ── internals ──────────────────────────────────────────────────────────────

/**
 * Resolve caller options to the `(level, canonical)` pair expected by
 * the WASM entrypoint.  An invalid `mode` is caught at compile time
 * via the `never` exhaustiveness guard; the runtime fallback below
 * only fires if a caller slips past `--strict` typing.
 */
function resolveOptions(
  opts?: ValidateOptions,
): { level: ValidateMode; canonical: boolean } {
  const mode: ValidateMode = opts?.mode ?? 'default';
  switch (mode) {
    case 'quick':
    case 'default':
    case 'checksum':
    case 'full':
      return { level: mode, canonical: opts?.canonical ?? false };
    default: {
      const _exhaustive: never = mode;
      throw new InvalidArgumentError(
        `validate: unknown mode "${String(_exhaustive)}"; expected quick | default | checksum | full`,
      );
    }
  }
}

function parseMessageReport(json: string): ValidationReport {
  try {
    return JSON.parse(json) as ValidationReport;
  } catch (err) {
    throw new InvalidArgumentError(
      `validate: WASM returned invalid JSON (${errorMessage(err)})`,
    );
  }
}
