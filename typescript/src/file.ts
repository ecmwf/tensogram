// (C) Copyright 2026- ECMWF and individual contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation nor
// does it submit to any jurisdiction.

/**
 * `TensogramFile` — read-only random access over a `.tgm` file, whether
 * it lives on a Node file system or behind an HTTP(S) URL.
 *
 * ## Design
 *
 * This is the Scope-B implementation: the whole file is read eagerly on
 * open, its message positions are indexed once via `scan()`, and then
 * random access is a constant-time slice + decode. This is simple, fast
 * for any realistic .tgm file size, and identical in behaviour between
 * Node and the browser.
 *
 * For very large files (several GB) where Range-based lazy access is
 * required, see the follow-up listed in `plans/TYPESCRIPT_WRAPPER.md`.
 * The public interface is designed so that a lazy backend can be added
 * later without breaking callers.
 */

import { decode, decodeMetadata, scan } from './decode.js';
import { InvalidArgumentError, IoError, ObjectError, rethrowTyped } from './errors.js';
import type {
  DecodedMessage,
  DecodeOptions,
  FileSource,
  FromUrlOptions,
  GlobalMetadata,
  MessagePosition,
  OpenFileOptions,
} from './types.js';

/**
 * Read-only view over a `.tgm` file / URL / byte buffer.
 *
 * Async iteration yields one `DecodedMessage` per message in the file,
 * in wire order. The caller is responsible for calling `close()` on
 * each yielded message to release WASM memory deterministically.
 */
export class TensogramFile implements AsyncIterable<DecodedMessage> {
  readonly #bytes: Uint8Array;
  readonly #positions: readonly MessagePosition[];
  readonly #source: FileSource;
  #closed = false;

  private constructor(bytes: Uint8Array, source: FileSource) {
    this.#bytes = bytes;
    this.#source = source;
    this.#positions = rethrowTyped(() => scan(bytes));
  }

  /**
   * Open a `.tgm` file from the local file system. Node-only — the
   * `node:fs/promises` module is loaded lazily so browser bundlers
   * can tree-shake this path.
   */
  static async open(
    path: string | URL,
    options: OpenFileOptions = {},
  ): Promise<TensogramFile> {
    if (typeof path !== 'string' && !(path instanceof URL)) {
      throw new InvalidArgumentError(
        'TensogramFile.open: path must be a string or file:// URL',
        'TensogramFile.open: path must be a string or file:// URL',
      );
    }
    let readFile: typeof import('node:fs/promises').readFile;
    try {
      ({ readFile } = await import('node:fs/promises'));
    } catch (cause) {
      throw withCause(
        new IoError(
          'TensogramFile.open requires Node; use fromUrl in browsers',
          'TensogramFile.open requires Node; use fromUrl in browsers',
        ),
        cause,
      );
    }

    let bytes: Uint8Array;
    try {
      const readOptions = options.signal ? { signal: options.signal } : undefined;
      const buf = await readFile(path, readOptions);
      bytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    } catch (err) {
      throw withCause(
        new IoError(
          `TensogramFile.open: ${describeError(err)}`,
          `TensogramFile.open: ${describeError(err)}`,
        ),
        err,
      );
    }
    return new TensogramFile(bytes, 'local');
  }

  /**
   * Open a `.tgm` file over HTTP(S). Works in any environment with a
   * fetch-compatible global (modern browsers, Node ≥ 20, Deno, Bun).
   *
   * The entire resource is downloaded in one request. Range requests
   * and lazy access are a Scope-C follow-up (see
   * `plans/TYPESCRIPT_WRAPPER.md`).
   */
  static async fromUrl(
    url: string | URL,
    options: FromUrlOptions = {},
  ): Promise<TensogramFile> {
    const fetchFn = options.fetch ?? globalThis.fetch;
    if (typeof fetchFn !== 'function') {
      throw new InvalidArgumentError(
        'TensogramFile.fromUrl: no fetch implementation is available',
        'TensogramFile.fromUrl: no fetch implementation is available',
      );
    }
    const init: RequestInit = {};
    if (options.headers !== undefined) init.headers = options.headers;
    if (options.signal !== undefined) init.signal = options.signal;

    let response: Response;
    try {
      response = await fetchFn(url, init);
    } catch (err) {
      throw withCause(
        new IoError(
          `TensogramFile.fromUrl: ${describeError(err)}`,
          `TensogramFile.fromUrl: ${describeError(err)}`,
        ),
        err,
      );
    }
    if (!response.ok) {
      throw new IoError(
        `TensogramFile.fromUrl: HTTP ${response.status} ${response.statusText || ''}`.trim(),
        `TensogramFile.fromUrl: HTTP ${response.status} ${response.statusText || ''}`.trim(),
      );
    }
    const buf = await response.arrayBuffer();
    return new TensogramFile(new Uint8Array(buf), 'remote');
  }

  /**
   * Wrap an already-loaded byte buffer. Useful for tests and for
   * callers that already have the file in memory (e.g. `File.arrayBuffer()`
   * in the browser).
   */
  static fromBytes(bytes: Uint8Array): TensogramFile {
    if (!(bytes instanceof Uint8Array)) {
      throw new InvalidArgumentError(
        'TensogramFile.fromBytes: expected a Uint8Array',
        'TensogramFile.fromBytes: expected a Uint8Array',
      );
    }
    // Defensive copy so later mutation of the caller's buffer doesn't
    // invalidate our scan result.
    return new TensogramFile(new Uint8Array(bytes), 'buffer');
  }

  /** Number of messages indexed from the file. */
  get messageCount(): number {
    return this.#positions.length;
  }

  /** Total byte length of the underlying buffer. */
  get byteLength(): number {
    return this.#bytes.byteLength;
  }

  /** Where these bytes came from. */
  get source(): FileSource {
    return this.#source;
  }

  /** Decode a single message by zero-based index. */
  async message(index: number, options?: DecodeOptions): Promise<DecodedMessage> {
    this.#assertOpen();
    const slice = this.rawMessage(index);
    return decode(slice, options);
  }

  /** Decode only the global metadata for a single message. */
  async messageMetadata(index: number): Promise<GlobalMetadata> {
    this.#assertOpen();
    const slice = this.rawMessage(index);
    return decodeMetadata(slice);
  }

  /** Return the raw bytes of a single message, zero-copy. */
  rawMessage(index: number): Uint8Array {
    this.#assertOpen();
    if (!Number.isInteger(index) || index < 0 || index >= this.#positions.length) {
      throw new ObjectError(
        `message index ${index} out of range (have ${this.#positions.length})`,
        `message index ${index} out of range (have ${this.#positions.length})`,
      );
    }
    const { offset, length } = this.#positions[index];
    return this.#bytes.subarray(offset, offset + length);
  }

  /** Iterate all messages in wire order. */
  async *[Symbol.asyncIterator](): AsyncIterator<DecodedMessage> {
    for (let i = 0; i < this.#positions.length; i++) {
      yield this.message(i);
    }
  }

  /** Release the backing buffer (does not affect already-decoded messages). */
  close(): void {
    this.#closed = true;
  }

  #assertOpen(): void {
    if (this.#closed) {
      throw new InvalidArgumentError(
        'TensogramFile has been closed',
        'TensogramFile has been closed',
      );
    }
  }
}

function describeError(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

/** Attach a `cause` to an error and return it for fluent chaining. */
function withCause<E extends Error>(err: E, cause: unknown): E {
  Object.defineProperty(err, 'cause', { value: cause, configurable: true });
  return err;
}
