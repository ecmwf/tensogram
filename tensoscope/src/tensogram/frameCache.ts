import type { CoordinateData, FieldStats } from './index';

export interface CachedFrame {
  data: Float32Array;
  coordinates: CoordinateData | null;
  stats: FieldStats;
  shape: number[];
}

type FrameKey = string;

function makeKey(msgIdx: number, objIdx: number): FrameKey {
  return `${msgIdx}:${objIdx}`;
}

export class FrameCache {
  private readonly cache = new Map<FrameKey, CachedFrame>();
  private readonly inflight = new Set<FrameKey>();
  private readonly queue: Array<() => Promise<void>> = [];
  private active = 0;
  private generation = 0;

  readonly lookAhead: number;
  readonly maxConcurrent: number;
  private readonly keepBehind = 2;

  constructor(isRemote: boolean) {
    this.lookAhead = isRemote ? 3 : 8;
    this.maxConcurrent = isRemote ? 2 : 4;
  }

  get(msgIdx: number, objIdx: number): CachedFrame | undefined {
    return this.cache.get(makeKey(msgIdx, objIdx));
  }

  put(msgIdx: number, objIdx: number, frame: CachedFrame): void {
    this.cache.set(makeKey(msgIdx, objIdx), frame);
  }

  prefetch(
    frames: ReadonlyArray<{ msgIdx: number; objIdx: number }>,
    currentIndex: number,
    decode: (msgIdx: number, objIdx: number) => Promise<CachedFrame>,
  ): void {
    const gen = this.generation;
    const end = Math.min(frames.length - 1, currentIndex + this.lookAhead);
    for (let i = currentIndex + 1; i <= end; i++) {
      const { msgIdx, objIdx } = frames[i];
      const k = makeKey(msgIdx, objIdx);
      if (this.cache.has(k) || this.inflight.has(k)) continue;
      this.inflight.add(k);
      this.queue.push(async () => {
        try {
          const result = await decode(msgIdx, objIdx);
          if (this.generation === gen) this.cache.set(k, result);
        } finally {
          this.inflight.delete(k);
        }
      });
    }
    this.drain();
    this.evict(frames, currentIndex);
  }

  private drain(): void {
    while (this.active < this.maxConcurrent && this.queue.length > 0) {
      const task = this.queue.shift()!;
      this.active++;
      task().finally(() => {
        this.active--;
        this.drain();
      });
    }
  }

  private evict(
    frames: ReadonlyArray<{ msgIdx: number; objIdx: number }>,
    currentIndex: number,
  ): void {
    const keep = new Set<FrameKey>();
    const lo = Math.max(0, currentIndex - this.keepBehind);
    const hi = Math.min(frames.length - 1, currentIndex + this.lookAhead);
    for (let i = lo; i <= hi; i++) {
      keep.add(makeKey(frames[i].msgIdx, frames[i].objIdx));
    }
    for (const k of this.cache.keys()) {
      if (!keep.has(k)) this.cache.delete(k);
    }
  }

  flush(): void {
    this.generation++;
    this.active = 0;
    this.cache.clear();
    this.inflight.clear();
    this.queue.length = 0;
  }
}

let _instance: FrameCache | null = null;

export function initFrameCache(isRemote: boolean): FrameCache {
  if (_instance) _instance.flush();
  _instance = new FrameCache(isRemote);
  return _instance;
}

export function getFrameCache(): FrameCache | null {
  return _instance;
}
