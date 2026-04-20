import { useCallback, useEffect, useRef, useState } from 'react';

export interface AnimationPlayerState {
  isPlaying: boolean;
  currentFrameIndex: number;
  speed: number;
  play: () => void;
  pause: () => void;
  stop: () => void;
  setSpeed: (fps: number) => void;
  setFrame: (index: number) => void;
}

/**
 * Manages play/pause/stop state and a setInterval-based timer.
 * Calls onFrameChange on each tick so the parent can trigger data fetching.
 * Loops back to frame 0 when the end is reached.
 */
export function useAnimationPlayer(
  totalFrames: number,
  onFrameChange: (index: number) => void,
): AnimationPlayerState {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [speed, setSpeedState] = useState(2); // fps

  // Keep a ref to onFrameChange so the interval doesn't capture a stale closure
  const onFrameChangeRef = useRef(onFrameChange);
  useEffect(() => {
    onFrameChangeRef.current = onFrameChange;
  }, [onFrameChange]);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearTimer = () => {
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  // Start/restart the interval whenever isPlaying or speed changes
  useEffect(() => {
    if (!isPlaying || totalFrames === 0) {
      clearTimer();
      return;
    }

    intervalRef.current = setInterval(() => {
      setCurrentFrameIndex((prev) => {
        const next = totalFrames > 0 ? (prev + 1) % totalFrames : 0;
        onFrameChangeRef.current(next);
        return next;
      });
    }, 1000 / speed);

    return clearTimer;
  }, [isPlaying, speed, totalFrames]);

  const play = useCallback(() => {
    if (totalFrames > 0) setIsPlaying(true);
  }, [totalFrames]);

  const pause = useCallback(() => {
    setIsPlaying(false);
  }, []);

  const stop = useCallback(() => {
    setIsPlaying(false);
    setCurrentFrameIndex(0);
    onFrameChangeRef.current(0);
  }, []);

  const setSpeed = useCallback((fps: number) => {
    setSpeedState(fps);
  }, []);

  const setFrame = useCallback((index: number) => {
    setCurrentFrameIndex(index);
    onFrameChangeRef.current(index);
  }, []);

  return { isPlaying, currentFrameIndex, speed, play, pause, stop, setSpeed, setFrame };
}
