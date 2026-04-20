import { useEffect } from 'react';

interface Handlers {
  onPlayPause: () => void;
  onStepBack: () => void;
  onStepForward: () => void;
  enabled: boolean;
}

export function useKeyboardShortcuts({
  onPlayPause,
  onStepBack,
  onStepForward,
  enabled,
}: Handlers) {
  useEffect(() => {
    if (!enabled) return;

    const handle = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;

      if (e.key === ' ') {
        e.preventDefault();
        onPlayPause();
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        onStepBack();
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        onStepForward();
      }
    };

    window.addEventListener('keydown', handle);
    return () => window.removeEventListener('keydown', handle);
  }, [enabled, onPlayPause, onStepBack, onStepForward]);
}
