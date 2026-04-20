/** Dialog for opening tensogram files -- local file picker or remote URL. */

import { useState, useRef, type FormEvent } from 'react';
import { useAppStore } from '../../store/useAppStore';

export function FileOpenDialog() {
  const [url, setUrl] = useState('');
  const [busy, setBusy] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const openLocalFile = useAppStore((s) => s.openLocalFile);
  const openUrl = useAppStore((s) => s.openUrl);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setLocalError(null);
    try {
      await openLocalFile(file);
    } catch (err) {
      setLocalError(String(err));
    } finally {
      setBusy(false);
    }
  };

  const handleUrlSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!url.trim()) return;
    setBusy(true);
    setLocalError(null);
    try {
      await openUrl(url.trim());
    } catch (err) {
      setLocalError(String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="file-open-dialog">
      <h2>Open File</h2>

      {/* Local file picker */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".tgm"
        onChange={handleFileChange}
        disabled={busy}
        style={{ display: 'none' }}
      />
      <button
        type="button"
        onClick={() => fileInputRef.current?.click()}
        disabled={busy}
        className="browse-btn"
        style={{ width: '100%', marginBottom: 8 }}
      >
        {busy ? 'Opening...' : 'Choose .tgm file'}
      </button>

      {/* Remote URL */}
      <form onSubmit={handleUrlSubmit} className="file-open-form">
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://... or s3://..."
          disabled={busy}
          className="path-input"
          aria-label="Remote URL"
        />
        <button type="submit" disabled={busy || !url.trim()} className="open-btn">
          {busy ? 'Opening...' : 'Open URL'}
        </button>
      </form>
      {localError && <p className="error-msg">{localError}</p>}
    </div>
  );
}
