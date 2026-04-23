/** Welcome modal shown on first load before a file is opened. */

import { useState, useRef, type FormEvent } from 'react';
import { useAppStore } from '../../store/useAppStore';
import logo from '../../assets/tensogram-logo.png';

export function WelcomeModal() {
  const [url, setUrl] = useState('');
  const [busy, setBusy] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const openLocalFile = useAppStore((s) => s.openLocalFile);
  const openUrl = useAppStore((s) => s.openUrl);
  const storeError = useAppStore((s) => s.error);
  const storeLoading = useAppStore((s) => s.loading);

  const displayError = localError ?? storeError;
  const isBusy = busy || storeLoading;

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
    <div className="modal-backdrop">
      <div className="modal-content welcome-modal">
        <div className="welcome-logo-area">
          <img src={logo} alt="Tensogram" className="welcome-logo" />
        </div>
        <div className="welcome-body">
          <p className="welcome-hint">Open a <code>.tgm</code> file to get started.</p>

          <input
            ref={fileInputRef}
            type="file"
            accept=".tgm"
            onChange={handleFileChange}
            disabled={isBusy}
            style={{ display: 'none' }}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isBusy}
            className="open-btn welcome-file-btn"
          >
            {isBusy ? 'Opening...' : 'Choose .tgm file'}
          </button>

          <div className="welcome-divider">
            <span>or</span>
          </div>

          <form onSubmit={handleUrlSubmit} className="file-open-form">
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https:// URL (use a presigned URL for private S3/GCS/Azure)"
              disabled={isBusy}
              className="path-input"
              aria-label="Remote URL"
            />
            <button type="submit" disabled={isBusy || !url.trim()} className="open-btn">
              {isBusy ? 'Opening...' : 'Open'}
            </button>
          </form>
          {displayError && <p className="error-msg">{displayError}</p>}
        </div>
      </div>
    </div>
  );
}
