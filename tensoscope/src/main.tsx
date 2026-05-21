import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { ensureInit } from './tensogram'

// Kick off WASM init early -- components that need it will await ensureInit()
// internally, but starting here warms the cache.
ensureInit();

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
