import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './room.css'

const mount = document.getElementById('app')!
createRoot(mount).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

// Crossfade over the static #room-hero shell (DIRECTION 4.5). The shell stays in place until the fade ends, so layout does not move.
const stage = document.getElementById('stage')
const shell = document.getElementById('room-hero')
requestAnimationFrame(() => requestAnimationFrame(() => {
  stage?.classList.add('is-ready')
  shell?.setAttribute('aria-hidden', 'true')
  const done = () => { if (shell) shell.hidden = true }
  shell?.addEventListener('transitionend', done, { once: true })
  window.setTimeout(done, 700)
}))
