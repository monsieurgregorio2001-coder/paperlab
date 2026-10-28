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

// Back restores scroll against the short shell, then React makes the page tall and the view jumps. The last height of this route
// is remembered per tab and reserved by the inline script in index.html, so the browser restores scroll against the real height.
// Nothing here scrolls on mount, and history.scrollRestoration stays "auto".
const hKey = () => `pl.h:${location.pathname}${location.hash}`
addEventListener('pagehide', () => {
  try {
    if (stage && shell?.hidden) sessionStorage.setItem(hKey(), String(Math.round(stage.getBoundingClientRect().height)))
  } catch { /* no storage */ }
})
window.setTimeout(() => document.documentElement.style.removeProperty('--stage-h'), 500)
