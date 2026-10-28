import { PRESETS } from './content'
import type { WeightsMode } from '../lib/designed'

export interface UiState {
  text: string
  /** 0 to 3 */
  head: number
  /** Selected word, or null for the default word. */
  word: number | null
  position: boolean
  scaling: boolean
  mask: boolean
  mode: WeightsMode
}

export const DEFAULT_STATE: UiState = {
  text: PRESETS[0], head: 0, word: null, position: true, scaling: true, mask: false, mode: 'designed',
}

const MAX_Q = 200

/** ?q= sentence, head= 1 to 4, w= word number, off= position,scaling, mask=on, weights=random. */
export function parseState(search: string): UiState {
  const p = new URLSearchParams(search)
  const q = p.get('q')
  const head = Number(p.get('head'))
  const w = Number(p.get('w'))
  const off = new Set((p.get('off') ?? '').split(',').map((s) => s.trim()))
  return {
    text: q !== null ? q.slice(0, MAX_Q) : DEFAULT_STATE.text,
    head: Number.isInteger(head) && head >= 1 && head <= 4 ? head - 1 : 0,
    word: Number.isInteger(w) && w >= 1 && w <= 16 ? w - 1 : null,
    position: !off.has('position'),
    scaling: !off.has('scaling'),
    mask: p.get('mask') === 'on',
    mode: p.get('weights') === 'random' ? 'random' : 'designed',
  }
}

export function serializeState(s: UiState): string {
  const p = new URLSearchParams()
  if (s.text !== DEFAULT_STATE.text) p.set('q', s.text)
  if (s.head !== 0) p.set('head', String(s.head + 1))
  if (s.word !== null) p.set('w', String(s.word + 1))
  const off = [!s.position && 'position', !s.scaling && 'scaling'].filter(Boolean)
  if (off.length) p.set('off', off.join(','))
  if (s.mask) p.set('mask', 'on')
  if (s.mode === 'random') p.set('weights', 'random')
  const out = p.toString()
  return out ? `?${out}` : ''
}
