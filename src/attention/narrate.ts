import { ALL_ON, runAttention, topTarget, type AttentionRun, type Switches, type WeightsMode } from '../lib/designed'

export type SwitchName = 'position' | 'scaling' | 'mask'

const pct = (p: number) => `${Math.round(p * 100)}%`
const q = (t: string) => `‘${t}’`

/** The default query word: the third word, or the last when the sentence is shorter. */
export function defaultWord(n: number): number {
  return Math.min(2, Math.max(0, n - 1))
}

/** Live readout: what the chosen word reads most in the chosen head. */
export function readout(run: AttentionRun, head: number, word: number): string {
  const n = run.tokens.length
  if (n === 0) return 'Type a sentence to see what each word reads.'
  if (n === 1) return `${q(run.tokens[0])} has no other word to read.`
  const row = run.heads[head].weights[word]
  const j = topTarget(row, word)
  return `${q(run.tokens[word])} reads ${q(run.tokens[j])} most.`
}

/** The row-sum line, from the weights on screen. */
export function rowSumLine(run: AttentionRun, head: number): string {
  const rows = run.heads[head].weights
  if (!rows.length) return 'Each row sums to 1.000'
  const worst = rows.reduce((m, r) => Math.max(m, Math.abs(r.reduce((a, b) => a + b, 0) - 1)), 0)
  const total = rows.reduce((a, r) => a + r.reduce((x, y) => x + y, 0), 0) / rows.length
  return worst < 5e-4 ? `Each row sums to ${total.toFixed(3)}` : `Rows sum to between ${(1 - worst).toFixed(3)} and ${(1 + worst).toFixed(3)}`
}

/**
 * The computed sentence for one switch. It compares the switch on with the switch off,
 * with the other switches as they are now, so the numbers are the ones the threads would show.
 */
export function consequence(text: string, mode: WeightsMode, current: Switches, which: SwitchName, head: number, word: number): string {
  const on = runAttention(text, mode, { ...current, [which]: true })
  const off = runAttention(text, mode, { ...current, [which]: false })
  const n = on.tokens.length
  if (n < 2) return 'Type two or more words to see what the switches change.'
  const name = on.tokens[word]
  if (which === 'scaling') {
    const without = Math.max(...off.heads[head].weights[word])
    const withIt = Math.max(...on.heads[head].weights[word])
    return `Without the scaling, ${q(name)} puts ${pct(without)} of its attention on one word. With it, ${pct(withIt)}.`
  }
  if (which === 'position') {
    const rowOn = on.heads[head].weights[word]
    const t = topTarget(rowOn, word)
    return `Without position, ${q(name)} puts ${pct(off.heads[head].weights[word][t])} of its attention on ${q(on.tokens[t])}. With it, ${pct(rowOn[t])}.`
  }
  if (word === n - 1) return `${q(name)} is the last word, so the causal mask changes nothing for it. Pick an earlier word.`
  const later = (r: AttentionRun) => r.heads[head].weights[word].slice(word + 1).reduce((a, b) => a + b, 0)
  return `Without the causal mask, ${q(name)} puts ${pct(later(off))} of its attention on later words. With it, ${pct(later(on))}.`
}

export const SWITCH_COPY: Record<SwitchName, { label: string; hint: string }> = {
  position: { label: 'Position', hint: 'Adds where each word sits' },
  scaling: { label: 'Scaling', hint: 'Divides scores by √d_k' },
  mask: { label: 'Causal mask', hint: 'Hides later words' },
}

export { ALL_ON }
