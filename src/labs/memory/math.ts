/**
 * Lab 6 — Memory. Forward passes of an Elman RNN and an LSTM, step by step.
 * Pure and deterministic. The trained weights come from `models/memory/train.py`
 * (bracket depth task) and are passed in; this file never fetches.
 *
 * Conventions match the training script: one-hot inputs over `vocab`, row-major [out][in] matrices,
 * LSTM gates stacked in the order i, f, g, o, h₀ = c₀ = 0.
 */

import type { Formula } from '../_shared/formula'

export type { Formula }
export { fill } from '../_shared/formula'

export type Matrix = number[][]

export interface RnnWeights {
  /** H × V */
  Wxh: Matrix
  /** H × H */
  Whh: Matrix
  bh: number[]
  /** C × H */
  Why: Matrix
  by: number[]
}

export interface LstmWeights {
  /** 4H × V, rows grouped as i, f, g, o. */
  W: Matrix
  /** 4H × H */
  U: Matrix
  b: number[]
  /** C × H */
  Wy: Matrix
  by: number[]
}

export interface Metrics {
  stepAccuracy: number
  sequenceAccuracy: number
}

export interface MemoryWeights {
  task: string
  vocab: string[]
  classes: string[]
  hidden: number
  rnn: RnnWeights & { metrics: { test: Metrics; long: Metrics } }
  lstm: LstmWeights & { metrics: { test: Metrics; long: Metrics } }
}

export interface RnnStep {
  t: number
  char: string
  input: number
  /** Pre-activation Wxh x + Whh h + bh. */
  z: number[]
  h: number[]
  logits: number[]
  probs: number[]
  pred: number
}

export interface LstmStep {
  t: number
  char: string
  input: number
  /** Input gate. */
  i: number[]
  /** Forget gate. */
  f: number[]
  /** Candidate cell value. */
  g: number[]
  /** Output gate. */
  o: number[]
  c: number[]
  h: number[]
  logits: number[]
  probs: number[]
  pred: number
}

/** σ(z) = 1 / (1 + e^(−z)), computed without overflow. */
export function sigmoid(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z))
  const e = Math.exp(z)
  return e / (1 + e)
}

/** Numerically stable softmax. */
export function softmax(v: readonly number[]): number[] {
  let max = -Infinity
  for (const x of v) if (x > max) max = x
  const e = v.map((x) => Math.exp(x - max))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((x) => x / s)
}

/** Index of the largest value (first one on ties). */
export function argmax(v: readonly number[]): number {
  let k = 0
  for (let i = 1; i < v.length; i++) if (v[i] > v[k]) k = i
  return k
}

/** M·v + b. */
function affine(M: Matrix, v: readonly number[], b: readonly number[]): number[] {
  return M.map((row, j) => {
    let s = b[j]
    for (let i = 0; i < row.length; i++) s += row[i] * v[i]
    return s
  })
}

/** M·onehot(k) + N·h + b, using a column of M for the one-hot input. */
function affineOneHot(M: Matrix, k: number, N: Matrix, h: readonly number[], b: readonly number[]): number[] {
  return N.map((row, j) => {
    let s = b[j] + M[j][k]
    for (let i = 0; i < row.length; i++) s += row[i] * h[i]
    return s
  })
}

/** Maps each character to its vocab index; characters outside the vocab map to `fallback` ('.' by default). */
export function encode(text: string, vocab: readonly string[], fallback = '.'): number[] {
  const fb = vocab.indexOf(fallback)
  return Array.from(text).map((ch) => {
    const k = vocab.indexOf(ch)
    return k >= 0 ? k : fb
  })
}

/** True nesting depth after each character of a bracket string ('(' +1, ')' −1, anything else 0). */
export function depths(text: string): number[] {
  let d = 0
  return Array.from(text).map((ch) => {
    if (ch === '(') d++
    else if (ch === ')') d--
    return d
  })
}

/** One Elman step: h = tanh(Wxh·x + Whh·h₋₁ + bh). */
export function rnnCell(w: RnnWeights, input: number, hPrev: readonly number[]): { z: number[]; h: number[] } {
  const z = affineOneHot(w.Wxh, input, w.Whh, hPrev, w.bh)
  return { z, h: z.map(Math.tanh) }
}

/** One LSTM step: gates from W·x + U·h₋₁ + b, then c = f⊙c₋₁ + i⊙g and h = o⊙tanh(c). */
export function lstmCell(
  w: LstmWeights,
  input: number,
  hPrev: readonly number[],
  cPrev: readonly number[],
): { i: number[]; f: number[]; g: number[]; o: number[]; c: number[]; h: number[] } {
  const H = hPrev.length
  const a = affineOneHot(w.W, input, w.U, hPrev, w.b)
  const i = a.slice(0, H).map(sigmoid)
  const f = a.slice(H, 2 * H).map(sigmoid)
  const g = a.slice(2 * H, 3 * H).map(Math.tanh)
  const o = a.slice(3 * H).map(sigmoid)
  const c = cPrev.map((cp, k) => f[k] * cp + i[k] * g[k])
  const h = c.map((ck, k) => o[k] * Math.tanh(ck))
  return { i, f, g, o, c, h }
}

/** Runs the RNN over a sequence of vocab indices and returns every step. */
export function traceRnn(w: RnnWeights, seq: readonly number[], vocab: readonly string[] = []): RnnStep[] {
  let h = new Array<number>(w.bh.length).fill(0)
  return seq.map((input, t) => {
    const cell = rnnCell(w, input, h)
    h = cell.h
    const logits = affine(w.Why, h, w.by)
    const probs = softmax(logits)
    return { t, char: vocab[input] ?? '', input, z: cell.z, h, logits, probs, pred: argmax(probs) }
  })
}

/** Runs the LSTM over a sequence of vocab indices and returns every step with its four gates. */
export function traceLstm(w: LstmWeights, seq: readonly number[], vocab: readonly string[] = []): LstmStep[] {
  const H = w.U[0]?.length ?? 0
  let h = new Array<number>(H).fill(0)
  let c = new Array<number>(H).fill(0)
  return seq.map((input, t) => {
    const cell = lstmCell(w, input, h, c)
    h = cell.h
    c = cell.c
    const logits = affine(w.Wy, h, w.by)
    const probs = softmax(logits)
    return { t, char: vocab[input] ?? '', input, ...cell, logits, probs, pred: argmax(probs) }
  })
}

/** Traces typed text through the chosen trained cell. */
export function trace(kind: 'rnn', weights: MemoryWeights, text: string): RnnStep[]
export function trace(kind: 'lstm', weights: MemoryWeights, text: string): LstmStep[]
export function trace(kind: 'rnn' | 'lstm', weights: MemoryWeights, text: string): RnnStep[] | LstmStep[] {
  const seq = encode(text, weights.vocab)
  return kind === 'rnn' ? traceRnn(weights.rnn, seq, weights.vocab) : traceLstm(weights.lstm, seq, weights.vocab)
}

/** Per-character and whole-sequence accuracy on bracket strings, comparing `classes[pred]` with `depths`. */
export function evaluate(kind: 'rnn' | 'lstm', weights: MemoryWeights, texts: readonly string[]): Metrics {
  let steps = 0
  let okSteps = 0
  let okSeqs = 0
  for (const text of texts) {
    const truth = depths(text)
    const preds = (kind === 'rnn' ? trace('rnn', weights, text) : trace('lstm', weights, text)).map((s) => s.pred)
    let all = true
    preds.forEach((p, t) => {
      steps++
      if (weights.classes[p] === String(truth[t])) okSteps++
      else all = false
    })
    if (all) okSeqs++
  }
  return { stepAccuracy: steps ? okSteps / steps : 0, sequenceAccuracy: texts.length ? okSeqs / texts.length : 0 }
}

// ---------- validation of fetched JSON ----------

function isNumArray(v: unknown, n?: number): v is number[] {
  return Array.isArray(v) && (n === undefined || v.length === n) && v.every((x) => typeof x === 'number' && Number.isFinite(x))
}

function isMatrix(v: unknown, rows: number, cols: number): v is Matrix {
  return Array.isArray(v) && v.length === rows && v.every((r) => isNumArray(r, cols))
}

function isMetrics(v: unknown): v is Metrics {
  if (typeof v !== 'object' || v === null) return false
  const m = v as Record<string, unknown>
  return typeof m.stepAccuracy === 'number' && typeof m.sequenceAccuracy === 'number'
}

function field(o: Record<string, unknown>, k: string): unknown {
  return o[k]
}

/** Checks shapes of fetched weights JSON and returns it typed, or throws with the first problem. */
export function loadWeights(json: unknown): MemoryWeights {
  const fail = (msg: string): never => {
    throw new Error('memory weights: ' + msg)
  }
  if (typeof json !== 'object' || json === null) return fail('not an object')
  const o = json as Record<string, unknown>
  const vocab = field(o, 'vocab')
  const classes = field(o, 'classes')
  const H = field(o, 'hidden')
  if (!Array.isArray(vocab) || !vocab.every((s) => typeof s === 'string')) return fail('vocab')
  if (!Array.isArray(classes) || !classes.every((s) => typeof s === 'string')) return fail('classes')
  if (typeof H !== 'number' || !Number.isInteger(H) || H < 1) return fail('hidden')
  const V = vocab.length
  const C = classes.length
  const rnn = field(o, 'rnn')
  const lstm = field(o, 'lstm')
  if (typeof rnn !== 'object' || rnn === null) return fail('rnn')
  if (typeof lstm !== 'object' || lstm === null) return fail('lstm')
  const r = rnn as Record<string, unknown>
  const l = lstm as Record<string, unknown>
  if (!isMatrix(r.Wxh, H, V)) fail('rnn.Wxh')
  if (!isMatrix(r.Whh, H, H)) fail('rnn.Whh')
  if (!isNumArray(r.bh, H)) fail('rnn.bh')
  if (!isMatrix(r.Why, C, H)) fail('rnn.Why')
  if (!isNumArray(r.by, C)) fail('rnn.by')
  if (!isMatrix(l.W, 4 * H, V)) fail('lstm.W')
  if (!isMatrix(l.U, 4 * H, H)) fail('lstm.U')
  if (!isNumArray(l.b, 4 * H)) fail('lstm.b')
  if (!isMatrix(l.Wy, C, H)) fail('lstm.Wy')
  if (!isNumArray(l.by, C)) fail('lstm.by')
  for (const [name, m] of [['rnn', r.metrics], ['lstm', l.metrics]] as const) {
    if (typeof m !== 'object' || m === null) fail(name + '.metrics')
    const mm = m as Record<string, unknown>
    if (!isMetrics(mm.test) || !isMetrics(mm.long)) fail(name + '.metrics')
  }
  return json as MemoryWeights
}

/** Formulas for the maths drawer, with `{name}` placeholders. */
export function describe(): Formula[] {
  return [
    { id: 'rnn', label: 'RNN step', template: 'hₜ = tanh(Wxh xₜ + Whh hₜ₋₁ + b)   (t = {t}, x = "{char}")' },
    { id: 'out', label: 'Read-out', template: 'yₜ = Why hₜ + by;  p = softmax(yₜ);  depth = {pred} with p = {p}' },
    { id: 'gates', label: 'LSTM gates', template: 'i = σ(Wᵢ x + Uᵢ h + bᵢ),  f = σ(W_f x + U_f h + b_f),  o = σ(Wₒ x + Uₒ h + bₒ)' },
    { id: 'cand', label: 'Candidate', template: 'g = tanh(W_g x + U_g h + b_g)' },
    { id: 'cell', label: 'Cell update', template: 'cₜ = f ⊙ cₜ₋₁ + i ⊙ g' },
    { id: 'hidden', label: 'Hidden state', template: 'hₜ = o ⊙ tanh(cₜ)' },
  ]
}
