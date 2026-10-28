import { describe, it, expect } from 'vitest'
import attentionFixture from '../../tests/fixtures/attention.json'
import { ALL_ON, DESIGNED_FEATURES, DESIGNED_HEADS, designedEmbedding, designedWeights, runAttention, topTarget } from './designed'
import { D_K, D_MODEL, N_HEADS, runModel, scaledDotProductAttention, type Matrix } from './transformer'
import { PRESETS } from '../attention/content'
import { firstMismatch } from '../labs/_shared/close'

const sum = (r: number[]) => r.reduce((a, b) => a + b, 0)

/** Plain loops, written without the kernel: softmax(Q K^T / sqrt(d_k)) V for a stated W_Q and W_K. */
function reference(X: Matrix, Wq: Matrix, Wk: Matrix, Wv: Matrix, causal: boolean, scale: boolean) {
  const n = X.length
  const proj = (W: Matrix) => X.map((x) => W[0].map((_, c) => x.reduce((s, v, r) => s + v * W[r][c], 0)))
  const Q = proj(Wq), K = proj(Wk), V = proj(Wv)
  const weights: number[][] = []
  const output: number[][] = []
  for (let i = 0; i < n; i++) {
    const s: number[] = []
    for (let j = 0; j < n; j++) {
      if (causal && j > i) { s.push(-Infinity); continue }
      let d = 0
      for (let c = 0; c < D_K; c++) d += Q[i][c] * K[j][c]
      s.push(scale ? d / Math.sqrt(D_K) : d)
    }
    const m = Math.max(...s)
    const e = s.map((v) => Math.exp(v - m))
    const z = sum(e)
    weights.push(e.map((v) => v / z))
    output.push(V[0].map((_, c) => weights[i].reduce((a, w, j) => a + w * V[j][c], 0)))
  }
  return { weights, output }
}

const SENTENCES = [...PRESETS, 'A', '?!', 'the the the the', 'One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen']

describe('designed heads equal the kernel for their stated weights', () => {
  for (const text of SENTENCES) {
    for (const mask of [false, true]) {
      for (const scaling of [true, false]) {
        it(`${JSON.stringify(text.slice(0, 24))} mask=${mask} scaling=${scaling}`, () => {
          const run = runAttention(text, 'designed', { position: true, scaling, mask })
          for (let h = 0; h < N_HEADS; h++) {
            const { Wq, Wk } = designedWeights(h)
            const ref = reference(run.X, Wq, Wk, run.Wv[h], mask, scaling)
            expect(firstMismatch(run.heads[h].weights, ref.weights, 1e-9)).toBeNull()
            expect(firstMismatch(run.heads[h].output, ref.output, 1e-9)).toBeNull()
          }
        })
      }
    }
  }
})

describe('each designed head does what it says', () => {
  it('head 1 puts at least 0.5 on the previous word in every preset', () => {
    for (const text of PRESETS) {
      for (const mask of [false, true]) {
        const run = runAttention(text, 'designed', { ...ALL_ON, mask })
        run.heads[0].weights.forEach((row, i) => {
          if (i > 0) expect(row[i - 1], `${text} word ${i} mask ${mask}`).toBeGreaterThanOrEqual(0.5)
        })
      }
    }
  })
  it('head 4 puts at least 0.5 on the next word in every preset when later words are visible', () => {
    for (const text of PRESETS) {
      const run = runAttention(text, 'designed', ALL_ON)
      run.heads[3].weights.forEach((row, i, rows) => {
        if (i < rows.length - 1) expect(row[i + 1], `${text} word ${i}`).toBeGreaterThanOrEqual(0.5)
      })
    }
  })
  it('head 2 puts its largest weight on the word itself or an identical word', () => {
    for (const text of PRESETS) {
      const run = runAttention(text, 'designed', ALL_ON)
      run.heads[1].weights.forEach((row, i) => {
        const best = topTarget(row, -1)
        expect(run.tokens[best].toLowerCase(), `${text} word ${i}`).toBe(run.tokens[i].toLowerCase())
      })
    }
  })
  it('head 3 puts most of its attention on punctuation when the sentence has some', () => {
    for (const text of PRESETS) {
      const run = runAttention(text, 'designed', ALL_ON)
      const punct = run.tokens.map((t) => /^[^\p{L}\p{N}]$/u.test(t))
      if (!punct.some(Boolean)) continue
      run.heads[2].weights.forEach((row) => {
        const share = sum(row.filter((_, j) => punct[j]))
        expect(share).toBeGreaterThan(punct.filter(Boolean).length / punct.length)
      })
    }
  })
  it('the stated weights read only the dimensions the rule names', () => {
    const rowsUsed = (W: Matrix) => W.map((r, i) => (r.some((v) => v !== 0) ? i : -1)).filter((i) => i >= 0)
    const { codeDims, punctuationDim, oneDim } = DESIGNED_FEATURES
    expect(rowsUsed(designedWeights(0).Wk)).toEqual([0, 1, 2, 3])
    expect(rowsUsed(designedWeights(0).Wq)).toEqual([0, 1, 2, 3])
    expect(rowsUsed(designedWeights(3).Wq)).toEqual([0, 1, 2, 3])
    expect(rowsUsed(designedWeights(1).Wq)).toEqual([...codeDims])
    expect(rowsUsed(designedWeights(1).Wk)).toEqual([...codeDims])
    expect(rowsUsed(designedWeights(2).Wq)).toEqual([oneDim])
    expect(rowsUsed(designedWeights(2).Wk)).toEqual([punctuationDim])
  })
  it('names four heads, each with a rule and weights in words', () => {
    expect(DESIGNED_HEADS.map((h) => h.name)).toEqual(['Previous word', 'Same word', 'Punctuation', 'Next word'])
    for (const h of DESIGNED_HEADS) { expect(h.rule.length).toBeGreaterThan(10); expect(h.weights.length).toBeGreaterThan(10) }
  })
  it('the same word has the same embedding, a different word does not', () => {
    expect(designedEmbedding('Cat')).toEqual(designedEmbedding('cat'))
    expect(designedEmbedding('cat')).not.toEqual(designedEmbedding('mat'))
    expect(DESIGNED_FEATURES.clearDims.every((d) => designedEmbedding('cat')[d] === 0)).toBe(true)
  })
})

describe('the switches break what they should', () => {
  it('without position, head 1 no longer finds the previous word', () => {
    const run = runAttention('The cat sat on the mat.', 'designed', { ...ALL_ON, position: false })
    expect(run.heads[0].weights[2][1]).toBeLessThan(0.5)
  })
  it('without the scaling, a head is sharper', () => {
    const on = runAttention('The cat sat on the mat.', 'designed', ALL_ON).heads[0].weights[2]
    const off = runAttention('The cat sat on the mat.', 'designed', { ...ALL_ON, scaling: false }).heads[0].weights[2]
    expect(Math.max(...off)).toBeGreaterThan(Math.max(...on))
  })
  it('with the mask, no word reads a later word', () => {
    const run = runAttention('The cat sat on the mat.', 'designed', { ...ALL_ON, mask: true })
    for (const hd of run.heads) hd.weights.forEach((row, i) => row.forEach((w, j) => { if (j > i) expect(w).toBe(0) }))
  })
  it('every row of every head sums to 1', () => {
    for (const text of SENTENCES) for (const mask of [false, true]) for (const scaling of [true, false]) for (const position of [true, false]) {
      const run = runAttention(text, 'designed', { position, scaling, mask })
      for (const hd of run.heads) for (const row of hd.weights) expect(Math.abs(sum(row) - 1)).toBeLessThan(1e-9)
    }
  })
})

describe('random weights mode is the first version engine', () => {
  it('matches runModel head for head', () => {
    for (const mask of [false, true]) {
      const a = runAttention('The cat sat on the mat.', 'random', { ...ALL_ON, mask })
      const b = runModel('The cat sat on the mat.', { causal: mask })
      expect(firstMismatch(a.heads.map((h) => h.weights), b.heads.map((h) => h.weights), 1e-12)).toBeNull()
      expect(firstMismatch(a.attnOut, b.attnOut, 1e-12)).toBeNull()
    }
    expect(D_MODEL).toBe(16)
  })
})

describe('NumPy fixture fidelity at 1e-6', () => {
  const fx = attentionFixture as unknown as {
    cases: Array<{ head: number; n: number; causal: boolean; scale: boolean; X: Matrix; Wv: Matrix; Wq: Matrix; Wk: Matrix; scores: (number | null)[][]; weights: Matrix; output: Matrix }>
  }
  it('has cases for every head', () => {
    expect(new Set(fx.cases.map((c) => c.head)).size).toBe(4)
    expect(fx.cases.length).toBeGreaterThan(20)
  })
  it('the stated W_Q and W_K are the ones NumPy builds from the rules', () => {
    for (const c of fx.cases) {
      const w = designedWeights(c.head)
      expect(firstMismatch(w.Wq, c.Wq, 1e-6)).toBeNull()
      expect(firstMismatch(w.Wk, c.Wk, 1e-6)).toBeNull()
    }
  })
  it('the kernel matches NumPy for scores, weights, and output', () => {
    for (const c of fx.cases) {
      const mm = (A: Matrix, B: Matrix) => A.map((r) => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)))
      const r = scaledDotProductAttention(mm(c.X, c.Wq), mm(c.X, c.Wk), mm(c.X, c.Wv), c.causal, c.scale)
      const scores = r.scores.map((row) => row.map((v) => (v === -Infinity ? null : v)))
      expect(firstMismatch(scores.map((row) => row.map((v) => v ?? 0)), c.scores.map((row) => row.map((v) => v ?? 0)), 1e-6)).toBeNull()
      expect(firstMismatch(r.weights, c.weights, 1e-6)).toBeNull()
      expect(firstMismatch(r.output, c.output, 1e-6)).toBeNull()
    }
  })
})
