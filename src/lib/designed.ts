/**
 * The Attention room's engine (ADR 0009). One attention kernel: scaledDotProductAttention in transformer.ts.
 *
 * Designed heads are true softmax(QK^T / sqrt(d_k)) V. Only the weights are hand-set: W_Q and W_K are written
 * below from named parameters, so a reader can check each head against its stated rule. W_V and W_O stay seeded.
 * The toy embeddings carry labelled designed features (see DESIGNED_FEATURES), as the room says on screen.
 */
import {
  D_K, D_MODEL, N_HEADS, addMatrices, hashString, matmul, mulberry32, positionalEncoding,
  scaledDotProductAttention, seededMatrix, tokenId, tokenize, embedToken,
  type AttentionResult, type Matrix,
} from './transformer'

export type WeightsMode = 'designed' | 'random'

export interface Switches {
  /** Add the sinusoidal positional encoding to the embeddings. */
  position: boolean
  /** Divide scores by sqrt(d_k). */
  scaling: boolean
  /** Hide later words from each word. */
  mask: boolean
}
export const ALL_ON: Switches = { position: true, scaling: true, mask: false }

export interface HeadInfo {
  id: 'previous' | 'same' | 'punctuation' | 'next'
  name: string
  /** The stated rule, in words. */
  rule: string
  /** The stated weights, in words. */
  weights: string
}

/** Embedding layout for designed mode. Dimensions the heads read. */
export const DESIGNED_FEATURES = {
  /** Dimensions 0 to 3 are zero in every embedding, so position can be read cleanly. */
  clearDims: [0, 1, 2, 3],
  /**
   * Dimensions 12 to 15 hold a four-number code of the word, the same for the same word. These dimensions carry
   * almost flat position waves, so the code is read without position leaking in. The second and fourth numbers
   * are opposites, which cancels the small flat part.
   */
  codeDims: [12, 13, 14, 15],
  /** Dimension 8 is 1 for a punctuation mark and 0 for a word. */
  punctuationDim: 8,
  /** Dimension 10 is 1 for every token. */
  oneDim: 10,
  /** The length of the code vector. */
  codeLength: 4,
} as const

export const HEAD_GAIN = { previous: 8, next: 8, same: 1, punctuation: 6 } as const

export const DESIGNED_HEADS: HeadInfo[] = [
  { id: 'previous', name: 'Previous word', rule: 'Each word reads the word just before it.',
    weights: 'W_Q turns the position waves back by one step. W_K copies them.' },
  { id: 'same', name: 'Same word', rule: 'Each word reads copies of itself, including itself.',
    weights: 'W_Q and W_K both copy the four-number word code.' },
  { id: 'punctuation', name: 'Punctuation', rule: 'Every word reads the punctuation marks.',
    weights: 'W_Q copies the all-ones feature. W_K copies the punctuation flag.' },
  { id: 'next', name: 'Next word', rule: 'Each word reads the word just after it.',
    weights: 'W_Q turns the position waves forward by one step. W_K copies them.' },
]

function zeros(r: number, c: number): Matrix {
  return Array.from({ length: r }, () => new Array<number>(c).fill(0))
}

/** The angle per position of the two sine waves in dimensions 0 to 3 (paper section 3.5, i = 0 and 1). */
export function positionFrequencies(): number[] {
  return [0, 1].map((i) => 1 / Math.pow(10000, (2 * i) / D_MODEL))
}

/** Hand-set W_Q and W_K for designed head h (0 to 3). */
export function designedWeights(h: number): { Wq: Matrix; Wk: Matrix } {
  const Wq = zeros(D_MODEL, D_K)
  const Wk = zeros(D_MODEL, D_K)
  const kind = DESIGNED_HEADS[h].id
  if (kind === 'previous' || kind === 'next') {
    const g = HEAD_GAIN[kind]
    const dir = kind === 'previous' ? -1 : 1
    positionFrequencies().forEach((w, f) => {
      const a = 2 * f
      Wk[a][a] = 1
      Wk[a + 1][a + 1] = 1
      // q = k shifted by dir positions: rotate the (sin, cos) pair by dir * w.
      const c = Math.cos(w), s = Math.sin(dir * w)
      Wq[a][a] = g * c
      Wq[a + 1][a] = g * s
      Wq[a][a + 1] = -g * s
      Wq[a + 1][a + 1] = g * c
    })
  } else if (kind === 'same') {
    DESIGNED_FEATURES.codeDims.forEach((d, c) => {
      Wq[d][c] = HEAD_GAIN.same
      Wk[d][c] = 1
    })
  } else {
    Wq[DESIGNED_FEATURES.oneDim][0] = HEAD_GAIN.punctuation
    Wk[DESIGNED_FEATURES.punctuationDim][0] = 1
  }
  return { Wq, Wk }
}

const PUNCT = /^[^\p{L}\p{N}]$/u

/** Toy embedding with designed features. Same word, same vector. */
export function designedEmbedding(token: string): number[] {
  const t = token.toLowerCase()
  const rand = mulberry32(hashString('designed:' + t))
  const out = embedToken(token).map((v) => v * 0.3)
  for (const d of DESIGNED_FEATURES.clearDims) out[d] = 0
  const gauss = () => {
    let s = 0
    for (let i = 0; i < 12; i++) s += rand()
    return s - 6
  }
  const g1 = gauss()
  const code = [gauss(), g1, gauss(), -g1]
  const norm = Math.hypot(...code) || 1
  DESIGNED_FEATURES.codeDims.forEach((d, c) => { out[d] = (DESIGNED_FEATURES.codeLength * code[c]) / norm })
  out[DESIGNED_FEATURES.punctuationDim] = PUNCT.test(token) ? 1 : 0
  out[DESIGNED_FEATURES.oneDim] = 1
  return out
}

export interface AttentionRun {
  mode: WeightsMode
  switches: Switches
  tokens: string[]
  ids: number[]
  total: number
  truncated: boolean
  embeddings: Matrix
  positional: Matrix
  X: Matrix
  heads: AttentionResult[]
  Wq: Matrix[]
  Wk: Matrix[]
  Wv: Matrix[]
  Wo: Matrix
  concat: Matrix
  attnOut: Matrix
}

export function runAttention(text: string, mode: WeightsMode = 'designed', switches: Switches = ALL_ON): AttentionRun {
  const { tokens, total, truncated } = tokenize(text)
  const embeddings = tokens.map((t) => (mode === 'designed' ? designedEmbedding(t) : embedToken(t)))
  const positional = tokens.map((_, p) => positionalEncoding(p))
  const X = switches.position ? addMatrices(embeddings, positional) : embeddings
  const s = 1 / Math.sqrt(D_MODEL)
  const Wq: Matrix[] = [], Wk: Matrix[] = [], Wv: Matrix[] = []
  const heads: AttentionResult[] = []
  for (let h = 0; h < N_HEADS; h++) {
    if (mode === 'designed') {
      const w = designedWeights(h)
      Wq.push(w.Wq); Wk.push(w.Wk)
    } else {
      Wq.push(seededMatrix(`Wq${h}`, D_MODEL, D_K, s)); Wk.push(seededMatrix(`Wk${h}`, D_MODEL, D_K, s))
    }
    Wv.push(seededMatrix(`Wv${h}`, D_MODEL, D_K, s))
    heads.push(scaledDotProductAttention(matmul(X, Wq[h]), matmul(X, Wk[h]), matmul(X, Wv[h]), switches.mask, switches.scaling))
  }
  const concat = tokens.map((_, i) => heads.flatMap((hd) => hd.output[i]))
  const Wo = seededMatrix('Wo', D_MODEL, D_MODEL, s)
  return {
    mode, switches, tokens, ids: tokens.map(tokenId), total, truncated, embeddings, positional, X,
    heads, Wq, Wk, Wv, Wo, concat, attnOut: matmul(concat, Wo),
  }
}

/** Where a word's attention goes: top word other than itself when there is one. */
export function topTarget(row: number[], self: number): number {
  let best = -1
  for (let j = 0; j < row.length; j++) {
    if (j === self && row.length > 1) continue
    if (best < 0 || row[j] > row[best]) best = j
  }
  return best
}

/** Mean weight each word receives from all words: the share that sets its type weight. */
export function received(weights: Matrix): number[] {
  const n = weights.length
  return weights.length ? weights[0].map((_, j) => weights.reduce((s, r) => s + r[j], 0) / n) : []
}
