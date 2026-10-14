/**
 * Pure math for the PaperLab toy Transformer encoder layer.
 * No DOM, no React, no randomness other than seeded hashes.
 * Matrices are row-major number[][].
 */

export type Matrix = number[][]

export const MAX_TOKENS = 16
export const D_MODEL = 16
export const N_HEADS = 4
export const D_K = D_MODEL / N_HEADS
export const D_FF = 32
export const VOCAB_SIZE = 1000

// ---------- Tokenizer ----------

export interface TokenizeResult {
  tokens: string[]
  /** Total tokens found before truncation. */
  total: number
  truncated: boolean
}

const TOKEN_RE = /[\p{L}\p{N}]+(?:['’][\p{L}]+)?|[^\s\p{L}\p{N}]/gu

/** Simplified word-level tokenizer: words (with inner apostrophes) and single punctuation marks. */
export function tokenize(text: string, max: number = MAX_TOKENS): TokenizeResult {
  const all = text.match(TOKEN_RE) ?? []
  return { tokens: all.slice(0, max), total: all.length, truncated: all.length > max }
}

// ---------- Seeded randomness ----------

/** FNV-1a 32-bit hash. */
export function hashString(s: string): number {
  let h = 0x811c9dc5
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

/** Mulberry32 PRNG returning floats in [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Approximately standard normal via sum of 12 uniforms (Irwin-Hall). */
function approxNormal(rand: () => number): number {
  let s = 0
  for (let i = 0; i < 12; i++) s += rand()
  return s - 6
}

export function tokenId(token: string): number {
  return hashString(token.toLowerCase()) % VOCAB_SIZE
}

export function embedToken(token: string, dModel: number = D_MODEL): number[] {
  const rand = mulberry32(hashString('emb:' + token.toLowerCase()))
  return Array.from({ length: dModel }, () => approxNormal(rand))
}

export function seededMatrix(name: string, rows: number, cols: number, scale: number): Matrix {
  const rand = mulberry32(hashString('w:' + name))
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => approxNormal(rand) * scale),
  )
}

export function seededVector(name: string, n: number, scale: number): number[] {
  return seededMatrix(name, 1, n, scale)[0]
}

// ---------- Positional encoding (Vaswani et al., section 3.5) ----------

/** PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d)). */
export function positionalEncoding(pos: number, dModel: number = D_MODEL): number[] {
  const out = new Array<number>(dModel)
  for (let k = 0; k < dModel; k++) {
    const i = Math.floor(k / 2)
    const angle = pos / Math.pow(10000, (2 * i) / dModel)
    out[k] = k % 2 === 0 ? Math.sin(angle) : Math.cos(angle)
  }
  return out
}

// ---------- Linear algebra ----------

export function matmul(a: Matrix, b: Matrix): Matrix {
  const n = a.length
  const m = b[0]?.length ?? 0
  const k = b.length
  const out: Matrix = Array.from({ length: n }, () => new Array<number>(m).fill(0))
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < k; p++) {
      const aip = a[i][p]
      for (let j = 0; j < m; j++) out[i][j] += aip * b[p][j]
    }
  }
  return out
}

export function transpose(a: Matrix): Matrix {
  if (a.length === 0) return []
  return a[0].map((_, j) => a.map((row) => row[j]))
}

export function addMatrices(a: Matrix, b: Matrix): Matrix {
  return a.map((row, i) => row.map((v, j) => v + b[i][j]))
}

export function addBias(a: Matrix, bias: number[]): Matrix {
  return a.map((row) => row.map((v, j) => v + bias[j]))
}

export function softmax(row: number[]): number[] {
  if (row.length === 0) return []
  let max = -Infinity
  for (const v of row) if (v > max) max = v
  const exps = row.map((v) => (v === -Infinity ? 0 : Math.exp(v - max)))
  const sum = exps.reduce((s, v) => s + v, 0)
  return exps.map((v) => v / sum)
}

export function layerNorm(x: Matrix, eps = 1e-5): Matrix {
  return x.map((row) => {
    const mean = row.reduce((s, v) => s + v, 0) / row.length
    const variance = row.reduce((s, v) => s + (v - mean) ** 2, 0) / row.length
    const inv = 1 / Math.sqrt(variance + eps)
    return row.map((v) => (v - mean) * inv)
  })
}

export function relu(x: Matrix): Matrix {
  return x.map((row) => row.map((v) => Math.max(0, v)))
}

// ---------- Attention ----------

export interface AttentionResult {
  Q: Matrix
  K: Matrix
  V: Matrix
  /** Scaled scores QK^T / sqrt(d_k), masked entries are -Infinity. */
  scores: Matrix
  weights: Matrix
  output: Matrix
}

/** Scaled dot-product attention for given Q, K, V. */
export function scaledDotProductAttention(
  Q: Matrix,
  K: Matrix,
  V: Matrix,
  causal = false,
): AttentionResult {
  const dk = Q[0]?.length ?? 1
  const raw = matmul(Q, transpose(K))
  const scores = raw.map((row, i) =>
    row.map((v, j) => (causal && j > i ? -Infinity : v / Math.sqrt(dk))),
  )
  const weights = scores.map(softmax)
  return { Q, K, V, scores, weights, output: matmul(weights, V) }
}

export function singleHeadAttention(
  X: Matrix,
  Wq: Matrix,
  Wk: Matrix,
  Wv: Matrix,
  causal = false,
): AttentionResult {
  return scaledDotProductAttention(matmul(X, Wq), matmul(X, Wk), matmul(X, Wv), causal)
}

// ---------- Full toy layer ----------

export interface ModelOptions {
  causal: boolean
}

export interface ModelRun {
  tokens: string[]
  total: number
  truncated: boolean
  ids: number[]
  embeddings: Matrix
  positional: Matrix
  /** embeddings + positional encoding */
  X: Matrix
  heads: AttentionResult[]
  concat: Matrix
  attnOut: Matrix
  /** LayerNorm(X + attention) */
  norm1: Matrix
  ffnHidden: Matrix
  ffnOut: Matrix
  /** LayerNorm(norm1 + FFN(norm1)) */
  final: Matrix
}

export function runModel(text: string, options: ModelOptions = { causal: false }): ModelRun {
  const { tokens, total, truncated } = tokenize(text)
  const embeddings = tokens.map((t) => embedToken(t))
  const positional = tokens.map((_, p) => positionalEncoding(p))
  const X = addMatrices(embeddings, positional)

  const heads: AttentionResult[] = []
  for (let h = 0; h < N_HEADS; h++) {
    const s = 1 / Math.sqrt(D_MODEL)
    heads.push(
      singleHeadAttention(
        X,
        seededMatrix(`Wq${h}`, D_MODEL, D_K, s),
        seededMatrix(`Wk${h}`, D_MODEL, D_K, s),
        seededMatrix(`Wv${h}`, D_MODEL, D_K, s),
        options.causal,
      ),
    )
  }
  const concat = tokens.map((_, i) => heads.flatMap((h) => h.output[i]))
  const Wo = seededMatrix('Wo', D_MODEL, D_MODEL, 1 / Math.sqrt(D_MODEL))
  const attnOut = matmul(concat, Wo)
  const norm1 = layerNorm(addMatrices(X, attnOut))

  const W1 = seededMatrix('W1', D_MODEL, D_FF, 1 / Math.sqrt(D_MODEL))
  const b1 = seededVector('b1', D_FF, 0.1)
  const W2 = seededMatrix('W2', D_FF, D_MODEL, 1 / Math.sqrt(D_FF))
  const b2 = seededVector('b2', D_MODEL, 0.1)
  const ffnHidden = relu(addBias(matmul(norm1, W1), b1))
  const ffnOut = addBias(matmul(ffnHidden, W2), b2)
  const final = layerNorm(addMatrices(norm1, ffnOut))

  return {
    tokens, total, truncated,
    ids: tokens.map(tokenId),
    embeddings, positional, X, heads, concat, attnOut, norm1, ffnHidden, ffnOut, final,
  }
}
