import { describe, it, expect } from 'vitest'
import {
  tokenize, softmax, positionalEncoding, runModel, singleHeadAttention, seededMatrix,
  layerNorm, D_MODEL, D_K, N_HEADS, D_FF, MAX_TOKENS, mulberry32, embedToken, tokenId, VOCAB_SIZE,
} from './transformer'

describe('tokenize', () => {
  it('handles empty and whitespace input', () => {
    expect(tokenize('').tokens).toEqual([])
    expect(tokenize('   \n\t ').tokens).toEqual([])
    expect(tokenize('').truncated).toBe(false)
  })
  it('splits words and punctuation', () => {
    expect(tokenize('Hello, world!').tokens).toEqual(['Hello', ',', 'world', '!'])
  })
  it('keeps inner apostrophes', () => {
    expect(tokenize("It's fine").tokens).toEqual(["It's", 'fine'])
  })
  it('handles punctuation only input', () => {
    expect(tokenize('?!...').tokens).toEqual(['?', '!', '.', '.', '.'])
  })
  it('truncates beyond 16 tokens', () => {
    const r = tokenize(Array.from({ length: 30 }, (_, i) => 'w' + i).join(' '))
    expect(r.tokens).toHaveLength(MAX_TOKENS)
    expect(r.total).toBe(30)
    expect(r.truncated).toBe(true)
  })
  it('does not flag exactly 16 tokens', () => {
    const r = tokenize(Array(16).fill('a').join(' '))
    expect(r.truncated).toBe(false)
  })
})

describe('softmax', () => {
  it('rows sum to 1 for random inputs', () => {
    const rand = mulberry32(42)
    for (let t = 0; t < 200; t++) {
      const row = Array.from({ length: 1 + Math.floor(rand() * 16) }, () => (rand() - 0.5) * 100)
      const s = softmax(row).reduce((a, b) => a + b, 0)
      expect(Math.abs(s - 1)).toBeLessThan(1e-6)
    }
  })
  it('is stable for large values', () => {
    const out = softmax([1000, 1000])
    expect(out[0]).toBeCloseTo(0.5, 9)
  })
})

describe('positionalEncoding', () => {
  it('matches the paper formula', () => {
    const d = 16
    for (const pos of [0, 1, 5, 15]) {
      const pe = positionalEncoding(pos, d)
      for (let i = 0; i < d / 2; i++) {
        const angle = pos / Math.pow(10000, (2 * i) / d)
        expect(pe[2 * i]).toBeCloseTo(Math.sin(angle), 12)
        expect(pe[2 * i + 1]).toBeCloseTo(Math.cos(angle), 12)
      }
    }
  })
  it('known values at position 0 and 1', () => {
    const p0 = positionalEncoding(0, 4)
    expect(p0).toEqual([0, 1, 0, 1])
    const p1 = positionalEncoding(1, 4)
    expect(p1[0]).toBeCloseTo(Math.sin(1), 12)
    expect(p1[1]).toBeCloseTo(Math.cos(1), 12)
    expect(p1[2]).toBeCloseTo(Math.sin(0.01), 12)
  })
})

describe('attention', () => {
  const X = Array.from({ length: 5 }, (_, i) => embedToken('t' + i))
  const mk = (n: string) => seededMatrix(n, D_MODEL, D_K, 0.25)
  it('has the correct shapes', () => {
    const r = singleHeadAttention(X, mk('q'), mk('k'), mk('v'))
    expect(r.scores).toHaveLength(5)
    expect(r.scores[0]).toHaveLength(5)
    expect(r.weights).toHaveLength(5)
    expect(r.output).toHaveLength(5)
    expect(r.output[0]).toHaveLength(D_K)
    r.weights.forEach((row) => expect(Math.abs(row.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-6))
  })
  it('causal mask zeroes future positions', () => {
    const r = singleHeadAttention(X, mk('q'), mk('k'), mk('v'), true)
    for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) expect(r.weights[i][j]).toBe(0)
    expect(r.weights[0][0]).toBeCloseTo(1, 9)
  })
})

describe('layerNorm', () => {
  it('gives zero mean and unit variance per row', () => {
    const [row] = layerNorm([[1, 2, 3, 4, 10]])
    const mean = row.reduce((a, b) => a + b, 0) / row.length
    const v = row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length
    expect(mean).toBeCloseTo(0, 9)
    expect(v).toBeCloseTo(1, 3)
  })
})

describe('runModel', () => {
  it('produces consistent shapes', () => {
    const r = runModel('The cat sat on the mat.')
    const n = r.tokens.length
    expect(n).toBe(7)
    expect(r.X).toHaveLength(n)
    expect(r.X[0]).toHaveLength(D_MODEL)
    expect(r.heads).toHaveLength(N_HEADS)
    expect(r.concat[0]).toHaveLength(D_MODEL)
    expect(r.ffnHidden[0]).toHaveLength(D_FF)
    expect(r.final).toHaveLength(n)
    expect(r.final[0]).toHaveLength(D_MODEL)
    r.ids.forEach((id) => { expect(id).toBeGreaterThanOrEqual(0); expect(id).toBeLessThan(VOCAB_SIZE) })
  })
  it('is deterministic', () => {
    expect(runModel('Attention is all you need')).toEqual(runModel('Attention is all you need'))
  })
  it('is case insensitive for ids and embeddings', () => {
    expect(tokenId('Cat')).toBe(tokenId('cat'))
    expect(embedToken('Cat')).toEqual(embedToken('cat'))
  })
  it('handles empty input', () => {
    const r = runModel('')
    expect(r.tokens).toEqual([])
    expect(r.final).toEqual([])
    expect(r.heads[0].weights).toEqual([])
  })
  it('handles a single token', () => {
    const r = runModel('hi')
    expect(r.heads[0].weights).toEqual([[1]])
  })
  it('contains no NaN for long input', () => {
    const r = runModel(Array(40).fill('word').join(' '))
    expect(r.tokens).toHaveLength(16)
    expect(r.final.flat().every(Number.isFinite)).toBe(true)
  })
})
