import { describe, it, expect } from 'vitest'
import {
  loadWeights, trace, traceRnn, traceLstm, evaluate, encode, depths, softmax, argmax, sigmoid,
  describe as formulas, type RnnWeights, type LstmWeights,
} from './math'
import { firstMismatch } from '../_shared/close'
import raw from '../../../tests/fixtures/memory.json'
import weightsJson from '../../../models/memory/weights.json'

interface Fixture {
  trained: { text: string; seq: number[]; depths: number[]; rnn: unknown[]; lstm: unknown[] }[]
  random: { seq: number[]; rnnWeights: RnnWeights; lstmWeights: LstmWeights; rnn: unknown[]; lstm: unknown[] }
  testSet: string[]
  longSet: string[]
}
const F = raw as unknown as Fixture
const random = F.random
const W = loadWeights(weightsJson)

// 1e-6 throughout (relative above 1).
const TOL = 1e-6

describe('memory: fixtures from scripts/fixtures.py', () => {
  for (const c of F.trained) {
    it(`trained RNN trace matches NumPy on "${c.text}"`, () => {
      expect(encode(c.text, W.vocab)).toEqual(c.seq)
      expect(firstMismatch(trace('rnn', W, c.text), c.rnn, TOL)).toBeNull()
    })
    it(`trained LSTM trace and gates match NumPy on "${c.text}"`, () => {
      expect(firstMismatch(trace('lstm', W, c.text), c.lstm, TOL)).toBeNull()
    })
    it(`both trained cells get every depth right on "${c.text}"`, () => {
      expect(depths(c.text)).toEqual(c.depths)
      expect(trace('rnn', W, c.text).map((s) => s.pred)).toEqual(c.depths)
      expect(trace('lstm', W, c.text).map((s) => s.pred)).toEqual(c.depths)
    })
  }
  it('RNN cell matches on small random weights', () => {
    expect(firstMismatch(traceRnn(random.rnnWeights, random.seq), random.rnn, TOL)).toBeNull()
  })
  it('LSTM cell matches on small random weights', () => {
    expect(firstMismatch(traceLstm(random.lstmWeights, random.seq), random.lstm, TOL)).toBeNull()
  })
})

describe('memory: the accuracy in models/memory/README.md', () => {
  for (const kind of ['rnn', 'lstm'] as const) {
    it(`${kind}: test set (${F.testSet.length} × 24 chars) accuracy equals weights.json metrics`, () => {
      expect(evaluate(kind, W, F.testSet)).toEqual(W[kind].metrics.test)
      expect(W[kind].metrics.test).toEqual({ stepAccuracy: 1, sequenceAccuracy: 1 })
    })
    it(`${kind}: long set (${F.longSet.length} × 96 chars) accuracy equals weights.json metrics`, () => {
      expect(evaluate(kind, W, F.longSet)).toEqual(W[kind].metrics.long)
      expect(W[kind].metrics.long).toEqual({ stepAccuracy: 1, sequenceAccuracy: 1 })
    })
  }
  it('the sets have the sizes the README states', () => {
    expect(F.testSet).toHaveLength(500)
    expect(F.longSet).toHaveLength(200)
    expect(F.testSet.every((s) => s.length === 24)).toBe(true)
    expect(F.longSet.every((s) => s.length === 96)).toBe(true)
    expect(weightsJson.testSet).toEqual({ size: 500, length: 24, seed: 12345 })
    expect(weightsJson.longSet).toEqual({ size: 200, length: 96, seed: 12346 })
    expect(weightsJson.seqLen).toBe(24)
  })
  it('weights are rounded to 6 decimals', () => {
    expect(weightsJson.decimals).toBe(6)
    const all = [W.rnn.Wxh, W.rnn.Whh, W.rnn.Why, W.lstm.W, W.lstm.U, W.lstm.Wy].flat(2)
      .concat(W.rnn.bh, W.rnn.by, W.lstm.b, W.lstm.by)
    for (const v of all) expect(Math.abs(v * 1e6 - Math.round(v * 1e6))).toBeLessThan(1e-6)
  })
})

describe('memory: properties', () => {
  it('weights.json is small and has the documented shape', () => {
    expect(JSON.stringify(weightsJson).length).toBeLessThan(50 * 1024)
    expect(W.vocab).toEqual(['(', ')', '.'])
    expect(W.classes).toEqual(['0', '1', '2', '3', '4'])
    expect(W.hidden).toBe(8)
  })
  it('loadWeights rejects bad shapes', () => {
    expect(() => loadWeights(null)).toThrow()
    expect(() => loadWeights({ ...W, hidden: 7 })).toThrow(/rnn\.Wxh/)
    expect(() => loadWeights({ ...W, lstm: { ...W.lstm, b: [1] } })).toThrow(/lstm\.b/)
  })
  it('gates stay in their ranges and the cell follows c = f⊙c₋₁ + i⊙g', () => {
    const t = trace('lstm', W, '((.)(()))(')
    let cPrev = new Array<number>(W.hidden).fill(0)
    for (const s of t) {
      for (const v of [...s.i, ...s.f, ...s.o]) expect(v >= 0 && v <= 1).toBe(true)
      for (const v of s.g) expect(Math.abs(v)).toBeLessThanOrEqual(1)
      s.c.forEach((c, k) => expect(c).toBeCloseTo(s.f[k] * cPrev[k] + s.i[k] * s.g[k], 12))
      s.h.forEach((h, k) => expect(h).toBeCloseTo(s.o[k] * Math.tanh(s.c[k]), 12))
      cPrev = s.c
    }
  })
  it('encodes characters outside the vocab as "."', () => {
    expect(encode('a(b)', W.vocab)).toEqual([2, 0, 2, 1])
    expect(trace('rnn', W, 'x(')[0].char).toBe('.')
  })
  it('empty input gives an empty trace', () => {
    expect(trace('rnn', W, '')).toEqual([])
    expect(trace('lstm', W, '')).toEqual([])
    expect(evaluate('lstm', W, [])).toEqual({ stepAccuracy: 0, sequenceAccuracy: 0 })
  })
  it('helpers', () => {
    expect(softmax([1000, 1000])).toEqual([0.5, 0.5])
    expect(argmax([0.1, 0.7, 0.7])).toBe(1)
    expect(sigmoid(0)).toBe(0.5)
    expect(depths('(()')).toEqual([1, 2, 1])
  })
  it('describe lists the drawer formulas', () => {
    expect(formulas().map((f) => f.id)).toEqual(['rnn', 'out', 'gates', 'cand', 'cell', 'hidden'])
  })
})
