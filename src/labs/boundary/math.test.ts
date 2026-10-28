import { describe, it, expect } from 'vitest'
import {
  sigmoid, logisticGD, trace, perceptron, decisionSampler, twoBlobs, xor, logLoss, logisticGradient,
  accuracy, boundaryLine, score, describe as formulas, type LabeledPoint, type LinearModel,
} from './math'
import { firstMismatch } from '../_shared/close'
import raw from '../../../tests/fixtures/boundary.json'

interface Fixture {
  datasets: Record<'twoBlobs' | 'twoBlobsWide' | 'xor', { n: number; seed: number; sep?: number; spread?: number; noise?: number; points: LabeledPoint[] }>
  sigmoid: { z: number[]; p: number[] }
  logistic: { dataset: 'twoBlobs' | 'xor'; lr: number; steps: number; trace: unknown[] }[]
  perceptron: { dataset: 'twoBlobsWide'; maxUpdates: number; steps: unknown[]; updates: number; converged: boolean; model: LinearModel }
  sampler: { model: LinearModel; bounds: { xMin: number; xMax: number; yMin: number; yMax: number }; n: number; xs: number[]; ys: number[]; p: number[][] }
}
const F = raw as unknown as Fixture

// 1e-6 throughout (relative above 1). The 50-step logistic traces pass at 1e-6 as well:
// the loss is convex and both sides use float64, so differences stay near machine precision.
const TOL = 1e-6

describe('boundary: fixtures from scripts/fixtures.py', () => {
  it('twoBlobs reproduces the NumPy port of the generator', () => {
    const d = F.datasets.twoBlobs
    expect(firstMismatch(twoBlobs(d.n, d.seed), d.points, TOL)).toBeNull()
    const w = F.datasets.twoBlobsWide
    expect(firstMismatch(twoBlobs(w.n, w.seed, { sep: w.sep, spread: w.spread }), w.points, TOL)).toBeNull()
  })
  it('xor reproduces the NumPy port of the generator', () => {
    const d = F.datasets.xor
    expect(firstMismatch(xor(d.n, d.seed, d.noise), d.points, TOL)).toBeNull()
  })
  it('sigmoid matches, including ±800', () => {
    expect(firstMismatch(F.sigmoid.z.map(sigmoid), F.sigmoid.p, TOL)).toBeNull()
    expect(sigmoid(-800)).toBe(0)
    expect(sigmoid(800)).toBe(1)
  })
  for (const c of F.logistic) {
    it(`logistic GD matches for 50 steps on ${c.dataset}`, () => {
      const t = logisticGD(F.datasets[c.dataset].points, { lr: c.lr, steps: c.steps })
      expect(t).toHaveLength(51)
      expect(firstMismatch(t, c.trace, TOL)).toBeNull()
    })
  }
  it('perceptron makes the same 10 updates', () => {
    const p = F.perceptron
    const r = perceptron(F.datasets[p.dataset].points, { maxUpdates: p.maxUpdates })
    expect(r.updates).toBe(10)
    expect(firstMismatch(r, { steps: p.steps, updates: p.updates, converged: p.converged, model: p.model }, TOL)).toBeNull()
  })
  it('decisionSampler matches the NumPy grid', () => {
    const s = F.sampler
    expect(firstMismatch(decisionSampler(s.model, s.bounds, s.n), { xs: s.xs, ys: s.ys, p: s.p }, TOL)).toBeNull()
  })
})

describe('boundary: properties', () => {
  const blobs = twoBlobs(60, 11)
  it('logistic GD separates two blobs and lowers the loss', () => {
    const t = trace(blobs, { lr: 1, steps: 300 })
    expect(t[t.length - 1].acc).toBe(1)
    expect(t[t.length - 1].loss).toBeLessThan(t[0].loss)
    expect(t[0].loss).toBeCloseTo(Math.LN2, 12)
  })
  it('logistic gradient matches central differences', () => {
    const m: LinearModel = { w: [0.4, -0.3], b: 0.1 }
    const g = logisticGradient(blobs, m)
    const h = 1e-6
    const d = (f: (e: number) => LinearModel) => (logLoss(blobs, f(h)) - logLoss(blobs, f(-h))) / (2 * h)
    expect(g.dw[0]).toBeCloseTo(d((e) => ({ w: [m.w[0] + e, m.w[1]], b: m.b })), 6)
    expect(g.dw[1]).toBeCloseTo(d((e) => ({ w: [m.w[0], m.w[1] + e], b: m.b })), 6)
    expect(g.db).toBeCloseTo(d((e) => ({ w: m.w, b: m.b + e })), 6)
  })
  it('the perceptron converges on separable data and every update follows the rule', () => {
    const r = perceptron(blobs, { maxEpochs: 1000 })
    expect(r.converged).toBe(true)
    expect(accuracy(blobs, r.model)).toBe(1)
    for (const s of r.steps) {
      const p = blobs[s.i]
      const t = p.label === 1 ? 1 : -1
      expect(s.misclassified).toBe(t * score(s.wBefore, p.x, p.y) <= 0)
      const exp = s.misclassified
        ? { w: [s.wBefore.w[0] + t * p.x, s.wBefore.w[1] + t * p.y], b: s.wBefore.b + t }
        : s.wBefore
      expect(firstMismatch(s.wAfter, exp, 1e-12)).toBeNull()
    }
  })
  it('the perceptron never converges on xor and stops at maxEpochs', () => {
    const r = perceptron(xor(40, 2), { maxEpochs: 20 })
    expect(r.converged).toBe(false)
    expect(r.steps).toHaveLength(40 * 20)
  })
  it('traces do not share arrays between steps', () => {
    const t = logisticGD(blobs, { lr: 0.5, steps: 3 })
    expect(t[0].w).not.toBe(t[1].w)
    const r = perceptron(blobs, { maxUpdates: 2 })
    expect(r.steps[0].wAfter.w).not.toBe(r.steps[1].wBefore.w)
  })
  it('generators are deterministic and balanced', () => {
    expect(twoBlobs(10, 4)).toEqual(twoBlobs(10, 4))
    expect(twoBlobs(10, 4)).not.toEqual(twoBlobs(10, 5))
    expect(twoBlobs(10, 4).filter((p) => p.label === 1)).toHaveLength(5)
    for (const p of xor(200, 3)) expect(p.label).toBe(p.x * p.y > 0 ? 1 : 0)
  })
  it('boundaryLine clips s = 0 to the bounds', () => {
    const b = { xMin: -1, xMax: 1, yMin: -1, yMax: 1 }
    const seg = boundaryLine({ w: [1, -1], b: 0 }, b)
    expect(seg).toEqual([{ x: -1, y: -1 }, { x: 1, y: 1 }])
    expect(boundaryLine({ w: [0, 1], b: -0.5 }, b)).toEqual([{ x: -1, y: 0.5 }, { x: 1, y: 0.5 }])
    expect(boundaryLine({ w: [0, 0], b: 1 }, b)).toBeNull()
    expect(boundaryLine({ w: [1, 0], b: -5 }, b)).toBeNull()
  })
  it('handles empty data', () => {
    expect(logLoss([], { w: [1, 1], b: 0 })).toBe(0)
    expect(perceptron([]).converged).toBe(true)
  })
  it('describe lists the drawer formulas', () => {
    expect(formulas().map((f) => f.id)).toEqual(['score', 'sigmoid', 'loss', 'grad', 'step', 'boundary', 'perceptron'])
  })
})
