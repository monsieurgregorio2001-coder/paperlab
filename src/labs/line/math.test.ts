import { describe, it, expect } from 'vitest'
import { olsFit, gdFit, trace, mse, mseGradient, mseSurface, predict, linspace, describe as formulas, fill, type Point } from './math'
import { firstMismatch } from '../_shared/close'
import fixture from '../../../tests/fixtures/line.json'

// Every comparison uses 1e-6 (relative above 1). The 20-step GD traces also pass at 1e-6:
// both sides run the same float64 updates on a convex bowl, so rounding differences shrink.
const TOL = 1e-6

describe('line: fixtures from scripts/fixtures.py', () => {
  for (const c of fixture.cases) {
    const points: Point[] = c.points
    it(`OLS matches NumPy lstsq on "${c.name}"`, () => {
      const m = olsFit(points)
      expect(firstMismatch({ ...m, loss: mse(points, m) }, c.ols, TOL)).toBeNull()
    })
    it(`GD trace matches for 20 steps on "${c.name}"`, () => {
      const t = gdFit(points, { lr: c.gd.lr, steps: c.gd.steps, init: c.gd.init })
      expect(t).toHaveLength(21)
      expect(firstMismatch(t, c.gd.trace, TOL)).toBeNull()
    })
  }
  it('MSE surface matches the NumPy grid', () => {
    const s = fixture.surface
    const hand: Point[] = fixture.cases[0].points
    const got = mseSurface(hand, [s.wRange[0], s.wRange[1]], [s.bRange[0], s.bRange[1]], s.n)
    expect(firstMismatch(got, { ws: s.ws, bs: s.bs, loss: s.loss }, TOL)).toBeNull()
    const flat = s.loss.flat()
    expect(got.min).toBeCloseTo(Math.min(...flat), 12)
    expect(got.max).toBeCloseTo(Math.max(...flat), 12)
  })
})

describe('line: properties', () => {
  const pts: Point[] = fixture.cases[1].points
  it('GD converges to the OLS fit', () => {
    const t = gdFit(pts, { lr: 0.1, steps: 2000, init: { w: 0, b: 0 } })
    const end = t[t.length - 1]
    const ols = olsFit(pts)
    expect(end.w).toBeCloseTo(ols.w, 8)
    expect(end.b).toBeCloseTo(ols.b, 8)
  })
  it('the OLS fit has zero gradient', () => {
    const g = mseGradient(pts, olsFit(pts))
    expect(Math.abs(g.dw)).toBeLessThan(1e-12)
    expect(Math.abs(g.db)).toBeLessThan(1e-12)
  })
  it('the gradient matches central differences', () => {
    const m = { w: 0.3, b: -0.2 }
    const g = mseGradient(pts, m)
    const h = 1e-6
    const nw = (mse(pts, { w: m.w + h, b: m.b }) - mse(pts, { w: m.w - h, b: m.b })) / (2 * h)
    const nb = (mse(pts, { w: m.w, b: m.b + h }) - mse(pts, { w: m.w, b: m.b - h })) / (2 * h)
    expect(g.dw).toBeCloseTo(nw, 6)
    expect(g.db).toBeCloseTo(nb, 6)
  })
  it('trace is gdFit', () => {
    const o = { lr: 0.05, steps: 5, init: { w: 1, b: 1 } }
    expect(trace(pts, o)).toEqual(gdFit(pts, o))
  })
  it('a learning rate that is too large diverges and the trace stops once the loss is not finite', () => {
    const t = gdFit(pts, { lr: 10, steps: 5000, init: { w: 0, b: 0 } })
    expect(t.length).toBeLessThan(5001)
    expect(Number.isFinite(t[t.length - 1].loss)).toBe(false)
  })
  it('handles empty, single and vertical point sets', () => {
    expect(olsFit([])).toEqual({ w: 0, b: 0 })
    expect(olsFit([{ x: 2, y: 5 }])).toEqual({ w: 0, b: 5 })
    expect(olsFit([{ x: 1, y: 0 }, { x: 1, y: 4 }])).toEqual({ w: 0, b: 2 })
    expect(mse([], { w: 1, b: 1 })).toBe(0)
    expect(gdFit([], { lr: 0.1, steps: 3, init: { w: 1, b: 2 } }).every((s) => s.w === 1 && s.b === 2)).toBe(true)
  })
  it('predict and linspace', () => {
    expect(predict({ w: 2, b: -1 }, 3)).toBe(5)
    expect(linspace(0, 1, 5)).toEqual([0, 0.25, 0.5, 0.75, 1])
    expect(linspace(4, 9, 1)).toEqual([4])
  })
  it('describe lists formulas with placeholders that fill', () => {
    const f = formulas()
    expect(f.map((x) => x.id)).toEqual(['model', 'loss', 'grad', 'step', 'ols'])
    expect(fill(f[0].template, { w: 0.5, b: 2 })).toBe('ŷ = 0.5 · x + 2')
    expect(fill('{a} {missing}', { a: 1 })).toBe('1 {missing}')
  })
})
