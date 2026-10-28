import { describe, it, expect } from 'vitest'
import {
  buildNetwork, forward, backward, sgdStep, trainEpoch, trace, decisionSurface, datasetLoss, accuracy, penalty,
  spiral, moons, circles, xor, cloneNetwork, predict, describe as formulas,
  type Activation, type Network, type LabeledPoint, type Regularization,
} from './math'
import { firstMismatch } from '../_shared/close'
import raw from '../../../tests/fixtures/net.json'

interface Weights { W: number[][][]; b: number[][] }
interface Case {
  name: string
  sizes: number[]
  hidden: Activation
  output: Activation
  seed: number
  init: Weights
  input: number[]
  label: number
  forward: { z: number[][]; a: number[][] }
  backward: { dW: number[][][]; db: number[][]; delta: number[][]; loss: number }
  sgd: { batch: LabeledPoint[]; lr: number; reg: Regularization; lambda: number; loss: number; gradW: number[][][]; gradB: number[][]; after: Weights & { dead: boolean[][][] } }
}
interface Fixture {
  datasets: Record<'spiral' | 'moons' | 'circles' | 'xor', { n: number; seed: number; noise: number; points: LabeledPoint[] }>
  cases: Case[]
  train: {
    dataset: { n: number; seed: number; noise: number }
    sizes: number[]; hidden: Activation; output: Activation; seed: number
    opts: { lr: number; batchSize: number; reg: Regularization; lambda: number }
    steps: number
    history: { loss: number; acc: number }[]
    final: Weights
  }
  surface: { bounds: { xMin: number; xMax: number; yMin: number; yMax: number }; n: number; xs: number[]; ys: number[]; out: number[][] }
}
const F = raw as unknown as Fixture

// 1e-6 throughout (relative above 1). The 3-epoch training run (12 SGD steps) also passes at 1e-6.
const TOL = 1e-6

const gens = { spiral, moons, circles, xor }

describe('net: fixtures from scripts/fixtures.py', () => {
  for (const [name, d] of Object.entries(F.datasets) as [keyof typeof gens, Fixture['datasets']['spiral']][]) {
    it(`${name} reproduces the NumPy port of the generator`, () => {
      expect(firstMismatch(gens[name](d.n, d.seed, d.noise), d.points, TOL)).toBeNull()
    })
  }
  for (const c of F.cases) {
    describe(c.name, () => {
      const net = buildNetwork(c.sizes, { hidden: c.hidden, output: c.output, seed: c.seed })
      it('initial weights match the seeded Xavier init', () => {
        expect(firstMismatch({ W: net.W, b: net.b }, c.init, TOL)).toBeNull()
      })
      it('forward matches every layer', () => {
        expect(firstMismatch(forward(net, c.input), c.forward, TOL)).toBeNull()
      })
      it('backward matches every weight gradient and delta', () => {
        const t = c.output === 'tanh' ? 2 * c.label - 1 : c.label
        const g = backward(net, forward(net, c.input), [t])
        expect(firstMismatch(g, c.backward, TOL)).toBeNull()
      })
      it(`one SGD step (${c.sgd.reg}, λ = ${c.sgd.lambda}) matches`, () => {
        const r = sgdStep(net, c.sgd.batch, { lr: c.sgd.lr, reg: c.sgd.reg, lambda: c.sgd.lambda })
        expect(firstMismatch(r.loss, c.sgd.loss, TOL)).toBeNull()
        expect(firstMismatch(r.grads, { dW: c.sgd.gradW, db: c.sgd.gradB }, TOL)).toBeNull()
        expect(firstMismatch({ W: r.net.W, b: r.net.b, dead: r.net.dead }, c.sgd.after, TOL)).toBeNull()
      })
    })
  }
  it('the L1 case marks dead weights, which then stay at 0', () => {
    const c = F.cases.find((x) => x.sgd.reg === 'l1')
    if (!c) throw new Error('no L1 case')
    const net = buildNetwork(c.sizes, { hidden: c.hidden, output: c.output, seed: c.seed })
    const o = { lr: c.sgd.lr, reg: c.sgd.reg, lambda: c.sgd.lambda }
    const one = sgdStep(net, c.sgd.batch, o).net
    const deadCount = one.dead.flat(2).filter(Boolean).length
    expect(deadCount).toBeGreaterThan(0)
    const two = sgdStep(one, c.sgd.batch, { ...o, reg: 'none' as const }).net
    one.dead.forEach((m, l) => m.forEach((r, j) => r.forEach((d, i) => { if (d) expect(two.W[l][j][i]).toBe(0) })))
  })
  it('three training steps on a spiral match, loss and accuracy included', () => {
    const t = F.train
    const data = spiral(t.dataset.n, t.dataset.seed, t.dataset.noise)
    const net = buildNetwork(t.sizes, { hidden: t.hidden, output: t.output, seed: t.seed })
    const tr = trace(net, data, t.opts, t.steps)
    expect(firstMismatch(tr.map((s) => ({ loss: s.loss, acc: s.acc })), t.history, TOL)).toBeNull()
    const end = tr[tr.length - 1].net
    expect(firstMismatch({ W: end.W, b: end.b }, t.final, TOL)).toBeNull()
  })
  it('decisionSurface matches the NumPy grid', () => {
    const t = F.train
    const data = spiral(t.dataset.n, t.dataset.seed, t.dataset.noise)
    let net = buildNetwork(t.sizes, { hidden: t.hidden, output: t.output, seed: t.seed })
    for (let s = 0; s < t.steps; s++) net = trainEpoch(net, data, t.opts)
    const s = F.surface
    expect(firstMismatch(decisionSurface(net, s.bounds, s.n), { xs: s.xs, ys: s.ys, out: s.out }, TOL)).toBeNull()
  })
})

/** Central-difference check of every weight and bias gradient for one example. */
function gradCheck(net: Network, input: number[], target: number): number {
  const g = backward(net, forward(net, input), [target])
  const loss = (n: Network) => backward(n, forward(n, input), [target]).loss
  const h = 1e-6
  let worst = 0
  net.W.forEach((m, l) => m.forEach((r, j) => r.forEach((_, i) => {
    const p = cloneNetwork(net)
    const q = cloneNetwork(net)
    p.W[l][j][i] += h
    q.W[l][j][i] -= h
    worst = Math.max(worst, Math.abs((loss(p) - loss(q)) / (2 * h) - g.dW[l][j][i]))
  })))
  net.b.forEach((r, l) => r.forEach((_, j) => {
    const p = cloneNetwork(net)
    const q = cloneNetwork(net)
    p.b[l][j] += h
    q.b[l][j] -= h
    worst = Math.max(worst, Math.abs((loss(p) - loss(q)) / (2 * h) - g.db[l][j]))
  }))
  return worst
}

describe('net: properties', () => {
  const combos: [Activation, Activation][] = [['tanh', 'sigmoid'], ['relu', 'tanh'], ['sigmoid', 'linear'], ['linear', 'sigmoid']]
  for (const [hidden, output] of combos) {
    it(`backprop matches central differences (${hidden} hidden, ${output} out)`, () => {
      const net = buildNetwork([2, 5, 4, 1], { hidden, output, seed: 3 })
      expect(gradCheck(net, [0.37, -0.81], output === 'tanh' ? -1 : 1)).toBeLessThan(1e-6)
    })
  }
  it('learns moons to at least 0.95 training accuracy in 300 steps', () => {
    const data = moons(200, 5, 0.05)
    const net = buildNetwork([2, 8, 8, 1], { seed: 2 })
    const tr = trace(net, data, { lr: 0.3, batchSize: 10 }, 300)
    expect(tr[tr.length - 1].acc).toBeGreaterThanOrEqual(0.95)
    expect(tr[tr.length - 1].loss).toBeLessThan(tr[0].loss)
  })
  it('L2 shrinks the weight penalty compared with no regularisation', () => {
    const data = circles(100, 1, 0.05)
    const net = buildNetwork([2, 6, 1], { seed: 9 })
    let a = net
    let b = net
    for (let s = 0; s < 50; s++) {
      a = trainEpoch(a, data, { lr: 0.3, batchSize: 10 })
      b = trainEpoch(b, data, { lr: 0.3, batchSize: 10, reg: 'l2', lambda: 0.05 })
    }
    expect(penalty(b, 'l2', 1)).toBeLessThan(penalty(a, 'l2', 1))
  })
  it('never mutates its inputs', () => {
    const net = buildNetwork([2, 4, 1], { seed: 1 })
    const snap = JSON.stringify(net)
    const data = xor(20, 1)
    sgdStep(net, data, { lr: 0.5, reg: 'l1', lambda: 0.1 })
    trainEpoch(net, data, { lr: 0.5, batchSize: 5 })
    expect(JSON.stringify(net)).toBe(snap)
  })
  it('rejects bad sizes and bce without a sigmoid output', () => {
    expect(() => buildNetwork([2])).toThrow()
    expect(() => buildNetwork([2, 0, 1])).toThrow()
    const net = buildNetwork([2, 3, 1], { output: 'tanh' })
    expect(() => backward(net, forward(net, [0, 0]), [1], 'bce')).toThrow()
  })
  it('datasets are deterministic and inside [-1.2, 1.2]; spiral, moons and circles alternate labels', () => {
    for (const g of [spiral, moons, circles, xor]) {
      const d = g(101, 7, 0.1)
      expect(d).toEqual(g(101, 7, 0.1))
      expect(d).toHaveLength(101)
      if (g !== xor) expect(Math.abs(d.filter((p) => p.label === 1).length - 50.5)).toBe(0.5)
      for (const p of d) {
        expect(Math.abs(p.x)).toBeLessThanOrEqual(1.2)
        expect(Math.abs(p.y)).toBeLessThanOrEqual(1.2)
      }
    }
  })
  it('accuracy, loss and predict agree on the output', () => {
    const net = buildNetwork([2, 3, 1], { seed: 5 })
    const data = xor(30, 6)
    expect(datasetLoss(net, [])).toBe(0)
    expect(accuracy(net, [])).toBe(0)
    const manual = data.filter((p) => (predict(net, [p.x, p.y])[0] >= 0.5 ? 1 : 0) === p.label).length / data.length
    expect(accuracy(net, data)).toBe(manual)
  })
  it('runs at least 60 training steps per second: 2 → 8×4 → 1, 200 points, batch 10', () => {
    const data = spiral(200, 1, 0.1)
    let net = buildNetwork([2, 8, 8, 8, 8, 1], { seed: 1 })
    const opts = { lr: 0.03, batchSize: 10, reg: 'l2' as const, lambda: 0.001 }
    for (let s = 0; s < 20; s++) net = trainEpoch(net, data, opts) // warm-up
    const steps = 300
    const t0 = performance.now()
    for (let s = 0; s < steps; s++) net = trainEpoch(net, data, opts)
    const perSecond = steps / ((performance.now() - t0) / 1000)
    console.log(`net: ${perSecond.toFixed(0)} training steps/s (200 points, [2,8,8,8,8,1], batch 10)`)
    expect(perSecond).toBeGreaterThanOrEqual(60)
  })
  it('describe lists the drawer formulas', () => {
    expect(formulas().map((f) => f.id)).toEqual(['layer', 'neuron', 'bce', 'mse', 'delta', 'step', 'l1', 'l2'])
  })
})
