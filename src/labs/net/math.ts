/**
 * Lab 3 — Net. A small fully connected network for 2D classification.
 * Written from scratch (not derived from TensorFlow Playground's nn.ts); the dead-weight
 * convention under L1 follows the Playground's behaviour.
 * Pure and deterministic: randomness only through seeded `mulberry32`. Functions never
 * mutate their inputs; a training step returns a new network.
 *
 * Indexing: layer l maps sizes[l] inputs to sizes[l + 1] outputs. W[l][j][i] is the weight
 * from neuron i of layer l to neuron j of layer l + 1; b[l][j] is that neuron's bias.
 */

import { mulberry32, uniform } from '../_shared/rand'
import type { Formula } from '../_shared/formula'
import type { Bounds, LabeledPoint } from '../boundary/math'

export type { Formula, Bounds, LabeledPoint }
export { fill } from '../_shared/formula'
export { xor } from '../boundary/math'

export type Activation = 'tanh' | 'relu' | 'sigmoid' | 'linear'
export type Loss = 'bce' | 'mse'
export type Regularization = 'none' | 'l1' | 'l2'

export interface Network {
  /** [inputs, ...hidden, outputs]. */
  sizes: number[]
  hidden: Activation
  output: Activation
  W: number[][][]
  b: number[][]
  /** dead[l][j][i] is true once L1 has pushed W[l][j][i] through zero; it then stays 0. */
  dead: boolean[][][]
}

export interface BuildOptions {
  hidden?: Activation
  output?: Activation
  seed?: number
}

export interface ForwardResult {
  /** z[l] = W[l]·a[l] + b[l], pre-activations of layer l + 1. */
  z: number[][]
  /** a[0] is the input; a[l + 1] = act(z[l]). The last entry is the output. */
  a: number[][]
}

export interface Gradients {
  dW: number[][][]
  db: number[][]
}

export interface BackwardResult extends Gradients {
  /** delta[l][j] = ∂loss/∂z[l][j]. */
  delta: number[][]
  loss: number
}

export interface SgdOptions {
  lr: number
  reg?: Regularization
  /** Regularisation rate λ. */
  lambda?: number
  /** Defaults to 'bce' for a sigmoid output, otherwise 'mse'. */
  loss?: Loss
}

export interface TrainOptions extends SgdOptions {
  batchSize: number
}

export interface SgdResult {
  net: Network
  /** Mean data loss over the batch before the update. */
  loss: number
  /** Mean data gradients over the batch, without regularisation. */
  grads: Gradients
}

export interface TraceStep {
  step: number
  net: Network
  loss: number
  acc: number
}

export interface Surface {
  xs: number[]
  ys: number[]
  /** out[i][j] is the network output at (xs[j], ys[i]). */
  out: number[][]
}

/** Applies an activation to one value. */
export function activate(f: Activation, z: number): number {
  switch (f) {
    case 'tanh':
      return Math.tanh(z)
    case 'relu':
      return z > 0 ? z : 0
    case 'sigmoid':
      return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z))
    case 'linear':
      return z
  }
}

/** Derivative of an activation, from its input z and output a. */
export function activateDer(f: Activation, z: number, a: number): number {
  switch (f) {
    case 'tanh':
      return 1 - a * a
    case 'relu':
      return z > 0 ? 1 : 0
    case 'sigmoid':
      return a * (1 - a)
    case 'linear':
      return 1
  }
}

/** Builds a network with Xavier-uniform weights, W ~ U(±√(6/(fanIn + fanOut))), drawn layer by layer, row by row; biases 0. */
export function buildNetwork(sizes: readonly number[], opts: BuildOptions = {}): Network {
  if (sizes.length < 2 || sizes.some((s) => !Number.isInteger(s) || s < 1)) {
    throw new Error('sizes must list at least two positive integers')
  }
  const rand = mulberry32(opts.seed ?? 1)
  const W: number[][][] = []
  const b: number[][] = []
  const dead: boolean[][][] = []
  for (let l = 0; l < sizes.length - 1; l++) {
    const fanIn = sizes[l]
    const fanOut = sizes[l + 1]
    const r = Math.sqrt(6 / (fanIn + fanOut))
    W.push(Array.from({ length: fanOut }, () => Array.from({ length: fanIn }, () => uniform(rand, -r, r))))
    b.push(new Array<number>(fanOut).fill(0))
    dead.push(Array.from({ length: fanOut }, () => new Array<boolean>(fanIn).fill(false)))
  }
  return { sizes: [...sizes], hidden: opts.hidden ?? 'tanh', output: opts.output ?? 'sigmoid', W, b, dead }
}

/** Deep copy of a network. */
export function cloneNetwork(net: Network): Network {
  return {
    sizes: [...net.sizes],
    hidden: net.hidden,
    output: net.output,
    W: net.W.map((m) => m.map((r) => [...r])),
    b: net.b.map((r) => [...r]),
    dead: net.dead.map((m) => m.map((r) => [...r])),
  }
}

/** Activation used by layer l (the last layer uses the output activation). */
function actOf(net: Network, l: number): Activation {
  return l === net.W.length - 1 ? net.output : net.hidden
}

/** Forward pass returning every layer's pre-activations and activations. */
export function forward(net: Network, input: readonly number[]): ForwardResult {
  const z: number[][] = []
  const a: number[][] = [[...input]]
  for (let l = 0; l < net.W.length; l++) {
    const Wl = net.W[l]
    const prev = a[l]
    const f = actOf(net, l)
    const zl = new Array<number>(Wl.length)
    const al = new Array<number>(Wl.length)
    for (let j = 0; j < Wl.length; j++) {
      const row = Wl[j]
      let s = net.b[l][j]
      for (let i = 0; i < row.length; i++) s += row[i] * prev[i]
      zl[j] = s
      al[j] = activate(f, s)
    }
    z.push(zl)
    a.push(al)
  }
  return { z, a }
}

/** Network output for one input. */
export function predict(net: Network, input: readonly number[]): number[] {
  const { a } = forward(net, input)
  return a[a.length - 1]
}

/** The loss a network trains with when none is given. */
export function defaultLoss(net: Network): Loss {
  return net.output === 'sigmoid' ? 'bce' : 'mse'
}

/** Training target for a 0/1 label: ±1 for a tanh output, otherwise the label itself. */
export function targetFor(net: Network, label: number): number {
  return net.output === 'tanh' ? 2 * label - 1 : label
}

/** Class threshold on the output: 0 for tanh, otherwise 0.5. */
export function threshold(net: Network): number {
  return net.output === 'tanh' ? 0 : 0.5
}

/** Per-output loss: bce = max(z,0) − t·z + ln(1 + e^(−|z|)) (needs a sigmoid output); mse = ½(a − t)². */
function outputLoss(loss: Loss, z: number, a: number, t: number): number {
  if (loss === 'bce') return Math.max(z, 0) - t * z + Math.log1p(Math.exp(-Math.abs(z)))
  return 0.5 * (a - t) * (a - t)
}

/** Backpropagation for one example: per-weight gradients of its loss, plus the deltas. */
export function backward(net: Network, fwd: ForwardResult, target: readonly number[], loss: Loss = defaultLoss(net)): BackwardResult {
  if (loss === 'bce' && net.output !== 'sigmoid') throw new Error('bce loss needs a sigmoid output')
  const L = net.W.length
  const delta: number[][] = new Array<number[]>(L)
  const zL = fwd.z[L - 1]
  const aL = fwd.a[L]
  let total = 0
  delta[L - 1] = zL.map((z, j) => {
    const t = target[j]
    total += outputLoss(loss, z, aL[j], t)
    return loss === 'bce' ? aL[j] - t : (aL[j] - t) * activateDer(net.output, z, aL[j])
  })
  for (let l = L - 2; l >= 0; l--) {
    const next = net.W[l + 1]
    const dn = delta[l + 1]
    delta[l] = fwd.z[l].map((z, i) => {
      let s = 0
      for (let j = 0; j < next.length; j++) s += next[j][i] * dn[j]
      return s * activateDer(net.hidden, z, fwd.a[l + 1][i])
    })
  }
  const dW = net.W.map((Wl, l) => Wl.map((_, j) => fwd.a[l].map((ai) => delta[l][j] * ai)))
  const db = delta.map((d) => [...d])
  return { dW, db, delta, loss: total }
}

/** Mean data loss over labelled 2D points. */
export function datasetLoss(net: Network, data: readonly LabeledPoint[], loss: Loss = defaultLoss(net)): number {
  if (data.length === 0) return 0
  let s = 0
  for (const p of data) {
    const { z, a } = forward(net, [p.x, p.y])
    s += outputLoss(loss, z[z.length - 1][0], a[a.length - 1][0], targetFor(net, p.label))
  }
  return s / data.length
}

/** Share of points whose output is on the label's side of the threshold. */
export function accuracy(net: Network, data: readonly LabeledPoint[]): number {
  if (data.length === 0) return 0
  const th = threshold(net)
  let ok = 0
  for (const p of data) if ((predict(net, [p.x, p.y])[0] >= th ? 1 : 0) === p.label) ok++
  return ok / data.length
}

/** Regularisation penalty: λ Σ|w| for L1, (λ/2) Σ w² for L2, over weights only (not biases). */
export function penalty(net: Network, reg: Regularization, lambda: number): number {
  if (reg === 'none' || lambda === 0) return 0
  let s = 0
  for (const Wl of net.W) for (const row of Wl) for (const w of row) s += reg === 'l1' ? Math.abs(w) : 0.5 * w * w
  return lambda * s
}

/**
 * One SGD step on a mini-batch: w ← w − η(ḡ + λ·r(w)) with r = sign(w) for L1 or w for L2.
 * Under L1 a weight whose update crosses zero is set to 0 and marked dead; dead weights never update again.
 */
export function sgdStep(net: Network, batch: readonly LabeledPoint[], opts: SgdOptions): SgdResult {
  const loss = opts.loss ?? defaultLoss(net)
  const reg = opts.reg ?? 'none'
  const lambda = opts.lambda ?? 0
  const next = cloneNetwork(net)
  const dW = net.W.map((m) => m.map((r) => new Array<number>(r.length).fill(0)))
  const db = net.b.map((r) => new Array<number>(r.length).fill(0))
  let total = 0
  for (const p of batch) {
    const fwd = forward(net, [p.x, p.y])
    const g = backward(net, fwd, [targetFor(net, p.label)], loss)
    total += g.loss
    for (let l = 0; l < dW.length; l++) {
      for (let j = 0; j < dW[l].length; j++) {
        db[l][j] += g.db[l][j]
        for (let i = 0; i < dW[l][j].length; i++) dW[l][j][i] += g.dW[l][j][i]
      }
    }
  }
  const n = batch.length
  if (n === 0) return { net: next, loss: 0, grads: { dW, db } }
  for (let l = 0; l < dW.length; l++) {
    for (let j = 0; j < dW[l].length; j++) {
      db[l][j] /= n
      next.b[l][j] = net.b[l][j] - opts.lr * db[l][j]
      for (let i = 0; i < dW[l][j].length; i++) {
        dW[l][j][i] /= n
        if (net.dead[l][j][i]) continue
        const w = net.W[l][j][i]
        const r = reg === 'l1' ? Math.sign(w) : reg === 'l2' ? w : 0
        let w2 = w - opts.lr * (dW[l][j][i] + lambda * r)
        if (reg === 'l1' && w !== 0 && w * w2 < 0) {
          w2 = 0
          next.dead[l][j][i] = true
        }
        next.W[l][j][i] = w2
      }
    }
  }
  return { net: next, loss: total / n, grads: { dW, db } }
}

/** One training step: a pass over the data in order, in mini-batches of `batchSize`. */
export function trainEpoch(net: Network, data: readonly LabeledPoint[], opts: TrainOptions): Network {
  let cur = net
  const bs = Math.max(1, Math.floor(opts.batchSize))
  for (let s = 0; s < data.length; s += bs) cur = sgdStep(cur, data.slice(s, s + bs), opts).net
  return cur
}

/** Runs `steps` training steps and returns a snapshot after each (entry 0 is the start). */
export function trace(net: Network, data: readonly LabeledPoint[], opts: TrainOptions, steps: number): TraceStep[] {
  const loss = opts.loss ?? defaultLoss(net)
  const out: TraceStep[] = [{ step: 0, net, loss: datasetLoss(net, data, loss), acc: accuracy(net, data) }]
  let cur = net
  for (let s = 1; s <= steps; s++) {
    cur = trainEpoch(cur, data, opts)
    out.push({ step: s, net: cur, loss: datasetLoss(cur, data, loss), acc: accuracy(cur, data) })
  }
  return out
}

/** Linearly spaced values from lo to hi inclusive, or [lo] when n ≤ 1. */
function axis(lo: number, hi: number, n: number): number[] {
  if (n <= 1) return [lo]
  return Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))
}

/** Samples the first output on an n × n grid over the bounds, for the decision terrain. */
export function decisionSurface(net: Network, bounds: Bounds, n: number): Surface {
  const xs = axis(bounds.xMin, bounds.xMax, n)
  const ys = axis(bounds.yMin, bounds.yMax, n)
  return { xs, ys, out: ys.map((y) => xs.map((x) => predict(net, [x, y])[0])) }
}

/**
 * Two interleaved spirals. Point k has label k mod 2 and index i = ⌊k/2⌋;
 * r = i / (n/2), θ = 3.5π·r + π·label, then (r sin θ, r cos θ) plus uniform noise (two draws).
 */
export function spiral(n: number, seed: number, noise = 0): LabeledPoint[] {
  const rand = mulberry32(seed)
  const half = n / 2
  const out: LabeledPoint[] = []
  for (let k = 0; k < n; k++) {
    const label: 0 | 1 = k % 2 === 0 ? 0 : 1
    const r = Math.floor(k / 2) / half
    const t = 3.5 * Math.PI * r + Math.PI * label
    const x = r * Math.sin(t) + uniform(rand, -noise, noise)
    const y = r * Math.cos(t) + uniform(rand, -noise, noise)
    out.push({ x, y, label })
  }
  return out
}

/**
 * Two moons. Point k has label k mod 2 and draws θ = π·u. Label 0: (cos θ, sin θ); label 1: (1 − cos θ, 0.5 − sin θ).
 * Then x ← (x − 0.5)/1.5, y ← (y − 0.25)/1.5 to fit in [−1, 1], plus uniform noise (two draws).
 */
export function moons(n: number, seed: number, noise = 0): LabeledPoint[] {
  const rand = mulberry32(seed)
  const out: LabeledPoint[] = []
  for (let k = 0; k < n; k++) {
    const label: 0 | 1 = k % 2 === 0 ? 0 : 1
    const t = Math.PI * rand()
    const x0 = label === 0 ? Math.cos(t) : 1 - Math.cos(t)
    const y0 = label === 0 ? Math.sin(t) : 0.5 - Math.sin(t)
    const x = (x0 - 0.5) / 1.5 + uniform(rand, -noise, noise)
    const y = (y0 - 0.25) / 1.5 + uniform(rand, -noise, noise)
    out.push({ x, y, label })
  }
  return out
}

/**
 * A disc inside a ring. Point k has label 1 (inner, r ∈ [0, 0.5)) when k is even, else label 0 (ring, r ∈ [0.7, 1)).
 * Draws r then θ ∈ [0, 2π), then uniform noise (two draws).
 */
export function circles(n: number, seed: number, noise = 0): LabeledPoint[] {
  const rand = mulberry32(seed)
  const out: LabeledPoint[] = []
  for (let k = 0; k < n; k++) {
    const label: 0 | 1 = k % 2 === 0 ? 1 : 0
    const r = label === 1 ? uniform(rand, 0, 0.5) : uniform(rand, 0.7, 1)
    const t = uniform(rand, 0, 2 * Math.PI)
    const x = r * Math.sin(t) + uniform(rand, -noise, noise)
    const y = r * Math.cos(t) + uniform(rand, -noise, noise)
    out.push({ x, y, label })
  }
  return out
}

/** Formulas for the maths drawer, with `{name}` placeholders. */
export function describe(): Formula[] {
  return [
    { id: 'layer', label: 'One layer', template: 'a⁽ˡ⁺¹⁾ = {act}(W⁽ˡ⁾ a⁽ˡ⁾ + b⁽ˡ⁾)' },
    { id: 'neuron', label: 'One neuron', template: 'a = {act}(Σᵢ wᵢ xᵢ + b) = {a}' },
    { id: 'bce', label: 'Cross-entropy loss', template: 'L = −(1/{n}) Σ [t ln p + (1 − t) ln(1 − p)] = {loss}' },
    { id: 'mse', label: 'Squared-error loss', template: 'L = (1/{n}) Σ ½ (a − t)² = {loss}' },
    { id: 'delta', label: 'Backpropagation', template: 'δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀ δ⁽ˡ⁺¹⁾) ⊙ {act}′(z⁽ˡ⁾);  ∂L/∂w = δ · a' },
    { id: 'step', label: 'SGD step', template: 'w ← w − {lr} · (∂L/∂w + {lambda} · r(w)),  batch of {batch}' },
    { id: 'l1', label: 'L1 penalty', template: '{lambda} · Σ |w|;  a weight that crosses zero is set to 0 and stays dead' },
    { id: 'l2', label: 'L2 penalty', template: '({lambda}/2) · Σ w²' },
  ]
}
