/**
 * Lab 2 — Boundary. Logistic regression and the perceptron on 2D points with two classes.
 * Pure and deterministic: randomness only through seeded `mulberry32`.
 * Labels are 0 or 1; the perceptron maps them to −1 and +1 internally.
 */

import { mulberry32, normal, uniform } from '../_shared/rand'
import type { Formula } from '../_shared/formula'

export type { Formula }
export { fill } from '../_shared/formula'

export interface LabeledPoint {
  x: number
  y: number
  label: 0 | 1
}

/** A linear score s = w₁x + w₂y + b, shared by both models. */
export interface LinearModel {
  w: [number, number]
  b: number
}

export interface Bounds {
  xMin: number
  xMax: number
  yMin: number
  yMax: number
}

export interface LogisticOptions {
  lr: number
  /** Number of updates; the trace has steps + 1 entries (entry 0 is `init`). */
  steps: number
  init?: LinearModel
}

export interface LogisticStep {
  step: number
  w: [number, number]
  b: number
  /** Mean binary cross-entropy at (w, b). */
  loss: number
  /** Share of points on the right side of p = 0.5, in [0, 1]. */
  acc: number
}

export interface PerceptronOptions {
  /** Step size η (1 is the classic rule). */
  lr?: number
  /** Stop after this many weight updates. */
  maxUpdates?: number
  /** Stop after this many passes over the data. */
  maxEpochs?: number
  init?: LinearModel
}

export interface PerceptronStep {
  /** Index of the point visited. */
  i: number
  epoch: number
  /** True when t·s ≤ 0 for this point, so the rule fired. */
  misclassified: boolean
  wBefore: LinearModel
  wAfter: LinearModel
}

export interface PerceptronResult {
  steps: PerceptronStep[]
  updates: number
  /** True when a full pass made no update. */
  converged: boolean
  model: LinearModel
}

export interface ProbabilityGrid {
  xs: number[]
  ys: number[]
  /** p[i][j] = σ(score) at (xs[j], ys[i]). */
  p: number[][]
}

const ZERO: LinearModel = { w: [0, 0], b: 0 }

/** Copies a model so traces never share arrays. */
function copy(m: LinearModel): LinearModel {
  return { w: [m.w[0], m.w[1]], b: m.b }
}

/** s = w₁x + w₂y + b. */
export function score(m: LinearModel, x: number, y: number): number {
  return m.w[0] * x + m.w[1] * y + m.b
}

/** σ(z) = 1 / (1 + e^(−z)), computed without overflow for large |z|. */
export function sigmoid(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z))
  const e = Math.exp(z)
  return e / (1 + e)
}

/** P(label = 1 | x, y) = σ(s). */
export function predictProba(m: LinearModel, x: number, y: number): number {
  return sigmoid(score(m, x, y))
}

/** Mean binary cross-entropy, using the stable form max(s,0) − t·s + ln(1 + e^(−|s|)). */
export function logLoss(points: readonly LabeledPoint[], m: LinearModel): number {
  if (points.length === 0) return 0
  let s = 0
  for (const p of points) {
    const z = score(m, p.x, p.y)
    s += Math.max(z, 0) - p.label * z + Math.log1p(Math.exp(-Math.abs(z)))
  }
  return s / points.length
}

/** Share of points with (s ≥ 0) matching label 1. */
export function accuracy(points: readonly LabeledPoint[], m: LinearModel): number {
  if (points.length === 0) return 0
  let ok = 0
  for (const p of points) if ((score(m, p.x, p.y) >= 0 ? 1 : 0) === p.label) ok++
  return ok / points.length
}

/** Gradient of the mean cross-entropy: (1/n) Σ (pᵢ − tᵢ)·[xᵢ, yᵢ, 1]. */
export function logisticGradient(
  points: readonly LabeledPoint[],
  m: LinearModel,
): { dw: [number, number]; db: number } {
  const n = points.length
  if (n === 0) return { dw: [0, 0], db: 0 }
  let g1 = 0
  let g2 = 0
  let gb = 0
  for (const p of points) {
    const r = predictProba(m, p.x, p.y) - p.label
    g1 += r * p.x
    g2 += r * p.y
    gb += r
  }
  return { dw: [g1 / n, g2 / n], db: gb / n }
}

/** Full-batch gradient descent on logistic regression; returns every step. */
export function logisticGD(points: readonly LabeledPoint[], opts: LogisticOptions): LogisticStep[] {
  let m = copy(opts.init ?? ZERO)
  const out: LogisticStep[] = []
  for (let step = 0; step <= opts.steps; step++) {
    out.push({ step, w: [m.w[0], m.w[1]], b: m.b, loss: logLoss(points, m), acc: accuracy(points, m) })
    if (step === opts.steps) break
    const g = logisticGradient(points, m)
    m = { w: [m.w[0] - opts.lr * g.dw[0], m.w[1] - opts.lr * g.dw[1]], b: m.b - opts.lr * g.db }
  }
  return out
}

/** Step-by-step logistic trace for the scene; the same as `logisticGD`. */
export function trace(points: readonly LabeledPoint[], opts: LogisticOptions): LogisticStep[] {
  return logisticGD(points, opts)
}

/**
 * Perceptron rule, visiting points in order: if t·s ≤ 0 (t = ±1) then w ← w + η·t·[x, y], b ← b + η·t.
 * Records every visit; stops on a clean pass, `maxUpdates` or `maxEpochs`.
 */
export function perceptron(points: readonly LabeledPoint[], opts: PerceptronOptions = {}): PerceptronResult {
  const lr = opts.lr ?? 1
  const maxUpdates = opts.maxUpdates ?? Infinity
  const maxEpochs = opts.maxEpochs ?? 100
  let m = copy(opts.init ?? ZERO)
  const steps: PerceptronStep[] = []
  let updates = 0
  let converged = false
  if (points.length === 0) return { steps, updates, converged: true, model: m }
  outer: for (let epoch = 0; epoch < maxEpochs; epoch++) {
    let fired = false
    for (let i = 0; i < points.length; i++) {
      const p = points[i]
      const t = p.label === 1 ? 1 : -1
      const wrong = t * score(m, p.x, p.y) <= 0
      const before = copy(m)
      if (wrong) {
        m = { w: [m.w[0] + lr * t * p.x, m.w[1] + lr * t * p.y], b: m.b + lr * t }
        updates++
        fired = true
      }
      steps.push({ i, epoch, misclassified: wrong, wBefore: before, wAfter: copy(m) })
      if (updates >= maxUpdates) break outer
    }
    if (!fired) {
      converged = true
      break
    }
  }
  return { steps, updates, converged, model: m }
}

/** Samples σ(score) on an n × n grid over the bounds, for the sigmoid sheet. */
export function decisionSampler(m: LinearModel, bounds: Bounds, n: number): ProbabilityGrid {
  const xs = axis(bounds.xMin, bounds.xMax, n)
  const ys = axis(bounds.yMin, bounds.yMax, n)
  return { xs, ys, p: ys.map((y) => xs.map((x) => predictProba(m, x, y))) }
}

/** Linearly spaced values from lo to hi inclusive, or [lo] when n ≤ 1. */
export function axis(lo: number, hi: number, n: number): number[] {
  if (n <= 1) return [lo]
  return Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))
}

/** The segment of the line s = 0 inside the bounds, or null when it misses them or w = 0. */
export function boundaryLine(
  m: LinearModel,
  bounds: Bounds,
): [{ x: number; y: number }, { x: number; y: number }] | null {
  const [a, c] = m.w
  const pts: { x: number; y: number }[] = []
  const eps = 1e-12
  const push = (x: number, y: number) => {
    if (x < bounds.xMin - eps || x > bounds.xMax + eps || y < bounds.yMin - eps || y > bounds.yMax + eps) return
    if (pts.some((q) => Math.abs(q.x - x) < 1e-9 && Math.abs(q.y - y) < 1e-9)) return
    pts.push({ x, y })
  }
  if (c !== 0) {
    push(bounds.xMin, -(a * bounds.xMin + m.b) / c)
    push(bounds.xMax, -(a * bounds.xMax + m.b) / c)
  }
  if (a !== 0) {
    push(-(c * bounds.yMin + m.b) / a, bounds.yMin)
    push(-(c * bounds.yMax + m.b) / a, bounds.yMax)
  }
  return pts.length >= 2 ? [pts[0], pts[1]] : null
}

/**
 * Two Gaussian blobs: label 1 around (sep/2, sep/2), label 0 around (−sep/2, −sep/2), standard deviation `spread`.
 * Labels alternate 1, 0, 1, … ; each point draws x then y with `normal`.
 */
export function twoBlobs(n: number, seed: number, opts: { sep?: number; spread?: number } = {}): LabeledPoint[] {
  const sep = opts.sep ?? 1
  const spread = opts.spread ?? 0.3
  const rand = mulberry32(seed)
  const out: LabeledPoint[] = []
  for (let k = 0; k < n; k++) {
    const label: 0 | 1 = k % 2 === 0 ? 1 : 0
    const c = label === 1 ? sep / 2 : -sep / 2
    const x = c + spread * normal(rand)
    const y = c + spread * normal(rand)
    out.push({ x, y, label })
  }
  return out
}

/**
 * XOR quadrants: x, y uniform in [−1, 1), pushed 0.1 away from each axis, label 1 when x·y > 0,
 * then uniform noise in [−noise, noise) added to each coordinate (two more draws, only when noise > 0).
 */
export function xor(n: number, seed: number, noise = 0): LabeledPoint[] {
  const rand = mulberry32(seed)
  const pad = 0.1
  const out: LabeledPoint[] = []
  for (let k = 0; k < n; k++) {
    let x = uniform(rand, -1, 1)
    let y = uniform(rand, -1, 1)
    x += x >= 0 ? pad : -pad
    y += y >= 0 ? pad : -pad
    const label: 0 | 1 = x * y > 0 ? 1 : 0
    if (noise > 0) {
      x += uniform(rand, -noise, noise)
      y += uniform(rand, -noise, noise)
    }
    out.push({ x, y, label })
  }
  return out
}

/** Formulas for the maths drawer, with `{name}` placeholders. */
export function describe(): Formula[] {
  return [
    { id: 'score', label: 'Score', template: 's = {w1} · x + {w2} · y + {b}' },
    { id: 'sigmoid', label: 'Sigmoid', template: 'p = σ(s) = 1 / (1 + e^(−s)) = {p}' },
    { id: 'loss', label: 'Cross-entropy', template: 'L = −(1/{n}) Σ [tᵢ ln pᵢ + (1 − tᵢ) ln(1 − pᵢ)] = {loss}' },
    { id: 'grad', label: 'Gradient', template: '∂L/∂w = (1/{n}) Σ (pᵢ − tᵢ) [xᵢ, yᵢ];  ∂L/∂b = (1/{n}) Σ (pᵢ − tᵢ)' },
    { id: 'step', label: 'Gradient step', template: 'w ← w − {lr} · ∂L/∂w;  b ← b − {lr} · ∂L/∂b' },
    { id: 'boundary', label: 'Decision line', template: '{w1} · x + {w2} · y + {b} = 0' },
    {
      id: 'perceptron',
      label: 'Perceptron rule',
      template: 'if t · s ≤ 0:  w ← w + {lr} · t · [x, y],  b ← b + {lr} · t   (t = ±1)',
    },
  ]
}
