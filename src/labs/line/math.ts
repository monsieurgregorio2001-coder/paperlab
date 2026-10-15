/**
 * Lab 1 — Line. Linear regression on 2D points.
 * Pure and deterministic: no DOM, no randomness. Loss is mean squared error,
 * L(w, b) = (1/n) Σ (w·xᵢ + b − yᵢ)².
 */

import type { Formula } from '../_shared/formula'

export type { Formula }
export { fill } from '../_shared/formula'

export interface Point {
  x: number
  y: number
}

export interface LineModel {
  w: number
  b: number
}

export interface GdOptions {
  /** Learning rate η. */
  lr: number
  /** Number of updates; the trace has steps + 1 entries (entry 0 is `init`). */
  steps: number
  /** Starting parameters. */
  init: LineModel
}

export interface GdStep {
  step: number
  w: number
  b: number
  /** MSE at (w, b). */
  loss: number
  /** ∂L/∂w at (w, b). */
  dw: number
  /** ∂L/∂b at (w, b). */
  db: number
}

export interface MseSurface {
  /** n values of w from wRange[0] to wRange[1] inclusive. */
  ws: number[]
  /** n values of b from bRange[0] to bRange[1] inclusive. */
  bs: number[]
  /** loss[i][j] is the MSE at (ws[j], bs[i]). */
  loss: number[][]
  min: number
  max: number
}

/** ŷ = w·x + b. */
export function predict(model: LineModel, x: number): number {
  return model.w * x + model.b
}

/** Mean squared error of the model on the points (0 for no points). */
export function mse(points: readonly Point[], model: LineModel): number {
  if (points.length === 0) return 0
  let s = 0
  for (const p of points) {
    const r = model.w * p.x + model.b - p.y
    s += r * r
  }
  return s / points.length
}

/** Gradient of the MSE: ∂L/∂w = (2/n) Σ rᵢxᵢ, ∂L/∂b = (2/n) Σ rᵢ with rᵢ = ŷᵢ − yᵢ. */
export function mseGradient(points: readonly Point[], model: LineModel): { dw: number; db: number } {
  const n = points.length
  if (n === 0) return { dw: 0, db: 0 }
  let sw = 0
  let sb = 0
  for (const p of points) {
    const r = model.w * p.x + model.b - p.y
    sw += r * p.x
    sb += r
  }
  return { dw: (2 * sw) / n, db: (2 * sb) / n }
}

/** Closed-form least squares: w = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)², b = ȳ − w·x̄; w = 0 when all x are equal. */
export function olsFit(points: readonly Point[]): LineModel {
  const n = points.length
  if (n === 0) return { w: 0, b: 0 }
  let mx = 0
  let my = 0
  for (const p of points) {
    mx += p.x
    my += p.y
  }
  mx /= n
  my /= n
  let sxy = 0
  let sxx = 0
  for (const p of points) {
    const dx = p.x - mx
    sxy += dx * (p.y - my)
    sxx += dx * dx
  }
  const w = sxx === 0 ? 0 : sxy / sxx
  return { w, b: my - w * mx }
}

/** Full-batch gradient descent; returns every step, stopping early once the loss is not finite. */
export function gdFit(points: readonly Point[], opts: GdOptions): GdStep[] {
  let w = opts.init.w
  let b = opts.init.b
  const out: GdStep[] = []
  for (let step = 0; step <= opts.steps; step++) {
    const loss = mse(points, { w, b })
    const { dw, db } = mseGradient(points, { w, b })
    out.push({ step, w, b, loss, dw, db })
    if (!Number.isFinite(loss)) break
    w -= opts.lr * dw
    b -= opts.lr * db
  }
  return out
}

/** Step-by-step trace for the scene; the same as `gdFit`. */
export function trace(points: readonly Point[], opts: GdOptions): GdStep[] {
  return gdFit(points, opts)
}

/** Linearly spaced values from lo to hi inclusive, or [lo] when n ≤ 1. */
export function linspace(lo: number, hi: number, n: number): number[] {
  if (n <= 1) return [lo]
  return Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))
}

/** Samples the MSE on an n × n grid over (w, b) for the 3D loss bowl. */
export function mseSurface(
  points: readonly Point[],
  wRange: readonly [number, number],
  bRange: readonly [number, number],
  n: number,
): MseSurface {
  const ws = linspace(wRange[0], wRange[1], n)
  const bs = linspace(bRange[0], bRange[1], n)
  let min = Infinity
  let max = -Infinity
  const loss = bs.map((b) =>
    ws.map((w) => {
      const l = mse(points, { w, b })
      if (l < min) min = l
      if (l > max) max = l
      return l
    }),
  )
  return { ws, bs, loss, min, max }
}

/** Formulas for the maths drawer, with `{name}` placeholders. */
export function describe(): Formula[] {
  return [
    { id: 'model', label: 'Prediction', template: 'ŷ = {w} · x + {b}' },
    { id: 'loss', label: 'Mean squared error', template: 'L = (1/{n}) Σ (ŷᵢ − yᵢ)² = {loss}' },
    {
      id: 'grad',
      label: 'Gradient',
      template: '∂L/∂w = (2/{n}) Σ (ŷᵢ − yᵢ) xᵢ = {dw};  ∂L/∂b = (2/{n}) Σ (ŷᵢ − yᵢ) = {db}',
    },
    { id: 'step', label: 'Gradient step', template: 'w ← w − {lr} · ∂L/∂w;  b ← b − {lr} · ∂L/∂b' },
    { id: 'ols', label: 'Closed form', template: 'w = Σ (xᵢ − x̄)(yᵢ − ȳ) / Σ (xᵢ − x̄)²;  b = ȳ − w · x̄' },
  ]
}
