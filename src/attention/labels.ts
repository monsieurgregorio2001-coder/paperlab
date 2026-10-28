import type { WeightsMode } from '../lib/designed'

export interface Truth {
  text: string
  /** The Scaling switch is hidden until the designed heads are verified (DIRECTION 6.1). */
  scalingAvailable: boolean
}

/**
 * The first-screen truth label. "Real math" appears only when the src/lib tests passed at build time
 * (ADR 0009). Otherwise the page says what it can prove and hides Scaling.
 */
export function truthLabel(verified: boolean, mode: WeightsMode): Truth {
  if (!verified) {
    return { text: mode === 'designed' ? 'Designed patterns · Toy size' : 'Random weights · Toy size', scalingAvailable: false }
  }
  return {
    text: mode === 'designed' ? 'Real math · Designed heads · Toy size' : 'Real math · Random weights · Toy size',
    scalingAvailable: true,
  }
}
