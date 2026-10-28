/**
 * Seeded randomness shared by the lab kernels.
 * `hashString` and `mulberry32` are copied unchanged from `src/lib/transformer.ts`.
 * `scripts/fixtures.py` ports all three functions bit for bit, so datasets and
 * initial weights can be checked against NumPy.
 */

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

/** Standard normal sample by Box-Muller, consuming exactly two draws: sqrt(-2 ln(1 - u1)) * cos(2π u2). */
export function normal(rand: () => number): number {
  const u1 = rand()
  const u2 = rand()
  return Math.sqrt(-2 * Math.log(1 - u1)) * Math.cos(2 * Math.PI * u2)
}

/** Uniform sample in [lo, hi). */
export function uniform(rand: () => number, lo: number, hi: number): number {
  return lo + (hi - lo) * rand()
}
