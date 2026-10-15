/**
 * Test helper: deep comparison of numbers inside arrays and objects.
 * Two numbers match when |a − b| ≤ tol · max(1, |b|), so the tolerance is absolute
 * near zero and relative for large values such as early losses.
 */

/** Returns the path and values of the first mismatch, or null when everything matches. */
export function firstMismatch(actual: unknown, expected: unknown, tol: number, path = '$'): string | null {
  if (typeof expected === 'number') {
    if (typeof actual !== 'number') return `${path}: expected number ${expected}, got ${String(actual)}`
    if (!(Math.abs(actual - expected) <= tol * Math.max(1, Math.abs(expected)))) {
      return `${path}: ${actual} vs ${expected} (diff ${Math.abs(actual - expected)})`
    }
    return null
  }
  if (Array.isArray(expected)) {
    if (!Array.isArray(actual)) return `${path}: expected array`
    if (actual.length !== expected.length) return `${path}: length ${actual.length} vs ${expected.length}`
    for (let i = 0; i < expected.length; i++) {
      const m = firstMismatch(actual[i], expected[i], tol, `${path}[${i}]`)
      if (m) return m
    }
    return null
  }
  if (typeof expected === 'object' && expected !== null) {
    if (typeof actual !== 'object' || actual === null) return `${path}: expected object`
    const a = actual as Record<string, unknown>
    for (const [k, v] of Object.entries(expected)) {
      const m = firstMismatch(a[k], v, tol, `${path}.${k}`)
      if (m) return m
    }
    return null
  }
  return Object.is(actual, expected) ? null : `${path}: ${String(actual)} vs ${String(expected)}`
}
