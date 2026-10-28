import { describe, it, expect } from 'vitest'
import { hashString, mulberry32, normal, uniform } from './rand'
import * as legacy from '../../lib/transformer'
import fixture from '../../../tests/fixtures/rand.json'

describe('shared rand', () => {
  it('matches the NumPy port of mulberry32 exactly', () => {
    const r = mulberry32(fixture.seed)
    expect(fixture.first5.map(() => r())).toEqual(fixture.first5)
  })
  it('matches the NumPy port of hashString exactly', () => {
    expect(hashString('paperlab')).toBe(fixture.hash.paperlab)
  })
  it('is the same generator as the v1 transformer kernel', () => {
    const a = mulberry32(7)
    const b = legacy.mulberry32(7)
    for (let i = 0; i < 100; i++) expect(a()).toBe(b())
    expect(hashString('Wq0')).toBe(legacy.hashString('Wq0'))
  })
  it('normal has mean near 0 and variance near 1 over 20 000 draws', () => {
    const r = mulberry32(1)
    let s = 0
    let s2 = 0
    const n = 20000
    for (let i = 0; i < n; i++) {
      const z = normal(r)
      s += z
      s2 += z * z
    }
    expect(Math.abs(s / n)).toBeLessThan(0.03)
    expect(Math.abs(s2 / n - 1)).toBeLessThan(0.03)
  })
  it('uniform stays in [lo, hi)', () => {
    const r = mulberry32(2)
    for (let i = 0; i < 1000; i++) {
      const u = uniform(r, -3, 5)
      expect(u).toBeGreaterThanOrEqual(-3)
      expect(u).toBeLessThan(5)
    }
  })
})
