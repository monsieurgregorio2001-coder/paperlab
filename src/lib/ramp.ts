/**
 * Colour ramps for heatmaps. Backgrounds are mixed in CSS from design tokens.
 * This module only decides how much to mix and which text colour keeps AA contrast,
 * using the same hex values as the tokens in styles.css.
 */
type RGB = [number, number, number]

const hex = (h: string): RGB => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)) as RGB

const THEMES = {
  light: { surface: '#FFFFFF', accent: '#2340B8', warm: '#B5562B', ink: '#17171B', onAccent: '#FFFFFF' },
  dark: { surface: '#17171C', accent: '#9DB0FF', warm: '#F0A07A', ink: '#ECEAE5', onAccent: '#0F0F12' },
}

function lum([r, g, b]: RGB): number {
  const f = (c: number) => {
    const s = c / 255
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
  }
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
}

function contrast(a: RGB, b: RGB): number {
  const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x)
  return (hi + 0.05) / (lo + 0.05)
}

const mix = (base: RGB, bg: RGB, m: number): RGB => base.map((c, i) => c * m + bg[i] * (1 - m)) as RGB

type ThemeName = keyof typeof THEMES
type Hue = 'accent' | 'warm'

/** Best text token and its contrast for a given mix in a theme. */
function bestText(theme: ThemeName, hue: Hue, m: number): { token: 'ink' | 'accent-ink'; ratio: number } {
  const t = THEMES[theme]
  const bg = mix(hex(t[hue]), hex(t.surface), m)
  const ink = contrast(bg, hex(t.ink))
  const on = contrast(bg, hex(t.onAccent))
  return ink >= on ? { token: 'ink', ratio: ink } : { token: 'accent-ink', ratio: on }
}

const MIN_RATIO = 4.6

/** Mix amounts (0 to 1) where text stays AA in both themes and both hues. */
const ALLOWED: number[] = (() => {
  const out: number[] = []
  for (let i = 0; i <= 100; i++) {
    const m = i / 100
    const ok = (['light', 'dark'] as ThemeName[]).every((th) =>
      (['accent', 'warm'] as Hue[]).every((h) => bestText(th, h, m).ratio >= MIN_RATIO),
    )
    if (ok) out.push(m)
  }
  return out
})()

export interface Shade {
  /** Mix amount of the hue into the surface, 0 to 1. */
  m: number
  hue: Hue
  light: 'ink' | 'accent-ink'
  dark: 'ink' | 'accent-ink'
}

/** Map a magnitude t in [0, 1] to a shade. Skips mixes where text contrast would fail AA. */
export function shade(t: number, hue: Hue = 'accent'): Shade {
  const c = Math.max(0, Math.min(1, t))
  const m = ALLOWED[Math.round(c * (ALLOWED.length - 1))]
  return { m, hue, light: bestText('light', hue, m).token, dark: bestText('dark', hue, m).token }
}

export function cellVars(s: Shade): Record<string, string | number> {
  return {
    '--m': s.m,
    '--hue': s.hue === 'accent' ? 'var(--accent)' : 'var(--warm)',
    '--tl': `var(--${s.light})`,
    '--td': `var(--${s.dark})`,
  }
}
