import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import type { CSSProperties, KeyboardEvent } from 'react'
import { received } from '../lib/designed'
import type { Matrix } from '../lib/transformer'

interface Anchor { x: number; y: number }

interface Props {
  tokens: string[]
  /** Weights of the chosen head with the switches as they are. */
  weights: Matrix
  /** Weights with every switch at its default. Drawn as a ghost when different. */
  baseline: Matrix
  word: number
  onSelect: (i: number) => void
}

const MIN_WGHT = 360
const MAX_WGHT = 820

/** Path from the top of one word to the top of another, rising above the sentence. */
function threadPath(a: Anchor, b: Anchor, self: boolean): string {
  if (self) return `M ${a.x - 14} ${a.y} C ${a.x - 22} ${a.y - 30}, ${a.x + 22} ${a.y - 30}, ${a.x + 14} ${a.y}`
  const lift = 22 + Math.min(40, Math.abs(b.x - a.x) * 0.14)
  return `M ${a.x} ${a.y} C ${a.x} ${a.y - lift * 1.5}, ${b.x} ${b.y - lift * 1.5}, ${b.x} ${b.y}`
}

export default function Sentence({ tokens, weights, baseline, word, onSelect }: Props) {
  const box = useRef<HTMLDivElement>(null)
  const refs = useRef<(HTMLButtonElement | null)[]>([])
  const [anchors, setAnchors] = useState<Anchor[]>([])
  const [size, setSize] = useState({ w: 0, h: 0 })

  const measure = useCallback(() => {
    const el = box.current
    if (!el) return
    setSize({ w: el.clientWidth, h: el.clientHeight })
    setAnchors(tokens.map((_, i) => {
      const b = refs.current[i]
      return b ? { x: b.offsetLeft + b.offsetWidth / 2, y: b.offsetTop + 6 } : { x: 0, y: 0 }
    }))
  }, [tokens])

  useLayoutEffect(measure, [measure])
  useEffect(() => {
    const el = box.current
    if (!el) return
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    void document.fonts?.ready.then(measure)
    return () => ro.disconnect()
  }, [measure])

  const rec = received(weights)
  const recMax = Math.max(...rec, 1e-9)
  const row = weights[word] ?? []
  const base = baseline[word] ?? []

  const onKey = (e: KeyboardEvent<HTMLButtonElement>) => {
    const n = tokens.length
    let next = -1
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = (word + 1) % n
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') next = (word - 1 + n) % n
    else if (e.key === 'Home') next = 0
    else if (e.key === 'End') next = n - 1
    if (next < 0) return
    e.preventDefault()
    onSelect(next)
    refs.current[next]?.focus()
  }

  return (
    <div className="sentence" ref={box}>
      <svg className="threads" width={size.w} height={size.h} aria-hidden="true" focusable="false">
        {tokens.map((_, j) => {
          const a = anchors[word], b = anchors[j]
          if (!a || !b) return null
          const d = threadPath(a, b, j === word)
          const w = row[j] ?? 0
          const g = base[j] ?? 0
          const ghost = Math.abs(g - w) > 0.03 && g > 0.03
          return (
            <g key={j}>
              <path d={d} className="ghost" style={{ opacity: ghost ? 0.7 : 0, strokeWidth: 1.5 + 6 * g } as CSSProperties} />
              <path d={d} className="thread" style={{ opacity: w < 0.02 ? 0 : 0.28 + 0.72 * w, strokeWidth: 1.5 + 8 * w } as CSSProperties} />
            </g>
          )
        })}
      </svg>
      <div className="words" role="radiogroup" aria-label="Word to read from">
        {tokens.map((t, i) => (
          <button
            key={i}
            ref={(el) => { refs.current[i] = el }}
            type="button"
            role="radio"
            aria-checked={i === word}
            tabIndex={i === word ? 0 : -1}
            className="word"
            data-w={t}
            style={{ fontWeight: Math.round(MIN_WGHT + (MAX_WGHT - MIN_WGHT) * (rec[i] ?? 0) / recMax) } as CSSProperties}
            onClick={() => onSelect(i)}
            onKeyDown={onKey}
          >
            {t}
          </button>
        ))}
      </div>
    </div>
  )
}
