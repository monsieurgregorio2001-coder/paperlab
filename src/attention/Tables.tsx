import type { CSSProperties } from 'react'
import type { Matrix } from '../lib/transformer'

interface TableProps {
  title: string
  note?: string
  rows: Matrix
  rowLabels: string[]
  colLabels: string[]
  heat?: boolean
}

const fmt = (v: number) => (v === -Infinity ? '−∞' : v.toFixed(2))

export function NumTable({ title, note, rows, rowLabels, colLabels, heat }: TableProps) {
  return (
    <figure className="numtable">
      <figcaption>
        <strong>{title}</strong>
        {note ? <span className="note"> {note}</span> : null}
      </figcaption>
      <div className="scroll" role="region" aria-label={`${title}, scrolls sideways`} tabIndex={0}>
        <table>
          <thead>
            <tr>
              <td />
              {colLabels.map((c, j) => <th key={j} scope="col">{c}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <th scope="row">{rowLabels[i]}</th>
                {r.map((v, j) => (
                  <td key={j} style={heat ? ({ '--heat': `${Math.round(Math.max(0, v) * 70)}%` } as CSSProperties) : undefined} className={heat ? 'heat' : undefined}>
                    {fmt(v)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </figure>
  )
}

export const indexLabels = (n: number, prefix = '') => Array.from({ length: n }, (_, i) => `${prefix}${i + 1}`)
