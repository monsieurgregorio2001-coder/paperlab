import type { CSSProperties } from 'react'
import type { Matrix } from '../../lib/transformer'
import { cellVars, shade } from '../../lib/ramp'

interface Props {
  title: string
  caption?: string
  rows: Matrix
  rowLabels: string[]
  showNumbers: boolean
  /** Fixed colour scale maximum. Defaults to the largest absolute value in the matrix. */
  scaleMax?: number
  colLabelPrefix?: string
}

/** Diverging cell: accent for positive, warm for negative, surface at zero. */
export function cellStyle(v: number, max: number): CSSProperties {
  const t = max > 0 ? Math.min(1, Math.abs(v) / max) : 0
  return cellVars(shade(t, v >= 0 ? 'accent' : 'warm')) as CSSProperties
}

function fmt(v: number): string {
  return v === -Infinity ? '−∞' : v.toFixed(2)
}

export default function MatrixView({ title, caption, rows, rowLabels, showNumbers, scaleMax, colLabelPrefix = 'd' }: Props) {
  const max = scaleMax ?? Math.max(0.0001, ...rows.flat().filter(Number.isFinite).map(Math.abs))
  const cols = rows[0]?.length ?? 0
  return (
    <figure className="matrix">
      <figcaption>
        <strong>{title}</strong>
        {caption ? <span className="muted"> {caption}</span> : null}
      </figcaption>
      <div className="scroll" tabIndex={0} role="region" aria-label={`${title}, scrollable table`}>
        <table className={showNumbers ? 'mt numbers' : 'mt'}>
          <thead>
            <tr>
              <th scope="col" className="corner"><span className="sr-only">Token</span></th>
              {Array.from({ length: cols }, (_, j) => (
                <th key={j} scope="col" className="colh">{colLabelPrefix}{j + 1}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                <th scope="row" className="rowh">{rowLabels[i]}</th>
                {row.map((v, j) => {
                  const finite = Number.isFinite(v)
                  const label = `${rowLabels[i]}, ${colLabelPrefix}${j + 1}: ${fmt(v)}`
                  return (
                    <td
                      key={j}
                      className={finite ? 'cell' : 'cell masked'}
                      style={finite ? cellStyle(v, max) : undefined}
                      title={label}
                      aria-label={label}
                    >
                      {showNumbers || !finite ? fmt(v) : null}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </figure>
  )
}
