import type { Matrix } from '../lib/transformer'

interface Props {
  title: string
  caption?: string
  rows: Matrix
  rowLabels: string[]
  showNumbers: boolean
  /** Fixed color scale maximum. Defaults to the largest absolute value in the matrix. */
  scaleMax?: number
  colLabelPrefix?: string
}

export function cellStyle(v: number, max: number): React.CSSProperties {
  const m = max > 0 ? Math.min(1, Math.abs(v) / max) : 0
  const pct = Math.round(m * 85)
  const base = v >= 0 ? 'var(--pos)' : 'var(--neg)'
  return { background: `color-mix(in srgb, ${base} ${pct}%, var(--cell-bg))`, color: m > 0.5 ? '#ffffff' : undefined }
}

function fmt(v: number): string {
  return v === -Infinity ? '-∞' : v.toFixed(2)
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
                {row.map((v, j) => (
                  <td
                    key={j}
                    style={cellStyle(v, max)}
                    title={`${rowLabels[i]}, ${colLabelPrefix}${j + 1}: ${fmt(v)}`}
                    aria-label={`${rowLabels[i]}, ${colLabelPrefix}${j + 1}: ${fmt(v)}`}
                  >
                    {showNumbers ? fmt(v) : null}
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
