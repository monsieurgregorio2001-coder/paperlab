import type { CSSProperties } from 'react'
import { cellVars, shade } from '../lib/ramp'

interface Props {
  title: string
  tokens: string[]
  weights: number[][]
  selected: number | null
  onSelect: (i: number) => void
}

export default function Heatmap({ title, tokens, weights, selected, onSelect }: Props) {
  return (
    <figure className="matrix heat">
      <figcaption>
        <strong>{title}</strong>
        <span className="muted"> Rows are queries and columns are keys. Each row sums to 1.</span>
      </figcaption>
      <div className="scroll" tabIndex={0} role="region" aria-label={`${title} attention heatmap, scrollable`}>
        <table className="mt numbers">
          <thead>
            <tr>
              <th className="corner" aria-hidden="true" />
              <th colSpan={tokens.length} className="axis">Key: the token being looked at</th>
            </tr>
            <tr>
              <th scope="col" className="corner axis">Query</th>
              {tokens.map((t, j) => (
                <th key={j} scope="col" className="colh tokh" title={t}>{t}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {weights.map((row, i) => (
              <tr key={i} className={selected === i ? 'sel' : undefined}>
                <th scope="row" className="rowh">
                  <button
                    type="button"
                    className="rowbtn"
                    aria-pressed={selected === i}
                    aria-label={`Highlight what ${tokens[i]} attends to`}
                    onClick={() => onSelect(i)}
                  >
                    {tokens[i]}
                  </button>
                </th>
                {row.map((w, j) => {
                  const label = `${tokens[i]} attends to ${tokens[j]}: ${w.toFixed(3)}`
                  return (
                    <td
                      key={j}
                      className={selected === i ? 'cell selcell' : 'cell'}
                      style={cellVars(shade(w)) as CSSProperties}
                      title={label}
                      aria-label={label}
                    >
                      {w.toFixed(2)}
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
