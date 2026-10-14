import { cellStyle } from './MatrixView'

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
        <span className="muted"> Rows are queries. Columns are keys. Each row sums to 1.</span>
      </figcaption>
      <div className="scroll" tabIndex={0} role="region" aria-label={`${title} attention heatmap, scrollable`}>
        <table className="mt numbers">
          <thead>
            <tr>
              <th scope="col" className="corner"><span className="sr-only">Query token</span></th>
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
                    <td key={j} tabIndex={-1} style={cellStyle(w, 1)} title={label} aria-label={label}
                        className={selected === i ? 'selcell' : undefined}>
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
