import { D_K, D_MODEL, N_HEADS } from '../lib/transformer'
import { DESIGNED_FEATURES, DESIGNED_HEADS, type AttentionRun } from '../lib/designed'
import { NumTable, indexLabels } from './Tables'
import { HERE, LEGACY_URL, PAPER, PAPER_SECTIONS } from './content'

interface Props {
  run: AttentionRun
  head: number
  mode: 'designed' | 'random'
  onMode: (m: 'designed' | 'random') => void
}

/** The drawer body: every matrix behind the threads, for the chosen head. Rendered only when open. */
export default function Numbers({ run, head, mode, onMode }: Props) {
  const h = run.heads[head]
  const t = run.tokens
  const rowL = t
  const dk = indexLabels(D_K, 'd')
  const dm = indexLabels(D_MODEL, 'd')
  const all = DESIGNED_FEATURES
  const info = DESIGNED_HEADS[head]
  return (
    <div className="numbers">
      <p className="paper-line">
        Paper: d_model {PAPER.dModel}, {PAPER.heads} heads, d_k {PAPER.dK}. Here: {HERE.dModel}, {HERE.heads}, {HERE.dK}.
      </p>

      <div className="mode" role="radiogroup" aria-label="Weights">
        {(['designed', 'random'] as const).map((m) => (
          <button key={m} type="button" role="radio" aria-checked={mode === m} tabIndex={mode === m ? 0 : -1} className="chip" onClick={() => onMode(m)}
            onKeyDown={(e) => { if (e.key.startsWith('Arrow')) { e.preventDefault(); onMode(m === 'designed' ? 'random' : 'designed') } }}>
            {m === 'designed' ? 'Designed heads' : 'Random weights'}
          </button>
        ))}
      </div>

      {mode === 'designed' ? (
        <div className="stated">
          <p><strong>Head {head + 1}, {info.name}.</strong> {info.rule} {info.weights}</p>
          <p className="note">
            Only W_Q and W_K are set by hand. W_V and W_O are seeded. The toy embeddings carry designed features:
            dimensions {all.clearDims[0]} to {all.clearDims[all.clearDims.length - 1]} are zero, so position can be read cleanly.
            Dimensions {all.codeDims[0]} to {all.codeDims[all.codeDims.length - 1]} hold a code of the word.
            Dimension {all.punctuationDim} flags punctuation. Dimension {all.oneDim} is 1 for every token.
          </p>
        </div>
      ) : (
        <p className="note">Random weights: all four heads use seeded numbers, so no pattern was set. This is the first version&apos;s engine.</p>
      )}

      <h3>Tokens</h3>
      <p className="note">Ids come from a hash of the lowercased word. A real model looks words up in a learned list instead.</p>
      <ul className="ids">
        {t.map((w, i) => <li key={i}><span>{w}</span><span className="figure">id {run.ids[i]}</span></li>)}
      </ul>

      <h3>Head {head + 1} step by step</h3>
      <p className="note">
        softmax(QK<sup>T</sup> / &radic;d<sub>k</sub>) V. Q = XW<sub>Q</sub>, K = XW<sub>K</sub>, V = XW<sub>V</sub>.
        {' '}<a href={PAPER_SECTIONS.dotProduct.url} target="_blank" rel="noreferrer">{PAPER_SECTIONS.dotProduct.label}</a>
      </p>
      <NumTable title="W_Q" note={`${D_MODEL} x ${D_K}`} rows={run.Wq[head]} rowLabels={dm} colLabels={dk} />
      <NumTable title="W_K" note={`${D_MODEL} x ${D_K}`} rows={run.Wk[head]} rowLabels={dm} colLabels={dk} />
      <NumTable title="Q" note={`${t.length} x ${D_K}`} rows={h.Q} rowLabels={rowL} colLabels={dk} />
      <NumTable title="K" note={`${t.length} x ${D_K}`} rows={h.K} rowLabels={rowL} colLabels={dk} />
      <NumTable title="V" note={`${t.length} x ${D_K}`} rows={h.V} rowLabels={rowL} colLabels={dk} />
      <NumTable title="Scores" note={run.switches.scaling ? 'QK^T / sqrt(d_k). Hidden words are minus infinity.' : 'QK^T, not scaled. Hidden words are minus infinity.'} rows={h.scores} rowLabels={rowL} colLabels={t} />
      <NumTable title="Softmax weights" note="Each row sums to 1." rows={h.weights} rowLabels={rowL} colLabels={t} heat />
      <NumTable title="Weights x V" note="This head's output." rows={h.output} rowLabels={rowL} colLabels={dk} />

      <h3>All {N_HEADS} heads</h3>
      <p className="note">
        The head outputs are joined side by side, then mixed by W_O.
        {' '}<a href={PAPER_SECTIONS.multiHead.url} target="_blank" rel="noreferrer">{PAPER_SECTIONS.multiHead.label}</a>
      </p>
      <NumTable title="Concat" note={`${t.length} x ${D_MODEL}`} rows={run.concat} rowLabels={rowL} colLabels={dm} />
      <NumTable title="W_O" note={`${D_MODEL} x ${D_MODEL}`} rows={run.Wo} rowLabels={dm} colLabels={dm} />
      <NumTable title="Concat x W_O" note="The attention output." rows={run.attnOut} rowLabels={rowL} colLabels={dm} />

      <p className="note">
        Position waves: <a href={PAPER_SECTIONS.position.url} target="_blank" rel="noreferrer">{PAPER_SECTIONS.position.label}</a>.
        {' '}Embeddings: <a href={PAPER_SECTIONS.embeddings.url} target="_blank" rel="noreferrer">{PAPER_SECTIONS.embeddings.label}</a>.
      </p>
      <p className="legacy"><a href={LEGACY_URL}>The whole layer, step by step (first version)</a></p>
    </div>
  )
}
