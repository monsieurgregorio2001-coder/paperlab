import { useEffect, useMemo, useRef, useState } from 'react'
import { MAX_TOKENS, N_HEADS, D_K, runModel, tokenize } from './lib/transformer'
import MatrixView from './components/MatrixView'
import Heatmap from './components/Heatmap'
import { PAPER_URL, PRESETS, STEPS } from './steps'

const PLAY_MS = 4000

export default function App() {
  const [text, setText] = useState(PRESETS[0])
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [causal, setCausal] = useState(false)
  const [activeHeads, setActiveHeads] = useState<boolean[]>([true, true, false, false])
  const [selected, setSelected] = useState<number | null>(null)
  const [numbers, setNumbers] = useState(false)

  const model = useMemo(() => runModel(text, { causal }), [text, causal])
  const n = model.tokens.length
  const empty = n === 0
  const last = STEPS.length - 1

  const stateRef = useRef({ step, playing })
  stateRef.current = { step, playing }

  // Autoplay
  useEffect(() => {
    if (!playing) return
    const id = window.setInterval(() => {
      if (stateRef.current.step >= last) setPlaying(false)
      else setStep((s) => s + 1)
    }, PLAY_MS)
    return () => window.clearInterval(id)
  }, [playing, last])

  const next = () => setStep((s) => Math.min(last, s + 1))
  const back = () => setStep((s) => Math.max(0, s - 1))
  const togglePlay = () => {
    if (!playing && step >= last) setStep(0)
    setPlaying((p) => !p)
  }

  // Keyboard
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      const tag = t?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      if (e.key === 'ArrowRight') { e.preventDefault(); setStep((s) => Math.min(last, s + 1)) }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); setStep((s) => Math.max(0, s - 1)) }
      else if (e.key === ' ' && tag !== 'BUTTON' && tag !== 'A') {
        e.preventDefault()
        const { step: s, playing: p } = stateRef.current
        if (!p && s >= last) setStep(0)
        setPlaying(!p)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [last])

  // Reset selection if it falls out of range
  useEffect(() => {
    if (selected !== null && selected >= n) setSelected(null)
  }, [n, selected])

  const info = STEPS[step]
  const { total, truncated } = tokenize(text)
  const labels = model.tokens

  return (
    <>
      <a className="skip" href="#stage">Skip to visualization</a>
      <header className="top">
        <div className="wrap">
          <h1>PaperLab</h1>
          <p className="sub">
            Walk through one layer of the Transformer from <a href={PAPER_URL} target="_blank" rel="noreferrer">Attention Is All You Need</a> (Vaswani et al., 2017).
          </p>
          <p className="toy" role="note"><strong>Toy model:</strong> real Transformer math on small random weights. Not a trained model.</p>
        </div>
      </header>

      <main className="wrap">
        <section aria-labelledby="in-h" className="card">
          <h2 id="in-h" className="h2">Your sentence</h2>
          <label htmlFor="sentence" className="sr-only">Sentence to visualize</label>
          <input
            id="sentence"
            type="text"
            value={text}
            onChange={(e) => { setText(e.target.value); setSelected(null) }}
            placeholder="Type a sentence"
            autoComplete="off"
            spellCheck={false}
            aria-describedby="count"
          />
          <p id="count" className={truncated ? 'count warn' : 'count'} aria-live="polite">
            {n} of {MAX_TOKENS} tokens used.
            {truncated ? ` Your sentence has ${total} tokens, so only the first ${MAX_TOKENS} are shown. The rest are ignored.` : ''}
          </p>
          <div className="presets" role="group" aria-label="Example sentences">
            <span className="muted">Try:</span>
            {PRESETS.map((p) => (
              <button key={p} type="button" className="chip-btn" onClick={() => { setText(p); setSelected(null) }}>
                {p.length > 28 ? p.slice(0, 26) + '...' : p}
              </button>
            ))}
          </div>
        </section>

        <nav className="controls" aria-label="Step controls">
          <div className="btns">
            <button type="button" onClick={back} disabled={step === 0}>Back</button>
            <button type="button" onClick={togglePlay} aria-pressed={playing}>{playing ? 'Pause' : 'Play'}</button>
            <button type="button" onClick={next} disabled={step === last}>Next</button>
          </div>
          <ol className="dots">
            {STEPS.map((s, i) => (
              <li key={s.id}>
                <button
                  type="button"
                  className="dot"
                  aria-current={i === step ? 'step' : undefined}
                  aria-label={`Go to step ${i + 1}, ${s.short}`}
                  onClick={() => setStep(i)}
                >
                  <span className="num">{i + 1}</span><span className="dl">{s.short}</span>
                </button>
              </li>
            ))}
          </ol>
          <p className="hint muted">Keyboard: left and right arrows change step, space plays or pauses.</p>
        </nav>

        <section id="stage" className="card stage" aria-labelledby="step-h" tabIndex={-1}>
          <h2 id="step-h" className="h2" aria-live="polite">{info.title}</h2>
          <div className="formula" role="math" aria-label={info.formulaLabel}>{info.formula}</div>
          <p>{info.explain}</p>
          <p className="muted">
            Paper reference: <a href={info.url} target="_blank" rel="noreferrer">{info.section}</a>
          </p>

          {empty ? (
            <div className="empty">
              <p><strong>Nothing to show yet.</strong></p>
              <p>Type a sentence above or pick one of the examples to see it travel through the Transformer.</p>
            </div>
          ) : (
            <>
              {step === 0 && (
                <div>
                  <ul className="chips" aria-label="Tokens and ids">
                    {model.tokens.map((t, i) => (
                      <li key={i} className="tok">
                        <span className="tw">{t}</span>
                        <span className="tid" title={`id ${model.ids[i]}`}>id {model.ids[i]}</span>
                      </li>
                    ))}
                  </ul>
                  <p className="muted small">Simplified word-level tokenizer. Real models use subword BPE.</p>
                </div>
              )}

              {step > 0 && (
                <label className="check">
                  <input type="checkbox" checked={numbers} onChange={(e) => setNumbers(e.target.checked)} />
                  Show numbers in cells
                </label>
              )}

              {step === 1 && (
                <div className="stack">
                  <MatrixView title="Embeddings E" caption="Toy weights from a seeded hash of each token." rows={model.embeddings} rowLabels={labels} showNumbers={numbers} />
                  <div className="op" aria-hidden="true">+</div>
                  <MatrixView title="Positional encoding PE" caption="Sine and cosine waves from the paper." rows={model.positional} rowLabels={labels} showNumbers={numbers} scaleMax={1} />
                  <div className="op" aria-hidden="true">=</div>
                  <MatrixView title="Input X" caption="E plus PE, fed into attention." rows={model.X} rowLabels={labels} showNumbers={numbers} />
                  <Legend />
                </div>
              )}

              {step === 2 && (
                <div className="stack">
                  <div className="toolbar">
                    <div role="group" aria-label="Attention heads" className="toggles">
                      {Array.from({ length: N_HEADS }, (_, h) => (
                        <button
                          key={h}
                          type="button"
                          className="toggle"
                          aria-pressed={activeHeads[h]}
                          onClick={() => setActiveHeads((a) => a.map((v, i) => (i === h ? !v : v)))}
                        >
                          Head {h + 1}
                        </button>
                      ))}
                    </div>
                    <label className="check">
                      <input type="checkbox" checked={causal} onChange={(e) => setCausal(e.target.checked)} />
                      Causal mask (tokens cannot see the future)
                    </label>
                  </div>
                  {!activeHeads.some(Boolean) && <p className="muted">Turn on at least one head to see its attention.</p>}
                  {model.heads.map((h, hi) =>
                    activeHeads[hi] ? (
                      <div key={hi} className="headblock">
                        <Heatmap title={`Head ${hi + 1}`} tokens={labels} weights={h.weights} selected={selected} onSelect={(i) => setSelected(selected === i ? null : i)} />
                        {selected !== null && selected < n && (
                          <AttendList label={labels[selected]} tokens={labels} row={h.weights[selected]} />
                        )}
                        <details>
                          <summary>Show Q, K, V and scaled scores for head {hi + 1}</summary>
                          <MatrixView title="Q" caption={`n x ${D_K}`} rows={h.Q} rowLabels={labels} showNumbers={numbers} />
                          <MatrixView title="K" caption={`n x ${D_K}`} rows={h.K} rowLabels={labels} showNumbers={numbers} />
                          <MatrixView title="V" caption={`n x ${D_K}`} rows={h.V} rowLabels={labels} showNumbers={numbers} />
                          <MatrixView title="Scores QK^T / sqrt(d_k)" caption={causal ? 'Masked entries are minus infinity.' : 'Before softmax.'} rows={h.scores} rowLabels={labels} showNumbers={numbers} colLabelPrefix="k" />
                        </details>
                      </div>
                    ) : null,
                  )}
                  <p className="muted small">
                    Click a token name at the start of a row to see what it attends to. Each head has its own small weights and learns a different pattern in real models. Here the patterns are random.
                    The head outputs ({N_HEADS} heads of {D_K} numbers) are joined and mixed by an output matrix W<sub>O</sub>.
                  </p>
                  <Legend attention />
                </div>
              )}

              {step === 3 && (
                <div className="stack">
                  <MatrixView title="After attention: LayerNorm(X + Attention)" rows={model.norm1} rowLabels={labels} showNumbers={numbers} />
                  <MatrixView title="Hidden layer: max(0, xW1 + b1)" caption="32 numbers per token. Zeros come from the ReLU." rows={model.ffnHidden} rowLabels={labels} showNumbers={numbers} colLabelPrefix="h" />
                  <MatrixView title="FFN(x)" caption="Back to 16 numbers." rows={model.ffnOut} rowLabels={labels} showNumbers={numbers} />
                  <MatrixView title="Layer output: LayerNorm(x + FFN(x))" caption="This feeds the next layer. The paper stacks 6." rows={model.final} rowLabels={labels} showNumbers={numbers} />
                  <Legend />
                </div>
              )}
            </>
          )}
        </section>
      </main>

      <footer className="wrap foot">
        <p>Toy model: real Transformer math on small random weights. Not a trained model.</p>
        <p>Inspired by Clement Mihailescu&apos;s algorithm visualizers and Two Minute Papers. Not affiliated.</p>
        <p>
          By Garfield Greg Lim III. MIT license. Paper: <a href={PAPER_URL} target="_blank" rel="noreferrer">arXiv 1706.03762</a>.
        </p>
      </footer>
    </>
  )
}

function Legend({ attention = false }: { attention?: boolean }) {
  return (
    <div className="legend" aria-label="Color legend">
      {attention ? (
        <><span className="sw" style={{ background: 'var(--cell-bg)' }} /> 0 <span className="sw" style={{ background: 'var(--pos)' }} /> 1 (more attention)</>
      ) : (
        <><span className="sw" style={{ background: 'var(--neg)' }} /> negative <span className="sw" style={{ background: 'var(--cell-bg)' }} /> zero <span className="sw" style={{ background: 'var(--pos)' }} /> positive</>
      )}
      <span className="muted"> Hover or focus a cell for its exact value.</span>
    </div>
  )
}

function AttendList({ label, tokens, row }: { label: string; tokens: string[]; row: number[] }) {
  const order = row.map((w, j) => ({ w, j })).sort((a, b) => b.w - a.w).slice(0, 5)
  return (
    <div className="attend" aria-live="polite">
      <p><strong>{label}</strong> attends most to:</p>
      <ul>
        {order.map(({ w, j }) => (
          <li key={j}>
            <span className="an">{tokens[j]}</span>
            <span className="bar"><span style={{ width: `${(w * 100).toFixed(1)}%` }} /></span>
            <span className="av">{(w * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
