import { useEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { cellVars, shade } from '../lib/ramp'
import { MAX_TOKENS, N_HEADS, D_K, runModel, tokenize } from '../lib/transformer'
import MatrixView from './components/MatrixView'
import Heatmap from './components/Heatmap'
import { CASE, HUB_URL, PAPER_URL, PRESETS, REPO_URL, STEPS } from './steps'

const PLAY_MS = 4000

export default function App() {
  const [text, setText] = useState(PRESETS[0])
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [causal, setCausal] = useState(false)
  const [activeHeads, setActiveHeads] = useState<boolean[]>([true, true, false, false])
  const [selected, setSelected] = useState<number | null>(null)
  const [numbers, setNumbers] = useState(false)
  const [aboutOpen, setAboutOpen] = useState(() => window.matchMedia('(min-width: 860px)').matches)

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

  // Keyboard: arrows change step, P toggles play. Space is left to the focused button.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      const tag = t?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      const inScroller = !!t?.closest?.('.scroll')
      if (e.key === 'ArrowRight' && !inScroller) { e.preventDefault(); setStep((s) => Math.min(last, s + 1)) }
      else if (e.key === 'ArrowLeft' && !inScroller) { e.preventDefault(); setStep((s) => Math.max(0, s - 1)) }
      else if (e.key === 'p' || e.key === 'P') {
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
      <header className="site">
        <div className="wrap bar">
          <p className="crumb">
            <a className="brand" href={HUB_URL}>Garfield Lim</a>
            <span className="slash" aria-hidden="true">/</span>
            <span className="proj">PaperLab</span>
          </p>
          <a className="codelink" href={REPO_URL} target="_blank" rel="noreferrer">Code</a>
        </div>
      </header>

      <main className="wrap">
        <div className="hero">
          <section aria-labelledby="in-h" className="intro">
            <h1>PaperLab</h1>
            <p className="lede">See attention happen. Type a sentence and step through one layer of the Transformer from <a href={PAPER_URL} target="_blank" rel="noreferrer">Attention Is All You Need</a>.</p>
            <h2 id="in-h" className="sr-only">Your sentence</h2>
            <label htmlFor="sentence" className="field-label">Your sentence</label>
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
              <span className="muted small">Try</span>
              {PRESETS.map((p) => (
                <button key={p} type="button" className="chip-btn" onClick={() => { setText(p); setSelected(null) }}>
                  {p.length > 28 ? p.slice(0, 26) + '...' : p}
                </button>
              ))}
            </div>
            <p className="toy" role="note"><span className="dot-badge" aria-hidden="true" />Toy model: real Transformer math on small random weights, not a trained model.</p>
          </section>

          <details className="about" open={aboutOpen} onToggle={(e) => setAboutOpen(e.currentTarget.open)}>
            <summary>About this project</summary>
            <dl>
              <dt>Problem</dt>
              <dd>{CASE.problem}</dd>
              <dt>Build</dt>
              <dd>{CASE.build}</dd>
              <dt>Result</dt>
              <dd>{CASE.result}</dd>
            </dl>
          </details>
        </div>

        <nav className="controls" aria-label="Step controls">
          <ol className="stepper">
            <li className="rule" aria-hidden="true"><span style={{ transform: `scaleX(${step / last})` }} /></li>
            {STEPS.map((s, i) => (
              <li key={s.id} className={i < step ? 'done' : i === step ? 'now' : undefined}>
                <button
                  type="button"
                  className="stp"
                  aria-current={i === step ? 'step' : undefined}
                  onClick={() => setStep(i)}
                >
                  <span className="num" aria-hidden="true">{i + 1}</span>
                  <span className="lab"><span className="sr-only">Step {i + 1}: </span>{s.short}</span>
                </button>
              </li>
            ))}
          </ol>
          <div className="btns">
            <button type="button" className="btn" onClick={back} disabled={step === 0}>Back</button>
            <button type="button" className="btn primary" onClick={togglePlay} aria-pressed={playing} aria-keyshortcuts="P">
              {playing ? 'Pause' : 'Play'}<kbd aria-hidden="true">P</kbd>
            </button>
            <button type="button" className="btn" onClick={next} disabled={step === last}>Next</button>
            <span className="status muted" aria-hidden="true">Step {step + 1} of {STEPS.length}</span>
          </div>
          <details className="kbd">
            <summary>Keyboard</summary>
            <ul>
              <li><kbd>Left</kbd> and <kbd>Right</kbd> arrows go to the previous or next step.</li>
              <li><kbd>P</kbd> plays or pauses the walkthrough.</li>
              <li><kbd>Tab</kbd> moves between controls, and <kbd>Space</kbd> or <kbd>Enter</kbd> activates the focused button.</li>
            </ul>
          </details>
        </nav>

        <section id="stage" className="card stage" aria-labelledby="step-h" tabIndex={-1}>
          <p className="eyebrow">Step {step + 1} of {STEPS.length}</p>
          <h2 id="step-h" className="h2" aria-live="polite">{info.title}</h2>
          <div className="formula" tabIndex={0} role="math" aria-label={info.formulaLabel}>{info.formula}</div>
          <p className="explain">{info.explain}</p>
          <p className="ref muted">
            Paper: <a href={info.url} target="_blank" rel="noreferrer">{info.section}</a>
          </p>

          {empty ? (
            <div className="empty">
              <p><strong>Nothing to show yet.</strong></p>
              <p>Type a sentence above or pick an example to see it travel through the Transformer.</p>
            </div>
          ) : (
            <div key={step} className="body">
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
                  <p className="muted small">A simplified word-level tokenizer. Real models use subword BPE.</p>
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
                      Causal mask, so tokens cannot see the future
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
                        <details className="more">
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
                    Click a token at the start of a row to see what it attends to most. Each head has its own small weights.
                    In a trained model the heads learn different patterns, but here they are random.
                    The head outputs ({N_HEADS} heads of {D_K} numbers each) are joined and mixed by an output matrix W<sub>O</sub>.
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
            </div>
          )}
        </section>
      </main>

      <footer className="site-foot">
        <div className="wrap">
          <p className="disclaimer">Toy model: real Transformer math on small random weights, not a trained model. Inspired by Clement Mihailescu&apos;s algorithm visualizers and Two Minute Papers. Not affiliated.</p>
          <p>
            Built by Garfield Greg Lim III
            <span aria-hidden="true"> · </span><a href={HUB_URL}>Portfolio</a>
            <span aria-hidden="true"> · </span><a href="https://github.com/monsieurgregorio2001-coder" target="_blank" rel="noreferrer">GitHub</a>
            <span aria-hidden="true"> · </span><a href="https://www.linkedin.com/in/garfield-lim-8239242a9" target="_blank" rel="noreferrer">LinkedIn</a>
            <span aria-hidden="true"> · </span><a href="mailto:lim.ggreg@outlook.com">lim.ggreg@outlook.com</a>
          </p>
        </div>
      </footer>
    </>
  )
}

const STOPS = Array.from({ length: 9 }, (_, i) => i / 8)

function Legend({ attention = false }: { attention?: boolean }) {
  return (
    <div className="legend" role="group" aria-label="Colour legend">
      {attention ? (
        <>
          <span className="lg-end">0</span>
          <span className="lg-ramp" aria-hidden="true">
            {STOPS.map((t) => <span key={t} className="cell" style={cellVars(shade(t)) as CSSProperties} />)}
          </span>
          <span className="lg-end">1</span>
          <span className="muted small">More attention</span>
        </>
      ) : (
        <>
          <span className="lg-end">Negative</span>
          <span className="lg-ramp" aria-hidden="true">
            {[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1].map((v) => (
              <span key={v} className="cell" style={cellVars(shade(Math.abs(v), v < 0 ? 'warm' : 'accent')) as CSSProperties} />
            ))}
          </span>
          <span className="lg-end">Positive</span>
          <span className="muted small">Neutral at zero. Scaled to the largest value in each matrix.</span>
        </>
      )}
      <span className="muted small">Hover or focus a cell for its exact value.</span>
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
            <span className="abar"><span style={{ width: `${(w * 100).toFixed(1)}%` }} /></span>
            <span className="av">{(w * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
