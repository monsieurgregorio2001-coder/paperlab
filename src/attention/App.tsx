import { useEffect, useMemo, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'
import { MAX_TOKENS } from '../lib/transformer'
import { DESIGNED_HEADS, runAttention, type WeightsMode } from '../lib/designed'
import Sentence from './Sentence'
import Numbers from './Numbers'
import Receipt from './Receipt'
import Built from './Built'
import { truthLabel } from './labels'
import { PAPER_URL, PRESETS } from './content'
import { SWITCH_COPY, consequence, defaultWord, readout, rowSumLine, type SwitchName } from './narrate'
import { parseState, serializeState, type UiState } from './url'


export default function App() {
  const [s, setS] = useState<UiState>(() => parseState(window.location.search))
  const [last, setLast] = useState<SwitchName>('scaling')
  const [open, setOpen] = useState(false)
  const patch = (p: Partial<UiState>) => setS((o) => ({ ...o, ...p }))

  const truth = truthLabel(__HEADS_VERIFIED__, s.mode)
  const scaling = truth.scalingAvailable ? s.scaling : true
  const switches = { position: s.position, scaling, mask: s.mask }
  const run = useMemo(() => runAttention(s.text, s.mode, switches), [s.text, s.mode, s.position, scaling, s.mask])
  const baseline = useMemo(() => runAttention(s.text, s.mode), [s.text, s.mode])
  const n = run.tokens.length
  const word = s.word !== null && s.word < n ? s.word : defaultWord(n)
  const head = s.head

  // The URL follows state with replaceState. Tab never calls this, because Tab changes no state.
  const first = useRef(true)
  useEffect(() => {
    if (first.current) { first.current = false; return }
    const q = serializeState({ ...s, scaling })
    if (q !== window.location.search) window.history.replaceState(null, '', `${window.location.pathname}${q}${window.location.hash}`)
  }, [s, scaling])

  const names = s.mode === 'designed' ? DESIGNED_HEADS.map((h) => h.name) : DESIGNED_HEADS.map((_, i) => `Head ${i + 1}`)
  const rule = s.mode === 'designed' ? DESIGNED_HEADS[head].rule : 'Seeded weights. No pattern was set.'

  const toggle = (k: SwitchName) => { patch({ [k]: !s[k] } as Partial<UiState>); setLast(k) }
  const shown: SwitchName[] = truth.scalingAvailable ? ['position', 'scaling', 'mask'] : ['position', 'mask']
  const sentence = n > 0 ? consequence(s.text, s.mode, switches, shown.includes(last) ? last : 'position', head, word) : 'Type a sentence to see what the switches change.'

  const topList = useMemo(() => {
    if (!n) return []
    return run.heads[head].weights[word].map((w, j) => ({ w, j })).sort((a, b) => b.w - a.w).slice(0, 4).filter((x) => x.w >= 0.005)
  }, [run, head, word, n])

  const onRailKey = (e: KeyboardEvent<HTMLButtonElement>) => {
    const keys: Record<string, number> = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }
    let next = -1
    if (e.key in keys) next = (head + keys[e.key] + 4) % 4
    else if (e.key === 'Home') next = 0
    else if (e.key === 'End') next = 3
    if (next < 0) return
    e.preventDefault()
    patch({ head: next })
    document.getElementById(`head-${next}`)?.focus()
  }

  return (
    <div className="app">
      <header className="top">
        <p className="house-label truth" id="truth">{truth.text}</p>
        <h1>Attention</h1>
        <p className="lede">Type a sentence. See what each word reads. Then break the math.</p>
      </header>

      <section className="entry" aria-label="Your sentence">
        <label htmlFor="sentence" className="field-label">Your sentence</label>
        <input
          id="sentence" type="text" value={s.text} autoComplete="off" spellCheck={false} maxLength={200}
          onChange={(e) => patch({ text: e.target.value, word: null })} aria-describedby="count"
        />
        <p id="count" className={run.truncated ? 'count warn' : 'count'}>
          {n} of {MAX_TOKENS} words and marks used.
          {run.truncated ? ` Your sentence has ${run.total}, so only the first ${MAX_TOKENS} are read.` : ''}
        </p>
        <div className="presets" role="group" aria-label="Example sentences">
          {PRESETS.map((p) => (
            <button key={p} type="button" className="chip" onClick={() => patch({ text: p, word: null })}>
              {p.length > 24 ? `${p.slice(0, 22)}...` : p}
            </button>
          ))}
        </div>
      </section>

      <section className="stage" aria-label="Attention">
        {n === 0 ? (
          <div className="empty"><p><strong>Nothing to read yet.</strong></p><p>Type a sentence or pick an example.</p></div>
        ) : (
          <Sentence tokens={run.tokens} weights={run.heads[head].weights} baseline={baseline.heads[head].weights} word={word} onSelect={(i) => patch({ word: i })} />
        )}
        <div className="readout">
          <p className="read" role="status">{readout(run, head, word)}</p>
          {topList.length > 0 && (
            <ul className="top-reads" aria-label="Largest weights for this word">
              {topList.map(({ w, j }) => (
                <li key={j}><span>{run.tokens[j]}</span> <span className="figure">{Math.round(w * 100)}%</span></li>
              ))}
            </ul>
          )}
          <p className="sum figure">{rowSumLine(run, head)}</p>
        </div>
      </section>

      <section className="rail-wrap" aria-label="Heads">
        <h2 className="small-h" id="rail-h">Head</h2>
        <div className="rail" role="radiogroup" aria-labelledby="rail-h">
          {names.map((nm, i) => (
            <button
              key={i} id={`head-${i}`} type="button" role="radio" aria-checked={i === head} tabIndex={i === head ? 0 : -1}
              className="rail-btn" onClick={() => patch({ head: i })} onKeyDown={onRailKey}
            >
              <span className="num figure" aria-hidden="true">{i + 1}</span>
              <span>{nm}</span>
            </button>
          ))}
        </div>
        <p className="rule">{rule}</p>
      </section>

      <section className="break" aria-labelledby="break-h">
        <h2 className="small-h" id="break-h">Break the math</h2>
        <div className="switches">
          {shown.map((k) => (
            <button key={k} type="button" role="switch" aria-checked={s[k] && (k !== 'scaling' || scaling)} className="sw" onClick={() => toggle(k)}>
              <span className="sw-track" aria-hidden="true"><span className="sw-thumb" /></span>
              <span className="sw-text"><span className="sw-name">{SWITCH_COPY[k].label}</span><span className="sw-hint">{SWITCH_COPY[k].hint}</span></span>
            </button>
          ))}
        </div>
        <p className="computed" role="status">{sentence}</p>
        {!truth.scalingAvailable && <p className="note small">Scaling stays hidden until the designed heads pass their tests.</p>}
      </section>

      <details className="house-drawer drawer" onToggle={(e) => setOpen(e.currentTarget.open)}>
        <summary>Show the numbers</summary>
        <div className="house-drawer-body">
          {open && <Numbers run={run} head={head} mode={s.mode as WeightsMode} onMode={(m) => patch({ mode: m })} />}
        </div>
      </details>

      <p className="source small">
        From <a href={PAPER_URL} target="_blank" rel="noreferrer">Attention Is All You Need</a>. This is one layer at toy size.
      </p>

      <Receipt />
      <Built />
    </div>
  )
}
