import { useEffect, useState } from 'react'

interface Evidence {
  commit: string
  ran_at: string
  runner: string
  tests: { id: string; label: string; passed: number; total: number; command: string }[]
  evals: { id: string; label: string; passed: number; total: number }[]
  budgets: { id: string; limit: number; actual: number }[]
  limits: string[]
}

type State = { kind: 'loading' } | { kind: 'missing' } | { kind: 'ok'; e: Evidence }

const kb = (b: number) => `${(b / 1000).toFixed(1)} KB`

export default function Receipt() {
  const [s, setS] = useState<State>({ kind: 'loading' })
  useEffect(() => {
    let live = true
    fetch(`${import.meta.env.BASE_URL}evidence.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((e: Evidence) => { if (live) setS({ kind: 'ok', e }) })
      .catch(() => { if (live) setS({ kind: 'missing' }) })
    return () => { live = false }
  }, [])

  return (
    <section className="receipt house-receipt" aria-labelledby="receipt-h">
      <div className="house-receipt-head">
        <h2 id="receipt-h" className="house-receipt-name">Receipt</h2>
        {s.kind === 'ok' && <span className="house-label">Commit {s.e.commit}</span>}
      </div>
      {s.kind === 'loading' && <p role="status">Loading the receipt.</p>}
      {s.kind === 'missing' && <p><span className="house-fig is-missing">not measured yet</span> The receipt is written by the build, so it is missing in a dev server.</p>}
      {s.kind === 'ok' && (
        <dl className="house-receipt-rows">
          <div>
            <dt>Tests</dt>
            <dd>
              <ul className="rows">
                {s.e.tests.map((t) => (
                  <li key={t.id}>
                    <span className="house-fig">{t.passed} of {t.total}</span> {t.label}
                    <code>{t.command}</code>
                  </li>
                ))}
              </ul>
            </dd>
          </div>
          <div>
            <dt>Evals</dt>
            <dd>{s.e.evals.length ? s.e.evals.map((x) => `${x.label}: ${x.passed} of ${x.total}`).join('. ') : 'None. Nothing here is model output to score.'}</dd>
          </div>
          <div>
            <dt>Budgets</dt>
            <dd>
              <ul className="rows">
                {s.e.budgets.map((b) => (
                  <li key={b.id}><span className="house-fig">{kb(b.actual)} of {kb(b.limit)}</span> {b.id === 'js-gz' ? 'JavaScript before interaction, gzipped' : b.id}</li>
                ))}
              </ul>
            </dd>
          </div>
          <div>
            <dt>What this does not do</dt>
            <dd>{s.e.limits.join(' ')}</dd>
          </div>
          <div>
            <dt>Run</dt>
            <dd className="small">{s.e.runner}, {s.e.ran_at}</dd>
          </div>
        </dl>
      )}
    </section>
  )
}
