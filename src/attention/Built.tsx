/** "How this was built". Entries come only from this repo's ADRs, commits, and PRs. */
const COLUMNS: { title: string; items: string[] }[] = [
  {
    title: 'What AI did',
    items: [
      'Wrote the v2 vision, ADRs 0001 to 0007, the ladder plan, and the v1 audit (commit c5854ea).',
      'Built tested maths kernels for four later labs, checked against NumPy at 1e-6 (Phase K, PR #2). They have no screen yet.',
      'Wrote the build step that produces this page’s receipt (commits 7785610 and 23e8ff9).',
      'Built the designed heads, their tests, and this room (branch phase-3a/attention).',
    ],
  },
  {
    title: 'What I decided',
    items: [
      'Wrote and published v1, a four-step walk through one encoder layer (commit 702bade). It stays as the first version.',
      'Chose a look of PaperLab’s own over the portfolio’s editorial one (ADR 0004).',
      'Set the accuracy policy: pure tested maths, labelled toys, no number a scene does not compute (ADR 0003).',
      'Set the tour rule: sandbox first, and a tour never locks it (ADR 0006).',
    ],
  },
  {
    title: 'What I rejected or corrected',
    items: [
      'The v1 audit rejected the step buttons, the table views, and the editorial styling, and found about 250 KB of font subsets that were never loaded.',
      'The hub quoted 19 tests, so the math tests got their own row. The old limits line said random weights until the heads changed.',
      'Not recorded for this project yet: pull request reviews. There are none on this repo.',
    ],
  },
]

const PROPOSED = [
  'ADR 0008: Attention ships first and the ladder waits.',
  'ADR 0009: Designed heads are true attention with hand-set weights, and the "Real math" label needs a green test.',
  'ADR 0010: The face is Recursive. The room uses the house Inter for now.',
]

export default function Built() {
  return (
    <section className="built" aria-labelledby="built-h">
      <h2 id="built-h">How this was built</h2>
      <div className="cols">
        {COLUMNS.map((c) => (
          <div key={c.title} className="col">
            <h3>{c.title}</h3>
            <ul>{c.items.map((t) => <li key={t}>{t}</li>)}</ul>
          </div>
        ))}
      </div>
      <div className="proposed">
        <h3>Proposed by Claude, waiting for Garfield&apos;s review</h3>
        <ul>{PROPOSED.map((t) => <li key={t}>{t}</li>)}</ul>
      </div>
    </section>
  )
}
