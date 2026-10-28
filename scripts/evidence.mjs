// Writes dist/evidence.json (DIRECTION 4.4, schema 1). Run after `npm run build`.
// Fails (exit 1) if any test fails, the output shape is invalid, or a budget is exceeded.
import { spawnSync, execSync } from 'node:child_process';
import { readdirSync, readFileSync, writeFileSync, mkdtempSync, existsSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { validateEvidence } from './evidence-schema.mjs';

const JS_GZ_LIMIT = 120000; // DIRECTION 8: PaperLab Attention, gzip
const dist = 'dist';
if (!existsSync(dist)) { console.error('dist/ missing: run npm run build first'); process.exit(1); }

const out = join(mkdtempSync(join(tmpdir(), 'evidence-')), 'vitest.json');
const run = spawnSync('npx', ['vitest', 'run', '--reporter=json', `--outputFile=${out}`], { stdio: ['ignore', 'inherit', 'inherit'] });
let report;
try { report = JSON.parse(readFileSync(out, 'utf8')); } catch { console.error('vitest produced no report'); process.exit(1); }
const passed = report.numPassedTests, total = report.numTotalTests;
// The math suite is src/lib (the attention kernel). Tooling tests are counted in their own row.
const count = (pick) => report.testResults.filter((r) => pick(r.name.replace(/\\/g, '/')))
  .flatMap((r) => r.assertionResults)
  .reduce((n, a) => ({ passed: n.passed + (a.status === 'passed'), total: n.total + 1 }), { passed: 0, total: 0 });
const math = count((n) => n.includes('/src/lib/'));
const tooling = count((n) => !n.includes('/src/lib/'));

const walk = (d) => readdirSync(d, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(join(d, e.name)) : [join(d, e.name)]));
const jsGz = walk(dist).filter((f) => f.endsWith('.js')).reduce((n, f) => n + gzipSync(readFileSync(f)).length, 0);

const commit = (process.env.GITHUB_SHA || execSync('git rev-parse HEAD').toString().trim()).slice(0, 7);
const runner = process.env.GITHUB_ACTIONS
  ? `GitHub Actions ${process.env.ImageOS || process.env.RUNNER_OS}, Node ${process.versions.node.split('.')[0]}`
  : `local, Node ${process.versions.node.split('.')[0]}`;

const evidence = {
  schema: 1,
  room: 'paperlab',
  commit,
  ran_at: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
  runner,
  tests: [
    { id: 'unit', label: 'Unit tests on the attention math', ...math, command: 'npx vitest run src/lib' },
    { id: 'tooling', label: 'Tests on the evidence tooling', ...tooling, command: 'npx vitest run scripts' },
  ],
  evals: [],
  figures: [
    { id: 'math-tests', value: math.total, unit: 'tests', label: 'unit tests on the attention math', source: 'src/lib/transformer.test.ts' },
    { id: 'js-gz', value: jsGz, unit: 'bytes', label: 'gzipped JavaScript in the production build', source: 'dist/**/*.js' },
  ],
  budgets: [{ id: 'js-gz', limit: JS_GZ_LIMIT, actual: jsGz }],
  limits: ['Does not train anything.', 'One layer.', 'Random weights, not learned ones.'],
};

const problems = validateEvidence(evidence);
writeFileSync(join(dist, 'evidence.json'), JSON.stringify(evidence, null, 2) + '\n');
console.log(JSON.stringify(evidence, null, 2));

if (problems.length) { console.error('evidence shape invalid:\n' + problems.join('\n')); process.exit(1); }
if (run.status !== 0 || passed !== total || report.numFailedTests > 0) { console.error('tests failed'); process.exit(1); }
for (const b of evidence.budgets) if (b.actual > b.limit) { console.error(`budget ${b.id} exceeded: ${b.actual} > ${b.limit}`); process.exit(1); }
