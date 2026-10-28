// Shape check for dist/evidence.json (DIRECTION 4.4, schema 1). Returns a list of problems.
const isStr = (v) => typeof v === 'string' && v.length > 0;
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

export function validateEvidence(e) {
  const p = [];
  if (!e || typeof e !== 'object') return ['not an object'];
  if (e.schema !== 1) p.push('schema must be 1');
  for (const k of ['room', 'commit', 'ran_at', 'runner']) if (!isStr(e[k])) p.push(`${k} must be a non-empty string`);
  if (isStr(e.ran_at) && Number.isNaN(Date.parse(e.ran_at))) p.push('ran_at must be an ISO date');
  const arr = (k, fn) => {
    if (!Array.isArray(e[k])) return p.push(`${k} must be an array`);
    e[k].forEach((x, i) => fn(x, `${k}[${i}]`));
  };
  const strs = (x, w, keys) => keys.forEach((k) => { if (!isStr(x?.[k])) p.push(`${w}.${k} must be a non-empty string`); });
  arr('tests', (x, w) => {
    strs(x, w, ['id', 'label', 'command']);
    if (!isNum(x?.passed) || !isNum(x?.total) || x.passed > x.total) p.push(`${w} needs numeric passed <= total`);
  });
  arr('evals', (x, w) => {
    strs(x, w, ['id', 'label', 'command', 'labelled_by', 'limits']);
    if (!isNum(x?.passed) || !isNum(x?.total)) p.push(`${w} needs numeric passed/total`);
  });
  arr('figures', (x, w) => {
    strs(x, w, ['id', 'unit', 'label', 'source']);
    if (!isNum(x?.value)) p.push(`${w}.value must be a number`);
  });
  arr('budgets', (x, w) => {
    strs(x, w, ['id']);
    if (!isNum(x?.limit) || !isNum(x?.actual)) p.push(`${w} needs numeric limit/actual`);
  });
  arr('limits', (x, w) => { if (!isStr(x)) p.push(`${w} must be a non-empty string`); });
  return p;
}
