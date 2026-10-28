import { describe, expect, it } from 'vitest';
// @ts-expect-error plain .mjs helper
import { validateEvidence } from './evidence-schema.mjs';

const good = {
  schema: 1, room: 'paperlab', commit: 'abc1234', ran_at: '2026-10-28T00:00:00Z', runner: 'x',
  tests: [{ id: 'unit', label: 'Unit', passed: 1, total: 1, command: 'npx vitest run' }],
  evals: [],
  figures: [{ id: 'tests', value: 1, unit: 'tests', label: 'l', source: 's' }],
  budgets: [{ id: 'js-gz', limit: 120000, actual: 1 }],
  limits: ['One layer.'],
};

describe('evidence schema', () => {
  it('accepts a valid file', () => expect(validateEvidence(good)).toEqual([]));
  it('rejects wrong schema, missing arrays and bad numbers', () => {
    expect(validateEvidence({ ...good, schema: 2 })).not.toEqual([]);
    expect(validateEvidence({ ...good, limits: undefined })).not.toEqual([]);
    expect(validateEvidence({ ...good, budgets: [{ id: 'js-gz', limit: 'x', actual: 1 }] })).not.toEqual([]);
    expect(validateEvidence({ ...good, tests: [{ ...good.tests[0], passed: 2 }] })).not.toEqual([]);
  });
});
