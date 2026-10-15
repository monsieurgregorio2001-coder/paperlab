# Tracker

Status values: `todo` · `doing` · `review` · `done` · `later` · `dropped`. Plan: [`plan/2026-10-15-ladder.md`](plan/2026-10-15-ladder.md).

## Phase 0 — Foundation
| Item | Status | Notes |
| --- | --- | --- |
| Docs tree, ADRs 0001–0007, plan, audit | review | this PR |
| Identity: tokens, type, mark, spectrum | todo | |
| Shell: landing ladder, router, chrome, About | todo | |
| Engine: `useScene`, WebGPU + WebGL fallback, DPR cap, frame probe | todo | |
| Tour runner, maths drawer, toy badge | todo | |
| Checks + CI | todo | |

## Phase K — Kernels
| Item | Status | Notes |
| --- | --- | --- |
| Line math + fixtures | review | `phase-k/kernels`. OLS, GD trace, MSE surface. Fixtures in `tests/fixtures/line.json` from `scripts/fixtures.py`. |
| Boundary math + fixtures | review | `phase-k/kernels`. Logistic GD trace, perceptron trace, sigmoid sampler, seeded `twoBlobs` and `xor`. |
| Net (MLP) math + fixtures | review | `phase-k/kernels`. Written from scratch, not derived from TF Playground's `nn.ts`, so no Apache header. L1 dead-weight rule follows the Playground. A test asserts ≥ 60 training steps/s for 2 → 8×4 → 1 on 200 points. |
| Memory (RNN/LSTM) math + fixtures | review | `phase-k/kernels`. Bracket-depth task; weights trained by `models/memory/train.py` and committed as a tiny exception (see its README). |

## Phase 1 — Fitting labs
| Item | Status | Notes |
| --- | --- | --- |
| Line | todo | |
| Boundary | todo | |
| Net | todo | |

## Phase 2 — Attention labs
| Item | Status | Notes |
| --- | --- | --- |
| Attention | todo | kernel exists |
| Block | todo | |
| Retire `/legacy/` | todo | |

## Phase 3 — Representation labs
| Item | Status | Notes |
| --- | --- | --- |
| Space | todo | embedding set + licence |
| Sight | todo | train + export CNN |
| Memory | todo | |

## Phase 4 — Frontier
| Item | Status | Notes |
| --- | --- | --- |
| Noise | todo | |
| Frontier live (Qwen3-0.6B) | todo | instrumentation + validator |
| Frontier static (DeepSeek-V3) | todo | cite every number |

## Phase 5 — Polish and launch
| Item | Status | Notes |
| --- | --- | --- |
| Live thumbnails, OG per lab | todo | |
| Hub case page + OG refresh | todo | |
| Post-launch audit | todo | |

## Later
| Item | Notes |
| --- | --- |
| gpt-oss sinks + alternating windows panel | if Phase 4 has room |
| Filipino captions | |
