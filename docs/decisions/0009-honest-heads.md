# ADR 0009 — Designed heads are true attention, hand-set

Status: proposed by Claude on the owner's delegation, 2026-10-28. Reversible.

## Context
The audit found a defect: a "designed patterns" idea could be drawn without the math behind it. The first version used random weights, which is honest but shows no pattern worth reading. Any "Real math" label must be backed by a test.

## Decision
- The four heads are true `softmax(QK^T / sqrt(d_k)) V` through the one kernel in `src/lib/transformer.ts`.
- Only W_Q and W_K are hand-set (`src/lib/designed.ts`). W_V and W_O stay seeded. The toy embeddings carry labelled designed features, listed on screen in "Show the numbers".
- Heads: Previous word, Same word, Punctuation, Next word.
- Tests: each head equals an independent plain-loop kernel for its stated weights, head 1 puts at least 0.5 on the previous word in every preset, and a NumPy fixture agrees at 1e-6.
- The label "Real math · Designed heads · Toy size" is compiled in only when `src/lib` tests pass at build time. Otherwise the page says "Designed patterns · Toy size" and hides the Scaling switch.
- A random weights mode keeps the first version's engine, labelled "Real math · Random weights · Toy size".
- The prototype's `model.js` stays retired. One attention kernel exists.
- The limits line reads "Hand-set heads, not learned ones." only while this ships.

## Consequences
The Position switch shows a real cause: the previous and next heads read position waves, and they stop working without them.
- The "it / animal" preset is dropped. A toy with hand-set heads cannot resolve a pronoun, so the sentence would promise something the room does not do.
- The token id copy no longer says the model "cannot read letters". Ids here come from a hash, and the room says a real model looks them up in a learned list.
- Paper links point to the page of the section they name (3.2.1 is on page 4, 3.2.2 and 3.4 on page 5, 3.5 on page 6).
