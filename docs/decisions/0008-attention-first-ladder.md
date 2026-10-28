# ADR 0008 — Attention ships first, the ladder waits

Status: proposed by Claude on the owner's delegation, 2026-10-28. Reversible.

## Context
The v2 ladder plan (`docs/plan/2026-10-15-ladder.md`) draws ten labs. Only the kernels for four of them exist and are tested. DIRECTION principle 6 says finished or absent: no rungs for labs that are not built.

## Decision
- v2.0 ships one lab, Attention, at `/paperlab/`.
- Line (3D loss bowl, three.js WebGPU with a WebGL2 fallback and a poster) follows after launch if it passes the bar.
- Boundary, Net, Memory, Space, Sight, Noise, Block, and Frontier wait. ADR 0005 stays accepted and unscheduled.
- No rungs are drawn. The lab switcher appears when there are two labs.
- The kernels from PR #2 are merged without UI. They are tested and counted in the receipt on their own row.

## Consequences
This amends the ladder plan: its Phase 0 shell and landing page are not built for v2.0. The first version of the app stays at `/paperlab/legacy/`.
