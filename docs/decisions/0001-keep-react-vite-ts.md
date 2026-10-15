# ADR 0001 — Keep React 18, TypeScript and Vite; add Three.js directly

Status: accepted · 2026-10-15

## Context
PaperLab v1 is React 18 + TS + Vite with 249 lines of tested pure math. The redesign adds ten labs and heavy 3D.

## Decision
Keep the stack. Add `three` (r186, MIT) used directly through a small `useScene` hook, not react-three-fiber (its WebGPU path is alpha and it would own the render loop). Labs register in a `labs/` registry; each lab is one folder with `math.ts` (pure, tested), `scene.ts` (Three.js), `Lab.tsx` (controls + captions) and `tour.ts`.

## Consequences
One render loop per lab, paused when off screen. React owns controls and text; Three.js owns the canvas. Bundle per lab is code-split so the landing page stays light.
