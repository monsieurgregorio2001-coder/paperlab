# ADR 0006 — Sandbox first, with an optional tour

Status: accepted · 2026-10-15

## Decision
Every lab opens on the sandbox with a default dataset already in motion. A "Tour" button starts a sequence of steps; each step sets the scene state, highlights one element, and shows one caption (≤ 40 words). Steps are data (`tour.ts`), replayable, keyboard-navigable, and never lock the sandbox. A "Maths" drawer shows the formulas rendered in a lightweight way (no KaTeX; HTML + Unicode, as v1 did) with the current values substituted live.

## Consequences
Tours are testable (each step's state is a fixture). Captions are copy-reviewed like any other text.
