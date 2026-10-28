# ADR 0002 — Three.js `three/webgpu` renderer with automatic WebGL2 fallback

Status: accepted · 2026-10-15

## Context
Owner chose "go big on 3D". Research (2026-10-15): Three.js r186 ships WebGPU and WebGL renderers behind one API with TSL shaders; WebGPU compute is still flaky on some iPhones; forum reports show WebGL winning on draw-call-heavy scenes.

## Decision
Use `WebGPURenderer` with `forceWebGL` fallback when `navigator.gpu` is missing or adapter request fails. Cap `devicePixelRatio` at 2 (1.5 on phones). Prefer `InstancedMesh` for grids of cubes/points and TSL compute for particle fields; no custom WGSL until a profile shows the need. Every scene has a frame budget (≤ 8 ms on an M1 Air at 1440×900) measured in the lab's check script.

## Consequences
One dependency for all 3D. Scenes must be written against the node-material API so they compile to both backends. A no-WebGL state (static SVG or canvas 2D) is mandatory per lab.
