# ADR 0005 — Frontier lab: Qwen3-0.6B live, SmolLM2-360M fallback, DeepSeek-V3 drawn to scale

Status: accepted · 2026-10-15

## Context
Research 2026-10-15: transformers.js 4.3 (Apache 2.0) runs Qwen3 on WebGPU; q4f16 Qwen3-0.6B needs ~1.4 GB VRAM (web-llm config), SmolLM2-360M ~376 MB. transformers.js has no general `output_attentions`; the ishamf/llm-visualizer instrumentation script (MIT) promotes post-RoPE Q/K/V to ONNX outputs with a validator. DeepSeek-V3's `config_671B.json` publishes every number needed to draw MLA and the MoE router.

## Decision
- Live model: Qwen3-0.6B, q4f16, via `@huggingface/transformers` on WebGPU, instrumented at export time so per-layer Q/K/V are outputs; attention is rebuilt in JS and validated against the fused output in CI on toy inputs. Users opt in to the download with the size shown; SmolLM2-360M is offered on phones or when the adapter is weak.
- Static diagram: DeepSeek-V3 (MLA + 256 experts, 8 active, group-limited routing) with every number cited to its config line; gpt-oss sinks and alternating windows as a second panel if time allows.
- No API calls; everything runs locally.

## Consequences
A one-off instrumentation step in `models/frontier/` (Python, documented). Model files are fetched from Hugging Face at runtime, not committed.
