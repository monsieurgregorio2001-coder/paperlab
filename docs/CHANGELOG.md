# Changelog

## Unreleased

### Changed
- House 2.0.1: lock, sync, CI ref (hub 6b2ee82), room links (Attention, First version) in the Menu, `docs/MOTION.md`.

### Added
- Attention room at `/paperlab/` (phase 3a): designed heads on the tested kernel (ADR 0009), Break the math switches, Show the numbers drawer, receipt from `evidence.json`, house 2.0.1 crumb and footer, `404.html`.
- First version kept at `/paperlab/legacy/`.
- `tests/fixtures/attention.json` from `scripts/attention_fixture.py`, checked at 1e-6.
- ADRs 0008 to 0010, proposed by Claude, reversible.
- `docs/` with vision, ADRs 0001–0007, the ladder plan, v1 audit, tracker.
- `CLAUDE.md` conventions.
- Phase K kernels for Line, Boundary, Net and Memory in `src/labs/*/math.ts`, tested against NumPy fixtures from `scripts/fixtures.py`; tiny RNN/LSTM weights from `models/memory/train.py`.
