# Vivek Rao Quant Research Workspace

This workspace keeps the 55 public repositories isolated from AMOS research and EA implementation.

## Layers

- `SOURCE_MIRROR/`: immutable source references and provenance. **READ ONLY**.
- `AMOS_EDGE_LAB/`: reusable quantitative EDGE extraction for the whole AMOS platform.
- `EA_RESEARCH_LAB/`: EA/BOT-oriented translations, experiments, and Nautilus validation.

## Promotion rule

`SOURCE_MIRROR -> AMOS_EDGE_LAB -> EA_RESEARCH_LAB`

Source code is never edited in place. Any adaptation must be copied into the downstream layer with provenance (source repo, commit SHA, license, extracted concept, modifications).

TickScalper reconstruction remains a separate mainline project. Vivek-derived modules may only enter it as explicitly labeled external enhancements after original-parity work.
