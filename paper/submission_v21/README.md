# PhaseGate NeurIPS 2026 Workshop Paper (Anonymous v21)

This directory contains the self-contained LaTeX source for the anonymous v21 paper.
For easier editing in Overleaf, prose is formatted with one sentence per physical source line.
These single line breaks do not create new LaTeX paragraphs or alter the rendered paper.

## Build

Compile `main.tex` with a modern LaTeX engine. The supplied paper was built with Tectonic. The two PDF figures are already included under `figures/`.

To regenerate the figures, run:

```bash
MPLBACKEND=Agg python make_paper_figures.py
```

The plotting script embeds TrueType fonts and uses a Times-family serif font.

## Contents

- `main.tex`: canonical paper source.
- `PhaseGate_NeurIPS2026_Workshop_Anonymous_Polished_v21.tex`: named copy of the canonical source.
- `neurips_2026.sty`: workshop style file.
- `figures/`: publication PDF figures and PNG previews.
- `make_paper_figures.py`: deterministic figure-generation script.
- `REVISION_NOTES_v21.md`: summary of the v21 abstract wording revision.

The paper contains five main-text pages plus one references page.
