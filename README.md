# figlib

A figure compiler for mathematical exposition.

A figure here is not a drawing — it is a **program** with a claim attached:

```python
CLAIM  = "one sentence: what the figure ARGUES, not what it draws"
THEME  = RISO           # CLEAN | RISO
FORMAT = COLUMN         # MARGIN 340 | COLUMN 680 | WIDE 1000 px page slot
PARAMS = {...}          # every tunable; no magic numbers below

def compute(p): ...     # numerics -> arrays. No drawing decisions.
def build(g):   ...     # arrays -> Scene (or Figure, for multi-panel)
def assertions(g): ...  # numerical gate on the SAME arrays that got drawn
```

`compute → build → autoplace → gates → render` produces SVG + PNG. What
makes it a compiler rather than a plotting wrapper is the gate stack:

- **numerical** — the program's own assertions on the plotted arrays
- **mechanical** — label collisions, clipping, type below 8.5 pt,
  annotation load over 22% of the canvas; each diagnostic carries a
  *computed* fix (`offset_px += (+0, -13)`), not a complaint
- **color** — per-channel: correspondence hues must be pairwise separable,
  an order ramp must be monotone in lightness, ink must clear a contrast floor
- **golden regression** — rendering is byte-deterministic, so any diff
  against the committed baselines in `figures/out/` is a real change
- **readback** — a cold agent sees only the PNG and says what it claims;
  if that doesn't match `CLAIM`, the figure is wrong however pretty it is

The invariant underneath all of it: **content code names meanings, never
appearance.** `Role.CONTENT`, `theme.ramp(t)`, `theme.categorical(i)`.
Themes map meanings to ink, so a theme change restyles the whole corpus in
one edit, and colors carry their channel into the scene where the gate can
hold each to its own standard.

## Running it

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and cairo
(`brew install cairo`).

```
make test                        # pytest
make check F=figures/vca_fig9_cassinian.py    # render + all gates
make regress                     # corpus-wide golden diff
make update                      # refresh committed baselines
```

## Layout

```
src/figlib/    the library (see figlib/__init__.py for the module map)
figures/       the corpus, one program per figure; out/ holds committed baselines
docs/          skill.md (how to write a figure) · architecture.md (the stack
               and the design step) · grammar.md · exposition.md
tests/
```

Start at `docs/skill.md`.
