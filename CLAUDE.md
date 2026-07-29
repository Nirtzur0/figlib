# sci-figures / figlib

A figure compiler for mathematical exposition. A figure is a **program**
(`figures/*.py`) that computes geometry, builds a typed `Scene`, and
declares numerical assertions about it. Rendering is deterministic;
correctness is enforced by gates, not by looking at the picture.

**If you are about to write or edit a figure, read `docs/skill.md` first.**
It is the model-facing entry point and it is short. Everything below is
routing.

## Read order

| when | file |
|---|---|
| writing/editing a figure — always, first | `docs/skill.md` |
| before writing code: the design step 0–9 | `docs/architecture.md` |
| a judge or readback flagged a visual defect | `docs/grammar.md` |
| a design step feels arbitrary | `docs/exposition.md` |
| working on figlib itself (build order, what's missing) | `docs/primitive-gaps.md` |
| what the corpus of reference figures actually demands | `docs/corpus-study.md` |
| in-flight design specs | `docs/superpowers/specs/` |

`docs/architecture.md` owns the invariants. When a doc and the code
disagree, the code wins and the doc is a bug — fix it.

## Layout

```
src/figlib/      the library. Flat modules, imported as `from figlib.X import Y`
                 (see figlib/__init__.py for the module map). No package
                 façade — there is one obvious module per concern.
figures/*.py     the corpus: one figure program per file
figures/out/     COMMITTED render baselines (svg+png). These are the golden
                 files `--regress` diffs against; they are not scratch.
figures/refs/    scanned originals for comparative judging
tests/           pytest over the library; tests/svgkit.py parses emitted SVG
docs/            everything above
```

## Commands

cairo needs `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`. The Makefile
exports it — **use the make targets, or the render will fail with an
opaque cairo error.**

```
make test                        pytest
make check F=figures/x.py        render + all deterministic gates (exit 1 on fail)
make check F="figures/x.py --report"   + textual layout inventory
make regress                     corpus-wide golden diff (exit 1 on drift)
make update                      refresh the committed svg+png baselines
```

Raw form, if you need flags the targets don't pass through:
`DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib uv run figcheck <args>`.
Bare `figcheck` is not on PATH.

## The invariants (violating these is a bug, not a style choice)

- **Content code never names a color, font, or stroke width.** It names
  meanings: `Role.CONTENT`, `theme.ramp(t)`, `theme.categorical(i)`,
  `theme.surface_shade(t)`. A hex literal in a figure means that figure
  silently stops retheming, and no gate will catch it.
- **Canvas units are display CSS pixels.** A figure declares its page slot
  (`FORMAT = MARGIN | COLUMN | WIDE`) and every absolute quantity is at
  final rendered size. If annotation doesn't fit, take a larger format or
  cut ink — never shrink type.
- **New capability enters as a producer of scene items, never as a new
  renderer.** `surface3d` and `plots` both follow this; so must anything
  new. One rendering path means gates and themes apply for free.
- **Diagnostics contain the fix.** Collisions print verified `offset_px`
  nudges; clipping prints the overrun per edge; load failures print a
  computed format verdict. Apply what the gate says; don't guess and
  re-render.
- **Assert what could be wrong, never what is true by construction.**
- **`figures/out/` changes are reviewed.** Any diff there is a real render
  change — the renderer is byte-deterministic. Look at what moved before
  running `make update`.

## Definition of done for a figure

Gates pass, `make regress` is clean, and a readback record exists
(`figcheck --readback-prompt` → cold agent → `readback.record()`, written
to `figures/out/<name>.readback.md`). A figure without a readback record
is not done.

## Vault

This project has no folder in the Obsidian projects estate yet. Judgment
truth (verdict, roadmap, design rationale) currently lives in
`docs/superpowers/specs/`; run `/project-sync` to scaffold
`projects/experiments/sci-figures/` if strategy-level state starts
outgrowing the repo.
