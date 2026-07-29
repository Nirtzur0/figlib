# Figure Gallery — Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the figure corpus into subject subdirectories, change the render contract so grain is ink and every figure ships both a transparent and a RISO-cream ground, and add the `EXPOSITION` field plus a generated gallery — leaving the tree green so 23 agents can be dispatched against it.

**Architecture:** Path resolution is centralized in `regress.py::discover_programs` and a new `regress.py::artifact_dir`, so subject subdirectories are additive rather than a refactor. The render-contract changes are two small edits in `render.py::_emit_grain` and `regress.py::variants`. `EXPOSITION` becomes a module-level field on the figure-program contract in `program.py`, checked by a new gate that ships **disabled** and is enabled in Phase 1.

**Tech Stack:** Python 3.12, `uv`, pytest, cairosvg + Pillow, `xml.etree` for SVG emission. cairo needs `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` — always use the make targets.

## Global Constraints

- **Use the make targets.** `make test`, `make check F=...`, `make regress`, `make update`. A raw `uv run figcheck` misses the cairo library path and fails opaquely.
- **Content code never names a color, font, or stroke width.** Figures name meanings: `Role.CONTENT`, `theme.ramp(t)`, `theme.categorical(i)`.
- **New capability enters as a producer of scene items, never as a new renderer.**
- **Assert what could be wrong, never what is true by construction.**
- **`figures/out/` changes are reviewed.** The renderer is byte-deterministic; any diff there is a real render change. Look at what moved before `make update`.
- **Test first.** Write the failing test, run it, watch it fail for the right reason, then implement.
- Cream is RISO's paper: `("#f2ede1", "#e9dfcd")`. Grain tile alpha caps at 25/255; RISO `grain = 0.5`.
- Subject bins are exactly: `complex`, `signals`, `linalg`, `dynamics`, `circuits`, `capability`. One level deep only.

**Ordering note (deviation from the spec):** the spec presents Part A (render) before Part B (move). This plan inverts them. A pure `git mv` must be invisible to the renderer, so `make regress` printing 24/24 MATCH *after* the move is the proof that path resolution is right. Doing the render change first would let the move's drift hide inside the render's drift.

---

## File Structure

**Modified:**
- `src/figlib/regress.py` — `discover_programs` walks one level deep; new `artifact_dir()`; `variants()` becomes unconditional. Owns all program→artifact path derivation.
- `src/figlib/cli.py` — `_regress_main` derives per-program out dirs via `artifact_dir`; single-figure run does the same.
- `src/figlib/render.py` — `_emit_grain` drops its transparency check.
- `src/figlib/program.py` — `load_program` accepts `EXPOSITION`/`PROBES`; `Report` carries `exposition`.
- `src/figlib/gates.py` — new `exposition_gate()`.
- `tests/test_core.py` — the groundless-grain assertion inverts.
- `Makefile` — new `gallery` target.
- `CLAUDE.md`, `docs/skill.md`, `docs/architecture.md` — invariant rewritten.

**Created:**
- `src/figlib/gallery.py` — generates `GALLERY.md` from program metadata. One responsibility: metadata → markdown. No rendering.
- `tests/test_gallery.py`
- `docs/friction/README.md` — the friction-record template.

**Moved (`git mv`):** 23 programs into 6 subject dirs, and their `figures/out/` artifacts to match.

---

### Task 1: Commit the in-flight library work

The working tree carries a tested, coherent change (transparency-by-default, `_ground()`, `opaque_variant`) with `make regress` already clean. It must be committed before anything moves, so later drift is attributable.

**Files:**
- Modify: none (commit only)

**Interfaces:**
- Consumes: nothing
- Produces: a clean working tree; `RISO_PAPER`/`CLEAN_PAPER` and `opaque_variant()` available in `figlib.theme`

- [ ] **Step 1: Confirm the tree is green before committing**

```bash
make test && make regress
```
Expected: `595 passed`, then `match=24  drift=0  new=0  error=0`

- [ ] **Step 2: Commit the library and doc changes**

```bash
git add -A
git commit -m "render: transparency by default; grain and casings get an explicit ground

Style.transparent defaults True — a figure is ink meant to land on
whatever page embeds it. Casings, halos and hollow fills need SOME
opaque colour even with no ground, so _ground() gives them white, the
same hostile ground the contrast gate already assumes.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 3: Verify the tree is clean**

```bash
git status --short
```
Expected: no output

---

### Task 2: `discover_programs` walks one level deep

**Files:**
- Modify: `src/figlib/regress.py:57-62`
- Test: `tests/test_regress.py`

**Interfaces:**
- Consumes: nothing
- Produces: `discover_programs(figures_dir) -> list[Path]` — now returns paths in subject subdirectories too, sorted, excluding `out/` and `_`-prefixed names. Callers get full paths, not just names.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_regress.py`, right after `test_discover_skips_private_and_out`:

```python
def test_discover_walks_subject_subdirectories(tmp_path):
    """The corpus bins programs by subject one level deep; out/ still holds
    artifacts, not programs, at any depth."""
    figs = tmp_path / "figures"
    (figs / "complex").mkdir(parents=True)
    (figs / "signals").mkdir()
    (figs / "out" / "complex").mkdir(parents=True)
    (figs / "toplevel.py").write_text("")
    (figs / "complex" / "alpha.py").write_text("")
    (figs / "complex" / "_helper.py").write_text("")
    (figs / "signals" / "beta.py").write_text("")
    (figs / "out" / "complex" / "gamma.py").write_text("")
    found = [str(p.relative_to(figs)) for p in discover_programs(figs)]
    assert found == ["complex/alpha.py", "signals/beta.py", "toplevel.py"]


def test_discover_does_not_recurse_two_levels(tmp_path):
    """One level only — a nested subject tree is not a thing, and silently
    picking up a scratch program two levels down would poison the sweep."""
    figs = tmp_path / "figures"
    (figs / "complex" / "scratch").mkdir(parents=True)
    (figs / "complex" / "alpha.py").write_text("")
    (figs / "complex" / "scratch" / "deep.py").write_text("")
    found = [str(p.relative_to(figs)) for p in discover_programs(figs)]
    assert found == ["complex/alpha.py"]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_regress.py -k "discover" -v
```
Expected: the two new tests FAIL (subdirectory programs are not found — `found == ["toplevel.py"]`); `test_discover_skips_private_and_out` still PASSES.

- [ ] **Step 3: Implement**

Replace `discover_programs` in `src/figlib/regress.py`:

```python
def discover_programs(figures_dir: str | Path) -> list[Path]:
    """Every figure program under figures_dir: *.py, no leading underscore,
    top level or one subject directory deep. `out/` holds artifacts, not
    programs, so it is excluded at every depth; nesting stops at one level
    so a scratch directory inside a subject cannot poison the sweep."""
    root = Path(figures_dir)
    found = [p for p in root.glob("*.py") if not p.name.startswith("_")]
    for sub in root.iterdir():
        if not sub.is_dir() or sub.name == "out" or sub.name.startswith((".", "_")):
            continue
        found += [p for p in sub.glob("*.py") if not p.name.startswith("_")]
    return sorted(found)
```

- [ ] **Step 4: Run the full suite**

```bash
make test
```
Expected: `597 passed`

- [ ] **Step 5: Commit**

```bash
git add src/figlib/regress.py tests/test_regress.py
git commit -m "regress: discover programs one subject directory deep

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `artifact_dir` mirrors the subject into `out/`

**Files:**
- Modify: `src/figlib/regress.py` (add `artifact_dir`; use it in `compare_figure`, `variants`, `sweep`, `update_baselines`)
- Modify: `src/figlib/cli.py:55-80` (`_regress_main`)
- Test: `tests/test_regress.py`

**Interfaces:**
- Consumes: `discover_programs` from Task 2
- Produces: `artifact_dir(program_path: Path, figures_dir: Path) -> Path`. A program at `figures/complex/x.py` maps to `figures/out/complex`; a program directly in `figures/x.py` maps to `figures/out`. `sweep` and `update_baselines` gain a `figures_dir` parameter and derive each program's out dir themselves rather than taking one flat `out_dir`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_regress.py`:

```python
def test_artifact_dir_mirrors_subject(tmp_path):
    from figlib.regress import artifact_dir

    figs = tmp_path / "figures"
    assert artifact_dir(figs / "complex" / "x.py", figs) == figs / "out" / "complex"
    assert artifact_dir(figs / "x.py", figs) == figs / "out"


def test_sweep_finds_baselines_in_subject_subdirs(tmp_path):
    """A subject-binned program's baseline lives in the mirrored out/ dir;
    the sweep must look there and not in a flat out/."""
    from figlib.program import run
    from figlib.regress import sweep

    figs = tmp_path / "figures"
    (figs / "complex").mkdir(parents=True)
    prog = figs / "complex" / "toy_regress.py"
    prog.write_text(TOY)
    run(prog, out_dir=figs / "out" / "complex")

    results = sweep(figs)
    assert [r.status for r in results] == ["match"]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_regress.py -k "artifact_dir or subject_subdirs" -v
```
Expected: first test FAILS with `ImportError: cannot import name 'artifact_dir'`; second FAILS with status `new` (it looks in a flat `figures/out/`).

- [ ] **Step 3: Implement `artifact_dir` and thread it through**

Add to `src/figlib/regress.py` next to `discover_programs`:

```python
def artifact_dir(program_path: str | Path, figures_dir: str | Path) -> Path:
    """Where this program's committed artifacts live: out/ with the
    program's subject directory mirrored into it. A program directly in
    figures/ maps to figures/out/, so nothing outside this repo changes."""
    program_path, root = Path(program_path), Path(figures_dir)
    out = root / "out"
    parent = program_path.parent.resolve()
    if parent == root.resolve():
        return out
    return out / parent.name
```

Rewrite `variants`, `sweep` and `update_baselines` to take `figures_dir` and derive per-program out dirs:

```python
def variants(program_path: Path, out_dir: Path) -> list[bool]:
    """Which renders the corpus commits for this program: always the plain
    (groundless) one, plus the papered one iff a `_paper` baseline exists."""
    if (out_dir / f"{program_path.stem}_paper.svg").exists():
        return [False, True]
    return [False]


def sweep(figures_dir: str | Path, out_dir: str | Path | None = None,
          programs: list[Path] | None = None) -> list[RegressResult]:
    """Compare every program against its committed baseline. `out_dir`
    forces a single flat directory (tests, ad-hoc runs); the default
    derives each program's directory with artifact_dir."""
    figures_dir = Path(figures_dir)
    paths = programs if programs is not None else discover_programs(figures_dir)
    results = []
    for p in paths:
        out = Path(out_dir) if out_dir else artifact_dir(p, figures_dir)
        results += [compare_figure(p, out, paper=t) for t in variants(p, out)]
    return results


def update_baselines(programs: list[Path], out_dir: str | Path | None = None,
                     figures_dir: str | Path | None = None) -> list[Path]:
    """Overwrite the committed SVG+PNG for each program. Returns SVG paths."""
    written: list[Path] = []
    for p in programs:
        out = Path(out_dir) if out_dir else artifact_dir(p, figures_dir or p.parent.parent)
        out.mkdir(parents=True, exist_ok=True)
        for t in variants(p, out):
            written.append(_render_to(p, out, t))
    return written
```

In `src/figlib/cli.py::_regress_main`, replace the flat `out_dir` derivation:

```python
def _regress_main(args) -> int:
    """--regress / --update: the golden-figure harness over the corpus."""
    from pathlib import Path

    from .regress import discover_programs, sweep, update_baselines

    figures_dir = Path(args.figures_dir)
    named = [Path(p) for p in args.program]
    programs = named or discover_programs(figures_dir)

    if args.update:
        written = update_baselines(programs, figures_dir=figures_dir)
        for p in written:
            print(f"UPDATED {p}")
        print(f"\n{len(written)} baseline(s) refreshed under {figures_dir / 'out'}")
        return 0

    results = sweep(figures_dir, programs=programs)
    for r in results:
        print(r.line())
    bad = [r for r in results if not r.ok]
    tally = {k: sum(r.status == k for r in results)
             for k in ("match", "drift", "new", "error")}
    print("\n" + "  ".join(f"{k}={v}" for k, v in tally.items()))
    return 1 if bad else 0
```

- [ ] **Step 4: Run the full suite**

```bash
make test
```
Expected: `599 passed`

- [ ] **Step 5: Verify the real corpus is untouched by the plumbing change**

```bash
make regress
```
Expected: `match=24  drift=0  new=0  error=0` — the corpus is still flat, so `artifact_dir` returns `figures/out` for every program.

- [ ] **Step 6: Commit**

```bash
git add src/figlib/regress.py src/figlib/cli.py tests/test_regress.py
git commit -m "regress: derive each program's out dir, mirroring its subject

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: The move — 23 programs into 6 subject bins

This task changes no code. Its exit criterion is that `make regress` is **still 24/24 MATCH**, which is what proves Tasks 2 and 3 were right.

**Files:**
- Move: 23 `figures/*.py` and their `figures/out/*` artifacts

**Interfaces:**
- Consumes: `artifact_dir` from Task 3
- Produces: the subject-binned tree every later task and every dispatched agent assumes

- [ ] **Step 1: Create the subject directories**

```bash
cd /Users/nirtzur/Documents/projects/sci-figures
mkdir -p figures/{complex,signals,linalg,dynamics,circuits,capability}
mkdir -p figures/out/{complex,signals,linalg,dynamics,circuits,capability}
```

- [ ] **Step 2: Move the programs and their artifacts**

`demo_flow_past_cylinder` bins to `complex/` — it is potential flow, Needham Ch12, and belongs beside `vca_fig12_flow_grid`.

```bash
move_fig () {   # $1 = subject, $2 = stem
  git mv "figures/$2.py" "figures/$1/$2.py"
  for f in figures/out/"$2".* figures/out/"$2"_paper.*; do
    [ -e "$f" ] && git mv "$f" "figures/out/$1/$(basename "$f")"
  done
}
for s in vca_fig4_zn_polar_grid vca_fig9_cassinian vca_fig12_flow_grid \
         vca_fig14_volcanoes vca_fig30_elliptic_checkerboard \
         demo_panels_zsquared demo_sphere_stereographic \
         demo_flow_past_cylinder fig09_exp_series_spiral; do move_fig complex "$s"; done
for s in sampling_aliasing polezero_response dft_matrix_basis; do move_fig signals "$s"; done
for s in matrix_four_views svd_low_rank; do move_fig linalg "$s"; done
for s in strogatz_saddle_node demo_basin_wash demo_ou_ensemble_field \
         diffusion_ode_vs_sde; do move_fig dynamics "$s"; done
for s in schematic_transformer_block induction_head_circuit qk_circuit_tensor; do
  move_fig circuits "$s"; done
for s in demo_solids_gradient demo_glyphs_annulus; do move_fig capability "$s"; done
```

- [ ] **Step 3: Verify nothing was left behind**

```bash
ls figures/*.py 2>/dev/null; ls figures/out/*.* 2>/dev/null; \
  find figures -name "*.py" -not -path "*/out/*" | wc -l
```
Expected: no stray `.py` or artifacts at either top level; the count is `23`.

- [ ] **Step 4: Prove the move is invisible to the renderer**

```bash
make regress
```
Expected: `match=24  drift=0  new=0  error=0`

If any figure reports `new`, its baseline did not move alongside its program — fix the placement, do NOT run `make update`, which would paper over the mistake by writing a fresh baseline.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "corpus: bin 23 figures by subject; regress unchanged at 24/24

A pure move — no renderer change, so the sweep still matching is the
proof that path derivation is right.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Grain is ink — it renders on transparent

**Files:**
- Modify: `src/figlib/render.py:174-183` (`_emit_grain`)
- Test: `tests/test_core.py:85-93`

**Interfaces:**
- Consumes: nothing
- Produces: page-wide grain on every render where `style.grain > 0`, regardless of `transparent`

- [ ] **Step 1: Invert the existing assertion**

`tests/test_core.py` currently encodes the behavior being removed. Replace the groundless assertions (around lines 89-93) with:

```python
        assert "url(#paper)" in papered and "url(#grain)" in papered
        # Groundless drops the PAPER but keeps the grain: grain is the riso
        # PRINT texture, a property of the ink, not of the sheet. A figure
        # run through a riso press is grainy whether or not you can see the
        # paper behind it.
        assert "url(#paper)" not in clear
        assert "url(#grain)" in clear
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/test_core.py -k "ground or paper or grain" -v
```
Expected: FAIL on `assert "url(#grain)" in clear` — the transparent render currently emits no page grain.

- [ ] **Step 3: Implement**

In `src/figlib/render.py`, replace `_emit_grain` (the whole function, comment included):

```python
def _emit_grain(root: ET.Element, defs: ET.Element, style: Style, w: float, h: float) -> None:
    # Grain is INK, not paper: it is the riso print texture, so it rides
    # every render whose theme asks for it — groundless included. The cost
    # is that the overlay is canvas-sized, so a transparent figure's
    # bounding rectangle is faintly visible against a host background. That
    # is bounded and small (tile alpha caps at 25/255, RISO grain is 0.5,
    # so ~5% at peak) and it is the accepted trade.
    grain = getattr(style, "grain", 0.0)
    if grain > 0:
        ET.SubElement(root, "rect", {
            "x": "0", "y": "0", "width": _fmt(w), "height": _fmt(h),
            "fill": f"url(#{_ensure_grain_pattern(defs)})", "opacity": _fmt(grain)})
```

- [ ] **Step 4: Run the full suite**

```bash
make test
```
Expected: `599 passed`

- [ ] **Step 5: Review the drift before accepting it**

```bash
make regress
```
Expected: `drift` on every RISO figure (19 of 23), reported as a small changed-pixel fraction. Open two or three PNGs and confirm the change is a faint uniform speckle over the whole canvas and nothing else moved.

- [ ] **Step 6: Re-baseline and commit**

```bash
make update
make regress
```
Expected: `match=24  drift=0  new=0  error=0`

```bash
git add -A
git commit -m "render: grain is ink, not paper — it survives a groundless render

The page overlay used to zero itself on transparent, reasoning that a
full-bleed speckle over no ground reads as the background the transparent
render exists to remove. The framing was wrong: grain is the riso PRINT
texture, a property of the ink. Suppressing it was the bug.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: RISO everywhere

Cream is RISO's paper, and four figures would otherwise emit a white paper variant.

**Files:**
- Modify: `figures/dynamics/diffusion_ode_vs_sde.py`, `figures/complex/vca_fig4_zn_polar_grid.py`, `figures/complex/fig09_exp_series_spiral.py` (add `THEME = RISO`), `figures/complex/vca_fig9_cassinian.py` (`CLEAN` → `RISO`)

**Interfaces:**
- Consumes: the subject-binned tree from Task 4
- Produces: every figure in the corpus themed RISO

- [ ] **Step 1: Confirm which figures are not RISO**

```bash
grep -L "THEME = RISO" $(find figures -name "*.py" -not -path "*/out/*")
```
Expected exactly these four: `figures/dynamics/diffusion_ode_vs_sde.py`, `figures/complex/vca_fig4_zn_polar_grid.py`, `figures/complex/fig09_exp_series_spiral.py`, `figures/complex/vca_fig9_cassinian.py`

- [ ] **Step 2: Set the theme on each**

For `vca_fig9_cassinian.py`, change the existing line `THEME = CLEAN` to `THEME = RISO` and update its import.

For the other three, which have no `THEME` line, add `THEME = RISO` immediately after the `CLAIM` assignment and add `RISO` to the `figlib.theme` import. Example shape (match each file's existing import style):

```python
from figlib.theme import RISO

CLAIM = "..."
THEME = RISO
```

- [ ] **Step 3: Check each figure still gates green**

```bash
for f in figures/dynamics/diffusion_ode_vs_sde.py figures/complex/vca_fig4_zn_polar_grid.py \
         figures/complex/fig09_exp_series_spiral.py figures/complex/vca_fig9_cassinian.py; do
  make check F=$f || echo "FAILED: $f"
done
```
Expected: `[PASS]` for all four. A RISO theme changes palette and ink weights, so the color gate or a label-collision gate may now fire — if so, apply the fix the diagnostic prints; do not revert the theme.

- [ ] **Step 4: Review the drift, re-baseline, verify**

```bash
make regress          # expect drift on exactly these four
make update
make regress
```
Expected after update: `match=24  drift=0  new=0  error=0`

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "figures: RISO on the four stragglers — cream is the house ground

Three had no THEME line and silently defaulted to CLEAN; cassinian chose
CLEAN for book fidelity. All four move to RISO so every figure's paper
variant is the same cream.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Every figure ships both grounds

**Files:**
- Modify: `src/figlib/regress.py` (`variants`)
- Test: `tests/test_regress.py`

**Interfaces:**
- Consumes: `artifact_dir` from Task 3
- Produces: `variants(program_path, out_dir) -> [False, True]` unconditionally; the corpus grows from 24 to 46 baselines

- [ ] **Step 1: Write the failing test**

Add to `tests/test_regress.py`:

```python
def test_every_program_commits_both_grounds(tmp_path):
    """Transparent for embedding, cream for standing alone — both are
    committed, so neither can silently rot."""
    from figlib.regress import update_baselines, variants

    figs = tmp_path / "figures"
    (figs / "complex").mkdir(parents=True)
    prog = figs / "complex" / "toy_regress.py"
    prog.write_text(TOY)
    out = figs / "out" / "complex"

    assert variants(prog, out) == [False, True]
    update_baselines([prog], figures_dir=figs)
    assert (out / "toy_regress.svg").exists()
    assert (out / "toy_regress_paper.svg").exists()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/test_regress.py -k both_grounds -v
```
Expected: FAIL — `variants` returns `[False]` because no `_paper` baseline exists yet.

- [ ] **Step 3: Implement**

Replace `variants` in `src/figlib/regress.py`:

```python
def variants(program_path: Path, out_dir: Path) -> list[bool]:
    """Both grounds, always: the groundless render for embedding in a
    document, and the theme's own cream paper for standing alone. Sniffing
    for an existing _paper baseline (the old rule) meant a figure that had
    never been papered could never start being papered."""
    return [False, True]
```

- [ ] **Step 4: Run the full suite**

```bash
make test
```
Expected: `600 passed`

- [ ] **Step 5: Generate the 22 new paper baselines**

```bash
make regress
```
Expected: 24 `MATCH` plus 22 `NEW` (the missing `_paper` variants). `NEW` is not a regression, so the exit code is 0.

```bash
make update
make regress
```
Expected: `match=46  drift=0  new=0  error=0`

- [ ] **Step 6: Spot-check a cream render**

Open `figures/out/complex/vca_fig9_cassinian_paper.png` and confirm the cream paper gradient and grain are present.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "corpus: every figure commits both grounds — transparent and cream

variants() sniffed for an existing _paper baseline, so a figure that had
never been papered could never start being papered. 24 baselines -> 46.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: `EXPOSITION` on the contract, gated but disabled

**Files:**
- Modify: `src/figlib/program.py:64-80` (`load_program`), `:32-48` (`Report`), `:114` (gate wiring)
- Modify: `src/figlib/gates.py` (add `exposition_gate`)
- Test: `tests/test_program.py`

**Interfaces:**
- Consumes: nothing
- Produces: `gates.exposition_gate(mod, enabled: bool = False) -> list[Diagnostic]`; `Report.exposition: str | None`. Figure programs may declare `EXPOSITION: str` (1-3 paragraphs) or, for `capability/` probes, `PROBES: str` (one line). The gate ships **disabled** and Phase 1 flips it.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_program.py`:

```python
import textwrap

import pytest

BARE = textwrap.dedent('''
    import numpy as np
    from figlib.scene import Curve, Scene
    from figlib.style import Role

    CLAIM = "The line rises."
    PARAMS = {"n": 9}

    def compute(p):
        x = np.linspace(0, 1, p["n"])
        return np.column_stack([x, x])

    def build(g):
        return Scene(items=[Curve(g, role=Role.CONTENT)])

    def assertions(g):
        assert g[-1, 1] > g[0, 1], "should rise"
''')

WORDS = " ".join(["mechanism"] * 40)


def test_exposition_gate_is_silent_when_disabled(tmp_path):
    """It ships disabled so Phase 0 can land it without reddening a corpus
    that has not written the field yet."""
    from figlib.gates import exposition_gate
    from figlib.program import load_program

    prog = tmp_path / "bare.py"
    prog.write_text(BARE)
    assert exposition_gate(load_program(prog)) == []


def test_exposition_gate_fires_on_a_missing_field(tmp_path):
    from figlib.gates import exposition_gate
    from figlib.program import load_program

    prog = tmp_path / "bare.py"
    prog.write_text(BARE)
    diags = exposition_gate(load_program(prog), enabled=True)
    assert [d.kind for d in diags] == ["exposition"]
    assert "EXPOSITION" in diags[0].detail


def test_exposition_gate_rejects_a_stub(tmp_path):
    """A one-liner is not the passage the figure serves — the floor is what
    stops the field decaying into a restated CLAIM."""
    from figlib.gates import exposition_gate
    from figlib.program import load_program

    prog = tmp_path / "stub.py"
    prog.write_text(BARE + '\nEXPOSITION = "It shows a line going up."\n')
    diags = exposition_gate(load_program(prog), enabled=True)
    assert [d.kind for d in diags] == ["exposition"]
    assert "40 words" in diags[0].detail


def test_exposition_gate_accepts_a_real_passage(tmp_path):
    from figlib.gates import exposition_gate
    from figlib.program import load_program

    prog = tmp_path / "full.py"
    prog.write_text(BARE + f'\nEXPOSITION = """{WORDS}"""\n')
    assert exposition_gate(load_program(prog), enabled=True) == []


def test_probes_satisfies_the_gate_for_a_capability_figure(tmp_path):
    """capability/ figures are probes of the renderer, not arguments about
    mathematics; forcing exposition on them would only produce fiction."""
    from figlib.gates import exposition_gate
    from figlib.program import load_program

    prog = tmp_path / "probe.py"
    prog.write_text(BARE + '\nPROBES = "gradient seams across a shared edge"\n')
    assert exposition_gate(load_program(prog), enabled=True) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_program.py -k "exposition or probes" -v
```
Expected: all five FAIL with `ImportError: cannot import name 'exposition_gate'`.

- [ ] **Step 3: Implement the gate**

Add to the end of `src/figlib/gates.py`:

```python
# An EXPOSITION shorter than this is a restated CLAIM, not the passage the
# figure serves. Set from the shortest honest paragraph, not from taste.
MIN_EXPOSITION_WORDS = 40


def exposition_gate(mod, enabled: bool = False) -> list[Diagnostic]:
    """Does this figure record the argument it serves?

    CLAIM says what the figure argues; EXPOSITION says what argument it is
    FOR. A figure whose exposition cannot be written has no reason to
    exist, so this is a cheap early kill — but it ships disabled, because
    enabling it before the corpus carries the field would make every
    figure un-renderable (regress raises on a failed gate).

    capability/ figures answer with a one-line PROBES instead: they are
    probes of what the renderer can do, not arguments about mathematics.
    """
    if not enabled:
        return []
    if getattr(mod, "PROBES", "").strip():
        return []
    text = getattr(mod, "EXPOSITION", "") or ""
    if not text.strip():
        return [Diagnostic("exposition", (
            "no EXPOSITION: state the passage this figure serves (>= "
            f"{MIN_EXPOSITION_WORDS} words), or PROBES for a capability figure"))]
    n = len(text.split())
    if n < MIN_EXPOSITION_WORDS:
        return [Diagnostic("exposition", (
            f"EXPOSITION is {n} words; the floor is {MIN_EXPOSITION_WORDS} words. "
            "Write the surrounding text that made the figure necessary, "
            "not a restatement of CLAIM"))]
    return []
```

- [ ] **Step 4: Accept the field on the contract and carry it on the Report**

In `src/figlib/program.py`, `Report` gains a field after `claim`:

```python
    exposition: str | None = None
```

Wire it in the `return Report(...)` call at the end of `run()`:

```python
    return Report(mod.__name__, mod.CLAIM, svg_path, png_path, diags,
                  built=built, style=style, width_px=width_px, notes=notes,
                  signals=sigs,
                  exposition=getattr(mod, "EXPOSITION", None)
                             or getattr(mod, "PROBES", None))
```

`Report` is a plain dataclass with defaults, and `exposition` is passed by keyword, so field ordering is not a constraint — but add it after `claim` anyway, where it reads.

Wire the gate into `run()` immediately after the `numerical` call:

```python
    diags = numerical(lambda: mod.assertions(geom))
    from .gates import exposition_gate
    diags += exposition_gate(mod, enabled=EXPOSITION_REQUIRED)
```

and define the switch near the top of `program.py`, under the imports:

```python
# Phase 1 flips this to True, once every figure carries the field. Landing
# it False lets the gate ship without reddening the corpus that has not
# written EXPOSITION yet.
EXPOSITION_REQUIRED = False
```

- [ ] **Step 5: Run the full suite and the corpus**

```bash
make test && make regress
```
Expected: `605 passed`, then `match=46  drift=0  new=0  error=0` — the gate is disabled, so nothing changes.

- [ ] **Step 6: Commit**

```bash
git add src/figlib/gates.py src/figlib/program.py tests/test_program.py
git commit -m "contract: EXPOSITION — the argument a figure serves, gated but disabled

CLAIM says what a figure argues; nothing said what argument it is FOR.
The gate ships disabled because regress raises on a failed gate, so
enabling it before the corpus carries the field would make every figure
un-renderable. Phase 1 flips EXPOSITION_REQUIRED.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: `make gallery`

**Files:**
- Create: `src/figlib/gallery.py`, `tests/test_gallery.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `discover_programs` (Task 2), `artifact_dir` (Task 3), `EXPOSITION`/`PROBES` (Task 8)
- Produces: `gallery.build_gallery(figures_dir: Path) -> str` returning markdown; `gallery.write_gallery(figures_dir: Path) -> Path` writing `figures/out/GALLERY.md`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gallery.py`:

```python
"""The gallery is generated from program metadata, never hand-edited."""

import textwrap

from figlib.gallery import build_gallery

PROG = textwrap.dedent('''
    import numpy as np
    from figlib.scene import Curve, Scene
    from figlib.style import Role

    CLAIM = "The line rises monotonically."
    EXPOSITION = """{exposition}"""
    PARAMS = {{"n": 9}}

    def compute(p):
        x = np.linspace(0, 1, p["n"])
        return np.column_stack([x, x])

    def build(g):
        return Scene(items=[Curve(g, role=Role.CONTENT)])

    def assertions(g):
        assert g[-1, 1] > g[0, 1], "should rise"
''')


def _corpus(tmp_path):
    figs = tmp_path / "figures"
    (figs / "complex").mkdir(parents=True)
    (figs / "signals").mkdir()
    (figs / "complex" / "alpha.py").write_text(
        PROG.format(exposition="Alpha exposition text."))
    (figs / "signals" / "beta.py").write_text(
        PROG.format(exposition="Beta exposition text."))
    return figs


def test_gallery_groups_by_subject_in_stable_order(tmp_path):
    md = build_gallery(_corpus(tmp_path))
    assert md.index("## complex") < md.index("## signals")


def test_gallery_carries_claim_and_exposition(tmp_path):
    md = build_gallery(_corpus(tmp_path))
    assert "The line rises monotonically." in md
    assert "Alpha exposition text." in md
    assert "Beta exposition text." in md


def test_gallery_links_both_grounds(tmp_path):
    """The card shows cream because it reads better as a card; the
    transparent render is the one people embed, so it must be reachable."""
    md = build_gallery(_corpus(tmp_path))
    assert "complex/alpha_paper.png" in md
    assert "complex/alpha.svg" in md


def test_gallery_does_not_import_or_render(tmp_path):
    """Metadata is read from source, so a corpus of 35 figures does not
    cost 35 renders to list. A program that would fail its gates still
    appears."""
    figs = _corpus(tmp_path)
    (figs / "complex" / "broken.py").write_text(
        PROG.format(exposition="Broken exposition text.")
        + "\nraise RuntimeError('import would explode')\n")
    md = build_gallery(figs)
    assert "broken" in md
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_gallery.py -v
```
Expected: all four FAIL with `ModuleNotFoundError: No module named 'figlib.gallery'`.

- [ ] **Step 3: Implement**

Create `src/figlib/gallery.py`:

```python
"""Generate the browsable corpus index from figure-program metadata.

Metadata is read from SOURCE, not by importing: listing 35 figures should
not cost 35 renders, and a figure that is mid-repair should still appear
in the index rather than vanishing from it.

    build_gallery(figures_dir) -> markdown
    write_gallery(figures_dir) -> path to figures/out/GALLERY.md
"""

from __future__ import annotations

import ast
from pathlib import Path

from .regress import artifact_dir, discover_programs

#: subjects in reading order; anything else sorts after, alphabetically
SUBJECT_ORDER = ["complex", "signals", "linalg", "dynamics", "circuits",
                 "capability"]

SUBJECT_BLURB = {
    "complex": "Conformal maps, the Riemann sphere, potential flow.",
    "signals": "Sampling, spectra, and the geometry of transfer functions.",
    "linalg": "Matrices as geometry: the four readings, low-rank structure.",
    "dynamics": "Flows, bifurcations, and stochastic trajectories.",
    "circuits": "Transformer internals as computation graphs.",
    "capability": "Renderer probes, not arguments — kept honest about it.",
}


def _string_field(tree: ast.Module, name: str) -> str | None:
    """Read a module-level string assignment without executing the module."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    return node.value.value
    return None


def _subject(program: Path, figures_dir: Path) -> str:
    parent = program.parent.resolve()
    return "" if parent == figures_dir.resolve() else parent.name


def _sort_key(subject: str) -> tuple[int, str]:
    return ((SUBJECT_ORDER.index(subject), "") if subject in SUBJECT_ORDER
            else (len(SUBJECT_ORDER), subject))


def build_gallery(figures_dir: str | Path) -> str:
    figures_dir = Path(figures_dir)
    out_root = figures_dir / "out"
    entries: dict[str, list[tuple[str, str, str]]] = {}
    for program in discover_programs(figures_dir):
        tree = ast.parse(program.read_text(), str(program))
        claim = _string_field(tree, "CLAIM") or "_(no CLAIM)_"
        prose = (_string_field(tree, "EXPOSITION")
                 or _string_field(tree, "PROBES")
                 or "_(no EXPOSITION yet)_")
        subject = _subject(program, figures_dir)
        rel = artifact_dir(program, figures_dir).relative_to(out_root)
        stem = str(rel / program.stem) if str(rel) != "." else program.stem
        entries.setdefault(subject, []).append((stem, claim, prose.strip()))

    lines = ["# Figure gallery", "",
             "Generated by `make gallery` — do not edit. Each card shows the "
             "cream render; the transparent SVG is the one to embed.", ""]
    for subject in sorted(entries, key=_sort_key):
        lines += [f"## {subject or 'unbinned'}", ""]
        if blurb := SUBJECT_BLURB.get(subject):
            lines += [blurb, ""]
        for stem, claim, prose in sorted(entries[subject]):
            name = Path(stem).name
            lines += [
                f"### {name}", "",
                f"![{name}]({stem}_paper.png)", "",
                f"**Claim.** {claim}", "",
                prose, "",
                f"[transparent svg]({stem}.svg) · "
                f"[cream svg]({stem}_paper.svg) · "
                f"[readback]({stem}.readback.md)", "",
            ]
    return "\n".join(lines).rstrip() + "\n"


def write_gallery(figures_dir: str | Path) -> Path:
    figures_dir = Path(figures_dir)
    path = figures_dir / "out" / "GALLERY.md"
    path.write_text(build_gallery(figures_dir))
    return path
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_gallery.py -v
```
Expected: 4 passed

- [ ] **Step 5: Add the make target**

In `Makefile`, add after the `update` target, and add `gallery` to `.PHONY`:

```make
gallery:
	$(FIGCHECK) --gallery
```

Add the flag to `src/figlib/cli.py` — an argument beside `--regress`:

```python
    ap.add_argument("--gallery", action="store_true",
                    help="regenerate figures/out/GALLERY.md from program metadata")
```

and a branch before the `--regress` dispatch in `main()`:

```python
    if args.gallery:
        from pathlib import Path

        from .gallery import write_gallery
        path = write_gallery(Path(args.figures_dir))
        print(f"gallery: {path}")
        return 0
```

Add the line to the `help` target's echo block:

```make
	@echo 'make gallery                   regenerate figures/out/GALLERY.md'
```

- [ ] **Step 6: Generate the real gallery and read it**

```bash
make gallery
```
Expected: `gallery: figures/out/GALLERY.md`. Open it — 23 entries in 6 subjects, every one currently showing `_(no EXPOSITION yet)_`, which is exactly the Phase 1 worklist.

- [ ] **Step 7: Run the full suite and commit**

```bash
make test && make regress
```
Expected: `609 passed`, then `match=46  drift=0  new=0  error=0`

```bash
git add -A
git commit -m "gallery: generate the corpus index from program metadata

Read from source with ast, not by importing: listing 35 figures should not
cost 35 renders, and a figure mid-repair should still appear in the index
rather than vanishing from it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Sync the docs to the new contract

The invariants live in prose that is now wrong in three places. `CLAUDE.md` says the code wins and the doc is a bug — so fix the docs.

**Files:**
- Create: `docs/friction/README.md`
- Modify: `CLAUDE.md`, `docs/skill.md`, `docs/architecture.md`

**Interfaces:**
- Consumes: everything above
- Produces: the friction-record template every dispatched agent will fill in

- [ ] **Step 1: Rewrite the grain invariant in `CLAUDE.md`**

Replace the `A figure has no ground` bullet:

```markdown
- **A figure has no ground.** `Style.transparent` defaults to `True`: ink on
  alpha, no paper rect — the embedding document owns the background.
  **Grain is ink, not paper: it always renders.** Every figure commits both
  grounds — `<name>.svg` groundless, `<name>_paper.svg` on the theme's
  cream. `figcheck --paper` renders the papered variant on demand.
```

- [ ] **Step 2: Update the layout block in `CLAUDE.md`**

Replace the `figures/*.py` and `figures/out/` lines:

```
figures/<subject>/*.py   the corpus, binned by subject: complex, signals,
                 linalg, dynamics, circuits, capability. One level deep.
figures/out/<subject>/   COMMITTED render baselines (svg+png, both grounds)
                 plus readback/judge records. Golden files for --regress;
                 not scratch. GALLERY.md is generated — never edit it.
```

- [ ] **Step 3: Add `EXPOSITION` to the contract in `docs/skill.md`**

In the contract code block, after the `CLAIM` line:

```python
CLAIM = "one sentence: what the figure ARGUES, not what it draws"
EXPOSITION = """1-3 paragraphs: the passage this figure SERVES — the
                surrounding text that made it necessary. Write it BEFORE
                the code; if it cannot be written there is no figure."""
```

and note beneath the block that `capability/` probes answer with a one-line `PROBES` instead, and that the gate is enabled from Phase 1 onward.

- [ ] **Step 4: Add the `make gallery` line to the loop in `docs/skill.md`**

```
make gallery                               regenerate figures/out/GALLERY.md
```

- [ ] **Step 5: Record the two-ground contract in `docs/architecture.md`**

Add to the invariants section:

```markdown
- **Two grounds, always.** `regress.variants` returns both for every
  program: the groundless render for embedding, and the theme's cream
  paper for standing alone. Sniffing for an existing `_paper` baseline
  meant a figure that had never been papered could never start being
  papered.
- **Artifacts mirror subjects.** `regress.artifact_dir` is the single
  derivation from program path to output directory. A program directly in
  `figures/` still maps to `figures/out/`, so host projects outside this
  repo are unaffected.
```

- [ ] **Step 6: Create the friction-record template**

Create `docs/friction/README.md`:

```markdown
# Friction records

One record per figure built or rebuilt, plus one per subject for a
readback-only pass. Written by the agent that did the work, at the time it
did the work — the knowledge of which primitive was missing otherwise dies
with the session.

Phase 4 ranks hand-rolled devices by (figures affected x lines
hand-rolled) and folds the ranking into `primitive-gaps.md`. The ranking
is mechanical over the structured fields below, so fill them in literally.

## Template

Copy this into `docs/friction/<name>.md`.

```markdown
# <figure> — friction record        agent · subject · date

## What the claim needed
One paragraph: the geometry and annotation the CLAIM demanded.

## What figlib gave for free
What worked without a fight. This section protects working machinery from
being "improved" later — say what you leaned on.

## What I hand-rolled
The load-bearing section. One bullet per device, with a line count:

- <device> — ~N lines in the figure program that should be a primitive

## Gate diagnostics that did NOT contain the fix
`CLAUDE.md` claims diagnostics contain the fix. Where that failed, quote
the diagnostic and say what you actually had to do. If it never failed,
write "none" — that is a real result.

## Renders to first green
N

Count every `make check` from first run to first PASS. This is the
quantitative handle that ranks friction across the corpus, so do not
estimate it.

## Proposed primitive
A signature, not prose. If nothing is worth building, write "none" and say
why the hand-rolling was inherent to this figure rather than general.
```
```

- [ ] **Step 7: Verify and commit**

```bash
make test && make regress && make gallery
```
Expected: `609 passed`; `match=46  drift=0  new=0  error=0`; gallery regenerated

```bash
git add -A
git commit -m "docs: grain is ink, subjects bin the corpus, and friction gets a template

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 0 exit criteria

All four must hold before a single agent is dispatched:

- [ ] `make test` — green
- [ ] `make regress` — `match=46  drift=0  new=0  error=0`
- [ ] `make gallery` — writes `figures/out/GALLERY.md`, 23 entries across 6 subjects
- [ ] `git status --short` — clean

Then Phase 1 dispatches 6 agents, one per subject.
