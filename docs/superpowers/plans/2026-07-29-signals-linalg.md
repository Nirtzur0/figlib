# Signals & Linear Algebra Vocabulary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the four inexpressible signal/LA devices (stems, impulses, cross marker, CellGrid, colorbar) as compositional producers, then prove them with three exemplar figures (sampling/aliasing stack, pole-zero → response, DFT as change of basis).

**Architecture:** Every addition is a producer (`→ list[Item]`) or a frozen geometry object that draws nothing (`CellGrid`). No plot-type classes. Spec: `docs/superpowers/specs/2026-07-29-signals-linalg-design.md`.

**Tech Stack:** numpy, existing figlib scene/plots/figure machinery, pytest, figcheck gates.

## Global Constraints

- Content code never names a color/font/stroke width — Roles and theme ramps only.
- Every cairo-touching command needs `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` (use make targets or prefix explicitly). Python is `uv run`.
- TDD: failing test → implement → pass → commit, every task.
- `figures/out/` diffs are reviewed, not blindly updated. Do not touch the untracked `src/figlib/place.py` / `tests/test_place.py` or modified `docs/primitive-gaps.md` (another session's work).

---

### Task 1: stems, impulses, cross marker (`plots.py`)

**Files:** Modify `src/figlib/plots.py`; Test `tests/test_stems.py`.

**Produces:** `stems(x, y, *, baseline=0.0, marker="circle", filled=True, size=0.03, xscale=None, yscale=None, role=Role.CONTENT, color=None, width_scale=1.0) -> list[Item]`; `impulses(x, weights, *, baseline=0.0, xscale=None, yscale=None, role=Role.CONTENT, color=None, width_scale=1.0) -> list[Vector]`; `markers(..., shape="cross")`.

- [ ] Write failing tests: stems item count (n Curves + n markers), stem tops at y, hollow-marker stems stop short of the rim (top ≤ y − ry), filled-marker stems reach y, scales applied once, length mismatch raises; impulses are Vectors with tip − tail = weight (negative points down), zero-weight skipped or zero-length (decide: raise on zero? no — emit nothing for |w| < eps? simplest: emit exactly what's given, callers own their data — emit all, tip=tail for 0); cross marker emits 2 open Curves per point, `filled=True` raises ValueError, cross present in shape set.
- [ ] Run `make test` — expect failures.
- [ ] Implement: cross in `_unit_shape` handled as a special case in `markers` (two segment Curves at ±45°, scaled by an area-equalizing factor ≈ visual weight match; use length factor 1.0 and document); stems = per-sample `Curve([[x, base], [x, y_stop]])` + `markers`; impulses = `Vector((x, base), (x, base + w))`.
- [ ] `make test` green. Commit `plots: stems, impulses, cross marker — the discrete-sequence types`.

### Task 2: `figlib/matrix.py` — SUPERSEDED, execute the matrix-layer plan instead

**This task does not run.** `CellGrid` and the matrix-layer spec's `Block`
were independently designed as the same object in the same new module.
They are merged into one type, `Block`, and the whole of
`src/figlib/matrix.py` is now built by
**`docs/superpowers/plans/2026-07-29-matrix-layer.md` Tasks 1–5**, which
absorb `map_into`, `edge`, `extent`, `center`, `grid_lines`, `brackets`,
`cell_fills`, and `diag_cells` (as the mask recipe `diagonal(b, offset,
wrap)`).

One behavioural change to note before writing Task 6: `cell` is a scalar
cell **width** and cell height is `cell * aspect`, with `aspect`
defaulting to `1.0` (square). `dft_matrix_basis.py` wants tall cells, so
it passes `aspect` explicitly — square cells are the default because they
are what make a drawn rectangle's aspect ratio equal its shape, which the
matrix-layer conformability gates rest on.

- [ ] Confirm the matrix-layer plan Tasks 1–5 are merged before starting
      Task 6; there is nothing to implement or commit here.

### Task 3: `plots.colorbar`

**Files:** Modify `src/figlib/plots.py`; Test `tests/test_colorbar.py`.

**Produces:** `colorbar(scale, ramp, *, at, length, thickness, orient="y", ticks=None, label=None, role=Role.ANNOTATION, n=64, tick_labels=True) -> list[Item]`.

- [ ] Failing tests: n FilledCurve slabs spanning length exactly, slab colors = ramp at slab centers monotone in t, axis items present along the outer edge with tick positions mapped through scale, horizontal orient transposes, Log10 scale works (dB-style), label emitted.
- [ ] Implement by reusing `axis()` with `at` = strip's outer edge. `make test` green.
- [ ] Commit `plots: colorbar — a ramp channel finally gets a scale`.

### Task 4: exemplar `figures/sampling_aliasing.py`

O&S replica stack: three shared-ω panels (Figure, grid=(3,1)): (a) triangle spectrum X(jω) + impulse-train comb drawn with `impulses`; (b) replicas at kω_s, ideal-lowpass rect dashed; (c) ω_s < 2ω_m, alias overlap = `band(ω, 0, min(f_k, f_{k±1}))` in ACCENT role. Axes via `plots.axis`, π-free tick labels stated in ω_s units via explicit `Ticks`.

- [ ] Write figure (CLAIM/PARAMS/compute/build/assertions). Assertions: all panels share one Scale object's range; replica k is an exact shift of the base triangle; overlap area > 0 iff ω_s < 2ω_m (compute both cases); comb spacing = ω_s.
- [ ] `make check F=figures/sampling_aliasing.py` until gates pass — apply printed fixes, don't guess.
- [ ] Commit with `figures/out/` baselines after `make update` for this figure only (`figcheck --update` equivalent via make check + update flow; inspect diff first).

### Task 5: exemplar `figures/polezero_response.py`

Panel [a]: z-plane, CONSTRUCTION unit circle, `markers(..., "cross")` poles / hollow circles zeros, chords from e^{jω₀} to each; panel [b]: |H(e^{jω})| via `series`, ω₀ marked. Assertion: |H(e^{jω₀})| recomputed as gain·∏|chords to zeros|/∏|chords to poles| matches the curve value at ω₀ to 1e-9; poles strictly inside unit circle.

- [ ] Write figure; `make check` to green; commit with baselines.

### Task 6: exemplar `figures/dft_matrix_basis.py`

8×8 `Block` (see Task 2 — `CellGrid` is merged into it), each column j a mini stem plot of Re e^{2πijk/8} via `map_into` (stems computed in local (u,v), mapped); `brackets`; right of the matrix: x[n] stems and |X[k]| stems (the decomposition); small unit circle with the 8 roots of unity, `theme.phase(2πj/8)` hue binding column headers to roots. Assertions: F·F*/8 = I; drawn |X[k]| equals |fft(x)| to 1e-12; column waveform samples equal Re F[:, j].

This is the one figure that wants non-square cells: pass `aspect` explicitly (a waveform needs vertical room), and do **not** call `check_square_cells` on it — a basis gallery is not arguing about dimensions. It is also the end-to-end exercise of the merged surface (`map_into` + `brackets` + `cell_fills` + `aspect != 1.0`), so run it after the matrix-layer plan is fully merged.

- [ ] Write figure; `make check` to green; commit with baselines.

### Task 7: corpus regression, readbacks, docs

- [ ] `make regress` — the only acceptable diffs are the three new figures' additions; pre-existing drift in demo_flow_past_cylinder / demo_panels_zsquared / strogatz_saddle_node is NOT ours to rebaseline.
- [ ] Readback records for the three figures: `figcheck --readback-prompt` → cold subagent per figure → `readback.record()` → `figures/out/<name>.readback.md`.
- [ ] Note the new modules in `docs/primitive-gaps.md`? NO — file is another session's dirty state; instead add one line each to the module map in `src/figlib/__init__.py` docstring if that is the convention (check first).
- [ ] Final commit; full `make test`.
