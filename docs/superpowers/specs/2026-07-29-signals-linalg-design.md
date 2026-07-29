# Signals & linear algebra vocabulary — design

Date: 2026-07-29. Status: approved (user: "implement them", with the
constraint that nothing bakes in rigid structure — the agent writing a
figure must be able to compose ANY abstract plot from the pieces).

## What this is

The corpus survey (Bracewell, Oppenheim & Schafer, Lyons, Strang, JOS,
3B1B) decomposes the whole signal-processing / linear-algebra figure
tradition into a small set of mechanisms. figlib already owns most of
them (panel pairs, RasterField, phase hue, schematic layer, band,
Figure.grid small multiples). What is *inexpressible* today — the
doctrine's test for a new primitive — is exactly four things:

1. the discrete-sequence idiom: **stems** (sample lollipops) and
   **impulses** (Bracewell delta-arrows, height = weight) as two
   visually distinct types;
2. the **cross marker** (× for poles; ○ zeros = existing hollow circle);
3. an **addressable matrix geometry** — where is cell (i, j), and how do
   arbitrary items land inside it — from which heatmaps, structure
   portraits (Toeplitz/circulant bands), and basis galleries (a mini
   waveform per column) are all *recipes*, not API;
4. a **colorbar** — every ramp channel in the corpus is currently
   unlabelled.

Everything else the tradition needs (aligned transform-pair rows,
shared-abscissa stacks, aliasing overlap shading, flip-and-slide
filmstrips, phasor circles, block diagrams) is already expressible and
enters as exemplar figures, per "recipes live as exemplar figure
programs, not API surface."

## Design principle (the user's constraint, made structural)

Every addition is one of exactly two kinds:

- a **producer**: pure function → `list[Item]` (or one Item). The call
  site keeps z-order, scene membership, roles, scales. Identical
  contract to `plots.series`, `plots.axis`, `surface3d.surface_items`.
- a **geometry object**: frozen dataclass that answers coordinate
  questions (`CellGrid.rect(i, j)`, `.map_into(i, j, pts)`) and draws
  nothing. The author composes items against it.

No plot-type classes, no layout containers, no "SpectrumPanel". An
agent reasoning about a figure it has never seen before gets coordinate
answers and item emitters, and writes the composition itself.

## Additions to `src/figlib/plots.py`

```python
def stems(x, y, *, baseline=0.0, marker="circle", filled=True,
          size=0.03, xscale=None, yscale=None, role=Role.CONTENT,
          color=None, width_scale=1.0) -> list[Item]
```
One vertical Curve from baseline to each sample plus a marker at the
head. The stem stops short of a hollow marker's rim (the phase_line
trick) so hollow reads hollow. Scales applied here, once. This is the
`x[n]` type: samples of a function.

```python
def impulses(x, weights, *, baseline=0.0, xscale=None, yscale=None,
             role=Role.CONTENT, color=None, width_scale=1.0) -> list[Vector]
```
One Vector per impulse, tail at baseline, tip at baseline + weight —
Bracewell's convention: height IS the weight, the arrowhead marks "this
is a measure, not a value". Negative weights point down. Distinct from
`stems` by construction: arrow vs lollipop is the distributional /
sampled type distinction.

`markers` gains shape `"cross"`: two crossing segments (Curves), never
fillable — `filled=True` with `"cross"` is a ValueError. Area-equalized
against the other shapes. `_SHAPE_AREA_SCALE` entry chosen so a × at
the same `size` carries the same visual weight as a ○.

Nothing else: overlap shading is `band` (already exists — the aliasing
overlap is `band(x, 0, np.minimum(f1, f2))`), shared-abscissa stacking
is `Figure` rows with equal `Scale` ranges asserted in `assertions()`.

## `src/figlib/matrix.py` — MERGED, owned elsewhere

**Superseded.** This spec's `CellGrid` and the matrix-layer spec's `Block`
were independently designed as the same object in the same new module:
frozen, draws-nothing, top-left origin, row 0 at the top. They are now one
type, **`Block`**, specified in `2026-07-29-matrix-layer-design.md` §1,
which absorbs `map_into`, `edge`, `extent`, and `center` verbatim from the
sketch above.

One substantive change to what was written here. `CellGrid` carried
`cell: (width, height)`; `Block` carries a scalar `cell` plus an explicit
`aspect` (height/width, default `1.0`). Square cells are what make a
drawn rectangle's aspect ratio *equal its shape*, which is what the
matrix-layer conformability gates rest on — so the basis gallery's
non-square cells become an opt-out stated on the record rather than the
unremarked default.

The producers sketched here stay, and land with the matrix-layer plan:

```python
def grid_lines(b, *, role=Role.FRAME, inner=True, outer=True) -> list[Curve]
def brackets(b, *, pad=None, tick=None, role=Role.ANNOTATION) -> list[Curve]
def cell_fills(b, *, ramp, vmin=None, vmax=None,
               role=Role.CONTENT, opacity=1.0) -> list[FilledCurve]
    # the vector path for small matrices; `heat` is the raster path for
    # large ones. Both stay; the docstrings state the tradeoff.
```

`diag_cells` is absorbed into the matrix-layer's mask recipes as
`diagonal(b, offset=0, wrap=False) -> np.ndarray`; a call site needing
index pairs uses `np.argwhere`. Row/col index labels, deleted-row strikes,
value annotations, and basis galleries remain compositions of
`center`/`map_into`/`edge` with MathLabel/Curve/stems.

## `colorbar` (in `plots.py`)

```python
def colorbar(scale, ramp, *, at, length, thickness, orient="y",
             ticks=None, label=None, role=Role.ANNOTATION, n=64,
             tick_labels=True) -> list[Item]
```
A strip of n FilledCurve slabs through `ramp` (vector, so it inherits
theme/gates; no raster needed at this size) plus a reused `axis()` along
its edge mapping `scale` (Linear or Log10 — a dB colorbar is just a
scale). `at` is the strip's lower-left in math coords. Pure producer;
the author places it like any annotation.

## Exemplar figures (the recipes)

1. `figures/sampling_aliasing.py` — O&S replica stack. Three
   shared-ω panels (Figure rows): iconic triangle spectrum; replicas at
   multiples of ω_s with the ideal-lowpass rect dashed over them;
   under-sampled case with the alias overlap as `band` of the pairwise
   minimum, ACCENT-filled. Impulse train drawn with `impulses`.
   Assertions: replica spacing = ω_s; overlap nonempty iff ω_s < 2ω_m;
   all three panels share the exact ω Scale.
2. `figures/polezero_response.py` — JOS geometric evaluation. Left
   panel: z-plane, unit circle, × poles / ○ zeros (`markers`), chords
   from a marked e^{jω₀} to each pole/zero. Right panel: |H(e^{jω})|
   curve (`series`) with ω₀ ticked and the value at ω₀ marked; assertion
   recomputes |H| from the drawn pole/zero positions as the product of
   chord lengths and checks the curve's marked value against it.
3. `figures/dft_matrix_basis.py` — Strang × JOS change of basis. An 8×8
   `CellGrid`: each column j carries a mini stem plot of
   Re e^{2πijk/8} via `map_into`; brackets; below or beside, one signal
   x and its DFT as `stems` of the coefficient magnitudes; roots of
   unity on a small unit circle with the phase hue binding column j to
   its root. Assertions: F @ conj(F).T = 8 I (unitarity, computed);
   drawn coefficients equal fft(x) up to tolerance.

Each figure passes gates + regress and gets a readback record (project
definition of done).

## Testing

TDD per task. Library tests in `tests/test_stems.py` /
`tests/test_matrixgrid.py` (geometry: rect/center/map_into round-trips,
diag wrap; producers: item counts, roles, stem-gap-at-hollow-marker,
cross rejects filled, colorbar tick alignment via tests/svgkit.py where
SVG-level). Figure correctness lives in each figure's `assertions()`,
per project doctrine.

## Out of scope

Spectrogram exemplar (RasterField already proven by demo_ou_ensemble),
block-diagram ≡ matrix "same operator two drawings" pairing (wants
`Connector(kind="eq")` — a natural follow-up), animation filmstrips
beyond Figure.grid, π-multiple/dB tick formatters (compose `Ticks`
by hand at the call site until a third figure needs them).
