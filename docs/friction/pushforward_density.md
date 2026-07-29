# pushforward_density — friction record

## What the claim needed

A panel pair where the SAME cells are drawn twice — once as a flat grid
with uniform density, once as the image under a nonlinear map T — and
where per-cell density is exactly the inverse of the drawn polygon's
area ratio, with mass conservation checked on the actual shoelace areas,
not restated from the analytic Jacobian.

## What figlib gave for free

- `Figure`/`Panel`/`Connector` with the z^2 exemplar's
  `geometry_extents`-ratio trick for `width_frac` — matching page scale
  across two panels whose geometry occupies different math extents came
  straight from imitating `demo_panels_zsquared.py`.
- `Correspondence` fit the two annotated cells exactly: key only the
  cells actually being tracked (`expanded-cell`, `compressed-cell`),
  leave the other 23 unkeyed, list both keys in `changes` since both
  their position AND their fill color are supposed to move — no
  friction here, the model matched the claim once I realized "don't key
  the whole grid."
  `Callout` (anchor/target/leader) turned out to be exactly the primitive
  for "annotate a cell too small to hold its own label" — a box outside
  the grid extent, an arrow to the cell, no hand-rolled leader geometry.
- `plots.colorbar` + `plots.linear` gave the density scale as vector ink
  with ticks for free, following `svd_low_rank.py`'s pattern of feeding
  it a `Scale` and `THEME.ramp` directly.
- The map T = (x(1+ky), y(1-kx)) was chosen so det J = 1 + k(y-x) is
  EXACTLY affine in (x, y), which makes every grid edge map to a
  straight segment (no curve sampling needed for cell boundaries) and
  makes the cell-centroid Jacobian evaluation exact for the cell
  integral — this is a modeling choice, not a figlib feature, but it
  removed an entire class of "is my polygon curved enough to need more
  samples" questions.

## What I hand-rolled

- A 4-line shoelace-area function (`_shoelace`) — not in `geometry.py`.
  Small, but every figure that needs "area of the actual drawn polygon"
  (as opposed to an analytic formula) will want this; it's ~4 lines
  each time it's reinvented.
- The `_t(density, g)` linear-normalize-and-clip-to-[0,1] helper for
  feeding `theme.ramp` — also ~4 lines, also generic enough that it
  shows up in every figure using ramp-as-order over a data range (this
  one didn't use `plots.linear.fwd` because the ramp input needs [0,1],
  not scale-fwd coordinates; a `Scale.norm(v) -> float in [0,1]` method
  would have removed it).

## Gate diagnostics that did NOT contain the fix

- The colorbar's tick-label text ("0.64", "1.00", "2.27") clipped past
  the panel's right edge on the first render. The `clipped` diagnostic
  named the overflow in px but the fix ("nudge offset_px back in") does
  not apply to `colorbar()`'s internally-computed tick label offsets —
  there is no `offset_px` parameter exposed at the call site to nudge.
  The actual fix was realizing `geometry_extents` (used by `Transform`
  to size the panel when `Scene.xlim` is unset) measures `MathLabel`
  ANCHOR points, not glyph width, so a narrow colorbar scene under-pads
  itself for its own tick text. Manually widening `bar.xlim` to include
  room for the label glyphs fixed it; the gate never said "your xlim is
  the problem," only "this text is 34px past the edge."
- The `label-on-ink` diagnostic on the first attempt (annotation
  `MathLabel`s placed at cell centers) correctly reported one label as
  unfixable by single-axis nudge — but a small 5x5-grid cell is simply
  too small to hold a two-quantity label regardless of nudge direction.
  The real fix (switch to `Callout` with an anchor outside the grid) was
  a representation change the diagnostic could not suggest, since it
  only reasons about nudging the existing anchor, not replacing the
  primitive.

## Renders to first green: 2

(First `make check` failed on 3 `clipped` + 2 `label-on-ink`; second
`make check`, after the `bar.xlim` widen and the `Callout` swap,
passed. One further `make check` after adding the `rho_X` uniform-value
label, prompted by the readback agent's flagged ambiguity, still
passed — 3 total renders.)

## Proposed primitive

`Scale.norm(v: ArrayLike) -> np.ndarray` — linear map from
`(scale.lo, scale.hi)` to `[0, 1]`, the missing counterpart to `.fwd()`
for feeding `theme.ramp(t)` directly from a data range instead of
hand-rolling `(v - lo) / (hi - lo)` clipped, in every figure that colors
by an order channel over a computed range.
