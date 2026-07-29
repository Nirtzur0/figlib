# illconditioned_descent — friction record

## What the claim needed

Three overlaid trajectories (GD, heavy-ball momentum, Newton) on the same
rotated elliptical quadratic, staying visually distinguishable at very
different iteration counts (58 vs 18 vs 1), plus honest per-method
iteration-count annotations and a numerical certificate that those counts,
the monotonicity claims, and the sqrt(kappa) speed-up are properties of
the actual simulated arrays, not restated theory.

## What figlib gave for free

- `RISO.categorical(i)` for the three-way correspondence hue, with
  `correspondence_cap=4` comfortably covering three series.
- `Curve(arrows=..., arrow_style="filled")` for sparse motion markers on a
  58-point polyline — no manual arrowhead geometry.
- `Checks` for a multi-assertion gate report (one run, every failure
  named) instead of first-assert-wins.
- The mechanical gate's `label-on-ink` / `arrow-on-mark` diagnostics with
  verified `offset_px` nudges — every real collision this figure hit was
  fixed by typing back the printed suggestion, no guess-and-rerender.

## What I hand-rolled

- **Exact closed-form ellipse level sets** (~15 lines): since the
  objective is a known quadratic, `x = R @ [sqrt(2c/mu)cos t,
  sqrt(2c/L)sin t]` is exact and avoids `marching_squares`'/`auto_levels`'
  grid-sampling error entirely. Not a gap — a quadratic's level sets are a
  closed form and using the generic contour path would have been the
  wrong tool — but it means `builders.auto_levels` gave nothing here; a
  figure whose objective is NOT closed-form would need it.
- **A floating legend block in the empty off-diagonal wedge** (~10
  lines): with 58 GD iterates packed into the valley band, no
  in-band anchor for the three method labels stayed collision-free
  (`label-on-ink` fired against the dense zigzag no matter where in-band
  it was pinned). Computing `f(candidate)/f0 > 1` to find points outside
  the outermost level curve, then placing three stacked labels there, was
  manual — the same device as `demo_basin_wash`'s floating ODE caption,
  but there was no primitive that finds "the nearest genuinely ink-free
  region" automatically; I found it by hand-checking candidate points
  against `f0`.
- **Choosing `u0` (start point, in eigenbasis coordinates) to make the
  zigzag legible** (~5 lines of reasoning, not code): the first attempt
  (`u0=(1.0,1.05)`, comparable eigen-components) produced a dense
  criss-crossing "fan/star" near the origin — mathematically correct but
  unreadable. Realizing that GD's convergence factor magnitude is
  *identical* in both eigen-directions at the optimal step (so
  `||x_k||/||x_0||` is exactly `rho^k` independent of `u0`'s direction)
  meant `u0` could be re-chosen purely for legibility (mostly along the
  shallow/valley direction, small perpendicular offset) without changing
  any annotated count. This is a real, reusable fact about steepest
  descent on quadratics, not something figlib could have told me.

## Gate diagnostics that did NOT contain the fix

None outright wrong, but one diagnostic under-informed a real problem:
the mechanical gate reported `label-on-ink` for the method labels pinned
near curve points, with a computed `offset_px` nudge — but the nudge it
offered only clears the *nearest* ink, not the fact that the entire
in-band region is dense ink at every plausible nudge distance. The gate
correctly flagged each individual collision; it had no way to say "give
up nudging, this position class is saturated, relocate structurally" —
that judgment (move to a floating legend) was mine.

## Renders to first green: 3

(fail: label-on-ink + arrow-on-mark from in-band labels; fail: same
class after a start-point redesign shifted geometry; pass: after moving
labels to a computed off-diagonal empty region.)

## Proposed primitive

`layout.find_ink_free_anchor(scene, near: XY | None, exclude_radius_px:
float) -> XY` — given the already-placed geometry, return a point
guaranteed clear of every drawn item's bbox (optionally biased toward
`near`), so a floating caption's anchor is computed rather than found by
manually testing candidate math-coordinates against the objective. Every
figure so far that needed this (`demo_basin_wash`'s ODE caption, this
one's method legend) solved it by hand with figure-specific knowledge
(corner math-coords, `f(x)/f0 > 1`) that a generic geometry query could
replace.
