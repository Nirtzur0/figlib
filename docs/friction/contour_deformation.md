# contour_deformation — friction record

## What the claim needed

A filmstrip of loops around a simple pole, all carrying the SAME value
(homotopy invariance), plus one loop dragged bodily across the pole
carrying a DIFFERENT value (the residue jump) — with the numbers on the
page backed by trapezoidal quadrature on the exact drawn point arrays,
not a caption asserting the theorem.

## What figlib gave for free

- `Curve.arrows` — tangent-oriented direction markers at arc-length
  fractions, filled head for "contour" — was exactly the Needham
  orientation convention the task called out; no hand-rolled arrowhead
  geometry needed.
- `plots.markers("cross", filled=False, ...)` was already the corpus's
  pole glyph (used in `polezero_response.py`); reused verbatim.
- `Correspondence`/`keyed` handled the "same value across [a]-[c]" claim
  almost for free once the fingerprint's exclusion of *position* was
  understood: the value label's identical LaTeX text across panels is a
  tracked facet, so the mechanical residual check enforces "the printed
  number is literally the same," which is half the figure's claim,
  outside the numerical gate entirely.
- `geometry_extents` honoring `scene.xlim/ylim` when set (rather than
  deriving from item extents) made a shared page scale across [a]-[c]
  automatic — no manual scale-matching arithmetic.
- The strogatz-filmstrip grid shape (`grid=(2,3)`, three small multiples
  + one full-width panel) transferred directly: same shape of argument
  (instances, then the one instance where the family's rule breaks).

## What I hand-rolled

- **Complex trapezoidal contour quadrature** — ~10 lines
  (`_contour_integral`): closed-loop complex trapezoid on an (N,) complex
  array, wrapped via `np.concatenate([z, z[:1]])`. Not figlib's job (it's
  pure numerics upstream of the Scene), but every future complex-analysis
  figure that gates a contour integral will rewrite this same ten lines.
  A `geometry.py` or `builders.py` helper (`complex_contour_integral(z, f)`)
  would be a one-line primitive instead.
- **Star-shaped wobbly loop parametrization** — ~6 lines
  (`_loop`): `r(theta) = R(1 + amp cos(k theta + phase))`. Figure-specific
  enough (the wobble is this figure's choice of "visibly different but
  still simple") that it probably shouldn't generalize.
- **Convergence-tolerance calibration by hand**: figured out empirically
  (a scratch script outside the figure) that trapezoidal quadrature at
  n=6000 samples was accurate to ~1.2e-6 for this geometry, then set
  `tol = 2e-6` with that margin stated in a comment. Nothing in
  `assertions()` or the gate runner suggested a starting `n` or told me
  the achieved accuracy — I had to bisect resolution externally before
  writing a single assertion tolerance.

## Gate diagnostics that did NOT contain the fix

- `arrow-on-mark` diagnostics named the panel's *current* arrow fraction
  and a numeric replacement (e.g. "arrows=(0.60,)" as the clear
  alternative), but that replacement is only valid for one panel's own
  label layout — since all panels share `PARAMS["arrow_fracs"]`, applying
  it would have fixed one panel and silently left (or created) a collision
  in another. The diagnostic doesn't say "this is per-instance, not
  per-figure"; I had to infer that from having four panels share one
  tuple. The fix that actually worked was moving the *label* off the ring
  (a bigger structural change: shrink glyph clutter near the loop) rather
  than chasing arrow fractions panel-by-panel.
- `label-on-ink: ... no free single-axis nudge; nearest ink-free region
  center (x, y) canvas px` gives a canvas-px target but not a math-coord
  one — for a label placed by formula from world coordinates (as this
  figure's `_crossing_panel` labels are), translating a canvas-px target
  back into an anchor shift meant re-deriving the panel's own transform by
  hand rather than just retyping a suggested `offset_px`. Ended up
  re-deriving the anchor geometrically (placing it below the lowest of
  three circle bottoms) instead of using the printed canvas point at all.

## Renders to first green: 3

## Proposed primitive

```python
def complex_contour_integral(z: np.ndarray, f: Callable[[np.ndarray], np.ndarray]) -> complex:
    """Complex trapezoidal quadrature of f over a closed loop given by the
    exact drawn point array z (N,), closed by wrapping to z[0]."""
```

in `geometry.py` (alongside other compute-layer numerics) — every complex-
analysis figure that gates a residue/Cauchy claim needs this same ten
lines, and it's pure numerics with no drawing decision in it, exactly the
kind of thing `geometry.py` already owns.
