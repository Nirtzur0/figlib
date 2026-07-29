# amplitwist — friction record

## What the claim needed

A 2x2 panel-pair-of-panel-pairs: an analytic map (`f(z) = 1/z`) taking a
small disc to a disc with the SAME ratio and turn angle in every radial
direction, next to a non-analytic real-linear map (`g(x,y) = (ax,by)`,
`a != b`) taking the same disc to an ellipse — ratio and the angle between
two tracked arrows both genuinely direction-dependent. Every load-bearing
number (ratio spread, turn spread, Cauchy-Riemann residual, the surviving
or destroyed right angle) had to come from finite differences on the
actual drawn/mapped arrow tips, never from `f'(z0)` restated symbolically.

## What figlib gave for free

- `Vector` already documents the ghost/filled convention "the amplitwist
  figures" by name in its own docstring — didn't need to invent a
  primitive for radial arrows.
- `AngleMark` + `MathLabel` for the tracked right angle, reused unchanged
  from `demo_panels_zsquared`'s θ0/2θ0 device.
- `Figure(grid=(2,2))` with two `Connector`s handled the whole page layout
  and the map-squiggle between each domain/codomain pair; no new figure
  machinery needed for a 4-panel composite.
- `correspond.py`'s residual gate caught two real design mistakes for free
  (see below) rather than letting them ship silently.

## What I hand-rolled

- **Per-pair shared page scale** (~15 lines): each domain/codomain window
  is independently zoomed to its own drawn disc radius (`pad * r_disc`),
  but the two panels in a pair must share ONE half-width so the
  amplification (a real size change) reads as a visible fact on the page
  and `correspond.py`'s default `frame="shared"` doesn't flag drift. This
  is a `max(r_domain, r_codomain)` computed once per pair and threaded
  into both `_panel()` calls — not a primitive gap, just a thing every
  amplitwist-shaped composite will need to redo.
- **Per-label offset table keyed by (analytic, mapped)** (~6 lines) to
  clear the label-on-ink gate against the 45°/135° accent arrows; this is
  ordinary layout debugging, not a missing primitive.

## Gate diagnostics that did NOT contain the fix

- `stale-change` / `fixed-set-rescaled` together: my first `CORRESPONDENCE`
  declaration listed every arrow key in `changes=`, on the theory that
  "the arrows visibly move under the map" makes them a declared
  difference. But `correspond.py`'s fingerprint deliberately excludes
  position — a key belongs in `changes` only if its ROLE/COLOR/DASH/TEXT
  differs across parts. Diagnosing this took reading `correspond.py`
  itself; the gate output (`stale-change: ... identical in every part`)
  states the symptom correctly but the fix (drop position-only keys from
  `changes` entirely, and match page scale instead of reaching for
  `frame=`) isn't in the message. `frame=<reason>` looked like the
  natural escape hatch and made `fixed-set-rescaled` go away, but it was
  the wrong fix — it papers over a scale mismatch the figure didn't need
  once the two panels in a pair share one window.

## Renders to first green: 3

## Proposed primitive

`figure.paired_half_width(pad, *radii) -> float` — the "shared zoom
window for a rescaling pair" computation (`pad * max(radii)`) recurs
whenever a composite's whole point is a size change (amplitwist, any
before/after magnification figure) and page-scale-shared is the default
`correspond.py` wants. Small enough that inlining was fine here; if a
second figure needs it, promote it.
