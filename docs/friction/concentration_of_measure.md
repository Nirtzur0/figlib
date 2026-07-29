# concentration_of_measure — friction record

## What the claim needed
A single shared axis across d = 2, 10, 100, 1000 without crushing the
small-d curves, PLUS a place to show both halves of the claim (absolute
shell width stays O(1); relative width shrinks like 1/√d) without a
second panel, PLUS a way to show the joint density's mode (the origin)
in the same frame as the radial density's mode (√(d−1)), which live in
conceptually different objects.

## What figlib gave for free
`theme.ramp(t)` for the ordered d channel, `plots.colorbar` reused
directly as a log-spaced legend for the ramp (no bespoke legend code),
`Callout` for the origin punchline (opaque box, leader, arrowhead —
exactly the "boxed annotation with pointer" idiom), and the mechanical
gate's collision diagnostics, which located both layout bugs precisely
(pixel offsets, exact overlapping label pairs) rather than requiring
visual hunting.

## What I hand-rolled
- The x-axis rescaling itself (u = ‖X‖/√d, density Jacobian √d) — this
  is the actual design decision the brief asked for, not a primitive
  gap: normalizing by √d puts every curve's typical radius at u = 1 by
  construction, which is what let one dashed line do the job of "mark
  √d for every d" instead of four separate marks. ~15 lines, all math,
  not boilerplate.
- Peak-normalizing each curve to height 1 for shape comparison (true
  peak height grows ∝ √d and would otherwise crush the d=2 curve on a
  shared linear y-axis the same way a raw x-axis would have crushed it)
  — admitted on the figure via the printed σ(‖X‖) values rather than
  drawn, since the true heights carry no separate information already
  in the width comparison.

## Gate diagnostics that did NOT contain the fix
None outright — every collision fired with a working free-nudge
suggestion or an exact overlap pair. The one indirect cost: the first
free-nudge diagnostic ("+= (+0,-4) or (+0,+49)") was numerically valid
but the real fix was upstream (shorten and reposition two annotation
blocks whose *width*, not just position, caused the collision) —
applying the literal offset would have just moved the collision
elsewhere given how long the LaTeX strings were.

## Renders to first green: 3
## Proposed primitive
none — `colorbar` + `ramp` + `Callout` covered everything; the only
real work was the rescaling math, which is the figure's content, not
library plumbing.
