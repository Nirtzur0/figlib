# winding_number — friction record

## What the claim needed

Two Scenes bound by one map: a z-plane contour Γ enclosing a double zero
`a` and a simple zero `b`, and its image `f(Γ)` in the w-plane, drawn so
the reader can *count* — not take on faith — that `f(Γ)` winds around
`w = 0` exactly 3 times. Direction ticks on both curves (same
parametrization fractions) so the count is unambiguous, and a numerical
gate that measures the winding number the same way a human would: unwrap
`arg f(z)` along the sampled contour and read off the net turn, then
cross-check against an independent computation (root-finding the same
cubic via `numpy.roots`, not restating "2+1=3").

## What figlib gave for free

- `Curve.arrows` (tangent-oriented ticks at arc-length fractions) was
  exactly the primitive for "direction markers on both curves" — no
  custom arrowhead geometry needed, and it fired the `arrow-on-mark`
  mechanical gate the moment a tick landed on a label, which is the
  right failure mode.
- `Callout` (boxed label + leader to a target point) was the correct
  primitive for the turn-count annotation `n = 3` pointing at the
  origin — matches the panel-pair exemplar idiom (`polezero_response.py`)
  for "a computed quantity, labelled where it's read, leadered to what
  it's about."
- The panel-pair + map-`Connector` grammar (`Figure`/`Panel`/`Connector`)
  needed zero adaptation; `polezero_response.py` was close enough to
  imitate directly (z-plane panel with markers + axes, second panel,
  `Connector(kind="map")`).
- `gates.Checks` let four independent numerical facts (contour clearance,
  winding == enclosed count, winding ~ integer, "drop `a`'s twin root and
  the sub-curve's own winding is 1") report in one run instead of
  stopping at the first failure.

## What I hand-rolled

- **Multiplicity marker, and its failure** — ~6 lines. First attempt: a
  filled dot plus a concentric hollow ring ("bullseye") at `a` to signal
  "double root" visually. Rendered as an EMPTY ring with no visible
  inner dot. Root cause (found by reading `render.py`, not by guessing):
  a hollow `Point`'s fill is `_ground(style)` — opaque white on a
  transparent theme, by design (the "eraser" convention documented in
  architecture.md for halos/casings) — so the second, hollow Point
  painted an opaque disc that erased the first, filled one underneath.
  This is also a grammar violation waiting to happen: a hollow dot
  already means "excluded" elsewhere in the corpus (open dot =
  excluded, filled = attained), so stacking one to mean "multiplicity"
  would have been a second, silently conflicting use of the same glyph.
  Fixed by dropping the ring entirely — a single filled dot plus the
  `(double)` text label, with the multiplicity's actual *work* carried
  by the w-plane panel (two of the three turns visibly collapse near
  `a`'s image neighborhood). No primitive gap here; this is a corpus
  convention (hollow = erasing eraser, not a stackable ring) that isn't
  written down anywhere a figure author would see it before hitting it.
- **Winding-number-by-unwrap** itself — ~10 lines of `compute()`. Not a
  primitive gap; this is figure-specific numerics (`np.unwrap(np.angle(w))`),
  and it should stay in the figure program.

## Gate diagnostics that did NOT contain the fix

None outright — `arrow-on-mark` and `label-on-ink` both printed the exact
offset/fraction to use, and I applied their numbers verbatim. The one
diagnostic that pointed at a symptom rather than a cause was purely
visual (the bullseye rendering as an empty ring): no gate flags "this
hollow marker just erased the mark under it," because color/contrast
gates check legibility, not compositing intent. Found by reading
`render.py`'s `Point` branch directly.

## Renders to first green: 2

First `make check` run failed on `arrow-on-mark` (arrow tick landing on
the `\Gamma` label and on the `n = 3` callout) and `label-on-ink`
(`f(\Gamma)` sitting on its own curve). Both fixes were the diagnostic's
own suggested numbers; the second run passed. (One further `make check`
run was needed after the bullseye-marker fix, discovered visually post-PASS
rather than by a gate — see above.)

## Proposed primitive

A signature, not prose:

```python
def bullseye(xy: XY, role: Role, *, inner_scale: float = 1.0,
             ring_scale: float = 2.0) -> tuple[Item, ...]:
    """A filled dot with a ring around it that reads as multiplicity/
    emphasis, NOT as the hollow-Point 'excluded' convention — draws the
    ring as a thin unfilled Curve (circle), never a hollow Point, so it
    never erases what's under it on a transparent theme."""
```

Rationale: the corpus already has one way to draw "two nested markers at
one location" (stack two `Point`s) and it is silently unsafe on
`Style.transparent = True` themes because hollow `Point.filled = False`
paints an opaque eraser disc by design. A `bullseye()` builder that emits
a filled `Point` plus a stroked, unfilled `Curve` ring (never a second
`Point`) would give correspondence-multiplicity or emphasis-marker
figures a stacking primitive that can't accidentally erase itself — worth
having if a second figure ever wants "mark this point as special without
reusing the hollow = excluded channel." Not urgent: one figure needing it
so far.
