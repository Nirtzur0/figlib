# inversion_in_circle — friction record

## What the claim needed

Five checkable clauses in one panel pair: circles through the center of
inversion become lines; circles missing the center stay circles; the
inversion circle |z|=R is the pointwise-fixed locus; a pair of circles
crossing at 90 degrees still crosses at 90 degrees after the map; and the
map is anticonformal, so that right angle's sense flips even though its
magnitude survives.

## What figlib gave for free

- `Curve`/`Point`/`MathLabel`/`RightAngleMark`/`Callout` covered every
  drawn object with no new primitive needed — a circle-preserving map
  figure turned out to need nothing beyond the existing vocabulary.
- The panel-pair + `Correspondence` machinery (`figure.py`,
  `correspond.py`) is exactly the right shape for "domain vs. image":
  `geometry_extents`-derived `width_frac` kept the two very
  differently-scaled panels (a bounded family of circles vs. a family of
  unbounded lines, windowed) sharing one page scale automatically.
- The `stale-change` / `label-on-ink` / `clipped` diagnostics each found a
  real defect (a declared-but-undrawn difference, three separate label
  collisions, an off-canvas anchor) and each one's printed fix
  (`offset_px += (...)`, "no free single-axis nudge; nearest ink-free
  region center (x,y)") was directly actionable — no guessing required to
  resolve any of them.
- `Correspondence`'s position-exclusion (architecture.md: "position is
  deliberately outside the fingerprint") was the right call ONCE
  understood: the two tracked circles' only cross-panel difference is
  where they sit, so listing them in `changes=` was wrong (nothing in the
  facet actually differs) — the correct binding leaves them keyed but
  *not* in `changes`, and only the annotation whose *text* differs
  (`90°₊`/`90°₋`) belongs there. This took one failed run to learn; the
  diagnostic message explained it precisely enough to fix without
  re-reading the source.

## What I hand-rolled

- **Singular-point circle parametrization** (~25 lines): a circle through
  the pole of inversion literally passes through the coordinate where
  `1/z̄` divides by zero. Handling it — a phi-grid that goes "the long
  way around" a small excluded arc, one continuous open curve, no
  discontinuity — isn't circle-specific geometry the library has a
  primitive for, and probably shouldn't (it's inversion-specific).
- **Finite-window masking for an unbounded image** (~10 lines): the image
  of a through-origin circle is a full line, and only a bounded segment
  of it can be drawn. Masking by projected distance along the line's own
  direction, then trusting the mask to stay contiguous (true here because
  the projected coordinate is monotone away from the one singularity),
  is a small but easy-to-get-subtly-wrong piece of reasoning that took a
  derivation on paper before coding.
- **Circle-circle intersection + tangent-by-finite-difference** (~35
  lines): finding where two engineered-orthogonal circles actually cross,
  then reading off tangent directions at that crossing from the SAME
  sample index in the domain and image arrays (so orientation comparisons
  are apples-to-apples) — general-purpose computational geometry with no
  home in the current library.
- **A small offline grid search** (not shipped in the figure, just a
  scratch script) to choose the second orthogonal circle's center so its
  crossing point with the tracked circle sits maximally far from every
  other drawn curve. This is the one piece of real friction: a
  *composition* problem (many circles from one family, one engineered
  pair, all sharing a crowded region near the pole) that no gate flags,
  because every individual curve is correct and every individual label
  is correctly placed — the defect only exists as a *reading*, caught by
  the cold-reader pass, not any mechanical check.

## Gate diagnostics that did NOT contain the fix

- None, mechanically. Every `label-on-ink` / `clipped` / `stale-change`
  diagnostic that fired pointed at an actionable, correct fix (a pixel
  offset, or an explanation of the facet-fingerprint semantics).
- The one defect no gate could have caught: the orthogonal pair's
  crossing point landing visually on top of the (unrelated) inversion
  circle and a third, unrelated family circle, so a cold reader attributed
  the drawn right angle to the wrong pair of curves. This is exactly the
  gap `correspond.py`'s docstring names — "the residual leaking through a
  channel no gate reaches" — except here the leak isn't a correspondence
  residual at all, it's plain visual crowding between objects that were
  never claimed to relate to each other. The mechanical gate has no
  concept of "these two curves are meant to be read together, and a third
  curve must not sit between them" — that's a design-time property
  (`docs/architecture.md` step 4, "traversal") not a computable one.

## Renders to first green: 14

## Proposed primitive

None with high confidence. The closest candidate is a compute-time
helper, not a gate: something like
`geometry.max_min_separation(candidate_points, other_curves) -> best_point`
— a small grid/optimization utility for "place this tracked feature at
the point in a candidate set farthest from every other drawn curve." It
would have turned a 20-line scratch script into one library call, but
it's thin enough (and specific enough to "a marked point must be visually
isolated") that I'm not confident it generalizes past this figure and the
one or two others in the corpus that hit the same problem
(`vca_fig9_cassinian.py`'s congested-corner Callouts are the same shape of
problem, solved by hand there too).
