# The visual grammar

Extracted from Needham (*Visual Complex Analysis*, Figs 4–14) and hardened
by adversarial judging of our recreations against the originals. Each rule
earned its place by a concrete failure.

Upstream of these rendering rules sits exposition.md — whether the
figure deserves to exist, which representation to choose, and how a
reader's attention traverses it. That layer decides; this one renders.

## The design step (before any code)

State, in order:

1. **CLAIM** — one sentence. What the figure argues, not what it draws.
2. **Geometry that encodes the claim** — which computed objects make the
   claim visible? If no computation generates the geometry, the figure is
   Class B (layout is the content); otherwise every curve comes from
   numerics.
3. **The mechanism annotation** — which quantities must a reader be able
   to *read off the figure* to reconstruct the derivation? This is the
   rule the judges enforced hardest: a figure that shows the
   correspondence but strips the quantitative labels has "traded
   mechanism for decoration" (verbatim judge verdict). Needham's text
   says "as we see from the figure" — the figure must support that
   sentence.
4. **The claim's hard half** — most claims have an easy half and a hard
   half (for z^n: angle multiplication is easy, radial r^n
   bunching is hard). Ink budget goes to the hard half. Our first
   z^n recreation demonstrated only the easy half and lost the judging.

## Rules with their justifying failures

- **Labels are mathematical objects, never captions.** iθ, r_1 r_2 = k²,
  e^{iθ} — measured on the figure itself. (Needham's uniform practice.)
- **Ink hierarchy:** solid = content, dashed = construction,
  dotted/faint = frame. Axes only when the axes are the content.
- **Annotations state the theorem:** right-angle marks, equal ticks,
  angle arcs with the angle named. If the claim is "these are
  perpendicular," draw the right-angle mark — and assert it numerically.
- **Color is a semantic channel or absent.**
  - Hue = correspondence (same hue = same object across panels/maps).
    This beats grayscale books — it's the one channel where we win.
  - Lightness = ordered quantity (radius, level k).
  - An accent = the distinguished object (the lemniscate, the separatrix).
  - Never decoration.
- **A mark the reader cannot see is not a mark.** Content ink clears 3:1
  against the paper it lands on, measured after compositing its opacity —
  the color gate checks this, so it is not a matter of taste. Scaffolding
  (construction, frame, muted) is deliberately quieter and only has to stay
  perceptible; an ensemble drawn to show a distribution is judged as the
  bundle it is, not per path. The failures: viridis' light end at 1.6:1 on
  white left the last rays of the z^n fan drawn but unreadable, and the riso
  ramp's mustard sat at 1.3:1 on cream — effectively unprinted.
- **Correspondence hues must survive color-blindness.** Same hue = same
  object only holds if a reader can separate the hues; the gate checks all
  pairs (not just neighbors — any two curves in a plane can end up side by
  side) under protanopia and deuteranopia. This caps how many hues a palette
  can carry: four for RISO, three for CLEAN. Past the cap, identity rides
  dash or facets — never a generated hue, which is indistinguishable from
  one already in use.
- **Show distributional claims, don't assert them.** A claim about an
  ensemble (marginals match, endpoints cluster) needs the ensemble on
  the figure (endpoint strips, dense mesh), not a silhouette plus trust.
  (Diffusion figure, readback confusion #3.)
- **A mesh, not samples, when the claim is about a transformation of a
  coordinate system.** Ten curves read as ten curves; forty read as a
  deforming grid. (Judge: "the recreation samples the map; the book
  shows it.")
- **Label the theorem, not the instance.** Boundary labels r^n, nθ even
  when the drawing instantiates n = 3; note the instance once
  ("here n = 3") near the map label.
- **Every mark carries its conventional meaning.** An open dot means
  excluded; using it decoratively is "actively wrong ink" (judge).
  Filled = attained, open = excluded, no exceptions.
- **Attach labels to what they name.** A label floating in whitespace
  gets read as naming the nearest big thing. Angle labels sit on their
  arcs; segment labels at segment midpoints, offset perpendicular.
- **Representative randomness is chosen honestly.** Stochastic paths are
  real draws, but the seed is selected so drawn paths stay in frame and
  hit the claim's cases (both modes). Selection is on legibility, never
  on the claim.
- **Make the mechanism geometric, not verbal.** If a caption states a
  geometric fact ("R = distance to the poles"), draw the object that
  embodies it (the |z|=1 circle draped on the surface, running into both
  poles). A stated fact the drawing could show is a defect. (Fig 14
  readback: "the single fact the figure exists to establish is the one
  thing left to the caption.")
- **Admit depictions' lies.** Infinite quantities rendered finite
  (truncated spires) get a small printed admission; an unlabeled
  truncation reads as a finite maximum.
- **One claim per figure.** Stages of an argument get [a]/[b] panels.
- **Design at display size.** Canvas px = page CSS px; declare the slot
  (FORMAT) and never scale type to fit. If annotation doesn't fit, the
  slot is too small or the ink budget too large — shrinking labels below
  8.5 pt trades legibility for density and the mechanical gate rejects
  it. (The volcano's 8 pt truncation admission was the first catch.)
- **Trim to content.** Located whitespace (an empty quadrant) is a
  design smell; axes end just past the content they support.

## The gates (definition of done)

1. **Numerical:** assertions on the same arrays that got plotted —
   defining equations satisfied to tolerance, solutions Richardson-
   verified against a finer integration, distributional claims checked
   on a large ensemble even when only a few paths are drawn.
2. **Mechanical:** no label collisions/clipping (exact bboxes).
3. **Color:** every mark visible on its paper (contrast measured after
   compositing opacity, floors following the ink hierarchy), and every
   correspondence hue pairwise separable under protanopia/deuteranopia.
   Palette-level properties — slot separation, ramp monotonicity — are
   asserted per theme in `tests/test_color_gate.py`, so a palette edit
   cannot silently regress the corpus.
4. **Readback:** a context-free model sees only the PNG and reports in
   two passes — a *glance read* (first impression: does the macro
   structure alone carry the claim?) and a *studied read* (the claim,
   confusion bullets, and which steps were verifiable by inspection
   vs. taken on trust). A missed glance read is a macro-structure
   failure even if the studied read recovers. Confusion bullets are
   design review — each one is either fixed or explicitly accepted.
5. **Comparative (when a reference exists):** a judge sees the original
   and the recreation and rules BOOK BETTER / COMPARABLE / RECREATION
   BETTER with named defects. Iterate until at least COMPARABLE.
