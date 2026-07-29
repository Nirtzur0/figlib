# Primitive gaps: what VCA demands that figlib cannot yet draw

Evidence base: a device inventory of Needham's *Visual Complex Analysis*,
Ch2–12 (~200 figures surveyed by four parallel readers, cross-checked by
direct reading of the Riemann-sphere, Möbius-classification, and
Poincaré-disc figures). Counts below are figures-gated-per-capability,
summed across chapters. This doc ranks what to build; grammar.md says how
to use it; architecture.md owns the invariants.

## The binding constraint

It is not any single mark. figlib is single-panel; Needham's page grammar
is fundamentally **panel-pair + connector**: [domain panel | labeled
squiggle arrow | codomain panel] inside one rounded frame, extended to
chains (composition), fan-ins (f + f̄ → brace → F), filmstrips
(deformation sequences), 2×2 taxonomies, and equation-of-figures panels
composed with = and + glyphs. Ch4–5: ~28 of 31 figures are multi-panel.
Without a panel layer, no VCA figure is reproducible *as printed*.

## Tier 1 — gates the most figures, unanimous across all chapters

1. **Multi-panel layout + page-coordinate annotation layer** (~90 figs).
   A Figure container holding 2–4 Scenes with independent transforms;
   panel tags [a]/[b]; rounded-frame furniture (theme-owned); and a page
   layer for things that span panels. Insets/zoom lenses fall out free
   (a small panel + leader). Keep it dumb: affine placement, no
   constraint solver — the mechanical gate already catches collisions.

2. **Direction markers on curves** (~50+ figs). Tangent-oriented
   arrowhead glyphs at arc-length parameters t, on any Curve. Two
   styles, used semantically by Needham: **filled** = integration
   contour / motion, **hollow triangle** = streamline. Subsumes: curved
   arrows (short arc + terminal head), oriented circles, signed angle
   arcs, rotation-dial glyphs, loop-orientation marks. Cheapest
   high-value item in the whole inventory.

3. **Connector glyphs** (~40 figs, needs 1). The labeled squiggle
   map-arrow between panels (often doubled forward/back), hollow block
   arrows for filmstrip steps, inline =/+ operators between mini-panels.
   One Connector(path, decoration, arrowhead, label) in page coords.

4. **Line-style channel** (~30 figs). dash/dot/weight as first-class,
   theme-routed. Needham's third ink channel: solid = subject, dashed =
   hidden/construction/dual, dotted = background family — and in places
   dash pattern carries *identity* across panels (bold/plain/dashed
   circles ↔ bold/plain/dashed lines). Implies a monochrome categorical
   channel: a style-triple (weight, dash, head-style) alternative to
   `categorical(i,n)` hue, so CLEAN can stay faithful where the book is.

## Tier 2 — major unlocks

5. **Sphere / closed-surface projector** (~20 figs; all of Ch3's Riemann
   sphere, half of Ch6, flows-on-surfaces in Ch10). surface3d's height
   field cannot represent a sphere (two-valued). Minimal version, same
   pattern as surface3d (producer of scene items, never a renderer):
   orthographic sphere = silhouette circle + great/small circles as
   ellipse arcs split at the silhouette into visible (solid) and hidden
   (dashed) sub-arcs — the hidden-line test on a sphere is a dot
   product. Curves draped on the sphere clip the same way; sphere ∩
   plane scenes need painter's ordering with the plane quad. Add a
   **wireframe isoline mode** (gridlines with hidden-line removal, no
   facet shading) — the book draws the pseudosphere family as mesh, not
   Lambert.

6. **Scene-builder helpers** (compute-side, pure functions → scene
   items; leverage without constraint):
   - `mapped_grid(f, u_lines, v_lines)` → grid Curves + *addressable
     cells* as FilledCurves. Unlocks tracked black cells (the book's
     orientation marker), Ford checkerboards (parity fill), branch
     partitions. Used in every chapter.
   - **Streamline generator**: integral curves of a plane field with
     singularity-aware seeding and flux-tube spacing (Ψ = nk) so
     *density encodes speed*; stagnation/saddle detection emitting
     markers. ~20 figures in Ch10–12 alone; also every future
     phase-portrait / gradient-flow / optimization-trajectory figure.

7. **Clipping** (~10 figs). Clip scene content to the panel frame or a
   disc; checkerboard portraits and unbounded hyperbolic regions run
   content off-frame deliberately.

## Tier 3 — cheap glyphs and attributes, add on first demand

- **Open/outline Vector variant** — the ghost-copy convention (white
  copy of the preimage arrow at the image point) is load-bearing
  pedagogy in Ch4. A fill flag.
- **Brace glyph** — Brace(p1, p2, side, label) for measured lengths.
- **Boxed callout label + leader line** — box flag on MathLabel + a
  leader Connector.
- **MathLabel rotation** — an angle param covers nearly all
  "text along a curve" uses; true textPath is rare (~2 figs).
- **Pattern fills** (stipple/hatch) on FilledCurve, theme-routed —
  monochrome faithfulness; hue substitutes in color themes.
- **Regions with holes** — even-odd fill on FilledCurve.
- **Composite glyph recipes** (not primitives): starburst singularity
  mark, ⊕/⊖ crossing marks, saddle glyph — figure-program idioms.

## Beyond VCA: what DS / ML / physics add that the book does not

The VCA tier-1/2 list already covers most of physics and ML figure
space: phase portraits, vector fields, conformal grids, flows,
correspondence panels, 3D surfaces. Genuinely new needs:

1. **Raster scalar-field primitive** (dense grid → theme ramp, embedded
   as an image in the SVG, everything else stays vector). Attention
   matrices, loss landscapes, PDE fields, spectrograms, kernels. No
   vector substitute at that density; the honest encoding for a dense
   field. Numerical gate applies to the array as usual.
2. **plots.py** (already planned): axes/ticks/series ON scene
   primitives. ML demands log scales (scaling laws), error
   bands/ensembles (FilledCurve), histograms/densities, scatter.
3. **Class B schematic layer** (already planned): nodes/ports/edges +
   layout — architectures, computational graphs, Markov chains, einsum
   / tensor-shape block diagrams. Mechanical gate matters most here.

Everything else on the ML wish-list (embedding scatters, training
dynamics, distribution morphs, bifurcation diagrams) decomposes into
tier 1–2 + these three.

## Design doctrine (the bitter-lesson answer)

The temptation is a recipe/template per device — a `poincare_disc()`, a
`riemann_sphere_figure()`. Wrong scaling. The device count grows with
every new book and field; the primitive count above is ~a dozen and then
plateaus, because devices are *compositions*.

- **Small orthogonal kernel, compositional closure.** Markers-on-curves
  subsumes five devices; panels+connectors subsume six page patterns.
  Add a primitive only when a device is *inexpressible*, never when it
  is merely verbose. Verbosity is the model's job to absorb.
- **Recipes live as exemplar figure programs, not API surface.** The
  device lexicon (checkerboard portrait, ghost-copy, infinitesimal
  idiom, contour+region+winding unit, background-family/foreground-
  actor…) goes into the skill as named devices each with one worked
  figure program. A menu the model imitates and mutates — priors, not
  constraints. In-context exemplars scale with model capability; baked
  API does not.
- **The gates are the search loop.** Readback + mechanical + numerical
  gates are the scalable oracle; iterating a figure program against
  them is where quality comes from, and it improves for free as models
  improve. Invest there (more gate sharpness, cheap iteration) before
  investing in layout intelligence. No constraint solvers, no
  auto-placement heuristics: model proposes, gates dispose.
- **Compute-side builders are free leverage.** mapped_grid and
  streamlines are numerics with one honest output; they constrain
  nothing and remove the error-prone boilerplate. Style-side
  automation is where over-constraint creeps in — keep style choices
  in the figure program where the model (and the grammar) can see them.

## Build order

1. Curve markers + line-style channel + open-arrow flag (one sitting;
   biggest unlock per line of code).
2. Panel/Figure container + page layer + Connector glyphs.
3. Sphere projector (+ wireframe mode for surface3d).
4. mapped_grid + streamline builders.
5. Glyph pack: brace, callout+leader, label rotation, pattern fills,
   holes, clipping.
6. Raster field primitive; plots.py and Class B when their first real
   figure demands them (per architecture.md).

Each step lands with a benchmark recreation that needs it (candidates:
Fig [30] elliptic checkerboard → 1+2+4; Fig [19] stereographic
projection → 3; a Ch12 flow-past-cylinder → 4) and passes the full gate
stack including comparative judging.
