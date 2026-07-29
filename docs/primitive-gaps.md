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

## Continuous ink channels (landed; not in the original survey)

The VCA inventory counts *marks*; what it cannot count is that every
Curve attribute was a per-item scalar — nothing could vary along a
stroke, and nothing could sit under one. Three channel families fix
that, each general (a profile/callable, never a device):

- **width_profile** on Curve: width multipliers interpolated along arc
  length, rendered as a filled offset polygon. Tapered strokes, comet
  trails, speed-encoded weight. Semantics: taper = motion/emphasis.
- **ramp_segments** (builders): one polyline → n equal-arc-length
  Curves whose color/opacity follow callables of t; butt caps so
  translucent joints don't double-draw. Semantics: fade = time/decay,
  hue-along-curve = an ordered coordinate.
- **casing**: paper-colored under-strokes. `MathLabel.halo` keeps a
  label legible on busy ink (never mutate shared glyph `<symbol>`s —
  stroke the `<use>`); `Curve.casing` makes a curve read as passing
  OVER earlier ink. Both skipped on transparent themes. This is the
  channel that breaks the density–legibility tradeoff: annotation load
  can rise without clarity falling.

## Containment and elision (from the transformer-circuits schematic corpus)

Read off the 2025 attention-QK figures. The organizing idea is not a
palette: **their schematics are never free-floating box-and-arrow graphs.
Every one is embedded in a coordinate system that means something** — x is
context position, anchored to a literal monospace strip of the prompt at
the page foot, y is depth. Layout is not chosen, it is determined, and the
reader gets two real axes before reading a label. `induction_head_circuit`
already works this way; `schematic_transformer_block` did not.

**Region (landed).** Grouping as a filled ground behind the nodes, with
nesting read off a contrast ladder and no line drawn. Deliberately NOT a
`Node`: an edge entering an unattached Node is a collision, and a region is
the one box edges are supposed to cross, so they stay distinct types and
neither check special-cases the other. Members are named, not inferred — a
derived bbox that always fits would check nothing — which buys two
structural gates: `region_containment_violations` (a group that clips its
own member is a content bug) and `region_nesting_violations` (grounds must
form a tree; a half-overlap makes a third unnamed region out of the
intersection and there is no honest way to draw it).

*The finding that cost the most to learn:* the corpus washes its grounds at
about **1.03:1**, and that value is unavailable here. The house fill floor
is `MIN_PERCEPTIBLE_CONTRAST` (1.3:1), and clearing it is a statement about
luminance alone — no hue escapes it, every compliant fill lands near L 0.70
— so over an area the size of a sublayer the ground becomes the heaviest
ink on the page (measured: 59%). The corpus gets away with it on pure white
under a controlled rasterizer; on grained paper 1.03:1 is nothing. The
resolution is the gate's own exemption: an *outlined* fill is exempt,
because the border is then the mark carrying the contrast. So a house
grouping ground is a CONSTRUCTION-dashed border over a sub-floor wash —
louder than the corpus, and honest about why. **Large-area fills and stroke
floors are in genuine tension; the floor is right and the area is the thing
to spend carefully.**

**Still missing, in payoff order.** All three are *honesty devices
implemented as marks*, which is the corpus's real lesson: where our figures
admit an elision in the module docstring, theirs admit it on the page.

1. **Set/elision mark** — the stacked-card shadow meaning "this node is
   many things abbreviated to one". `Multi-Head Attention` is exactly this
   lie and currently only line 53 of its docstring says so.
2. **Declared truncation** — the diamond terminator and the arrow into
   `…`: this continues and I am choosing to stop. Cutting gets drawn.
3. **Unknown-mechanism node** — a box literally labelled `???`.
   Uncertainty as a node, not a caption hedge.

Two smaller ones, both cheap: a **labelled operator junction** (their K/Q
glyph is a named binary operator sitting ON an edge with its argument roles
annotated — generalizes the circled `+`, which is role-blind), and a
**mono/sans register split** where typeface encodes epistemic status
(monospace = literal model input, sans = human interpretation). The
register channel is free for us; every figure is currently all-serif-math.

`Role.MUTED` already covers their off-focus channel — same geometry,
desaturated, for "same object, not the subject" (`induction_head_circuit`
uses it for the non-accented ensemble).

## Transcript distillation, round 2 (measured, not recalled)

Mined all 14 session logs plus 41 subagent logs: **4.2M output tokens, 55
agents, 173 edits to figure programs**. Method and script shape matter
more than the numbers, so: parse the JSONL, reconstruct per-figure
`edit -> figcheck -> verdict` sequences, cluster diagnostic lines
verbatim, and split edits at each figure's FIRST PASS.

**The finding that survived checking.** Raw counts say `faint-ink` is 83%
of all diagnostics (260 events). That is a trap. Bucketed by hour, 196 of
them fall inside a single hour and carry `floor 1.5` — a floor that no
longer exists (it is 1.3 now, re-anchored on RISO's dotted frame at
1.44:1). That was ONE systemic ramp/theme defect hit ~200 times and then
fixed at the theme level. It is already compiled in and is not an
opportunity. *A frequency count over a transcript corpus measures how long
a bug lived, not how much a class of work costs.* Bucket by time and check
whether the constant still exists before believing any of it.

**What is actually durable.** 66 of 173 figure-program edits (**38%**)
touch label placement, and unlike faint-ink they are spread evenly across
every figure in the corpus. 16 are pure numeric nudges — identical code,
only the numbers moved. And **50% of all figure edits land AFTER the
figure's first PASS**: the gates are ceilings, so passing them says
"not broken", never "done", and the remaining loop is the agent looking
at a PNG and judging.

`autoplace` already closes the easy half (an un-pinned label that collides
gets a verified `offset_px` nudge). The expensive half is a label whose
position IS a parameter of the geometry — text set along an arc, a tag at
a fraction of a curve. Those are `pin=True` by construction, `autoplace`
is forbidden to move them, so they get hand-searched. One figure spent
four render cycles settling a single radius (1.17 -> 1.32 -> 1.40 -> 1.52)
with the clearance arithmetic done in prose each round.

**`place.py` (landed).** `place_on_locus` scans an author-stated locus and
returns the point with the most room, plus the achieved clearance as a
number the program can assert on; `label_clearance` answers "how much room
does this label have" so it stops being computed by hand. The split is the
doctrine's: the author chooses the LOCUS (meaning), the library chooses the
POINT on it (a 1-D scan with one answer, same species as `boundary_toward`),
and the gate holds the result.

*Its boundary, found by trying it on the figure that motivated it and
failing.* Retrofitting VCA Fig [19] did not work and was reverted:
- Placements are **greedy and sequential** — each becomes an obstacle for
  the next, so two words on competing loci starve each other, and the
  answer depends on order. The hand-tuning had solved them jointly.
- Under **free rotation** the objective misbehaves: maximizing clearance
  happily picks an orientation whose AABB is enormous but sits in a void.
- Everything measures **rotated AABBs**, so a word at 39° cannot fit a
  pocket that its glyphs would clear easily.
So v1 is honestly scoped to ONE label on a locus, which is the common
case. Jointly-placed sets stay hand-authored. Fixing this properly means
either tight glyph-hull clearance or a joint pass, and a joint pass is the
constraint solver this doctrine forbids — so it needs a decision, not a
patch.

**The post-PASS loop, classified — and a refuted hypothesis.** The "50% of
edits land after first PASS" number is real but does NOT mean 50% waste.
Classifying those 40 edits (this session's own excluded) by what they
changed:

| share | class | gateable? |
|---|---|---|
| 32% | structure / new content — annotations added, geometry reworked | no: design |
| 22% | ink / style channel | mostly no (see below) |
| 19% | label placement | partly — `place.py`, `autoplace` |
| 17% | params / geometry, pure numeric | partly — some already `arrow-on-mark` |
| 10% | exposition, assertions, imports | no |

The hypothesis on the table was a **salience gate**: declare the claim's
subject, rank ink groups by weight x length x contrast, assert the subject
ranks first. Reading all 9 ink/style edits verbatim kills it. They are:
occlusion correctness (a fill opacity raised so the cylinder actually
hides what is behind it), theme-routing hygiene (`as_floor`), a contrast
fix the colour gate *had already caught* (0.55 -> 0.75, "2.19:1 on
white"), dash-as-honesty (a chord inside the sphere), a private-to-public
API swap, and exactly ONE micro-tuning of weight and dash period. **One of
forty edits was a salience adjustment.** A gate for it would fire almost
never and would add a declaration to every figure to catch it.

So: the post-PASS loop is mostly irreducible design work plus gates
correctly catching regressions the author introduced after the first PASS.
That is the loop working, not waste. Do not build a "done" gate; the
compressible parts (labels, some numeric nudges) are already being
attacked directly, and the rest is the job.

*Instrument caveat:* the classifier keys on which tokens changed and
misfiles some edits (one assertions rewrite landed in ink/style). Treat
the shares as +/-5% and re-read the verbatim samples before acting on any
of them — the verbatim read is what refuted the hypothesis, not the table.

**A cheap API-gap signal, worth automating.** When a figure program
imports an underscore-prefixed figlib symbol, that is a missing public
parameter. It happened twice (`_emit`, `_split_signed` in the sphere demo,
both later replaced by `drape(..., signed=)`). Grepping `figures/` for
`import.*\b_[a-z]` is a one-line lint that turns a private-symbol reach
into a public-API to-do.

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

## Module-authoring rules (distilled from the capability-layer build)

For anyone extending figlib itself. Each rule closed out a real
debugging class in the builder transcripts.

- **The `*_ink()` rule.** A new item whose ink is derived (not literal
  points) ships ONE canvas-px resolver in render.py returning geometry
  + any derived MathLabels; gates.py imports it. A gate that re-derives
  geometry is a bug (this is how brace/callout/connector stay
  drift-free between render and gate).
- **Ink math is single-sourced in render.py.** A producer module that
  needs it (figure.py needs arrowheads) imports lazily inside the
  function. Never duplicate glyph geometry; never a module-level
  back-import.
- **Visibility predicates classify the boundary as visible (`>=`).**
  Uniform resampling WILL land samples exactly on a silhouette; `>`
  splits a fully-visible run into three.
- **numpy 2.x dropped 2-D `np.cross`** — use `geometry.cross2d`.
- **Rasterizer quirks registry** (the PNG is the figure of record):
  cairosvg ignores `image-rendering: pixelated` (pre-upsample rasters
  instead). Add new quirks here as found.
- **Hidden-line styling comes from `style.hidden_variant(role)`**, not
  local constants — the per-theme gate test keeps every hidden variant
  above the contrast floor when palettes change.
- **Tests assert SVG via `tests/svgkit.py`** (`tag()`, `find_by()`,
  `path_cmd_counts()`) — never raw `e.tag`: ElementTree namespaces
  every tag on parse, and that trap cost 30k tokens once.
- **Parallel-agent hygiene.** Shared-file edits are targeted Edits
  after a fresh read, never Write-rewrites. If pytest fails but a
  direct run passes: rerun once (mid-edit state), then check SVG
  namespacing, before any cache theory.

## Matrices (landed; invisible to the VCA survey)

The evidence base for this document was Needham, and Needham has no
matrices — so the largest gap for a machine-learning corpus was never
counted. `matrix.py` closes it, sourced from Hiranabe's *The Art of Linear
Algebra* (graphic notes on Strang), whose grammar turns out to be small
and orthogonal in exactly the way the doctrine above predicts: one
rectangle with four readings, and every factorization reducing to a sum
of rank-1 rectangles.

**The line the module had to justify.** `dft_matrix_basis` addresses an
8×8 grid with a two-line `center` closure and needs nothing from here —
correctly, because addressing is cheap to rewrite and the doctrine says
absorb verbosity. (An earlier `CellGrid` design was cut for exactly this
reason; `map_into`, `edge`, `grid_lines` and `brackets` went with it,
having no consumer.) What is *not* cheap to rewrite is the gating.
`check_expr` evaluates a drawn factorization in numpy and `check_conformable`
reads inner dimensions off the drawn term list; neither can live inside
one figure, because a gate is the shared oracle. Both need shape and
values to be one object surviving `compute()` → `assertions()`. **The
test for a new module is not "is this device inexpressible" but "does
this make a new class of claim checkable".**

**Shape is the thing to make geometric.** Drawing a matrix at its own
aspect ratio is not stylistic: it is what makes a non-conformable product
undrawable and lets the picture be *proved* rather than trusted. A figure
that merely arranges rectangles cannot be gated at all.

**Two marks cost a rewrite each, both for the same reason — the mark
asserted something the data did not.** `rank1` first painted one column
and one row of the *result* block; but the summand is the whole
rectangle, not that cross, so the mark was false. It now draws what rank
1 means — every column is `u` scaled by `v[j]` — which is true, and
differs per summand. And `svd_low_rank` first drew a noisy causal
attention map, which is *not* low rank (rank 4 kept 62%), so its CLAIM
was false; a Gaussian Gram matrix is (99.7%), and an assertion now pins
that so the claim cannot rot.

Still open, deliberately deferred: **einsum / tensor-network diagrams**
(Penrose notation). A tensor has legs, not a 2-D shape to draw to scale,
so it belongs on `schematic.py` as named nodes with contracted edges —
forcing it into `Block` would corrupt the shape-is-geometry invariant.
References: *An introduction to graphical tensor notation for mechanistic
interpretability* (arXiv 2402.01790) and *Named Tensor Notation* (arXiv
2102.13196).
