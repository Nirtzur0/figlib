# Architecture

How the system stays unified as it grows: one content model, one theme
layer, one gate protocol, and a thinking layer that decides what to draw
before anything gets drawn.

## The stack

```
CLAIM (one sentence)
  │  design step (prompts, below)
  ▼
figure program        figures/*.py — compute() → build() → assertions()
  │
  ▼
Scene                 typed primitives in math coords, semantic Roles
  │                   (2D primitives; 3D arrives as depth-sorted 2D items
  │                    via surface3d — same scene, same gates)
  ▼
Theme                 appearance ONLY: paper, inks per Role, ramps,
  │                   categorical hues, surface shading, grain
  ▼
autoplace             deterministic free-nudge solver: un-pinned labels
  │                   move ≤ 24 px clear of collisions and the frame;
  │                   the rest falls through to the gate
  ▼
render                SVG (+ PNG) — owns every bbox
  ▼
gates                 numerical / mechanical / readback / comparative
```

**Constructor vs. gate.** A check whose fix is computable gets promoted
into the pipeline; a check that stays data-dependent stays a gate. The
auto-place pass is the first promotion: the mechanical gate used to
*compute* a verified collision-free nudge and print it for the model to
type back in — now the same computation is applied before the gate
measures, and gates + render see one solved scene. The gate still owns
what the solver won't touch: pinned labels (`MathLabel.pin` — position
that IS meaning), moves past the 24 px budget (a label that far from its
anchor is a design defect, not a layout defect), and derived boxes
(brace labels, callout boxes, panel tags). The second promotion is
format derivation (see Sizing): undeclared `FORMAT` is solved from the
annotation census, and the load gate's "larger Format" advice became a
computed verdict instead of an assumption.

**The invariant that keeps everything consistent:** content code never
names a color, font, or stroke width. It names *meanings* — `Role.CONTENT`,
`Role.ACCENT2`, `theme.ramp(t)`, `theme.categorical(i)`,
`theme.surface_shade(t)`. Themes map meanings to appearance. Any figure
re-renders under any theme; a theme change is a one-line edit that
restyles the entire corpus.

## Themes

`theme.py` defines the contract; two implementations:

- **CLEAN** — white paper, near-black ink, restrained blue/red accents.
  For contexts where the figure sits inside dense text.
- **RISO** — the house look. Warm cream paper with a subtle vertical
  gradient, printed-grain overlay, sun-bleached saturated palette
  (indigo / brick / mustard / sage / plum), 3D facets shaded through a
  multi-hue ramp rather than dark→light. Risograph / Anthropic-circuits
  adjacent: granular, warm, flat-but-dimensional.

Semantic channels a theme must provide (this is the whole interface):

| channel | meaning | example |
|---|---|---|
| `ink(Role)` | line work by semantic role | content/construction/annotation |
| `ramp(t)` | ordered quantity → color | level k, radius, height |
| `categorical(i)` | correspondence hue | same ray before/after a map |
| `surface_shade(t)` | 3D facet lighting → color | the volcano ramp |
| `paper`, `grain` | ground and texture | riso cream + speckle |

**No ground by default.** `Style.transparent` is `True`: a render emits ink
on alpha, no paper rect, no page-wide grain overlay. A figure is meant to land
on whatever document embeds it, and the ground is that document's decision.
Two things are *not* ground and survive: grain inside a fill
(`FilledCurve.grain`), and the paper-coloured erasers — casings, halos,
hollow marker fills, callout backing. An eraser needs an opaque colour to
be an eraser, so groundless it paints white (`render._ground`), matching the
hostile ground `paper_stops()` already assumes. Cost: embed on a dark page
and those read as white patches. The colour-free fix is an alpha knockout
mask, which the renderer does not do yet.
The theme's own ground is the DEFAULT. To drop it: `transparent_variant(THEME)`,
the prebuilt `RISO_CLEAR` / `CLEAN_CLEAR`, or `figcheck --transparent`, which
writes `<name>_transparent.svg`. `--regress` checks both grounds for every
figure, unconditionally. The contrast gate assumes white under a groundless
render — the hostile case for light ink.

Rules: hue = correspondence, lightness = order, accent = the
distinguished object, never decoration (see grammar.md). A theme changes
*which* colors carry those channels, never *what* the channels mean.

**Channels are tagged, so they can be gated.** `categorical()` / `ramp()` /
`surface_shade()` return a `Hue` — a `str` subclass carrying its channel —
so a color's provenance survives into the Scene and the color gate can hold
each channel to its own standard: correspondence hues must be pairwise
separable, an order ramp must be monotone in lightness, a shading ramp is
exempt from contrast (its lit end is *meant* to approach the paper).
A bare `'#rrggbb'` in a figure is ungated on the correspondence check, which
is correct — it is not claiming to encode identity.

**Correspondence slots are fixed, never interpolated.** `categorical(i)`
indexes a fixed order; it does not take `n`. Interpolating `i/(n-1)` made a
series' hue a function of how many series shared the figure, so the same
object took different hues in two panels with different counts — breaking
the one property the channel exists to carry — and shrank neighbour
separation to ΔE 7.9 by n = 8. Past `theme.correspondence_cap` slots (4 for
RISO, 3 for CLEAN — the all-pairs limit these palettes actually support)
identity must ride a second channel: `Curve.dash`, facets, or a folded tail.

## Sizing (format.py)

**The invariant: canvas units are display CSS pixels.** A figure declares
its slot on the page (`FORMAT = MARGIN | COLUMN | WIDE` — 340 / 680 /
1000 px), and every absolute quantity — label pt, stroke px, arrowheads,
dot radii — is thereby at final rendered size. 11 pt is 11 pt in every
figure; nothing is designed at one size and read at another.

Consequences enforced by the mechanical gate:
- no label below 8.5 pt (`tiny-label`);
- no label taller than 18% of the canvas (`label-scale`);
- total label area under 22% of the canvas (`annotation-load`).

When a gate fires, the fix is **never smaller type** — but "larger
Format" is not assumed to help, because the ladder mostly doesn't:
MARGIN → COLUMN genuinely quarters the load fraction (same ink, 4× the
canvas), while COLUMN → WIDE is near-invariant *by design* (WIDE's
ink_scale 1.45 tracks its width ratio 1000/680 — reading-size parity,
so canvas and ink grow together). The runner therefore *computes* the
verdict: a load failure carries either `FORMAT = X would carry this
load` or `no format carries this load — trim annotation`, evaluated
per candidate (`format.smallest_carrier`).

`FORMAT` is a declaration of the page slot — an external constraint,
policed, never silently overridden. A program that declares none gets
the derived format (`format.derive_format`): the smallest carrier at or
above COLUMN, reported in the run notes. MARGIN is never derived —
taking the margin slot is page-layout intent, not a sizing consequence.
Single-panel figures default to COLUMN; two-panel comparisons and dense
3D take WIDE; a wrapped side figure takes MARGIN and correspondingly
less annotation.

## The three figure classes, one pipeline

- **Computed geometry (Class A)** — compute() produces the curves;
  everything so far. 3D is not a separate class: `surface3d` projects,
  depth-sorts, and shades into ordinary 2D items, so gates apply
  unchanged.
- **Data plots** — a thin `plots.py` layer (axes, ticks, series) built
  ON the scene primitives, not beside them. An axis is Curves + Ticks +
  MathLabels; a series is a Curve with a categorical hue. No second
  rendering path, so plots inherit theme + gates for free. (Built:
  `Axes` owns the data→scene transform — log scales happen there, so
  scene coords stay affine — with 1/2/5 and decade tick locators,
  frame/grid furniture, line/band/scatter/hist emitters, xlabel/ylabel.
  Axis titles sit outside the geometry extents by design: the figure
  program reserves margin via scene lims, and the mechanical gate's
  `clipped` check is what catches the miss.)
- **Schematics (Class B)** — nodes, ports, edges with a layout pass;
  compute() is empty and layout is the content. Same scene, same theme,
  same mechanical gate (label collisions matter most here). (Built:
  `schematic.py`. Typed `Node`/`Port`/`Edge` with five edge kinds and a
  chosen route — straight, one elbow, Bezier through stated via points.
  The layout pass DERIVES three things and nothing else: the boundary
  point where an edge attaches (`connect`/`trim_at_boundary` — edges
  terminate ON boxes, never at centers), a box that holds its own typeset
  label (`auto_node`), and a layered placement from a longest-path
  ranking of the DAG (`RankLayout`/`rank_positions`, even spacing within
  a rank, explicit lane and explicit position each winning locally). No
  force-directed step, no packing, no crossing minimization — the graph
  is *reported* by `crossings` / `clearance_violations` / `label_overflow`
  and the figure program decides. Benchmarks: `induction_head_circuit`,
  `schematic_transformer_block`.)

The rule that keeps the architecture clean: **new capability enters as a
producer of scene items, never as a new renderer.** surface3d proved the
pattern.

## Composites (correspond.py)

`Figure/Panel/Connector` places the parts of a composite and draws a
squiggle between them. It does not know what the squiggle *asserts* — and
a composite's claim is never a picture, it is a predicate over a binding
(exposition.md, "Composites"). So the relation is declared, not drawn:

- **The binding is a name on the geometry.** `Curve(..., key="unit-circle")`,
  or `keyed("branch", *plots.series(...))` for the producers that emit a
  group. Items sharing a key ARE the same object. Written where the
  geometry is built, so the two halves cannot drift apart the way
  retyped literals do.
- **`CORRESPONDENCE = [Correspondence(parts, varies, changes, frame)]`**
  at module level names the parts, the one axis of variation in prose,
  and the keys allowed to move.
- **The gate reports the residual**: a key that differs and was not
  declared; a key declared in `changes` that never actually moves (the
  figure states a difference it does not draw); page-scale drift across a
  shared frame; and — surviving a declared rescale — the fixed set drawn
  at two sizes.

Position is deliberately outside the fingerprint (moving is usually the
claim), and so is opacity (a legibility channel, not an identity one).
The declaration is fully silent when a program has none, so independent
panels are never forced to invent a relation. And there is no `kind=`:
before/after, transport, instance-in-family and analogy are settings of
(binding, variation), not types — see exposition.md for why naming them
would be the overfit.

Scope: multi-panel `Figure`s. Single-scene composites — a ghosted
pre-image, a word-scale inset — bind within one Scene and need a selector
for "part" that does not exist yet; `diffusion_ode_vs_sde` is the corpus
figure waiting on it.

## The thinking layer (what actually decides quality)

Everything above is plumbing. Quality is decided before code, by the
design step — the prompts that will form the skill's core. The theory
behind each step lives in exposition.md; the steps, in order:

0. **Earn the figure.** Write the one-sentence prose version that would
   make the drawing unnecessary. If it succeeds, no figure (prose is
   preferred — the reader runs their own imagery). If it fails, name
   the specific inference the reader must make and check the figure
   will make it *perceptual* — premises co-located, conclusion readable
   as a visual feature. A figure that redraws the prose fails here.
1. **CLAIM.** One sentence, the claim the figure argues. If you cannot
   write it, there is no figure yet. The claim may be delivered as a
   provocation (an unexplained phenomenon, a drawn counterexample) —
   but it exists.
2. **Representation.** Which primitive makes the claim's hypothesis
   checkable by eye (triangles, not rectangles)? What is the expert's
   *private* picture of this, the one that never gets published — draw
   that. Fix the abstraction rung (instance / trajectory / family /
   behavior map), one rung per figure, anchored by a bound concrete
   case. For a ∀-claim, a diverse population treated uniformly; for a
   limit or deformation, the one frame where the conclusion becomes
   obvious. If the picture saturates mid-argument, change
   representation rather than decorate.
3. **Size and slot.** Which page slot (Format)? Annotation load and
   slot are chosen together — a MARGIN figure carries one label, a
   WIDE comparison carries the theorem. Ink budget goes to the claim's
   HARD half (grammar.md).
4. **Traversal.** Script the read: where does attention enter (the
   claim), what path does adjacency force, what does a 3-second glance
   yield vs. a 30-second study? Everything used in one inference sits
   at one location. Delete off-message content first, then signal the
   skeleton.
5. **Mechanism annotation.** Which quantities must a cold reader be able
   to read OFF the figure to reconstruct the argument? Draw them, on
   the elements they describe; a caption stating what the drawing could
   show is a defect.
6. **Reader effort.** Which verification is delegated to the reader
   (its answer already on the figure), and which step is kept? Spend
   nothing on decoding, deliberately on the punchline inference.
7. **Channel assignment.** Map the claim's structure onto the theme's
   channels: what is correspondence (hue), what is order (ramp), what is
   THE object (accent), what is scaffolding (construction ink). The
   quantity carrying the claim gets the top of the perceptual hierarchy
   (position > length > slope > area > shading).
7b. **Binding (composites only).** If the figure has parts, what is the
   same object across them, and what is the ONE thing that varies? Key
   the shared objects; declare `varies` and `changes`. If more than one
   thing differs, the reader cannot attribute either — cut until one
   does. The fixed set gets one page size.
8. **Honesty pass.** What does the depiction lie about (truncated
   infinities, selected seeds, unequal panel scales)? Admit each on the
   figure or fix it. Then audit the *accidental* assertions: what do
   layout coincidences, unencoded positions, and axis ranges claim?
9. **Gate plan.** Which numerical assertions certify the geometry, and
   what should the cold reader's glance read and studied read each say?

Steps 0–6 are figure design; 7 is theming; 8–9 are verification. The
skill = these prompts + exposition.md + grammar.md + the figure-program
template + the gate harness. Model-facing wording lives with the skill when distilled
(M5); this file is the architectural home.

## Judged results so far (why we believe the pipeline)

Fig [4] z^n grid: COMPARABLE to the book. Fig [9] Cassinians:
RECREATION BETTER. Fig [14] volcanoes: readback-clean, mechanism drawn
(the |z|=1 circle running into the poles). TikZ evaluated head-on
(pgfplots probe) and declined: annotation-poor in 3D, no gates, slow
loop; kept only as a possible future export target.
