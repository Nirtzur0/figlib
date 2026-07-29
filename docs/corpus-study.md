# Corpus study: mechanisms from the exemplar sources

What we studied and what to steal — not templates, mechanisms. Sources:
transformer-circuits.pub (pixel-level read of the 2025 QK article + full
taxonomy sweep), Needham VCA beyond our recreated figures (inversion /
Riemann-sphere chapters), and a survey of the other canonical figure
corpora (VDGF, Tufte, Byrne, Strogatz, MacKay, Feynman, distill.pub,
3Blue1Brown, Bret Victor, Bostock, Nelsen, Bishop PRML).

The test for inclusion here: a mechanism earns a place if it handles a
CLAIM-SHAPE our current grammar cannot, or if it names a primitive we
should build. Grain of salt applied — these corpora skew toward their
own materials; we generalize the way of thinking, not the diagram type.

## Directly observed at transformer-circuits (pixels, not captions)

1. **The annotated derivation.** An equation IS the figure: each term of
   a long expansion carries a prose gloss anchored beneath it
   ("Feature-Error Interactions — these terms correspond to…"). The
   claim-shape is "this decomposition is exhaustive and each term has a
   meaning." Typography does the layout; there is no drawing at all.
2. **Evidence panels with typed relations.** Raw data (token strips
   with per-token activation highlighting — an inline heatmap over
   text) embedded as first-class figure content; panels joined by a
   typed edge vocabulary: pointed double arrow = excitation, flat-capped
   bar = inhibition (gene-regulation notation); a margin column of
   claim-first interpretation ("Whose opposite am I taking?"). Lesson:
   show the evidence, type the relations, interpret in the margin.
3. **Rails-and-paths schematics.** Dashed vertical rails (residual
   stream per token), faint curved ensemble edges (all attention paths),
   ONE accent-colored example path, reusable mini-glyphs (attention-head
   barcode), a legend. Our "distinguished object against the ensemble"
   rule applied to computation graphs.
4. Their color discipline confirms ours: one warm accent family where
   intensity = activation magnitude (lightness = order), two reserved
   hues for the two edge types (hue = meaning), everything else gray.

## The transformer-circuits taxonomy (full sweep, 2021–2025)

Eighteen recurring mechanisms across Framework, Induction Heads, Toy
Models, Monosemanticity, Scaling, Biology, attention-qk. The five
load-bearing ones:

1. **Colored-token text** — ground truth for "this unit means X": real
   dataset excerpts, per-token saturation ∝ activation, examples
   sampled across activation *deciles* (the honest move — not just top
   activations).
2. **The feature card** — a standardized evidentiary atom (activating
   examples + activation histogram + logit weights + ablations) reused
   at every scale, cited inline by ID. Amortizes reader learning;
   industrializes thousands-of-features surveys.
3. **Intervention before/after panels** — how correlational claims get
   upgraded to mechanism claims: baseline vs steered transcripts,
   dose–response strips across steering scale, captions stating the
   *predicted* change (each panel a registered test).
4. **Attribution graph + curated supernode schematic** — the two-tier
   split between raw evidence (complete, interactive, overwhelming) and
   claim (hand-curated static schematic), with every schematic linking
   to the raw graph it summarizes. Token sequence stays the x-axis so
   the graph reads like the prompt.
5. **Annotated training/scaling curves** — metric curve paired with a
   mechanism indicator on a shared x-axis so co-occurrence is visible;
   the exception model visually isolated. Their standard template for
   "ability = mechanism."

Also in the vocabulary: interaction-matrix heatmaps swept as small
multiples (a heatmap becomes a phase narrative), phase diagrams with
analytic boundaries over empirical grids, feature-geometry stick plots
whose quantitative plateaus are annotated with tiny polytope icons,
per-head scatters where opacity manages a 100+ point population,
operand×operand lookup heatmaps where the *pattern class is the
interpretation* (diagonals = sum-features), cross-run correlation
scatters whose named outliers index into case studies, and
deliberately informal toy schematics where symbol placement replaces a
legend.

Cross-cutting signatures worth adopting outright: (a) color binds
semantics across figure, equation, AND prose — a concept keeps its hue
everywhere; (b) scale is handled by opacity and decile-sampling, never
silent truncation; (c) captions state falsifiable predictions, not
descriptions.

## From Needham beyond Figs 4–14

- **Proof by transport.** Map the hard configuration into a domain
  where the claim is trivial, show both panels joined by the mapping
  arrow (orthogonal-diagonal quadrilateral → rectangle under inversion;
  Ptolemy via inversion). The figure argues by exhibiting the
  simplifying map, not by decorating the hard side.
- **Cross-section of a symmetric 3D scene.** Fig [21]: draw the 2D
  slice through the axis of symmetry, state that rotation generates the
  full picture. Honest dimension reduction — cheaper and often clearer
  than full 3D.
- **Marked angles at corresponding points** to assert conformality:
  the same angle drawn at p and at its image, equality visible.

## The six mechanisms to add to grammar.md

Our current six rules are all about rendering ONE figure honestly. The
gaps are families, time, procedures, and proof:

1. **Probe-object semantics** (Needham, 3b1b). Show an operator by its
   action on one standardized familiar object (grid, disk, basis pair,
   tangent vector), pre-image ghosted at low alpha in the same frame.
   Ghosting is the static encoding of before→after.
2. **Parameter-swept small multiples, with the continuum limit**
   (Tufte, Strogatz, Bishop, Victor). A family = one frame repeated
   along a swept parameter, everything else pixel-identical; when the
   sweep is dense, collapse into a bifurcation-style diagram. Comparison
   and abstraction are the same operation at two densities.
3. **State-space compilation** (Strogatz, Victor). Compile
   behavior-over-time into geometry: phase portraits with filled=stable
   / open=unstable fixed points, flow arrows, solid/dashed branch
   semantics; parameter planes colored by outcome. Honesty clause: mark
   which features are topological (guaranteed) vs metric (schematic).
4. **Worked-instance trace** (MacKay, Bostock). For any algorithm, run
   it on one concrete instance and show intermediate states, with the
   driving data structure visually foregrounded. Schematics assert;
   traces demonstrate.
5. **Color as referential noun** (Byrne, 3b1b, Carter). Promote
   hue=correspondence to a binding table: one hue per named object,
   declared once, enforced across every panel, equation, and inline
   reference. Scarcity (≤4–5 bound hues) is what keeps it referential.
6. **Declared exaggeration** (Feynman, Needham's ultimate equality).
   When an infinitesimal or small angle is drawn finite for legibility,
   annotate the distortion and its vanishing order. Sibling of honest
   truncation, for deliberate geometric lies.

Runner-up: **word-scale insets** (sparklines, MacKay pictograms,
Victor's pinned thumbnails) — tiny concrete graphics embedded at the
exact point in a larger diagram where the object lives. This is a
composition primitive several benchmarks need.

## Primitives these imply (build order, each against a benchmark)

| primitive | what it is | first benchmark that forces it |
|---|---|---|
| ghost pass | render any scene subset at low alpha as pre-image | 3b1b determinant frame |
| small-multiples layout | N pixel-identical panels + shared/swept annotation | PRML Fig 1.4 (M = 0,1,3,9) |
| plots.py (axes/ticks/series/band) | already designed; data claims need it | Feynman two-slit P₁,P₂,P₁₂ |
| typed-edge schematic layer | nodes/ports/edges with arrow-type semantics (excite/inhibit/map-to) | TC rails-and-paths |
| annotated derivation | ziamath equation + term-anchored gloss blocks | TC QK score expansion |
| heatmap + colorbar | matrix/field as color grid, theme.ramp | GP kernel matrix |
| inset/pin | a small scene embedded at a point of a larger one | Victor behavior map |

## Benchmark shortlist (stress-test our bar, in order)

1. **Strogatz saddle-node composite** (ẋ = r + x²): three phase-line
   portraits above the bifurcation diagram, solid/dashed branches,
   panels linked by shared conventions. Tests: small multiples,
   state-space compilation, plots.py. Class A throughout — everything
   computed.
2. **3b1b determinant frame**: unit square → parallelogram, ghost grid,
   color-bound matrix columns ↔ transformed basis vectors, area
   annotated. Tests: ghost pass, color-as-noun binding.
3. **TC-style rails-and-paths schematic** (e.g. induction-head circuit):
   our first true Class B figure. Tests: typed edges, glyph reuse,
   distinguished path.
4. **Feynman two-slit composite**: apparatus schematic + three linked
   probability curves including the counterfactual classical sum.
   Tests: schematic+plot composition, wrong-hypothesis curve.
5. **VDGF spherical-triangle holonomy**: parallel transport around a
   geodesic triangle, vector drawn at each stage, rotation = angular
   excess. Tests: surface3d + draped vectors + angle-between-copies
   annotation. Hardest; do last.

Also strong: MacKay noisy-channel decode (raster-in-vector, traces),
Byrne I.47 (proof filmstrip, inline glyphs), AM-GM proof without words
(inequality carried entirely by geometry), Minard 1812 (multivariate
data on one mark).
