# sci-figures — research-grade scientific figures as programs

**Date:** 2026-07-29
**Status:** draft for review
**Owner:** Nir

## Problem

AI-generated math/physics figures fail in two ways. Class A figures (phase portraits, spirals, level sets, trajectories) fail because the model draws curves that *look like* the mathematics instead of computing them — the curve carries no information and reads as a cartoon. Class B figures (free-body diagrams, block diagrams, commutative diagrams) fail on arbitrary geometry, label collisions, and crowding. Every existing tool (OpenTikZ, thesis-figure skills, TikZ template packs) is a Class B tool skewed to ML-architecture boxes.

Reading Needham's *Visual Complex Analysis* (Figures 5–14) revises the frame: the best figures are **A+B fused** — computed geometry wearing a schematic annotation layer that states the theorem on the figure (right-angle marks, labels that are mathematical objects: $iz$, $e^{i\theta}$, $2\cos\theta$). The architecture is therefore one pipeline, not two tools: computation is the geometry source, annotation is a first-class layer, and Class B is the degenerate case with an empty compute step.

The scarce resource is upstream of rendering: **figure design judgment** — deciding which geometry encodes the claim. The system must force that decision to be made explicitly, then verify the render against it.

## Thesis

The unit of work is a **figure program**, not an image:

```
CLAIM (one sentence) → params → compute() → Scene → layout → SVG/PNG → gates
```

Re-runnable, revisable, diffable. Geometry comes from numerics, never from drawn curves. Done is decided by three gates, only one of which involves a model.

## Substrate decision

**Custom SVG emitter** (user's call, 2026-07-29). We own every primitive and every bounding box, so the mechanical gate is exact and the visual grammar is not constrained by an existing library's annotation model. Math typography via **ziamath** (pure-Python LaTeX → SVG paths with real font metrics — exact glyph bboxes for collision detection). PNG rasterization via resvg (or cairosvg fallback) for the readback gate. numpy/scipy for the compute layer.

Rejected: matplotlib layer (annotation model would constrain the grammar; bboxes heuristic), Python→TikZ (two languages, slow compile loop, gates must re-parse compiled output).

## Components

Each has one purpose, a typed interface, and is testable alone.

- **`figlib/geometry.py`** — compute layer. ODE integration, series partial sums, vector fields, level sets, fixed points. Returns geometry in math coordinates as pure data. No drawing.
- **`figlib/scene.py`** — scene graph of typed primitives: `Curve`, `Vector`, `VectorChain`, `Point`, `MathLabel`, `AngleMark`, `RightAngleMark`, `Brace`, `Tick`, `ConstructionLine`, `PanelFrame`, `Panel`. All in math coords. Styled by **semantic role** (content / construction / annotation / frame), never raw stroke settings.
- **`figlib/style.py`** — the house style. Needham's ink hierarchy: solid = content, dashed = construction, dotted = frame; axes only when axes are the content. Modern twist: a small semantic palette (Distill-adjacent — muted structural ink, one or two saturated accents that carry meaning, e.g. term index or Re/Im), line-weight scale, arrowhead spec, type scale.
- **`figlib/typeset.py`** — LaTeX math labels → SVG path groups + exact bboxes (ziamath/ziafont).
- **`figlib/layout.py`** — math→canvas mapping, margins solved from real content extents, label anchoring and nudging with collision awareness.
- **`figlib/render.py`** — SVG emission; PNG rasterization for gates and preview.
- **`figlib/gates.py`** — the three gates (below), failing loudly with actionable diagnostics.

### Figure program contract

One Python file per figure:

```python
CLAIM = "The partial sums of e^{iθ} form a right-angle-turning vector chain that converges; the same series at real θ runs off along a ray."
PARAMS = {...}
def compute(p) -> Geometry: ...      # numerics only
def build(g) -> Scene: ...           # primitives + annotation
def assertions(g): ...               # numerical gate: residuals, invariants
```

Rendered SVG + PNG live beside the source. Figures are revised by editing the program.

## The three gates

1. **Numerical (deterministic).** Assertions on the computed content: ODE residual below tolerance along the plotted trajectory, $|f(x^*)|<\epsilon$ at a marked fixed point, opposite-sign eigenvalues at a marked saddle, series partial sums matching the closed form. Asserts on the *same arrays that were plotted*.
2. **Mechanical (deterministic).** Label bbox collisions, clipping against the canvas, ink density, axis coverage, contrast. Exact, because the emitter owns every bbox. Free — no model.
3. **Readback (model).** Rasterize; a fresh context-free agent sees only the PNG — no code, no note, no reasoning — and answers "what does this figure claim?" Pass iff the readback matches `CLAIM`. This is the cold-reader test; it catches "caricature," which the other gates cannot.

Gates do not silently auto-fix. Layout may nudge labels; if it can't solve, the author sees the diagnostic — an unsolvable layout is a design problem.

## Visual grammar (seed, grown empirically)

Extracted from Needham Figs 5–14; extended as exemplars demand:

- Labels are mathematical objects, never captions; quantities are measured on the figure itself.
- Ink hierarchy: solid content / dashed construction / dotted frame.
- Right-angle marks and equal-length ticks encode claims.
- One claim per figure; `[a]/[b]` panels for stages of an argument.
- Axes absent unless they are the content.
- Color is semantic or absent.

This grammar spec lives in `docs/grammar.md` and becomes the prompt core of the eventual skill.

## Method: empirical first, skill last

The figure pulls the library into existence — primitives are built when a benchmark figure demands them, not speculatively.

1. **M1 — emitter skeleton.** Paths, arrowheads, math labels, one panel. Render anything correct.
2. **M2 — benchmark: recreate Needham Figure [9]** (partial-sum spiral of $e^{i\theta}$ vs the $e^\theta$ ray). Ground truth on paper; fully computable; exercises the annotation grammar hard. Iterate until the readback gate passes and side-by-side with the book holds up, restyled modern.
3. **M3 — gates hardened** into a `figcheck` harness runnable on any figure program.
4. **M4 — fresh-topic test:** diffusion models — probability-flow ODE vs SDE trajectories sharing marginals. No ground-truth image; tests whether the *design step* (claim → geometry) generalizes.
5. **M5 — distill into a skill:** grammar spec + house style + figure-program template + gate harness + the design-step prompt (claim → geometry that encodes it → what to compute → what to annotate).

Deferred deliberately: corpus scraping (ten hand-understood exemplars beat a thousand unattributed SVGs; scraping is later template-library enrichment), TikZ export, Class-B layout solver (constraint-based placement) until a Class-B benchmark figure forces it.

## Repo layout

```
figlib/           # the library
figures/          # figure programs + rendered SVG/PNG side by side
exemplars/        # Needham recreations + notes on the grammar move each one taught
docs/             # this spec, grammar.md
tests/            # geometry vs analytic answers, SVG snapshots, gate tests on broken figures
```

## Testing

- Geometry against known analytic answers (e.g., partial sums vs closed form).
- Scene→SVG snapshot tests.
- Gate tests with deliberately broken figures (colliding labels, wrong fixed point, clipped content).
- The benchmark figures are the integration tests.

## Success criteria

A figure passes when (a) numerical assertions hold, (b) mechanical checks are clean, and (c) a cold reader states the intended claim from the image alone. The project passes when the Fig [9] recreation reaches that bar and the diffusion figure reaches it without a ground-truth image to imitate.
