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
render                SVG (+ PNG) — owns every bbox
  ▼
gates                 numerical / mechanical / readback / comparative
```

**The invariant that keeps everything consistent:** content code never
names a color, font, or stroke width. It names *meanings* — `Role.CONTENT`,
`Role.ACCENT2`, `theme.ramp(t)`, `theme.categorical(i, n)`,
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
| `categorical(i, n)` | correspondence hue | same ray before/after a map |
| `surface_shade(t)` | 3D facet lighting → color | the volcano ramp |
| `paper`, `grain` | ground and texture | riso cream + speckle |

Rules: hue = correspondence, lightness = order, accent = the
distinguished object, never decoration (see grammar.md). A theme changes
*which* colors carry those channels, never *what* the channels mean.

## The three figure classes, one pipeline

- **Computed geometry (Class A)** — compute() produces the curves;
  everything so far. 3D is not a separate class: `surface3d` projects,
  depth-sorts, and shades into ordinary 2D items, so gates apply
  unchanged.
- **Data plots** — a thin `plots.py` layer (axes, ticks, series) built
  ON the scene primitives, not beside them. An axis is Curves + Ticks +
  MathLabels; a series is a Curve with a categorical hue. No second
  rendering path, so plots inherit theme + gates for free. (Not built
  yet; build when the first real plot figure demands it.)
- **Schematics (Class B)** — nodes, ports, edges with a layout pass;
  compute() is empty and layout is the content. Same scene, same theme,
  same mechanical gate (label collisions matter most here). (Design
  decided, build against the first Class B benchmark.)

The rule that keeps the architecture clean: **new capability enters as a
producer of scene items, never as a new renderer.** surface3d proved the
pattern.

## The thinking layer (what actually decides quality)

Everything above is plumbing. Quality is decided before code, by the
design step — the prompts that will form the skill's core. In order:

1. **CLAIM.** One sentence, the claim the figure argues. If you cannot
   write it, there is no figure yet.
2. **Encoding choice.** Which geometry makes the claim *visible*?
   (Trajectories? level sets? a mesh deforming? a surface? an endpoint
   ensemble?) Ask: what would Needham draw; what does the claim's HARD
   half need (grammar.md).
3. **Mechanism annotation.** Which quantities must a cold reader be able
   to read OFF the figure to reconstruct the argument? Draw them; a
   caption stating what the drawing could show is a defect.
4. **Channel assignment.** Map the claim's structure onto the theme's
   channels: what is correspondence (hue), what is order (ramp), what is
   THE object (accent), what is scaffolding (construction ink).
5. **Honesty pass.** What does the depiction lie about (truncated
   infinities, selected seeds, unequal panel scales)? Admit each on the
   figure or fix it.
6. **Gate plan.** Which numerical assertions certify the geometry, and
   what should the cold reader say?

Steps 1–3 are figure design; 4 is theming; 5–6 are verification. The
skill = these prompts + grammar.md + the figure-program template + the
gate harness. Model-facing wording lives with the skill when distilled
(M5); this file is the architectural home.

## Judged results so far (why we believe the pipeline)

Fig [4] z^n grid: COMPARABLE to the book. Fig [9] Cassinians:
RECREATION BETTER. Fig [14] volcanoes: readback-clean, mechanism drawn
(the |z|=1 circle running into the poles). TikZ evaluated head-on
(pgfplots probe) and declined: annotation-poor in 3D, no gates, slow
loop; kept only as a possible future export target.
