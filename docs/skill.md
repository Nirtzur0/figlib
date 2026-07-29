# Writing a figure (the router)

The model-facing entry point. This file is deliberately short: it tells
you the contract, the loop, and where to look — it does not repeat the
theory. Read on demand:

- **architecture.md** — the stack and the design step (0–9). Read the
  design step *before* writing any code, every time.
- **grammar.md** — the visual rules with their justifying failures. Read
  during design; consult when a judge or readback flags a defect.
- **exposition.md** — why the design step is what it is. Background;
  read when a step feels arbitrary.
- **primitive-gaps.md** — maintainer-facing build order. Not needed to
  write a figure.

## The contract

A figure program is one module in `figures/`:

```python
CLAIM = "one sentence: what the figure ARGUES, not what it draws"
THEME = RISO            # optional; CLEAN | RISO (theme.py)
FORMAT = COLUMN         # MARGIN 340 | COLUMN 680 | WIDE 1000 px slot
PARAMS = {...}          # every tunable, no magic numbers in compute()

def compute(p): ...     # numerics -> arrays; NO drawing decisions
def build(g): ...       # arrays -> Scene, or Figure for multi-panel
def assertions(g): ...  # numerical gate on the SAME arrays that got drawn
```

Canvas px = display CSS px: the figure renders at the size it is read.
If annotation doesn't fit, take a larger Format or cut ink — never
shrink type.

## The loop

```
figcheck figures/<name>.py            # render + all deterministic gates
figcheck figures/<name>.py --report   # textual layout inventory
figcheck --regress                    # corpus-wide golden diff (exit 1 on drift)
figcheck --update [figures/<name>.py] # refresh the committed SVG+PNG baselines
```

`--regress` re-renders every figure and diffs the SVG text against
`figures/out/`; on mismatch it rasterizes both and prints the changed-pixel
fraction and RMS, so cosmetic drift (0.1% px) reads differently from a
redrawn figure. Rendering is byte-deterministic, so any diff is real. If
you change the renderer, run it, look at what moved, then `--update`.

1. Design first (architecture.md steps 0–9), then write the program.
2. Run figcheck. **Diagnostics contain the fix**: collisions print
   verified `offset_px` nudges, clipping prints the overrun per edge,
   faint-ink names the failing hex and floor. Apply them; don't guess.
3. Debug layout with `--report` (label bboxes, geometry extents,
   margins, nearest-neighbor gaps) — text before pixels. Look at the
   PNG only when the gates pass and you need judgment, not coordinates.
4. **Readback gate — mandatory, not optional.** Spawn a context-free
   agent (no CLAIM, no conversation) on the PNG with the prompt from
   `figcheck --readback-prompt`; write the result with
   `readback.record()`. Every confusion bullet is design review: fix it
   or explicitly accept it in the record. A figure without a readback
   record is not done.
5. Comparative gate when a reference exists: judge sees original +
   recreation, rules BOOK BETTER / COMPARABLE / RECREATION BETTER with
   named defects. Iterate to at least COMPARABLE.

Environment: cairo needs `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`
(the Makefile encodes this).

## Conventions that bite

- **Coordinates.** Scene anchors are math coords, +y UP. `offset_px` is
  canvas px, +y DOWN. Getting the sign wrong is the most common layout
  bug; the collision diagnostics speak offset_px, so trust them.
- **Color is a theme channel or absent.** `color=` on any item accepts
  ONLY theme channel output — `theme.ramp(t)` (ordered quantity),
  `theme.categorical(i, n)` (correspondence), `theme.surface_shade(t)`
  (3D lighting) — never a hex literal. Roles (`Role.ACCENT1`,
  `Role.CONSTRUCTION`, ...) cover everything else. A hardcoded hex means
  the figure silently stops retheming, and the color gate will not save
  you from that; it only catches invisibility and hue collapse.
- **The accent is THE object.** The distinguished curve gets an accent
  role and no color override; the ramp flows around it.
- **Assert what could be wrong, never what is true by construction.**
  Good: an independent identity on the plotted arrays, a
  finer-integration cross-check, "the drawn subset satisfies the
  defining equation". Theater: restating a mathematical fact the
  computation didn't produce (`assert abs(abs(1j) - 1.0) < 1e-15`).
  For several independent checks use `gates.Checks` so one run reports
  every failure, not just the first.
- **Dash carries meaning**: solid = content, dashed = construction /
  hidden, dotted = frame. `dash=` can carry identity across panels.
  Open dot = excluded, filled = attained — no decorative exceptions.

## Device -> exemplar index

Imitate the nearest exemplar; mutate, don't invent structure from
scratch. Each of these passed the full gate stack.

| device | exemplar |
|---|---|
| level-set family + defining-property annotation | `vca_fig9_cassinian.py` |
| conformal grid / mesh under a map | `vca_fig4_zn_polar_grid.py` |
| 3D surface, draped curves, world-anchored labels | `vca_fig14_volcanoes.py` |
| panel pair + map connector (Needham page grammar) | `demo_panels_zsquared.py` |
| sphere, hidden-line arcs | `demo_sphere_stereographic.py` |
| streamlines / flow field | `demo_flow_past_cylinder.py` |
| dual-family conformal grid (solid + subordinate dashed) | `vca_fig12_flow_grid.py` |
| checkerboard motion portrait (bipolar net, orbit cells) | `vca_fig30_elliptic_checkerboard.py` |
| brace / callout / pattern fill / raster field | `demo_glyphs_annulus.py` |
| stochastic ensemble, honest seeds | `diffusion_ode_vs_sde.py` |
| series / partial-sum geometry | `fig09_exp_series_spiral.py` |
| rails + typed paths (circuits idiom) | `induction_head_circuit.py` |
| ranked node/edge schematic, main path as a spine | `schematic_transformer_block.py` |

## 3D idioms (surface3d)

- Everything 3D becomes depth-tagged 2D items; `compose(*groups)` merges
  far-to-near. New capability is always a *producer of scene items*,
  never a new renderer.
- `label3(latex, xyz, cam, ...)` / `vector3(tail3, tip3, cam, ...)` —
  world-space anchors; never hand-project and relabel.
- `as_floor(group)` forces a ground plane behind everything — no magic
  depth constants.
- `depth_bias` lifts a curve lying ON the surface above its own quads
  (0.04–0.06 is the working range).
- `wireframe_items(..., hidden="dashed")` for mesh-style surfaces with
  hidden-line removal; `surface_items(shade=THEME.surface_shade)` for
  Lambertian facets.

## Multi-panel (figure.py)

`build()` returns `Figure(panels=[Panel(scene, tag="[a]"), ...],
connectors=[Connector(0, 1, kind="map", label=r"z \mapsto z^2")])`.
Layout is dumb affine slots — the mechanical gate catches what collides,
figure-wide. Correspondence hues must hold ACROSS panels; the color
gate pools them.

## Distilled traps (mined from ~800k tokens of builder transcripts)

Each line below cost a builder agent a full render-inspect-fix loop.
Read once; they apply to nearly every figure.

- **Count the reference first.** For recreations: count per-family
  curves, markers-per-curve, and labels in the original BEFORE
  compute(); set densities from the counts and record them in PARAMS.
  The agent that counted needed half the iterations of the one that
  tuned by eye.
- **Two families, one claim.** Primary family: solid CONTENT, sparse
  hollow markers (~2/curve). Secondary family: SAME role (it is
  content and must pool in the contrast ensemble — CONSTRUCTION is
  wrong), `width_scale=0.5`, `dash="fine-dashed"`, no markers.
- **Level ladders are computed, not chosen.** Use `auto_levels()` for
  spacing/count; your judgment is only the target density, the
  distinguished level (exclude it from the ladder; restyle it as THE
  object), and the on-figure admission when equal spacing carries a
  claim (`ΔΨ = ΔΦ`).
- **Never verify orientation by zooming.** `assert_tangents_align()`
  in assertions() replaces the zoom-and-stare cycle.
- **3D: place by computation, verify by assertion.** Every free point
  and label goes through `anchor()`/`label3()` in compute(), with
  asserted screen-space facts (outside the limb, min separation).
  Never adjust world coordinates against the PNG.
- **Planes in 3D scenes are line-art** (rims only). A translucent
  sheet fill cannot clear the fill-contrast floor on RISO and hides
  what it cuts; an opaque fill must be intended occlusion.
- **Far-side landmarks are drawn, muted.** A structurally necessary
  point behind a closed surface: muted dot, dashed connecting
  construction, ANNOTATION label. Literal occlusion loses the
  mechanism.
- **The empty border band is the margin, not a bug.** The transform
  pads xlim/ylim symmetrically (~60 px at WIDE). `--report` prints
  the real margins; don't investigate blank strips.
- **The PNG is the figure of record.** cairosvg silently ignores some
  SVG hints (`image-rendering: pixelated`, others). Never rely on a
  rendering hint; bake the effect into geometry/pixels.
- **Conventional glyphs use their canonical construction.** An
  invented brace read as an S-wave; the TeX construction read as a
  brace. Verify any new glyph with a zoomed crop (`--zoom`), then pin
  its shape in a test (path command counts).
