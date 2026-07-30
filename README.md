<p align="center">
  <img src="docs/brand/wordmark.png" alt="figlib — figures as gated programs" width="420">
</p>

<p align="center"><b>An agent skill for scientific figures — and the library it draws with.</b><br>
<sub>Reason about what to draw, build it as a program, prove it says what it claims.</sub></p>

<p align="center">
  <a href="#install-the-skill">Install</a> ·
  <a href="#what-the-library-draws">What it draws</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#what-gets-decided-before-any-code">How it decides</a> ·
  <a href="#what-the-compiler-checks">What it checks</a> ·
  <a href="#gallery">Gallery</a> ·
  <a href="docs/skill.md">Docs</a>
</p>

<p align="center">
  <a href="figures/out/GALLERY.md"><img src="figures/out/complex/vca_fig14_volcanoes.png" alt="the modular surface of 1/(1+z^2)" width="840"></a>
</p>

<p align="center"><sub><b>Why a real Taylor series stops converging at a radius nothing on the real line explains.</b><br>
The graph of 1/(1+x²) is the tranquil real slice of a surface that erupts at z = ±i. The radius is the distance to the poles you cannot see.</sub></p>

---

Ask a coding agent for a figure today and you get a matplotlib script. It
will run, it will produce a picture, and nothing in the loop can tell you
whether the arrowhead points the right way, whether the type is legible at
the size it will be read, or whether the picture argues the thing you meant.
The agent can't tell either — it never sees the render.

figlib is the missing loop. It is a **skill an agent loads** plus **the
library that skill writes against**, and together they make a figure something
a model can reason about, build, and verify:

- **Reason.** A ten-step design procedure runs *before* any code — earn the
  figure, name the claim, pick the representation and the abstraction rung,
  script the read, assign the perceptual channels, plan the gates. It is
  written as prompts, so it costs a model almost nothing to run every time.
- **Build.** A figure is a program: `compute → build → autoplace → gates →
  render`. Curves and fields, 3D surfaces, multi-panel pages, schematics,
  matrices, tensor diagrams, derivations — all in one scene model, so every
  gate and every theme applies to all of it.
- **Verify.** The program declares assertions about the arrays it plotted.
  Layout, colour and contrast are checked mechanically, and each failure comes
  back as a *computed fix* — `offset_px += (+0, -13)`, not "labels overlap".
  Then a second agent, given only the PNG and no context, says what the figure
  claims. If that misses the claim, the figure is wrong however pretty it is.

That last step is the point. Nothing else in an agent's toolchain closes the
loop between "I drew it" and "it says what I meant."

## Install the skill

Python 3.12+, [uv](https://docs.astral.sh/uv/), and cairo.

```bash
brew install cairo                                     # or your platform's
git clone https://github.com/Nirtzur0/figlib && cd figlib
uv sync && make test

uv tool install --editable .                           # figcheck on PATH

SKILL=~/.claude/skills/scientific-figures                # or ~/.codex/skills/…
mkdir -p "$SKILL/references"
ln -s "$PWD/docs/skill.md" "$SKILL/SKILL.md"
ln -s "$PWD/docs/"{architecture,grammar,exposition,corpus-study}.md \
      "$SKILL/references/"
```

Symlinks, not copies: the repo stays the single source, so a change to the
design step reaches every project on the machine the moment it is committed.

The host project takes **no dependency on figlib** — `figcheck` loads a figure
program into its own environment and writes the artifacts next to it. So the
skill works in a paper repo, a blog, a lecture course, a notebook project:
ask for a figure, and one lands in `figures/`, gated, with its baselines and
its readback record.

<sub>On macOS, cairo needs `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`;
wrap `~/.local/bin/figcheck` in two lines that export it, or the render dies
with an opaque error. The Makefile does this for you inside the repo.</sub>

## What the library draws

One scene model underneath all of it. **New capability enters as a producer of
scene items, never as a new renderer** — which is why a gate written once
applies to a phase portrait, a transformer schematic and an einsum diagram
alike, and why one theme edit restyles every figure ever written.

| | what it gives you | module |
|---|---|---|
| **Plane geometry** | curves, filled regions with grain and gradients, points, vectors, braces, callouts, raster fields | `scene` |
| **Fields and flows** | streamlines from a stream function, level ladders, stagnation points, phase portraits | `builders`, `geometry` |
| **Data plots** | axes with linear/log scales, series, bands, colorbars — as scene items, so the gates still apply | `plots` |
| **3D surfaces** | orthographic projection to depth-sorted 2D, chromatic light→shadow ramps in OKLCh, labels anchored in 3-space | `surface3d`, `sphere3d`, `shading` |
| **Multi-panel pages** | Panel/Connector grammar, plus a *declared* correspondence whose residual is measured — if two things differ between panels, the reader can't attribute either | `figure`, `correspond` |
| **Schematics** | typed Node/Port/Edge with a ranked layout pass — transformer blocks, circuits, pipelines | `schematic` |
| **Matrices** | blocks drawn at their own aspect ratio, structure and value encoders, expression rows, shape gates | `matrix` |
| **Tensor networks** | einsum as a Penrose diagram — and the spec is *derived from the drawing*, then gated against `np.einsum` | `tensor` |
| **Derivations** | annotated rows: terms at real typeset widths, operators midway, sans glosses hung beneath | `derivation` |
| **Word-scale insets** | a whole scene affinely embedded in another scene's coordinates | `inset` |

Type is real math: LaTeX through ziamath, with exact metrics, so the collision
gate measures the glyphs the renderer draws rather than an estimate of them.

## Quickstart

A figure is one module — the contract an agent writes to. This is
[`docs/examples/first_figure.py`](docs/examples/first_figure.py), trimmed only
of its prose:

```python
CLAIM = ("The tangent to y = x^2 at x = a meets the axis at a/2 — the "
         "subtangent is half the abscissa, for every a.")

EXPOSITION = """..."""             # the passage this figure is FOR, ≥40 words
                                   # written BEFORE the code, and gated

FORMAT = COLUMN                    # MARGIN 340 | COLUMN 680 | WIDE 1000 px
THEME  = RISO
PARAMS = {"a": 1.5, "xlim": (-0.9, 2.2), ...}     # no magic numbers below

def compute(p):                    # numerics -> arrays. No drawing decisions.
    a, lo, hi = p["a"], *p["reach"]
    x = np.linspace(*p["curve_x"], 400)
    t = np.array([a - lo, a + hi])
    return {"parabola": np.column_stack([x, x * x]),
            "tangent":  np.column_stack([t, 2.0 * a * (t - a) + a * a]),
            "foot": (a, a * a), "p": p}

def build(g):                      # arrays -> Scene. Roles, never colours.
    s = Scene(xlim=..., ylim=..., height_px=400)
    s.add(Curve(g["parabola"], role=Role.CONTENT))
    s.add(Curve(g["tangent"],  role=Role.ACCENT1))
    s.add(Point(g["foot"], filled=True, role=Role.ACCENT1))
    s.add(MathLabel(r"(a,\,a^2)", g["foot"], ha="right", va="bottom"))
    s.add(Brace((a / 2, -0.6), (a, -0.6), side=-1.0,   # the claim, as a LENGTH
                label=r"\text{subtangent} = \tfrac{a}{2}"))
    return s

def assertions(g):                 # the gate, on the arrays that got drawn
    (x0, y0), (x1, y1) = g["tangent"]
    root = x0 - y0 * (x1 - x0) / (y1 - y0)     # where the DRAWN line hits y=0
    assert abs(root - g["p"]["a"] / 2.0) < 1e-12
```

```
$ figcheck docs/examples/first_figure.py

[PASS] first_figure: The tangent to y = x^2 at x = a meets the axis at a/2 …
  svg: docs/examples/out/first_figure.svg
  png: docs/examples/out/first_figure.png
  placed: '\tfrac{a}{2}' offset_px += (+3, +0)
  clearance: tightest labels 'y = x^2' 2.2px, '\tfrac{a}{2}' 6.3px
```

<p align="center">
  <img src="docs/examples/out/first_figure.png" alt="tangent to a parabola meeting the axis at a/2" width="680">
</p>

Note what the assertion does *not* do: it never recomputes `a/2` from theory
and compares it to itself. It reads the endpoints of the line that was
actually drawn, solves for where that segment crosses the axis, and checks
*that*. Every assertion in the corpus is written this way — against the
plotted arrays, so a build error and a math error both surface as a failure.

## What gets decided before any code

The design step, in full, is in
[`docs/architecture.md`](docs/architecture.md#the-thinking-layer-what-actually-decides-quality);
the reasoning behind each step is in [`docs/exposition.md`](docs/exposition.md).
The shape of it:

| step | the question it forces |
|---|---|
| **0 · earn it** | write the prose that makes the figure unnecessary. If the prose works, no figure. |
| **1 · claim** | one sentence the figure *argues*. Can't write it? There is no figure yet. |
| **2 · representation** | which primitive makes the claim checkable by eye — and at which rung: instance, trajectory, family, behavior map? |
| **3 · slot** | MARGIN, COLUMN or WIDE. Chosen *with* the annotation budget, never after. |
| **4 · traversal** | where the eye enters, what path adjacency forces, what a 3-second glance yields versus a 30-second study. |
| **5 · mechanism** | which quantities a cold reader must read *off* the figure to reconstruct the argument. |
| **6 · reader effort** | which verification is delegated to the reader, and which single inference is kept for them. |
| **7 · channels** | claim structure onto theme channels: hue = correspondence, lightness = order, accent = *the* object. |
| **8 · honesty** | what the depiction lies about — truncated infinities, selected seeds, unequal panel scales — plus the accidental assertions layout makes for free. |
| **9 · gate plan** | which numerical assertions certify the geometry, and what the cold readback should say. |

This is the half no plotting library has, and the half that decides quality.
Steps 0 and 1 routinely conclude the figure should not exist; step 2 is where
a good figure and a competent one diverge; step 8 is the one nobody does by
hand. A person runs this once, on the figure they care about. An agent runs it
on every figure, because as prompts it costs almost nothing — which is the
whole argument for putting the procedure in the skill rather than in a style
guide someone might read.

## What the compiler checks

| gate | what it holds |
|---|---|
| **numerical** | the program's own assertions, on the arrays that were drawn |
| **mechanical** | label collisions, clipping, type under 8.5 pt, annotation over 22% of the canvas |
| **color** | correspondence hues pairwise separable · an order ramp monotone in lightness · ink clearing a contrast floor on the actual ground |
| **golden regression** | rendering is byte-deterministic, so any diff against the committed baseline is a real change |
| **readback** | a cold agent sees only the PNG and says what it claims; if that misses `CLAIM`, the figure is wrong however pretty it is |

Every mechanical diagnostic carries a *computed* fix, not a complaint — the
difference between a message a human interprets and one an agent can apply:

```
[FAIL] poincare_disc: …
  label-collision: "L" overlaps "P" by 11px — offset_px += (+0, -13)
  clipping: geodesic overruns the frame by 4.2px on the right
```

When a fix becomes fully computable it stops being a gate and moves into the
pipeline. Collision nudges did: `autoplace` now applies the solved offset
before the gate measures. What's left in the mechanical gate is what a solver
shouldn't silently fix — a pinned label whose position *is* meaning, a move
past the 24 px budget, which is a design defect wearing a layout defect's
clothes.

The last gate is the one that can't be computed. `figcheck --readback-prompt`
emits a prompt for a **second agent that has never seen the claim, the code or
the conversation** — only the PNG. It says what the figure asserts, what it
can read off, and what confused it. Every confusion bullet is design review:
fix it, or write down why you accepted it. The record lands in
`<name>.readback.md` beside the baselines, and a figure without one is not
done.

That is the closed loop. Design → program → computed repairs → an independent
reader. An agent can run all four without a human in the middle deciding
whether the picture looks right.

## Semantics, not appearance

`Role.CONTENT`, `theme.ramp(t)`, `theme.categorical(i)`, `theme.surface_shade(t)`.
No figure in this repo contains a hex literal, a font name, or a stroke width.

That is not tidiness. It is what makes the color gate possible: because a
color arrives carrying its *channel* — ordered, categorical, cyclic, depth —
the gate can hold each to its own standard. A ramp must be monotone in
lightness; a categorical set must be pairwise separable; neither test means
anything applied to an anonymous `#4c72b0`. It also means one theme edit
restyles the corpus, and a figure that quietly stops retheming is a bug the
corpus can find.

A figure is a printed page. The default ground is the RISO theme's white
stock with its grain — and grain is **ink, not paper**, so it rides the
groundless render too. Every figure commits both: `<name>.svg` on the stock,
and `<name>_transparent.svg` on alpha for a document that owns its own
background. `figcheck` checks both unconditionally, because a contrast gate
only earns its keep on a ground that is actually there.

## Gallery

Every figure below came out of that loop — designed against the ten steps,
declared as a program, gated, and read back cold by an agent that saw only the
picture. The readback records are committed next to the baselines, confusions
and all.

<!-- gallery:start -->

**36 figures.** Every thumbnail links to its claim, the passage it serves, and both renders — or browse [the full gallery](figures/out/GALLERY.md).

### complex

*Conformal maps, the Riemann sphere, contour integration.*

<p><a href="figures/out/GALLERY.md#amplitwist"><img src="figures/out/complex/amplitwist.png" alt="amplitwist" title="amplitwist" height="227"></a> <a href="figures/out/GALLERY.md#contour_deformation"><img src="figures/out/complex/contour_deformation.png" alt="contour_deformation" title="contour_deformation" height="227"></a> <a href="figures/out/GALLERY.md#demo_flow_past_cylinder"><img src="figures/out/complex/demo_flow_past_cylinder.png" alt="demo_flow_past_cylinder" title="demo_flow_past_cylinder" height="227"></a></p>

<p align="center"><a href="figures/out/GALLERY.md#demo_panels_zsquared"><img src="figures/out/complex/demo_panels_zsquared.png" alt="demo_panels_zsquared" title="demo_panels_zsquared" height="230"></a></p>

<p align="center"><a href="figures/out/GALLERY.md#demo_sphere_stereographic"><img src="figures/out/complex/demo_sphere_stereographic.png" alt="demo_sphere_stereographic" title="demo_sphere_stereographic" height="230"></a></p>

<p align="center"><a href="figures/out/GALLERY.md#fig09_exp_series_spiral"><img src="figures/out/complex/fig09_exp_series_spiral.png" alt="fig09_exp_series_spiral" title="fig09_exp_series_spiral" height="230"></a></p>

<p><a href="figures/out/GALLERY.md#inversion_in_circle"><img src="figures/out/complex/inversion_in_circle.png" alt="inversion_in_circle" title="inversion_in_circle" height="222"></a> <a href="figures/out/GALLERY.md#mobius_classification"><img src="figures/out/complex/mobius_classification.png" alt="mobius_classification" title="mobius_classification" height="222"></a> <a href="figures/out/GALLERY.md#poincare_disc"><img src="figures/out/complex/poincare_disc.png" alt="poincare_disc" title="poincare_disc" height="222"></a></p>

<p align="center"><a href="figures/out/GALLERY.md#vca_fig12_flow_grid"><img src="figures/out/complex/vca_fig12_flow_grid.png" alt="vca_fig12_flow_grid" title="vca_fig12_flow_grid" height="230"></a> <a href="figures/out/GALLERY.md#vca_fig14_volcanoes"><img src="figures/out/complex/vca_fig14_volcanoes.png" alt="vca_fig14_volcanoes" title="vca_fig14_volcanoes" height="230"></a></p>

<p><a href="figures/out/GALLERY.md#vca_fig30_elliptic_checkerboard"><img src="figures/out/complex/vca_fig30_elliptic_checkerboard.png" alt="vca_fig30_elliptic_checkerboard" title="vca_fig30_elliptic_checkerboard" height="222"></a> <a href="figures/out/GALLERY.md#vca_fig4_zn_polar_grid"><img src="figures/out/complex/vca_fig4_zn_polar_grid.png" alt="vca_fig4_zn_polar_grid" title="vca_fig4_zn_polar_grid" height="222"></a></p>

<p><a href="figures/out/GALLERY.md#vca_fig9_cassinian"><img src="figures/out/complex/vca_fig9_cassinian.png" alt="vca_fig9_cassinian" title="vca_fig9_cassinian" height="193"></a> <a href="figures/out/GALLERY.md#winding_number"><img src="figures/out/complex/winding_number.png" alt="winding_number" title="winding_number" height="193"></a></p>

### signals

*Sampling, spectra, and the geometry of transfer functions.*

<p align="center"><a href="figures/out/GALLERY.md#dft_matrix_basis"><img src="figures/out/signals/dft_matrix_basis.png" alt="dft_matrix_basis" title="dft_matrix_basis" height="230"></a> <a href="figures/out/GALLERY.md#polezero_response"><img src="figures/out/signals/polezero_response.png" alt="polezero_response" title="polezero_response" height="230"></a></p>

<p align="center"><a href="figures/out/GALLERY.md#sampling_aliasing"><img src="figures/out/signals/sampling_aliasing.png" alt="sampling_aliasing" title="sampling_aliasing" height="230"></a></p>

### linalg

*Matrices as geometry: the four readings, low rank, conditioning.*

<p align="center"><a href="figures/out/GALLERY.md#conditioning_ellipse"><img src="figures/out/linalg/conditioning_ellipse.png" alt="conditioning_ellipse" title="conditioning_ellipse" height="230"></a> <a href="figures/out/GALLERY.md#matrix_four_views"><img src="figures/out/linalg/matrix_four_views.png" alt="matrix_four_views" title="matrix_four_views" height="230"></a></p>

<p><a href="figures/out/GALLERY.md#svd_low_rank"><img src="figures/out/linalg/svd_low_rank.png" alt="svd_low_rank" title="svd_low_rank" height="206"></a></p>

### dynamics

*Flows, bifurcations, and stochastic trajectories.*

<p align="center"><a href="figures/out/GALLERY.md#demo_basin_wash"><img src="figures/out/dynamics/demo_basin_wash.png" alt="demo_basin_wash" title="demo_basin_wash" height="230"></a> <a href="figures/out/GALLERY.md#demo_ou_ensemble_field"><img src="figures/out/dynamics/demo_ou_ensemble_field.png" alt="demo_ou_ensemble_field" title="demo_ou_ensemble_field" height="230"></a></p>

<p align="center"><a href="figures/out/GALLERY.md#diffusion_ode_vs_sde"><img src="figures/out/dynamics/diffusion_ode_vs_sde.png" alt="diffusion_ode_vs_sde" title="diffusion_ode_vs_sde" height="230"></a></p>

<p align="center"><a href="figures/out/GALLERY.md#saddle_node_behavior_map"><img src="figures/out/dynamics/saddle_node_behavior_map.png" alt="saddle_node_behavior_map" title="saddle_node_behavior_map" height="230"></a> <a href="figures/out/GALLERY.md#strogatz_saddle_node"><img src="figures/out/dynamics/strogatz_saddle_node.png" alt="strogatz_saddle_node" title="strogatz_saddle_node" height="230"></a></p>

### optim

*What actually governs convergence.*

<p align="center"><a href="figures/out/GALLERY.md#illconditioned_descent"><img src="figures/out/optim/illconditioned_descent.png" alt="illconditioned_descent" title="illconditioned_descent" height="230"></a></p>

### probability

*Densities under maps, and where the mass really lives.*

<p align="center"><a href="figures/out/GALLERY.md#concentration_of_measure"><img src="figures/out/probability/concentration_of_measure.png" alt="concentration_of_measure" title="concentration_of_measure" height="230"></a></p>

<p align="center"><a href="figures/out/GALLERY.md#prml_polyfit_multiples"><img src="figures/out/probability/prml_polyfit_multiples.png" alt="prml_polyfit_multiples" title="prml_polyfit_multiples" height="230"></a> <a href="figures/out/GALLERY.md#pushforward_density"><img src="figures/out/probability/pushforward_density.png" alt="pushforward_density" title="pushforward_density" height="230"></a></p>

### statmech

*Order parameters and phase transitions.*

<p align="center"><a href="figures/out/GALLERY.md#ising_transition"><img src="figures/out/statmech/ising_transition.png" alt="ising_transition" title="ising_transition" height="230"></a></p>

### infotheory

*Typicality, and why the mode is not the story.*

<p align="center"><a href="figures/out/GALLERY.md#typical_set"><img src="figures/out/infotheory/typical_set.png" alt="typical_set" title="typical_set" height="230"></a></p>

### circuits

*Transformer internals as computation graphs.*

<p align="center"><a href="figures/out/GALLERY.md#induction_head_circuit"><img src="figures/out/circuits/induction_head_circuit.png" alt="induction_head_circuit" title="induction_head_circuit" height="230"></a> <a href="figures/out/GALLERY.md#qk_circuit_tensor"><img src="figures/out/circuits/qk_circuit_tensor.png" alt="qk_circuit_tensor" title="qk_circuit_tensor" height="230"></a></p>

<p><a href="figures/out/GALLERY.md#qk_score_derivation"><img src="figures/out/circuits/qk_score_derivation.png" alt="qk_score_derivation" title="qk_score_derivation" height="203"></a></p>

<p align="center"><a href="figures/out/GALLERY.md#schematic_transformer_block"><img src="figures/out/circuits/schematic_transformer_block.png" alt="schematic_transformer_block" title="schematic_transformer_block" height="230"></a></p>

<!-- gallery:end -->

## Commands

In any project, through the installed tool:

```
figcheck figures/x.py                render + every gate, exit 1 on failure
         --report                    + a textual layout inventory
         --zoom x0,y0,x1,y1:4        a magnified crop, for judging detail
         --readback-prompt           the prompt for the cold-reader agent
         --transparent               ink on alpha instead of the stock
figcheck --regress --figures-dir figures    golden diff over THIS project
figcheck --update  --figures-dir figures    refresh its baselines
```

Inside this repo, use the make targets — they cover the corpus and export
cairo's library path, which a bare `uv run figcheck` misses:

```
make test                            pytest
make check F=figures/optim/x.py      render + every gate, exit 1 on failure
  F="figures/optim/x.py --report"    + a textual layout inventory
  F="figures/optim/x.py --transparent"   ink on alpha instead of the stock
make regress                         corpus-wide golden diff
make update                          refresh the committed baselines
make gallery                         regenerate GALLERY.md + this README grid
make brand                           redraw the wordmark, mark and social card
```

## Layout

```
src/figlib/     the library — flat modules, one per concern
                (see figlib/__init__.py for the map)
figures/<subject>/*.py
                the corpus: complex · signals · linalg · dynamics · optim ·
                probability · statmech · infotheory · circuits
figures/out/    committed render baselines, both grounds, plus readback records
docs/skill.md   THE skill — symlinked into ~/.claude/skills and ~/.codex/skills
docs/           architecture.md (the stack and the design step) · grammar.md
                (visual rules, each with the failure that justifies it) ·
                exposition.md (why the design step is what it is)
                brand/ — the wordmark and social card, drawn by the library
                they brand (same ink, same grain, same math face)
tests/
```

Start at [`docs/skill.md`](docs/skill.md). It is short, it is the contract, and
it is the file your agent actually loads.
