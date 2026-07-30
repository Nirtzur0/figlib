<h1 align="center">figlib</h1>

<p align="center"><b>A figure compiler for mathematical exposition.</b></p>

<p align="center">
  <a href="figures/out/GALLERY.md"><img src="figures/out/complex/vca_fig14_volcanoes.png" alt="the modular surface of 1/(1+z^2)" width="840"></a>
</p>

<p align="center"><sub><b>Why a real Taylor series stops converging at a radius nothing on the real line explains.</b><br>
The graph of 1/(1+x²) is the tranquil real slice of a surface that erupts at z = ±i. The radius is the distance to the poles you cannot see.</sub></p>

---

A figure here is not a drawing. It is a **program with a claim attached**, and
the claim is checked.

```python
CLAIM      = "one sentence: what the figure ARGUES, not what it draws"
EXPOSITION = "the passage this figure is FOR — written before the code"
THEME      = RISO        # CLEAN | RISO   (riso on cream is the default look)
FORMAT     = WIDE        # MARGIN 340 | COLUMN 680 | WIDE 1000 px page slot
PARAMS     = {...}       # every tunable; no magic numbers below this line

def compute(p):    ...   # numerics -> arrays. No drawing decisions.
def build(g):      ...   # arrays -> Scene (or Figure, for multi-panel)
def assertions(g): ...   # the gate, on the SAME arrays that got drawn
```

`compute → build → autoplace → gates → render` emits SVG + PNG. What makes it a
compiler rather than a plotting wrapper is that `assertions` runs against the
plotted arrays, not a re-derivation of them:

```python
def assertions(g):
    for pan in g["panels"]:
        r, xs = pan["r"], pan["fixed"]
        assert len(xs) == (2 if r < 0 else 1 if r == 0 else 0)
        for x, stable in zip(xs, pan["stable"]):
            assert abs(r + x**2) < 1e-12                 # it IS a fixed point
            assert stable == (2 * x < 0)                 # from f'(x*), not a table
        for a in pan["arrows"]:                          # every drawn arrow
            assert a.rightward == (r + a.midpoint**2 > 0)
```

If the arrowhead points the wrong way, the build fails. Not the review — the
build.

## The gate stack

| gate | what it holds |
|---|---|
| **numerical** | the program's own assertions, on the arrays that were drawn |
| **mechanical** | label collisions, clipping, type under 8.5 pt, annotation over 22% of the canvas |
| **color** | correspondence hues pairwise separable · an order ramp monotone in lightness · ink clearing a contrast floor on the actual ground |
| **golden regression** | rendering is byte-deterministic, so any diff against the committed baseline is a real change |
| **readback** | a cold agent sees only the PNG and says what it claims; if that misses `CLAIM`, the figure is wrong however pretty it is |

Every mechanical diagnostic carries a *computed* fix, not a complaint:

```
[FAIL] poincare_disc: ...
  label-collision: "L" overlaps "P" by 11px — offset_px += (+0, -13)
  clipping: geodesic overruns the frame by 4.2px on the right
```

## Content code names meanings, never appearance

`Role.CONTENT`, `theme.ramp(t)`, `theme.categorical(i)`, `theme.surface_shade(t)`.
No figure in this repo contains a hex literal, a font name, or a stroke width.

That is not tidiness. It is what makes the color gate possible: because a color
arrives carrying its *channel* — ordered, categorical, cyclic, depth — the gate
can hold each to its own standard. A ramp must be monotone in lightness; a
categorical set must be pairwise separable; neither test makes sense on an
anonymous `#4c72b0`. It also means one theme edit restyles the whole corpus, and
a figure that quietly stops retheming is a bug the corpus can find.

A figure is a printed page. The default ground is the riso theme's cream, with
grain — and grain is **ink, not paper**, so it rides the groundless render too.
Every figure commits both: `<name>.svg` on cream, and `<name>_transparent.svg`
on alpha for a document that owns its own background. `figcheck` checks both
unconditionally, because a contrast gate only earns its keep on a ground that
is actually there.

## Gallery

33 figures. Click any thumbnail for its claim, the passage it serves, and both
renders — or browse [**the full gallery**](figures/out/GALLERY.md).

<!-- gallery:start -->

### complex

*Conformal maps, the Riemann sphere, contour integration.*

<p><a href="figures/out/GALLERY.md#amplitwist"><img src="figures/out/complex/amplitwist.png" alt="amplitwist" height="227"></a> <a href="figures/out/GALLERY.md#contour_deformation"><img src="figures/out/complex/contour_deformation.png" alt="contour_deformation" height="227"></a> <a href="figures/out/GALLERY.md#demo_flow_past_cylinder"><img src="figures/out/complex/demo_flow_past_cylinder.png" alt="demo_flow_past_cylinder" height="227"></a></p>

<p><sub>amplitwist · contour_deformation · demo_flow_past_cylinder</sub></p>

<p><a href="figures/out/GALLERY.md#demo_panels_zsquared"><img src="figures/out/complex/demo_panels_zsquared.png" alt="demo_panels_zsquared" height="164"></a> <a href="figures/out/GALLERY.md#demo_sphere_stereographic"><img src="figures/out/complex/demo_sphere_stereographic.png" alt="demo_sphere_stereographic" height="164"></a></p>

<p><sub>demo_panels_zsquared · demo_sphere_stereographic</sub></p>

<p><a href="figures/out/GALLERY.md#fig09_exp_series_spiral"><img src="figures/out/complex/fig09_exp_series_spiral.png" alt="fig09_exp_series_spiral" height="255"></a></p>

<p><sub>fig09_exp_series_spiral</sub></p>

<p><a href="figures/out/GALLERY.md#inversion_in_circle"><img src="figures/out/complex/inversion_in_circle.png" alt="inversion_in_circle" height="222"></a> <a href="figures/out/GALLERY.md#mobius_classification"><img src="figures/out/complex/mobius_classification.png" alt="mobius_classification" height="222"></a> <a href="figures/out/GALLERY.md#poincare_disc"><img src="figures/out/complex/poincare_disc.png" alt="poincare_disc" height="222"></a></p>

<p><sub>inversion_in_circle · mobius_classification · poincare_disc</sub></p>

<p><a href="figures/out/GALLERY.md#vca_fig12_flow_grid"><img src="figures/out/complex/vca_fig12_flow_grid.png" alt="vca_fig12_flow_grid" height="237"></a> <a href="figures/out/GALLERY.md#vca_fig14_volcanoes"><img src="figures/out/complex/vca_fig14_volcanoes.png" alt="vca_fig14_volcanoes" height="237"></a></p>

<p><sub>vca_fig12_flow_grid · vca_fig14_volcanoes</sub></p>

<p><a href="figures/out/GALLERY.md#vca_fig30_elliptic_checkerboard"><img src="figures/out/complex/vca_fig30_elliptic_checkerboard.png" alt="vca_fig30_elliptic_checkerboard" height="222"></a> <a href="figures/out/GALLERY.md#vca_fig4_zn_polar_grid"><img src="figures/out/complex/vca_fig4_zn_polar_grid.png" alt="vca_fig4_zn_polar_grid" height="222"></a></p>

<p><sub>vca_fig30_elliptic_checkerboard · vca_fig4_zn_polar_grid</sub></p>

<p><a href="figures/out/GALLERY.md#vca_fig9_cassinian"><img src="figures/out/complex/vca_fig9_cassinian.png" alt="vca_fig9_cassinian" height="193"></a> <a href="figures/out/GALLERY.md#winding_number"><img src="figures/out/complex/winding_number.png" alt="winding_number" height="193"></a></p>

<p><sub>vca_fig9_cassinian · winding_number</sub></p>

### signals

*Sampling, spectra, and the geometry of transfer functions.*

<p><a href="figures/out/GALLERY.md#dft_matrix_basis"><img src="figures/out/signals/dft_matrix_basis.png" alt="dft_matrix_basis" height="230"></a> <a href="figures/out/GALLERY.md#polezero_response"><img src="figures/out/signals/polezero_response.png" alt="polezero_response" height="230"></a></p>

<p><sub>dft_matrix_basis · polezero_response</sub></p>

<p align="center"><a href="figures/out/GALLERY.md#sampling_aliasing"><img src="figures/out/signals/sampling_aliasing.png" alt="sampling_aliasing" height="299"></a></p>

<p align="center"><sub>sampling_aliasing</sub></p>

### linalg

*Matrices as geometry: the four readings, low rank, conditioning.*

<p><a href="figures/out/GALLERY.md#conditioning_ellipse"><img src="figures/out/linalg/conditioning_ellipse.png" alt="conditioning_ellipse" height="257"></a> <a href="figures/out/GALLERY.md#matrix_four_views"><img src="figures/out/linalg/matrix_four_views.png" alt="matrix_four_views" height="257"></a></p>

<p><sub>conditioning_ellipse · matrix_four_views</sub></p>

<p><a href="figures/out/GALLERY.md#svd_low_rank"><img src="figures/out/linalg/svd_low_rank.png" alt="svd_low_rank" height="206"></a></p>

<p><sub>svd_low_rank</sub></p>

### dynamics

*Flows, bifurcations, and stochastic trajectories.*

<p><a href="figures/out/GALLERY.md#demo_basin_wash"><img src="figures/out/dynamics/demo_basin_wash.png" alt="demo_basin_wash" height="250"></a> <a href="figures/out/GALLERY.md#demo_ou_ensemble_field"><img src="figures/out/dynamics/demo_ou_ensemble_field.png" alt="demo_ou_ensemble_field" height="250"></a></p>

<p><sub>demo_basin_wash · demo_ou_ensemble_field</sub></p>

<p><a href="figures/out/GALLERY.md#diffusion_ode_vs_sde"><img src="figures/out/dynamics/diffusion_ode_vs_sde.png" alt="diffusion_ode_vs_sde" height="240"></a> <a href="figures/out/GALLERY.md#strogatz_saddle_node"><img src="figures/out/dynamics/strogatz_saddle_node.png" alt="strogatz_saddle_node" height="240"></a></p>

<p><sub>diffusion_ode_vs_sde · strogatz_saddle_node</sub></p>

### optim

*What actually governs convergence.*

<p align="center"><a href="figures/out/GALLERY.md#illconditioned_descent"><img src="figures/out/optim/illconditioned_descent.png" alt="illconditioned_descent" height="299"></a></p>

<p align="center"><sub>illconditioned_descent</sub></p>

### probability

*Densities under maps, and where the mass really lives.*

<p><a href="figures/out/GALLERY.md#concentration_of_measure"><img src="figures/out/probability/concentration_of_measure.png" alt="concentration_of_measure" height="227"></a> <a href="figures/out/GALLERY.md#pushforward_density"><img src="figures/out/probability/pushforward_density.png" alt="pushforward_density" height="227"></a></p>

<p><sub>concentration_of_measure · pushforward_density</sub></p>

### statmech

*Order parameters and phase transitions.*

<p align="center"><a href="figures/out/GALLERY.md#ising_transition"><img src="figures/out/statmech/ising_transition.png" alt="ising_transition" height="299"></a></p>

<p align="center"><sub>ising_transition</sub></p>

### infotheory

*Typicality, and why the mode is not the story.*

<p align="center"><a href="figures/out/GALLERY.md#typical_set"><img src="figures/out/infotheory/typical_set.png" alt="typical_set" height="299"></a></p>

<p align="center"><sub>typical_set</sub></p>

### circuits

*Transformer internals as computation graphs.*

<p><a href="figures/out/GALLERY.md#induction_head_circuit"><img src="figures/out/circuits/induction_head_circuit.png" alt="induction_head_circuit" height="251"></a> <a href="figures/out/GALLERY.md#qk_circuit_tensor"><img src="figures/out/circuits/qk_circuit_tensor.png" alt="qk_circuit_tensor" height="251"></a></p>

<p><sub>induction_head_circuit · qk_circuit_tensor</sub></p>

<p align="center"><a href="figures/out/GALLERY.md#schematic_transformer_block"><img src="figures/out/circuits/schematic_transformer_block.png" alt="schematic_transformer_block" height="299"></a></p>

<p align="center"><sub>schematic_transformer_block</sub></p>

<!-- gallery:end -->

## Running it

Python 3.12+, [uv](https://docs.astral.sh/uv/), and cairo (`brew install cairo`).

```
make test                            pytest
make check F=figures/optim/x.py      render + every gate, exit 1 on failure
  F="figures/optim/x.py --report"    + a textual layout inventory
  F="figures/optim/x.py --transparent"   ink on alpha instead of cream
make regress                         corpus-wide golden diff
make update                          refresh the committed baselines
make gallery                         regenerate GALLERY.md + this README grid
```

`figcheck` also installs as a standalone tool, so figures can be authored from
any project without that project depending on figlib.

## Layout

```
src/figlib/     the library — flat modules, one per concern
                (see figlib/__init__.py for the map)
figures/<subject>/*.py
                the corpus: complex · signals · linalg · dynamics · optim ·
                probability · statmech · infotheory · circuits
figures/out/    committed render baselines, both grounds, plus readback records
docs/           skill.md (how to write a figure) · architecture.md (the stack,
                and the 0–9 design step) · grammar.md · exposition.md
tests/
```

Start at [`docs/skill.md`](docs/skill.md). It is short, and it is the contract.
