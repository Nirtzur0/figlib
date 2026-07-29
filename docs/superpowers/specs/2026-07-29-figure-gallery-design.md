# Figure gallery: reorganize, uplift, expand — and mine the friction

Status: approved design, not yet implemented.

## Where we are

`make regress` is 24/24 MATCH and 595 tests pass. Nothing is broken. What
is actually wrong is narrower and more interesting:

- **12 of 23 figure programs have no readback record**, so by the repo's
  own definition of done they are not done.
- **Five figures were authored before the library could draw.** Commit
  order stratifies the corpus cleanly: the capability layer (panels,
  connectors, sphere projector, glyph pack) landed at commit 26, gradient
  shading at 30, the plots layer at 42, the matrix layer at 52, tensor at
  60. `fig09_exp_series_spiral` (8), `diffusion_ode_vs_sde` (9),
  `vca_fig4_zn_polar_grid` (11), `vca_fig9_cassinian` (13) and
  `vca_fig14_volcanoes` (17) predate all of it.
- **`figures/out/` is flat**: 65 files, four kinds of artifact
  (`.svg`, `.png`, `.readback.md`, `.judge.md`, `.zoom.png`) interleaved
  across six unrelated subject areas.
- **Nothing records what a figure is for.** `CLAIM` says what the figure
  argues; nothing says what argument it serves.
- **Nothing records what it cost to build.** Each figure taught us
  something about which primitive was missing, and that knowledge died
  with the session that built it.

The last point is the one that compounds. This design treats the gallery
work as the occasion to instrument the library, not just to redecorate
the corpus.

## Goals

1. A browsable gallery grouped by subject, each entry carrying the figure,
   its claim, and the prose it serves.
2. Every figure at definition-of-done: gates green, readback recorded.
3. Wider subject coverage — 12 new figures, half from Needham, half from
   areas the corpus has zero reach into.
4. A ranked, evidence-backed list of the primitives that would make the
   next figure cheaper, derived from 35 independent build records rather
   than from one person's memory.

## Non-goals

- No renames. `demo_*` names stay. Renaming is churn with no payoff; the
  gallery surfaces `CLAIM`, and preserving the corpus literally is worth
  more than tidy filenames.
- No constraint solver, no interactive gallery, no new renderer. New
  capability enters as a producer of scene items (architecture.md).
- No unrelated refactoring of figlib.

---

## Part A — Render contract: grain is ink, and every figure ships two grounds

Two changes, both requested, both altering the contract rather than a
figure.

### A1. Page-wide grain renders on transparent

`render.py::_emit_grain` currently zeroes the page overlay when
`transparent=True`, justified in-comment as: over no ground the overlay
"composites as a full-bleed speckle field, which reads as exactly the
background the transparent render exists to remove."

That reasoning is coherent but the framing behind it is wrong. Grain is
not the paper's texture — it is the **riso print texture, a property of
the ink**. A figure run through a riso press carries grain whether or not
you can see the paper. Under that framing the suppression was the bug.

So `_emit_grain` emits unconditionally, driven only by `style.grain > 0`.

The honest cost: the grain rect is canvas-sized, so the figure's bounding
rectangle becomes faintly visible against any host background. The
magnitude is small — `grain_tile_datauri` caps alpha at 25/255 and RISO's
`grain` is 0.5, so peak contribution is roughly 5%. **Accepted as
specified.** If a readback flags a visible rectangle edge, the fallback is
an alpha feather over the canvas margin; it is not built speculatively.

`CLAUDE.md`'s invariant is rewritten. From:

> A figure has no ground. `Style.transparent` defaults to `True`: ink on
> alpha, no paper rect and no page-wide grain

to:

> A figure has no ground. `Style.transparent` defaults to `True`: ink on
> alpha, no paper rect. **Grain is ink, not paper — it always renders.**

### A2. Every figure emits both grounds

`regress.py::variants` currently sniffs for an existing `_paper` baseline
to decide whether a papered render exists. That is why exactly one figure
(`vca_fig14_volcanoes`) has one. It becomes unconditional:

```
<name>.svg / .png         transparent + grain
<name>_paper.svg / .png   RISO cream (#f2ede1 -> #e9dfcd) + grain
```

Baselines go 24 -> 46 now, ~70 once the new figures land.

### A3. RISO everywhere

Cream is already the house ground: RISO's paper *is* that cream, and 19 of
23 figures already set `THEME = RISO`. The four that do not would emit a
white paper variant instead of a cream one. All four move to RISO:

- `diffusion_ode_vs_sde`, `vca_fig4_zn_polar_grid`,
  `fig09_exp_series_spiral` — no `THEME` line at all, defaulting to CLEAN.
  Oversight.
- `vca_fig9_cassinian` — explicitly `THEME = CLEAN`, plausibly a
  book-fidelity choice (Needham's Cassinian figure is line art on white).
  Confirmed by the owner: move it to RISO like the rest.

CLEAN survives as a theme; no figure in the corpus uses it. It stays
because the contrast gate exercises white paper and because a future
book-faithful monochrome figure may want it.

---

## Part B — Reorganization: subject subdirectories

Programs and artifacts both bin by subject, mirrored. `git mv` throughout
so history follows.

```
figures/
  complex/     vca_fig4_zn_polar_grid, vca_fig9_cassinian, vca_fig12_flow_grid,
               vca_fig14_volcanoes, vca_fig30_elliptic_checkerboard,
               demo_panels_zsquared, demo_sphere_stereographic,
               demo_flow_past_cylinder, fig09_exp_series_spiral        (9)
  signals/     sampling_aliasing, polezero_response, dft_matrix_basis  (3)
  linalg/      matrix_four_views, svd_low_rank                         (2)
  dynamics/    strogatz_saddle_node, demo_basin_wash,
               demo_ou_ensemble_field, diffusion_ode_vs_sde            (4)
  circuits/    schematic_transformer_block, induction_head_circuit,
               qk_circuit_tensor                                       (3)
  capability/  demo_solids_gradient, demo_glyphs_annulus               (2)
  out/<subject>/<name>.{svg,png,readback.md,judge.md,zoom.png}
```

`capability/` is an honest bin: those two are probes of what the renderer
can do, not arguments about mathematics. They are exempt from `EXPOSITION`
(Part C) and carry a one-line `PROBES` string instead.

`demo_flow_past_cylinder` bins to `complex/`, not `dynamics/` — it is
potential flow, i.e. Needham Ch12, and it belongs beside `vca_fig12_flow_grid`.

### Library changes

Path resolution is centralized in two places, so this is additive, not a
refactor:

- `regress.py::discover_programs` — currently `glob("*.py")` at top level.
  Becomes: top-level `*.py` plus one level deep, excluding `out/` and any
  `_`-prefixed name. One level only; nested subjects are not a thing.
- New `regress.py::artifact_dir(program_path, figures_dir) -> Path` —
  mirrors the program's subject directory into `out/`. Every out-path
  derivation (`compare_figure`, `update_baselines`, `cli.py`'s `--zoom`
  and single-figure run) goes through it. A program directly in
  `figures/` maps to `figures/out/` unchanged, so nothing outside this
  repo breaks.

**Exit criterion for Part B: `make regress` prints 46/46 MATCH.** The move
is only correct if it is invisible to the renderer.

---

## Part C — `EXPOSITION` as a field, and a generated gallery

The requested "paragraph explaining what the figure explains" becomes a
module-level field, not a sidecar file, because it is the same kind of
object as `CLAIM` — a claim about the figure's job — and because a
sidecar rots the moment the figure changes.

```python
CLAIM      = "one sentence: what the figure ARGUES"
EXPOSITION = """The 1-3 paragraph passage this figure serves: the
surrounding text that made it necessary. For a VCA recreation, grounded
in Needham's actual argument at that point in the book. For the rest,
the motivating text of the idiom it borrows."""
```

A gate checks presence and non-triviality (a word-count floor and a
rejection of the placeholder string), so a figure without exposition fails
`make check` — the same standing as a missing readback record. That is
what stops the field from rotting.

The gate cannot be *enabled* in Phase 0: `regress.py::_render_to` raises
on a failed gate, so switching it on before any figure has an
`EXPOSITION` would make the corpus un-renderable and Phase 0's own exit
criterion unreachable. So Phase 0 **implements** the gate behind an
off-by-default flag and Phase 1 **enables** it as its final act, once all
23 figures carry the field. Landing the gate disabled is not a
compromise — the switch flip is Phase 1's exit criterion, which is the
only thing that makes it real.

`capability/` figures satisfy the gate with `PROBES` instead.

**Ordering claim, binding on the 12 new figures: `EXPOSITION` is written
before any code.** The figure serves an argument. If the paragraph cannot
be written, there is no figure to make — that is a cheap early kill and
it is the point.

### `make gallery`

Generates `figures/out/GALLERY.md` from every program's `EXPOSITION`,
`CLAIM`, `FORMAT`, `THEME` and its rendered PNG, grouped by subject.
Generated, never hand-edited. Each card shows the cream variant (it reads
better as a card) and links the transparent one and the readback record.

---

## Part D — Delegation, and the friction record

One agent per figure, end to end: read the design step in
`architecture.md`, write `EXPOSITION` first, compute -> Scene ->
assertions, drive gates green, run the readback loop, write a friction
record. Agents within a phase touch disjoint files and run in parallel
without worktrees. The only shared artifact is the generated gallery,
regenerated once at the end.

23 agents run across the four working phases, producing 23 friction
records: one per figure **built or rebuilt** (5 in Phase 2, 12 in Phase
3), plus one per subject in Phase 1, where the agent is reading and
readbacking rather than building and the friction is of a different kind
(what the readback loop cost, what the gates failed to catch).

Every agent writes `docs/friction/<name>.md` to a fixed template. This
is the compounding artifact and the reason to do the work with agents at
all:

```markdown
# <figure> — friction record        agent · subject · date

## What the claim needed
## What figlib gave for free           <- what works; do not break it
## What I hand-rolled                  <- the load-bearing section
  - <device> — ~N lines in the figure program that should be a primitive
## Gate diagnostics that did NOT contain the fix
## Renders to first green: N
## Proposed primitive (signature, not prose)
```

`Renders to first green` is the quantitative handle: it ranks friction
across the records without anyone reading them closely. "Gate diagnostics that
did NOT contain the fix" audits the *diagnostics contain the fix*
invariant from `CLAUDE.md`, which has never been measured.

Phase 4 ranks hand-rolled devices by (figures affected x lines
hand-rolled), folds the ranking into `primitive-gaps.md`, and builds the
top few.

---

## Part E — The figure slate

### Uplift (5) — same claim, current tools

The five that predate the capability layer, listed with what they were
built without. Each keeps its `CLAIM` and rebuilds the geometry against
panels, connectors, the glyph pack, gradients, the plots layer and the
line-style channel.

| figure | authored at | built without |
|---|---|---|
| `fig09_exp_series_spiral` | 8 | panels, connectors, glyphs, gradients, plots |
| `diffusion_ode_vs_sde` | 9 | same |
| `vca_fig4_zn_polar_grid` | 11 | same |
| `vca_fig9_cassinian` | 13 | same |
| `vca_fig14_volcanoes` | 17 | panels, connectors, gradients, plots |

### New from Needham (6)

Chapters the corpus has not touched. Each is a slate entry, not a
prescription — an agent may substitute with a written justification in
its friction record.

1. **Mobius classification** — elliptic / parabolic / hyperbolic /
   loxodromic as a 2x2 taxonomy of flows and fixed points. Exercises the
   panel layer hardest.
2. **Inversion in a circle** — orthogonal circles preserved; panel-pair
   joined by a labeled connector.
3. **Winding number / argument principle** — contour, image contour,
   zeros counted by turns.
4. **Contour deformation** — homotopy past a pole as a filmstrip.
5. **Poincare disc** — hyperbolic geodesics and a tessellation.
6. **The amplitwist** — an infinitesimal disc mapped to a rotated,
   scaled disc; Needham's signature idea, as a panel-pair.

### New beyond the book (6)

Subjects with zero current coverage.

7. **Ill-conditioned descent** — gradient descent vs momentum vs Newton
   on a quadratic, trajectories over level sets, condition number
   annotated.
8. **Conditioning as geometry** — the unit circle mapped to an ellipse by
   `A`, singular values as semi-axes, error amplification read off the
   axis ratio. Pairs with `svd_low_rank`.
9. **Pushforward density** — how a density transforms under a map, with
   the Jacobian shown as area distortion.
10. **The typical set** — asymptotic equipartition; sequence space with
    the typical shell (MacKay).
11. **Ising transition** — magnetization against temperature with sample
    configurations at three temperatures, as small multiples.
12. **Concentration of measure** — the norm distribution of Gaussian
    samples against dimension; the surprise is the point.

---

## Phases

Each phase has an exit criterion. Work stops cleanly after any of them.

| # | work | agents | exit criterion |
|---|---|---|---|
| 0 | Commit the in-flight library WIP. Part A render contract. Part B move + path resolution. Part C field, `make gallery`, exposition gate implemented but **disabled**. Re-baseline. | 0 | `make test` green; `make regress` 46/46 |
| 1 | `EXPOSITION` for all 23; readback for the 12 missing records; fix what readbacks flag; **enable the exposition gate** | 6 (one per subject) | gate enabled and every figure green under it; 23 readback records exist |
| 2 | Uplift the 5 | 5 | 5 rebuilt figures gate green with readbacks; drift on those 5 reviewed and re-baselined deliberately |
| 3a | 6 new from Needham | 6 | 6 new figures gate green with readbacks + friction records |
| 3b | 6 new beyond the book | 6 | as above |
| 4 | Friction synthesis -> `primitive-gaps.md`; build top-ranked primitives; regenerate gallery; final regress | 0 | ranked gap list committed; `make regress` clean at ~70 baselines; `GALLERY.md` current |

Phases 3a and 3b are independent and may run together (12 agents) if
throughput matters more than review granularity.

## Risks

- **Grain-on-transparent shows a rectangle edge.** Accepted; readbacks in
  Phase 1 are the detector; the alpha feather is the fallback.
- **23 agents produce stylistically divergent figures.** The device ->
  exemplar index in `skill.md` is the existing control: agents are told to
  mutate the nearest exemplar, not invent structure. Phase 4 reviews the
  corpus as a whole for drift.
- **The friction records become 23 files nobody reads.** `Renders to
  first green` plus the (figures x lines) ranking is the mitigation: the
  synthesis is mechanical over the structured fields, not a close read.
- **Subject bins are contested.** `demo_flow_past_cylinder` in `complex/`
  and the existence of `capability/` are the two judgment calls. Both are
  cheap to revisit — `git mv` plus a re-baseline.
