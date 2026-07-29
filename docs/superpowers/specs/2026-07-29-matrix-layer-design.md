# Matrix layer design

`src/figlib/matrix.py` — matrices and matrix operations as drawable,
gateable objects. A producer of scene items, following `surface3d` /
`plots` / `schematic`. No new renderer.

## The gap

`primitive-gaps.md` surveyed Needham, and Needham has no matrices. So the
gap was never counted. Three quarters of the substrate already exists:

- `RasterField` (`scene.py:203`) — dense array → theme ramp. Attention
  maps already draw.
- `Figure` / `Panel` / `Connector` (`figure.py`) — the equation-of-figures
  page grammar.
- `mapped_grid` (`builders.py:78`) — grid under a map.
- `schematic.py` — typed nodes / ports / edges.

What is missing is the **structure encoding**: a matrix drawn as a shaped
rectangle partitioned into columns, rows, blocks, and masked cells.

## Reference basis

Three traditions, and they are mutually incompatible encodings of the same
object. Mixing them in one figure is the standard failure.

**Structure — Hiranabe, *The Art of Linear Algebra* (with Strang).**
<https://github.com/kenjihiranabe/The-Art-of-Linear-Algebra>. The best
single source found; its grammar is small and orthogonal.

- One rectangle, **aspect ratio = m:n**, with four readings of it: opaque
  whole, dot lattice of `mn` entries, `n` vertical bands, `m` horizontal
  bands. Choosing the reading *is* the argument.
- Hue is a fixed 4-slot categorical: green = columns, magenta = rows,
  blue dots = scalars, gray = unstructured whole. Exactly
  `theme.correspondence_cap` for RISO.
- Triangular and diagonal structure is drawn as the **silhouette of the
  filled region**, never as printed zeros. `L` is a descending staircase;
  `Λ` is a dot on the diagonal.
- Rank-1 outer product = a full rectangle painted by a column stripe
  crossed with a row stripe.
- Every factorization (`CR`, `LU`, `QR`, `QΛQᵀ`, `UΣVᵀ`) then reduces to
  the *same* picture: a sum of rank-1 rectangles.
- **Equation of figures**: block pictures in a row joined by `=` and `+`.

Trefethen & Bau's block diagrams (full vs. reduced QR / SVD) are the same
grammar with less color. Wilkinson diagrams are the `mask` sub-case.

**Value — Hinton diagrams.**
<https://matplotlib.org/stable/gallery/specialty_plots/hinton_demo.html>.
Signed magnitude as square **area**, sign as fill. Still the honest
small-matrix encoding; a monotone ramp cannot carry sign.

**Action — Strang's four subspaces, Trefethen Lecture 4 (unit sphere →
hyperellipse), 3Blue1Brown's grid-under-a-map.** Already expressible via
`mapped_grid` and ordinary curves. Nothing to build.

## Out of scope

Einsum / tensor-network diagrams — *An introduction to graphical tensor
notation for mechanistic interpretability* (<https://arxiv.org/pdf/2402.01790>),
*Named Tensor Notation* (<https://arxiv.org/pdf/2102.13196>). A tensor has
legs, not a 2-D shape to draw to scale; forcing it into `Block` would
corrupt the shape-is-geometry invariant. Separate spec, against
`schematic.py`.

## 0a. Revision: the merge is withdrawn

The signals plan finished without building `matrix.py` at all.
`dft_matrix_basis.py` does its cell geometry inline — a two-line `center`
closure and a three-line bracket construction — and states the case in its
docstring: *"matrices need no primitive, only addresses."* That is a
correct application of the doctrine (add a primitive only when a device is
*inexpressible*, never when merely verbose), and the figure passes its
gates.

So the merge described in §0b below is withdrawn. `CellGrid` was never
built and now has no consumer, and importing its surface into `Block` on
speculation would be API for zero call sites. **Cut: `aspect`, `map_into`,
`edge`, `grid_lines`, `brackets`, `cell_fills`, `check_square_cells`.**
`cell` returns to a scalar, so shape-as-aspect-ratio is unconditional
again — which is strictly stronger than the `aspect`-with-opt-out version.

What survives the same argument, and why the layer is still worth
building: **their argument holds for addressing and fails for gating.** A
`center(n, k)` closure is genuinely cheap to rewrite per figure. A gate is
not — `check_expr` evaluates a drawn factorization and `check_conformable`
reads inner dimensions off the drawn term list, and neither can exist as
local arithmetic inside one figure, because a gate is the shared oracle.
Those gates need shape and values to be one object that survives from
`compute()` into `assertions()`. That object is `Block`, and nothing
smaller does the job.

Retained from the signals spec: `diagonal(b, offset, wrap=False)`, because
circulant wrap is real index arithmetic that is easy to get wrong and it
costs three lines.

## 0b. Merged with the signals & linear-algebra spec (superseded by §0a)

`2026-07-29-signals-linalg-design.md` independently specified `CellGrid`
in the same new module `src/figlib/matrix.py`: the same frozen, draws-
nothing, top-left-origin, row-0-at-top cell geometry. Two names for one
object is exactly the drift `architecture.md` exists to prevent, so they
are **one type, `Block`**, owned by this spec. That spec now defers here
for the geometry and keeps only its own producers.

The merge is not cosmetic. `CellGrid` carried `cell: (width, height)` —
non-square cells, needed for the DFT basis gallery where each cell holds a
mini waveform. `Block` needs a scalar cell, because square cells are what
make the drawn aspect ratio equal the shape and the conformability gates
mean anything. Resolution: scalar `cell` plus an explicit `aspect`
(cell height / cell width, default `1.0`). `aspect == 1.0` is the matrix
case and the default; a gallery opts out by saying so, on the record.

## 1. The core object — shape is geometry

```python
@dataclass(frozen=True)
class Block:
    """A matrix drawn to shape: n*cell wide, m*cell*aspect tall.

    Draws nothing — it answers coordinate questions and the author
    composes items against it.
    """
    m: int
    n: int
    origin: XY = (0.0, 0.0)           # TOP-left, math coords
    cell: float = 1.0                 # cell WIDTH in math units
    aspect: float = 1.0               # cell height / width; 1.0 = square
    values: np.ndarray | None = None  # optional; unlocks value encoders
    name: str = ""
```

`m` and `n` are two ints rather than a `shape` tuple because the whole
point is that `A.n == B.m` reads as the conformability question it is.

Derived geometry only, emitting no items: `rect()`, `span(j0, j1, i0, i1)`,
`cols(j0, j1)`, `rows(i0, i1)`, `cell_rect(i, j)`, `center(i, j)`,
`sub(rslice, cslice) -> Block`, `at(origin, cell=None) -> Block`,
`width`, `height`, `extent`, plus two from the signals spec:

- `map_into(i, j, pts, *, src=((0,1),(0,1)))` — affine-map points from a
  local `(u, v)` frame into cell `(i, j)`, `v` UP within the cell. The
  inset bridge: compute a mini stem plot or waveform in local coords, map
  it, emit it. This is what makes "basis gallery" a recipe rather than a
  primitive.
- `edge(side, *, pad=0.0)` — the `(2, 2)` segment along `"left"` /
  `"right"` / `"top"` / `"bottom"`; the anchor line for brackets, index
  labels, and dimension braces.

`extent` returns `(x0, x1, y0, y1)` and is named for the `RasterField`
parameter it is passed to.

Same discipline as `schematic.connect`: geometry is derived once, render
and gates both read it, so they cannot drift.

This single decision is what buys the gates. A non-conformable product is
not drawable, and the `values` slot makes the picture and the numbers the
same object.

Row 0 is at the **top** (`origin` is top-left, `+y` up in math coords),
matching `RasterField`'s existing row-0-at-`y1` rule and matrix index
convention.

## 2. Encoders

Each returns `list[Item]`. Style comes from roles and theme channels only;
no encoder names a color.

**Structure**

| function | device | items |
|---|---|---|
| `outline(b, role)` | the whole matrix | Curve rim, optional wash |
| `bands(b, axis, ...)` | `n` column or `m` row bands | FilledCurve per band |
| `lattice(b)` | `mn` entries as dots | Point grid |
| `mask(b, M)` | triangular / banded / sparsity / causal | FilledCurve per run of true cells |
| `rank1(b, j, i)` | outer product `a_j b_iᵀ` | column band + row band + faint whole |
| `grid_lines(b, ...)` | ruled cells | Curve per inner/outer rule |
| `brackets(b, ...)` | the `[ ]` glyphs | two 3-segment Curves |

`mask` takes a bool array or a predicate `(i, j) -> bool`. `tri("lower")`,
`banded(lo, hi)`, `diagonal(k, wrap=False)` and `causal()` are three-line
mask helpers inside the module — recipes, not primitives. `wrap` on
`diagonal` is what makes a circulant portrait a one-liner; a call site
that needs index pairs rather than a mask uses `np.argwhere`.

`grid_lines` and `brackets` come from the signals spec.

**Value**

| function | encoding |
|---|---|
| `heat(b, ...)` | `RasterField` over `b.extent` — the dense path |
| `cell_fills(b, ...)` | one FilledCurve per cell through a ramp — the vector path |
| `hinton(b)` | one square per entry, **area** ∝ \|v\|, fill by sign |
| `entries(b, fmt)` | MathLabel per cell; small `m, n` only |

`heat` and `cell_fills` (the latter from the signals spec) are the same
encoding at two densities and both stay: a 32×32 attention map wants one
raster, an 8×8 wants vector cells that inherit theming and gates. The
docstrings state the tradeoff; the author chooses.

`hinton` is the only genuinely new mark in the module.

## 3. The expression row

```python
def expr(terms: Sequence[Block | str], *, cell: float, gap: float
        ) -> tuple[list[Item], list[Block]]
```

Terms are `Block`s and operator strings (`"="`, `"+"`, `"@"`). Laid out
left to right in **one Scene, one coordinate system, one shared `cell`**,
on a common vertical center. Operators are MathLabels in the gaps. The
returned Blocks are the placed ones — the figure program draws into them
and passes them to the gates.

**Rejected: `Figure` / `Panel`.** Panels carry independent transforms, so
a 3×2 and a 2×4 could render at different cell sizes — silently
destroying the one property the entire grammar rests on. The blocks must
share coordinates.

## 4. Gates

Exported checkers, called from a figure's `assertions()` via
`gates.Checks` so one run reports every failure.

- `check_conformable(checks, terms)` — walk the **drawn** term list: every
  adjacent `A @ B` has `A.n == B.m`; every `+` has equal-shaped operands;
  both sides of `=` are equal-shaped.
- `check_cell_uniform(checks, blocks)` — one `(cell, aspect)` across the
  figure. Two matrices at different scales make their shapes
  incomparable. `expr` guarantees it within a row; a hand-placed block
  alongside is what this catches. (Replaces an earlier
  `check_shape_faithful`: with `width = n*cell`, drawn aspect ratio
  equals `m/n` by construction, so asserting it would be gate theater.)
- `check_no_overlap(checks, blocks)` — no two drawn matrices share area;
  a `gap` too small collides two blocks and the operator then reads as
  inside one of them.
- `check_square_cells(checks, blocks)` — every block claiming to argue
  about *shape* has `aspect == 1.0`. A gallery that opts out is fine; a
  factorization figure that opts out is drawing a lie about dimensions.
- `check_expr(checks, terms, rtol)` — when every Block carries `values`,
  evaluate the drawn expression in numpy and assert it holds. Draw
  `A = UΣVᵀ` and the gate *proves the picture is of a true
  factorization*. This is the load-bearing gate and the reason the layer
  belongs in figlib rather than in matplotlib.
- `check_hinton_area(checks, b, items)` — square areas proportional to
  `|v|`, guarding the side ∝ `|v|` bug that silently squares the encoding.

All four assert what could be wrong: the drawn geometry and the drawn
arrays, never a fact true by construction.

## 5. Theming

No new theme surface. New semantic *uses* of existing channels:

| meaning | channel |
|---|---|
| columns | `theme.categorical(0)` |
| rows | `theme.categorical(1)` |
| entries / scalars | `theme.categorical(2)` |
| unstructured whole | `Role.MUTED` wash |
| Hinton sign | `Role.ACCENT1` / `Role.ACCENT2` |
| value magnitude | `theme.ramp(t)` |

Four correspondence slots is exactly `correspondence_cap` for RISO. If the
assignment reads wrong it is a theme edit, not a figure edit.

## 6. Benchmarks

Definition of done per `CLAUDE.md`: gates pass, `make regress` clean, a
readback record exists.

1. **`figures/matrix_four_views.py`** — one 3×2 read four ways, then `AB`
   as a sum of rank-1 rectangles. Exercises `bands`, `lattice`, `rank1`,
   `expr`, `check_conformable`.
2. **`figures/svd_low_rank.py`** — a real array: `A ≈ Σ σᵢuᵢvᵢᵀ` as heat
   blocks in an expression row, truncation error annotated on the figure.
   Exercises `heat`, `expr`, `check_expr` at `rtol`. Hinton lands here as
   the small-matrix inset.

The signals spec's `figures/dft_matrix_basis.py` is the third benchmark
and the one that exercises the merged surface end to end — `map_into`,
`brackets`, `cell_fills`, and `aspect != 1.0` — so it stays owned by that
spec and lands with its plan.

## 7. Tests

`tests/test_matrix.py`, written before the implementation:

- `Block` geometry: `rect`/`cols`/`rows`/`cell_rect` corner arithmetic,
  `sub` composition, row 0 at top.
- Each gate fires on a constructed violation and stays silent on a valid
  case — including a deliberately non-conformable `expr` and a factorization
  perturbed past `rtol`.
- `hinton` area proportionality across a signed array.
- SVG assertions via `tests/svgkit.py` (`tag`, `find_by`,
  `path_cmd_counts`), never raw `e.tag`.

## Follow-on

Once landed, `docs/skill.md`'s device → exemplar index gains two rows, and
`primitive-gaps.md` gains a short section recording that the matrix
structure encoding was a gap the VCA survey could not have found.
