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

## 1. The core object — shape is geometry

```python
@dataclass(frozen=True)
class Block:
    """A matrix drawn to shape. The rectangle is m*cell tall by n*cell
    wide, so the drawn aspect ratio IS (m, n)."""
    m: int
    n: int
    origin: XY = (0.0, 0.0)           # top-left, math coords
    cell: float = 1.0
    values: np.ndarray | None = None  # optional; unlocks value encoders
    name: str = ""
```

Derived geometry only, emitting no items: `rect()`, `cols(j0, j1)`,
`rows(i0, i1)`, `cell_rect(i, j)`, `cell_center(i, j)`,
`sub(rslice, cslice) -> Block`, `width`, `height`, `bbox`.

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

`mask` takes a bool array or a predicate `(i, j) -> bool`. `tri("lower")`,
`banded(k)`, `causal()` are three-line helpers built on it inside the
module — recipes, not primitives.

**Value**

| function | encoding |
|---|---|
| `heat(b, ...)` | `RasterField` over `b.rect()` — exists, wired to Block |
| `hinton(b)` | one square per entry, **area** ∝ \|v\|, fill by sign |
| `entries(b, fmt)` | MathLabel per cell; small `m, n` only |

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
- `check_shape_faithful(checks, b)` — drawn aspect ratio equals `m/n`
  (catches a hand-set `cell` or `origin` override).
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
