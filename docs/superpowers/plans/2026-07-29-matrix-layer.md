# Matrix Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `src/figlib/matrix.py` — matrices and matrix operations as drawable, gateable scene items — plus two benchmark figures.

**Architecture:** One frozen `Block` dataclass carries `(m, n)` and derives all geometry, so the drawn rectangle's aspect ratio *is* the matrix shape. Encoder functions turn a `Block` into existing scene items (`FilledCurve`, `Point`, `MathLabel`, `RasterField`) — a producer of scene items, never a new renderer. An `expr` layout places Blocks and operator glyphs in one shared coordinate system, and five checkers gate the result, the strongest being `check_expr`, which evaluates the drawn factorization in numpy.

**Tech Stack:** Python 3.11+, numpy, existing figlib (`scene`, `style`, `gates`, `theme`). pytest. No new dependencies.

Spec: `docs/superpowers/specs/2026-07-29-matrix-layer-design.md`.

**This plan also builds all of `src/figlib/matrix.py` for the signals &
linear-algebra plan.** That spec independently designed `CellGrid` — the
same frozen, draws-nothing, top-left-origin cell geometry in the same new
module. The two are merged into one type, `Block`;
`docs/superpowers/plans/2026-07-29-signals-linalg.md` Task 2 is superseded
and defers here. The merge shows up in three places below: `Block` gains
`aspect`, `map_into`, and `edge` (Task 1); Task 2 gains `grid_lines`,
`brackets`, and `diagonal(..., wrap=)`; Task 3 gains `cell_fills`.

## Global Constraints

- **Never name a color, font, or stroke width in `matrix.py` or in a figure.** Encoders take `role: Role` and `color: str | None`; the *figure program* supplies `THEME.categorical(i)` / `THEME.ramp(t)`. `matrix.py` imports nothing from `theme.py`.
- **Math coords, `+y` UP.** `Block.origin` is the **top-left** corner, so row 0 is at the top — matching `RasterField`'s row-0-at-`y1` rule (`scene.py:207`).
- **Canvas units are display CSS px.** Figures declare `FORMAT`; never shrink type to fit.
- **Assert what could be wrong, never what is true by construction.**
- **Tests assert SVG via `tests/svgkit.py`** (`svg_root`, `tag`, `find_by`, `path_cmd_counts`) — never raw `e.tag`; ElementTree namespaces every tag on parse.
- **Commands** (cairo needs the Makefile's `DYLD_FALLBACK_LIBRARY_PATH`):
  - `make test`
  - `make check F=figures/<name>.py`
  - `make regress`
  - `make update F=figures/<name>.py`
- **`figures/out/` diffs are real.** Look at what moved before `make update`.
- Work on branch `matrix-layer`.

## File Structure

| file | responsibility |
|---|---|
| `src/figlib/matrix.py` (create) | `Block` + encoders + `expr` + checkers. One module: these all speak `Block` geometry and would drift if split. |
| `tests/test_matrix.py` (create) | Geometry arithmetic, encoder item counts, each checker firing and staying silent, SVG assertions. |
| `figures/matrix_four_views.py` (create) | Benchmark 1: structure grammar. |
| `figures/svd_low_rank.py` (create) | Benchmark 2: value encoders + `check_expr` on real arrays. |
| `figures/out/*.svg`, `*.png` (create) | Committed baselines. |
| `figures/out/*.readback.md` (create) | Mandatory readback records. |
| `src/figlib/__init__.py` (modify) | Add `matrix` to the module map docstring. |
| `docs/skill.md` (modify) | Two rows in the device → exemplar index. |
| `docs/primitive-gaps.md` (modify) | Record the gap the VCA survey could not have found. |
| `docs/superpowers/specs/2026-07-29-matrix-layer-design.md` (modify) | Amend §4 — see Task 5. |

---

### Task 1: `Block` — shape as geometry

**Files:**
- Create: `src/figlib/matrix.py`
- Test: `tests/test_matrix.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Block(m, n, origin=(0.0, 0.0), cell=1.0, aspect=1.0, values=None, name="")`, frozen. `cell` is the cell **width**; cell height is `cell * aspect`. Properties `cell_w`, `cell_h`, `width`, `height` (all `float`), `extent: tuple[float,float,float,float]` as `(x0, x1, y0, y1)`. Methods `rect() -> np.ndarray` (4,2); `span(j0, j1, i0, i1) -> np.ndarray` (4,2); `cols(j0, j1=None)`; `rows(i0, i1=None)`; `cell_rect(i, j)`; `center(i, j) -> tuple[float,float]`; `sub(rows: slice, cols: slice) -> Block`; `at(origin, cell=None) -> Block`; `map_into(i, j, pts, *, src=((0,1),(0,1))) -> np.ndarray`; `edge(side, *, pad=0.0) -> np.ndarray` (2,2). Module constant `XY = tuple[float, float]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_matrix.py`:

```python
"""The matrix layer: Block geometry, encoders, the expression row, gates."""

import numpy as np
import pytest

from figlib.matrix import Block


def test_block_extent_puts_row_zero_at_the_top():
    b = Block(2, 3)                      # origin (0,0) is the TOP-LEFT
    assert b.width == 3.0
    assert b.height == 2.0
    assert b.extent == (0.0, 3.0, -2.0, 0.0)
    # row 0 sits above row 1 in math coords (+y up)
    assert b.center(0, 0)[1] > b.center(1, 0)[1]


def test_block_center_and_rect():
    b = Block(2, 3, origin=(10.0, 5.0), cell=2.0)
    assert b.center(0, 0) == (11.0, 4.0)
    assert b.center(1, 2) == (15.0, 2.0)
    r = b.rect()
    assert r.shape == (4, 2)
    assert r[:, 0].min() == 10.0 and r[:, 0].max() == 16.0
    assert r[:, 1].min() == 1.0 and r[:, 1].max() == 5.0


def test_block_aspect_stretches_cells_vertically_only():
    b = Block(2, 3, cell=1.0, aspect=2.5)
    assert b.cell_w == 1.0 and b.cell_h == 2.5
    assert b.width == 3.0 and b.height == 5.0
    assert b.extent == (0.0, 3.0, -5.0, 0.0)
    # square is the default, because shape-as-aspect-ratio depends on it
    assert Block(2, 3).aspect == 1.0


def test_map_into_lands_a_local_frame_in_one_cell_with_v_up():
    b = Block(2, 2, cell=2.0)
    # the unit square maps to cell (0,1): x in [2,4], y in [-2,0]
    corners = np.array([[0.0, 0.0], [1.0, 1.0]])
    out = b.map_into(0, 1, corners)
    assert tuple(out[0]) == (2.0, -2.0)      # (u,v)=(0,0) -> bottom-left
    assert tuple(out[1]) == (4.0, 0.0)       # (u,v)=(1,1) -> top-right


def test_map_into_honors_a_custom_source_frame():
    b = Block(1, 1, cell=1.0)
    # a waveform in u in [0,8], v in [-1,1] fills the cell
    pts = np.array([[0.0, -1.0], [8.0, 1.0], [4.0, 0.0]])
    out = b.map_into(0, 0, pts, src=((0.0, 8.0), (-1.0, 1.0)))
    assert tuple(out[0]) == (0.0, -1.0)
    assert tuple(out[1]) == (1.0, 0.0)
    assert tuple(out[2]) == (0.5, -0.5)      # the midpoint stays central


def test_edge_returns_the_side_segment_with_padding():
    b = Block(2, 3)
    left = b.edge("left")
    assert left.shape == (2, 2)
    assert left[0, 0] == 0.0 and left[1, 0] == 0.0
    assert sorted(left[:, 1]) == [-2.0, 0.0]
    # pad pushes the segment OUTWARD from the block
    assert b.edge("left", pad=0.5)[0, 0] == -0.5
    assert b.edge("right", pad=0.5)[0, 0] == 3.5
    assert b.edge("top", pad=0.5)[0, 1] == 0.5
    assert b.edge("bottom", pad=0.5)[0, 1] == -2.5


def test_block_cols_and_rows_span_the_full_cross_dimension():
    b = Block(2, 3)
    c = b.cols(1)
    assert c[:, 0].min() == 1.0 and c[:, 0].max() == 2.0
    assert c[:, 1].min() == -2.0 and c[:, 1].max() == 0.0
    r = b.rows(0)
    assert r[:, 0].min() == 0.0 and r[:, 0].max() == 3.0
    assert r[:, 1].min() == -1.0 and r[:, 1].max() == 0.0
    # a range form: columns [0, 2)
    c2 = b.cols(0, 2)
    assert c2[:, 0].min() == 0.0 and c2[:, 0].max() == 2.0


def test_block_sub_composes_and_carries_values():
    V = np.arange(12, dtype=float).reshape(3, 4)
    b = Block(3, 4, values=V, name="A")
    s = b.sub(slice(1, 3), slice(2, 4))
    assert (s.m, s.n) == (2, 2)
    assert s.origin == (2.0, -1.0)          # shifted right 2, down 1
    assert np.array_equal(s.values, V[1:3, 2:4])
    # the sub-block's own cell (0,0) is the parent's cell (1,2)
    assert s.center(0, 0) == b.center(1, 2)


def test_block_rejects_a_values_shape_mismatch():
    with pytest.raises(ValueError, match="values shape"):
        Block(2, 3, values=np.zeros((3, 2)))


def test_block_rejects_a_nonpositive_shape():
    with pytest.raises(ValueError, match="positive"):
        Block(0, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `make test` (or `uv run pytest tests/test_matrix.py -v`)
Expected: FAIL — `ModuleNotFoundError: No module named 'figlib.matrix'`

- [ ] **Step 3: Write the implementation**

Create `src/figlib/matrix.py`:

```python
"""Matrices as drawable, gateable objects.

The organizing idea, taken from Hiranabe's *The Art of Linear Algebra*
(graphic notes on Strang): a matrix is a rectangle whose **aspect ratio is
its shape**, and the interesting choice is which of four readings of that
rectangle the argument needs — the opaque whole, the `mn` entries, the `n`
column bands, or the `m` row bands. Every factorization (CR, LU, QR,
QLQ^T, USV^T) then reduces to the same picture: a sum of rank-1 rectangles.

Shape-as-geometry is what makes the layer gateable rather than decorative.
A non-conformable product cannot be drawn to scale, and a `Block` that
carries its `values` makes the picture and the numbers the same object —
so `check_expr` can evaluate the drawn factorization and prove it holds.

This module is a producer of scene items. It imports no theme and names no
color: encoders take a `Role` and an optional `color`, and the figure
program supplies `THEME.categorical(i)` / `THEME.ramp(t)`.

Coordinates: math coords, +y UP, and `Block.origin` is the **top-left**
corner — so row 0 is at the top, matching `RasterField`'s row-0-at-`y1`
rule and ordinary matrix index convention.

`Block` is also the signals & linear-algebra spec's `CellGrid`: the two
were designed independently as the same addressable cell geometry, and
merged rather than left to drift. That is where `map_into`, `edge`,
`grid_lines`, `brackets`, and `cell_fills` come from.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

XY = tuple[float, float]


@dataclass(frozen=True)
class Block:
    """A matrix drawn to shape: `n*cell` wide, `m*cell*aspect` tall.

    Draws nothing — it answers coordinate questions and the author
    composes items against it.

    `aspect` is cell height / cell width. It defaults to 1.0, and that
    default is load-bearing: square cells are what make the drawn
    rectangle's aspect ratio EQUAL the shape (m, n), which is what the
    conformability gates rest on. A basis gallery whose cells hold mini
    waveforms opts out by passing `aspect` explicitly — on the record,
    and `check_square_cells` is how a figure that must not opt out says so.

    `values` is optional. Supplying it unlocks the value encoders (`heat`,
    `cell_fills`, `hinton`, `entries`) and the `check_expr` gate; a purely
    structural figure leaves it None.
    """
    m: int
    n: int
    origin: XY = (0.0, 0.0)            # TOP-left, math coords
    cell: float = 1.0                  # cell WIDTH
    aspect: float = 1.0                # cell height / width
    values: np.ndarray | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if self.m <= 0 or self.n <= 0:
            raise ValueError(
                f"Block shape must be positive, got ({self.m}, {self.n})")
        if self.cell <= 0 or self.aspect <= 0:
            raise ValueError(
                f"Block cell and aspect must be positive, got "
                f"{self.cell} and {self.aspect}")
        if self.values is not None:
            v = np.asarray(self.values)
            if v.shape != (self.m, self.n):
                raise ValueError(
                    f"Block {self.name or '?'}: values shape {v.shape} != "
                    f"({self.m}, {self.n})")

    # --- derived geometry (no scene items) -------------------------------

    @property
    def cell_w(self) -> float:
        return self.cell

    @property
    def cell_h(self) -> float:
        return self.cell * self.aspect

    @property
    def width(self) -> float:
        return self.n * self.cell_w

    @property
    def height(self) -> float:
        return self.m * self.cell_h

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(x0, x1, y0, y1) — named for the RasterField parameter it feeds."""
        ox, oy = self.origin
        return (ox, ox + self.width, oy - self.height, oy)

    def span(self, j0: int, j1: int, i0: int, i1: int) -> np.ndarray:
        """Closed rect (4, 2) over column range [j0, j1) x row range [i0, i1)."""
        ox, oy = self.origin
        x0, x1 = ox + j0 * self.cell_w, ox + j1 * self.cell_w
        y1, y0 = oy - i0 * self.cell_h, oy - i1 * self.cell_h
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)

    def rect(self) -> np.ndarray:
        return self.span(0, self.n, 0, self.m)

    def cols(self, j0: int, j1: int | None = None) -> np.ndarray:
        return self.span(j0, j0 + 1 if j1 is None else j1, 0, self.m)

    def rows(self, i0: int, i1: int | None = None) -> np.ndarray:
        return self.span(0, self.n, i0, i0 + 1 if i1 is None else i1)

    def cell_rect(self, i: int, j: int) -> np.ndarray:
        return self.span(j, j + 1, i, i + 1)

    def center(self, i: int, j: int) -> XY:
        ox, oy = self.origin
        return (ox + (j + 0.5) * self.cell_w, oy - (i + 0.5) * self.cell_h)

    def sub(self, rows: slice, cols: slice) -> "Block":
        """A sub-block: still a Block, so encoders and gates compose on it."""
        ri = range(*rows.indices(self.m))
        ci = range(*cols.indices(self.n))
        ox, oy = self.origin
        return Block(
            len(ri), len(ci),
            origin=(ox + ci.start * self.cell_w, oy - ri.start * self.cell_h),
            cell=self.cell, aspect=self.aspect,
            values=None if self.values is None else self.values[rows, cols],
            name=f"{self.name}[{ri.start}:{ri.stop},{ci.start}:{ci.stop}]"
                 if self.name else "")

    def at(self, origin: XY, cell: float | None = None) -> "Block":
        """The same matrix placed elsewhere — what `expr` uses to lay out."""
        return replace(self, origin=(float(origin[0]), float(origin[1])),
                       cell=self.cell if cell is None else float(cell))

    def map_into(self, i: int, j: int, pts: np.ndarray, *,
                 src: tuple[tuple[float, float], tuple[float, float]]
                 = ((0.0, 1.0), (0.0, 1.0))) -> np.ndarray:
        """Affine-map points from a local (u, v) frame into cell (i, j),
        with v UP within the cell.

        The inset bridge: compute a mini stem plot, waveform, or anything
        else in whatever local coordinates it is natural in, map it, emit
        it. This is what makes "basis gallery" a recipe rather than a
        primitive — the cell is just a viewport.
        """
        (u0, u1), (v0, v1) = src
        du = (u1 - u0) or 1.0
        dv = (v1 - v0) or 1.0
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        r = self.span(j, j + 1, i, i + 1)   # [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
        x0, y0 = float(r[0, 0]), float(r[0, 1])
        return np.column_stack([
            x0 + (p[:, 0] - u0) / du * self.cell_w,
            y0 + (p[:, 1] - v0) / dv * self.cell_h,
        ])

    def edge(self, side: str, *, pad: float = 0.0) -> np.ndarray:
        """The (2, 2) segment along one side, `pad` pushed OUTWARD.

        The anchor line for brackets, index labels, dimension braces.
        """
        x0, x1, y0, y1 = self.extent
        if side == "left":
            return np.array([[x0 - pad, y1], [x0 - pad, y0]], dtype=float)
        if side == "right":
            return np.array([[x1 + pad, y1], [x1 + pad, y0]], dtype=float)
        if side == "top":
            return np.array([[x0, y1 + pad], [x1, y1 + pad]], dtype=float)
        if side == "bottom":
            return np.array([[x0, y0 - pad], [x1, y0 - pad]], dtype=float)
        raise ValueError(
            f"edge side must be left|right|top|bottom, got {side!r}")


def _nm(b: Block) -> str:
    return b.name or f"{b.m}x{b.n}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make test`
Expected: PASS — 10 tests in `tests/test_matrix.py`, no regressions elsewhere.

- [ ] **Step 5: Commit**

```bash
git add src/figlib/matrix.py tests/test_matrix.py
git commit -m "matrix: Block — a matrix drawn at its own aspect ratio

Frozen dataclass carrying (m, n) and optional values; all geometry
derived, none emitted, so render and gates cannot drift. Origin is
top-left so row 0 is at the top, matching RasterField.

Merges the signals spec's CellGrid: same object, same module, designed
twice. map_into and edge come from there. Its cell (width, height)
becomes scalar cell + explicit aspect, because square cells are what
make the drawn aspect ratio equal the shape."
```

---

### Task 2: Structure encoders

**Files:**
- Modify: `src/figlib/matrix.py` (append)
- Test: `tests/test_matrix.py` (append)

**Interfaces:**
- Consumes: `Block` and its geometry methods from Task 1.
- Produces:
  - `outline(b, *, role=Role.CONTENT, wash=0.0) -> list`
  - `bands(b, axis="col", *, color=None, role=Role.CONTENT, opacity=1.0, gap=0.0) -> list[FilledCurve]`
  - `lattice(b, *, color=None, role=Role.CONTENT, radius_scale=1.0) -> list[Point]`
  - `mask(b, M, *, color=None, role=Role.CONTENT, opacity=1.0) -> list[FilledCurve]` where `M` is a `(m, n)` bool array or a callable `(i, j) -> bool`
  - `tri(b, side="lower", k=0) -> np.ndarray`
  - `banded(b, lo, hi) -> np.ndarray`
  - `diagonal(b, offset=0, *, wrap=False) -> np.ndarray`
  - `causal(b) -> np.ndarray`
  - `grid_lines(b, *, role=Role.FRAME, inner=True, outer=True) -> list[Curve]`
  - `brackets(b, *, pad=None, tick=None, role=Role.ANNOTATION) -> list[Curve]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_matrix.py`:

```python
from figlib.matrix import (bands, banded, brackets, causal, diagonal,
                           grid_lines, lattice, mask, outline, tri)
from figlib.scene import FilledCurve, Point
from figlib.style import Role


def test_outline_emits_a_closed_rim_and_an_optional_wash():
    b = Block(2, 3)
    assert len(outline(b)) == 1
    items = outline(b, wash=0.3)
    assert len(items) == 2
    assert isinstance(items[0], FilledCurve)      # wash first, under the rim
    assert items[0].opacity == 0.3
    assert items[1].closed is True


def test_bands_emit_one_fill_per_column_or_row():
    b = Block(2, 3)
    cols = bands(b, "col")
    rows = bands(b, "row")
    assert len(cols) == 3 and len(rows) == 2
    assert all(isinstance(f, FilledCurve) for f in cols)
    # a column band is full height, one cell wide
    c = cols[0].pts
    assert c[:, 0].max() - c[:, 0].min() == 1.0
    assert c[:, 1].max() - c[:, 1].min() == 2.0


def test_bands_gap_insets_each_band_without_moving_its_center():
    b = Block(2, 3)
    c = bands(b, "col", gap=0.2)[1].pts
    assert c[:, 0].min() == pytest.approx(1.1)
    assert c[:, 0].max() == pytest.approx(1.9)


def test_lattice_emits_one_dot_per_entry_at_cell_centers():
    b = Block(2, 3)
    dots = lattice(b)
    assert len(dots) == 6
    assert all(isinstance(d, Point) for d in dots)
    assert dots[0].xy == b.center(0, 0)


def test_mask_merges_consecutive_true_cells_in_a_row():
    b = Block(2, 3)
    M = np.array([[True, True, True], [False, True, False]])
    fills = mask(b, M)
    # row 0 is one merged run of three, row 1 is one single cell
    assert len(fills) == 2
    wide = fills[0].pts
    assert wide[:, 0].max() - wide[:, 0].min() == 3.0


def test_mask_accepts_a_predicate():
    b = Block(2, 2)
    fills = mask(b, lambda i, j: i == j)
    assert len(fills) == 2


def test_tri_banded_and_causal_are_masks_over_the_index_grid():
    b = Block(3, 3)
    assert tri(b, "lower").sum() == 6
    assert tri(b, "upper").sum() == 6
    assert banded(b, 0, 0).sum() == 3          # the diagonal
    assert banded(b, -1, 1).sum() == 7         # tridiagonal
    assert np.array_equal(causal(b), tri(b, "lower"))


def test_diagonal_wraps_for_a_circulant_portrait():
    b = Block(4, 4)
    assert diagonal(b, 0).sum() == 4
    assert diagonal(b, 1).sum() == 3           # the superdiagonal is short
    # wrapped, EVERY diagonal of a square block has exactly n cells —
    # which is what makes a circulant portrait a one-liner
    assert diagonal(b, 1, wrap=True).sum() == 4
    assert diagonal(b, -3, wrap=True).sum() == 4
    # np.argwhere is the index-pair form, for placing markers per cell
    assert np.argwhere(diagonal(b, 0)).shape == (4, 2)


def test_grid_lines_counts_inner_rules_and_the_outer_frame():
    b = Block(2, 3)
    inner = grid_lines(b, outer=False)
    assert len(inner) == 1 + 2                 # 1 horizontal, 2 vertical
    both = grid_lines(b)
    assert len(both) == len(inner) + 1         # + one closed outer rect
    assert both[-1].closed is True


def test_brackets_emit_two_three_segment_glyphs_flanking_the_block():
    b = Block(2, 3)
    left, right = brackets(b, pad=0.1, tick=0.2)
    assert left.pts.shape == (4, 2) and right.pts.shape == (4, 2)
    # the left bracket's spine sits left of the block, ticks point right
    assert left.pts[:, 0].min() == pytest.approx(-0.1)
    assert left.pts[:, 0].max() == pytest.approx(0.1)
    assert right.pts[:, 0].max() == pytest.approx(3.1)
    assert right.pts[:, 0].min() == pytest.approx(2.9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_matrix.py -v`
Expected: FAIL — `ImportError: cannot import name 'outline' from 'figlib.matrix'`

- [ ] **Step 3: Write the implementation**

Append to `src/figlib/matrix.py` (and extend the import line at the top):

```python
from typing import Callable, Sequence

from .scene import Curve, FilledCurve, MathLabel, Point, RasterField
from .style import Role

Mask = np.ndarray | Callable[[int, int], bool]


# --- structure encoders --------------------------------------------------
# The four readings of one rectangle. Which one you draw IS the argument.
# Color is the caller's: pass THEME.categorical(0) for columns,
# categorical(1) for rows, categorical(2) for entries — the fixed
# correspondence order, four slots on RISO.

def outline(b: Block, *, role: Role = Role.CONTENT,
            wash: float = 0.0) -> list:
    """The matrix as one whole: a closed rim, optionally over a MUTED wash.

    `wash` is the "unstructured whole" reading — a matrix with no internal
    structure yet asserted.
    """
    items: list = []
    if wash > 0.0:
        items.append(FilledCurve(b.rect(), role=Role.MUTED, opacity=wash,
                                 outline=False))
    items.append(Curve(b.rect(), role=role, closed=True))
    return items


def _inset(pts: np.ndarray, d: float) -> np.ndarray:
    """Shrink an axis-aligned rect by d/2 on every side."""
    if d <= 0.0:
        return pts
    cx = (pts[:, 0].min() + pts[:, 0].max()) / 2.0
    cy = (pts[:, 1].min() + pts[:, 1].max()) / 2.0
    hx = max((pts[:, 0].max() - pts[:, 0].min()) / 2.0 - d / 2.0, 1e-9)
    hy = max((pts[:, 1].max() - pts[:, 1].min()) / 2.0 - d / 2.0, 1e-9)
    return np.array([[cx - hx, cy - hy], [cx + hx, cy - hy],
                     [cx + hx, cy + hy], [cx - hx, cy + hy]], dtype=float)


def bands(b: Block, axis: str = "col", *, color: str | None = None,
          role: Role = Role.CONTENT, opacity: float = 1.0,
          gap: float = 0.0) -> list[FilledCurve]:
    """The matrix as `n` column vectors or `m` row vectors.

    `gap` insets each band (math units) so the bands read as separate
    vectors rather than as a solid block.
    """
    if axis not in ("col", "row"):
        raise ValueError(f"bands axis must be 'col' or 'row', got {axis!r}")
    k = b.n if axis == "col" else b.m
    return [FilledCurve(
                _inset(b.cols(t) if axis == "col" else b.rows(t), gap),
                role=role, color=color, opacity=opacity, outline=False)
            for t in range(k)]


def lattice(b: Block, *, color: str | None = None,
            role: Role = Role.CONTENT,
            radius_scale: float = 1.0) -> list[Point]:
    """The matrix as `mn` scalars: a dot at every cell center."""
    return [Point(b.center(i, j), role=role, color=color,
                  radius_scale=radius_scale)
            for i in range(b.m) for j in range(b.n)]


def _as_array(b: Block, M: Mask) -> np.ndarray:
    if callable(M):
        return np.array([[bool(M(i, j)) for j in range(b.n)]
                         for i in range(b.m)], dtype=bool)
    A = np.asarray(M, dtype=bool)
    if A.shape != (b.m, b.n):
        raise ValueError(
            f"mask shape {A.shape} != block shape ({b.m}, {b.n})")
    return A


def mask(b: Block, M: Mask, *, color: str | None = None,
         role: Role = Role.CONTENT,
         opacity: float = 1.0) -> list[FilledCurve]:
    """Structure as the silhouette of a filled region — never as printed
    zeros. Triangular, banded, sparsity, causal-attention masks.

    Consecutive true cells within a row merge into one rect: fewer items,
    and no hairline seams breaking up what should read as one staircase.
    """
    A = _as_array(b, M)
    out: list[FilledCurve] = []
    for i in range(b.m):
        j = 0
        while j < b.n:
            if not A[i, j]:
                j += 1
                continue
            j0 = j
            while j < b.n and A[i, j]:
                j += 1
            out.append(FilledCurve(b.span(j0, j, i, i + 1), role=role,
                                   color=color, opacity=opacity,
                                   outline=False))
    return out


# Mask recipes: three lines each, built on `mask`. Idioms, not primitives.

def tri(b: Block, side: str = "lower", k: int = 0) -> np.ndarray:
    """Triangular mask; `k` shifts the diagonal (k=-1 excludes it for lower)."""
    i, j = np.indices((b.m, b.n))
    return (j - i <= k) if side == "lower" else (j - i >= k)


def banded(b: Block, lo: int, hi: int) -> np.ndarray:
    """Diagonals `lo <= j - i <= hi`. banded(b, 0, 0) is the diagonal."""
    i, j = np.indices((b.m, b.n))
    return (j - i >= lo) & (j - i <= hi)


def diagonal(b: Block, offset: int = 0, *, wrap: bool = False) -> np.ndarray:
    """The k-th diagonal. `wrap` closes it modulo n — the circulant case,
    where every diagonal has exactly n cells and a Toeplitz portrait
    becomes a one-liner. `np.argwhere` on the result gives index pairs,
    for a call site placing one marker per cell.
    """
    i, j = np.indices((b.m, b.n))
    if not wrap:
        return (j - i) == offset
    return ((j - i) % b.n) == (offset % b.n)


def causal(b: Block) -> np.ndarray:
    """The attention mask: position i may attend to j <= i."""
    return tri(b, "lower", 0)


# --- cell furniture (from the signals spec) ------------------------------

def grid_lines(b: Block, *, role: Role = Role.FRAME, inner: bool = True,
               outer: bool = True) -> list[Curve]:
    """Ruled cells: `m-1` horizontal and `n-1` vertical inner rules, plus
    the closed outer rect. The furniture that makes a small matrix read as
    a table of entries rather than a filled region."""
    x0, x1, y0, y1 = b.extent
    out: list[Curve] = []
    if inner:
        for i in range(1, b.m):
            y = y1 - i * b.cell_h
            out.append(Curve(np.array([[x0, y], [x1, y]]), role=role))
        for j in range(1, b.n):
            x = x0 + j * b.cell_w
            out.append(Curve(np.array([[x, y0], [x, y1]]), role=role))
    if outer:
        out.append(Curve(b.rect(), role=role, closed=True))
    return out


def brackets(b: Block, *, pad: float | None = None,
             tick: float | None = None,
             role: Role = Role.ANNOTATION) -> list[Curve]:
    """The two square-bracket glyphs, as 3-segment Curves flanking the
    block. `pad` is the gap from the block edge, `tick` the length of the
    horizontal serifs; both default to a fraction of the cell width."""
    p = 0.25 * b.cell_w if pad is None else pad
    t = 0.30 * b.cell_w if tick is None else tick
    x0, x1, y0, y1 = b.extent
    left = np.array([[x0 - p + t, y1], [x0 - p, y1], [x0 - p, y0],
                     [x0 - p + t, y0]], dtype=float)
    right = np.array([[x1 + p - t, y1], [x1 + p, y1], [x1 + p, y0],
                      [x1 + p - t, y0]], dtype=float)
    return [Curve(left, role=role), Curve(right, role=role)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make test`
Expected: PASS — 20 tests in `tests/test_matrix.py`.

- [ ] **Step 5: Commit**

```bash
git add src/figlib/matrix.py tests/test_matrix.py
git commit -m "matrix: structure encoders — outline, bands, lattice, mask, furniture

The four readings of one rectangle (Hiranabe's grammar). mask merges
runs within a row so a triangular silhouette reads as a staircase, not
a grid of squares; tri/banded/diagonal/causal are recipes on it, not
primitives. grid_lines and brackets come from the merged signals spec."
```

---

### Task 3: `rank1` and the value encoders

**Files:**
- Modify: `src/figlib/matrix.py` (append)
- Test: `tests/test_matrix.py` (append)

**Interfaces:**
- Consumes: `Block`, `_inset` from Tasks 1–2.
- Produces:
  - `rank1(b, j, i, *, col_color=None, row_color=None, role=Role.CONTENT, ground=0.12) -> list`
  - `heat(b, *, ramp=None, vmin=None, vmax=None, opacity=1.0, interp=False) -> RasterField`
  - `cell_fills(b, *, ramp, vmin=None, vmax=None, role=Role.CONTENT, opacity=1.0) -> list[FilledCurve]`
  - `hinton(b, *, vmax=None, pos_role=Role.ACCENT1, neg_role=Role.ACCENT2, max_frac=0.92) -> list[FilledCurve]`
  - `entries(b, *, fmt="{:.2g}", role=Role.ANNOTATION, size_pt=None) -> list[MathLabel]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_matrix.py`:

```python
from figlib.matrix import cell_fills, entries, heat, hinton, rank1
from figlib.scene import MathLabel


def test_rank1_paints_a_ground_crossed_by_one_column_and_one_row():
    b = Block(3, 3)
    items = rank1(b, 1, 2)
    assert len(items) == 3
    ground, col, row = items
    assert ground.opacity == pytest.approx(0.12)
    assert col.pts[:, 0].min() == 1.0 and col.pts[:, 0].max() == 2.0
    assert row.pts[:, 1].min() == -3.0 and row.pts[:, 1].max() == -2.0


def test_heat_covers_exactly_the_block_extent():
    V = np.arange(6, dtype=float).reshape(2, 3)
    b = Block(2, 3, values=V)
    r = heat(b)
    assert r.extent == b.extent
    assert np.array_equal(r.values, V)


def test_heat_needs_values():
    with pytest.raises(ValueError, match="carries no values"):
        heat(Block(2, 3))


def test_cell_fills_is_the_vector_path_one_fill_per_cell():
    V = np.array([[0.0, 1.0], [2.0, 3.0]])
    b = Block(2, 2, values=V)
    fills = cell_fills(b, ramp=lambda t: f"#{int(255 * t):02x}0000")
    assert len(fills) == 4
    # normalized over the data range: min -> t=0, max -> t=1
    assert fills[0].color == "#000000"
    assert fills[-1].color == "#ff0000"
    assert np.array_equal(fills[0].pts, b.cell_rect(0, 0))


def test_cell_fills_honors_an_explicit_range():
    b = Block(1, 2, values=np.array([[0.0, 1.0]]))
    seen = []
    cell_fills(b, ramp=lambda t: seen.append(t) or "#000000",
               vmin=-1.0, vmax=1.0)
    assert seen == [0.5, 1.0]


def test_hinton_encodes_magnitude_as_area_not_as_side():
    # a 4x magnitude ratio must give a 4x AREA ratio, i.e. a 2x side ratio
    b = Block(1, 2, values=np.array([[1.0, 4.0]]))
    sq = hinton(b)
    assert len(sq) == 2
    side = [float(s.pts[:, 0].max() - s.pts[:, 0].min()) for s in sq]
    assert side[1] / side[0] == pytest.approx(2.0)


def test_hinton_splits_sign_across_two_roles_and_skips_zeros():
    b = Block(1, 3, values=np.array([[1.0, -1.0, 0.0]]))
    sq = hinton(b)
    assert len(sq) == 2                       # the zero emits nothing
    assert sq[0].role is Role.ACCENT1
    assert sq[1].role is Role.ACCENT2


def test_entries_labels_every_cell_at_its_center():
    b = Block(2, 2, values=np.array([[1.0, 2.0], [3.0, 4.0]]))
    labs = entries(b)
    assert len(labs) == 4
    assert all(isinstance(el, MathLabel) for el in labs)
    assert labs[0].anchor == b.center(0, 0)
    assert labs[0].latex == "1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_matrix.py -v`
Expected: FAIL — `ImportError: cannot import name 'rank1' from 'figlib.matrix'`

- [ ] **Step 3: Write the implementation**

Append to `src/figlib/matrix.py`:

```python
import math


def rank1(b: Block, j: int, i: int, *, col_color: str | None = None,
          row_color: str | None = None, role: Role = Role.CONTENT,
          ground: float = 0.12) -> list:
    """The outer product mark: the whole rectangle as a faint ground,
    crossed by column `j` and row `i`.

    This is the one mark every factorization reduces to. In A = sum_k
    a_k b_k^T, term k is drawn on a result-shaped block with column k of
    the left factor and row k of the right factor marked, so the reader
    can see *which* column and row produced this summand.
    """
    return [
        FilledCurve(b.rect(), role=Role.MUTED, opacity=ground, outline=False),
        FilledCurve(b.cols(j), role=role, color=col_color, opacity=1.0,
                    outline=False),
        FilledCurve(b.rows(i), role=role, color=row_color, opacity=1.0,
                    outline=False),
    ]


# --- value encoders ------------------------------------------------------

def _values(b: Block) -> np.ndarray:
    if b.values is None:
        raise ValueError(f"Block {_nm(b)} carries no values")
    return np.asarray(b.values, dtype=float)


def heat(b: Block, *, ramp=None, vmin: float | None = None,
         vmax: float | None = None, opacity: float = 1.0,
         interp: bool = False) -> RasterField:
    """Magnitude as the ordered ramp, over exactly the block's rectangle.

    Row 0 renders at the top in both conventions, so cell (i, j) of the
    array is cell (i, j) of the Block — a structure overlay (`mask`,
    `bands`) lands on the entries it describes.
    """
    return RasterField(_values(b), extent=b.extent, ramp=ramp, vmin=vmin,
                       vmax=vmax, opacity=opacity, interp=interp)


def cell_fills(b: Block, *, ramp, vmin: float | None = None,
               vmax: float | None = None, role: Role = Role.CONTENT,
               opacity: float = 1.0) -> list[FilledCurve]:
    """The same encoding as `heat`, one FilledCurve per cell instead of one
    raster.

    Two paths, and the author chooses: a 32x32 attention map wants `heat`
    (no vector substitute at that density), an 8x8 wants `cell_fills`,
    whose cells stay vector — individually addressable, crisp at any zoom,
    and visible to the color gate. Emission is row-major.
    """
    V = _values(b)
    lo = float(V.min()) if vmin is None else float(vmin)
    hi = float(V.max()) if vmax is None else float(vmax)
    d = (hi - lo) or 1.0
    return [FilledCurve(b.cell_rect(i, j), role=role,
                        color=ramp(float((V[i, j] - lo) / d)),
                        opacity=opacity, outline=False)
            for i in range(b.m) for j in range(b.n)]


def hinton(b: Block, *, vmax: float | None = None,
           pos_role: Role = Role.ACCENT1, neg_role: Role = Role.ACCENT2,
           max_frac: float = 0.92) -> list[FilledCurve]:
    """One square per nonzero entry: AREA proportional to |v|, fill by sign.

    A monotone ramp cannot carry sign — that is the whole reason this
    encoding survives for small signed matrices. Area, not side: side
    proportional to |v| squares the encoding and badly understates small
    entries, which is the classic implementation bug (`check_hinton_area`
    exists to catch it).

    Emission order is row-major over nonzero entries; the gate relies on it.
    """
    V = _values(b)
    peak = float(np.abs(V).max()) if vmax is None else float(vmax)
    if peak <= 0.0:
        return []
    out: list[FilledCurve] = []
    for i in range(b.m):
        for j in range(b.n):
            v = float(V[i, j])
            if v == 0.0:
                continue
            side = b.cell * max_frac * math.sqrt(min(abs(v) / peak, 1.0))
            cx, cy = b.center(i, j)
            h = side / 2.0
            out.append(FilledCurve(
                np.array([[cx - h, cy - h], [cx + h, cy - h],
                          [cx + h, cy + h], [cx - h, cy + h]], dtype=float),
                role=pos_role if v > 0 else neg_role,
                opacity=1.0, outline=False))
    return out


def entries(b: Block, *, fmt: str = "{:.2g}",
            role: Role = Role.ANNOTATION,
            size_pt: float | None = None) -> list[MathLabel]:
    """The numbers themselves. Only honest for small m, n — the mechanical
    gate's annotation-load check is what tells you when you have overrun it.
    """
    V = _values(b)
    return [MathLabel(fmt.format(float(V[i, j])), b.center(i, j),
                      role=role, ha="center", va="center", size_pt=size_pt)
            for i in range(b.m) for j in range(b.n)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make test`
Expected: PASS — 28 tests in `tests/test_matrix.py`.

- [ ] **Step 5: Commit**

```bash
git add src/figlib/matrix.py tests/test_matrix.py
git commit -m "matrix: rank1 mark and the value encoders

rank1 is the mark every factorization reduces to. heat wires RasterField
to Block extent (row 0 at top in both, so overlays land on the right
cells); cell_fills is the same encoding at vector density for small
matrices; hinton carries sign, which no monotone ramp can, with area —
not side — proportional to |v|."
```

---

### Task 4: The expression row

**Files:**
- Modify: `src/figlib/matrix.py` (append)
- Test: `tests/test_matrix.py` (append)

**Interfaces:**
- Consumes: `Block`, `Role`, `MathLabel`.
- Produces:
  - `OPERATORS: dict[str, str]` — `{"=": "=", "+": "+", "-": "-", "@": r"\cdot", "→": r"\mapsto"}`
  - `expr(terms, *, cell=1.0, gap=0.5, op_gap=1.2, origin=(0.0, 0.0), op_role=Role.ANNOTATION, op_size_pt=None) -> tuple[list[MathLabel], list[Block]]`
  - `bounds(blocks, *, pad=0.0) -> tuple[tuple[float,float], tuple[float,float]]`

`terms` is a `Sequence[Block | str]`. The returned Blocks are the **placed** ones — origins and `cell` are overridden — and are what the figure program draws into and hands to the gates.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_matrix.py`:

```python
from figlib.matrix import bounds, expr


def test_expr_places_blocks_left_to_right_on_a_common_center():
    A, B = Block(3, 2, name="A"), Block(2, 4, name="B")
    ops, placed = expr([A, "@", B, "=", Block(3, 4, name="C")],
                       cell=1.0, gap=0.5, op_gap=1.0)
    assert [p.name for p in placed] == ["A", "B", "C"]
    # left to right, no overlap
    assert placed[0].extent[1] <= placed[1].extent[0]
    assert placed[1].extent[1] <= placed[2].extent[0]
    # vertically centered on y = 0 regardless of m
    for p in placed:
        assert (p.extent[2] + p.extent[3]) / 2 == pytest.approx(0.0)


def test_expr_forces_one_shared_cell_so_inner_dimensions_line_up():
    A = Block(3, 2, cell=9.0, name="A")     # a stale cell must not survive
    B = Block(2, 4, cell=0.1, name="B")
    _, placed = expr([A, "@", B], cell=2.0)
    assert all(p.cell == 2.0 for p in placed)
    # A's 2 columns and B's 2 rows are drawn at the same physical size
    assert placed[0].width / placed[0].n == placed[1].height / placed[1].m


def test_expr_emits_one_label_per_operator():
    ops, placed = expr([Block(2, 2), "=", Block(2, 2), "+", Block(2, 2)])
    assert len(ops) == 2
    assert [o.latex for o in ops] == ["=", "+"]
    assert all(isinstance(o, MathLabel) for o in ops)
    # each operator sits between the blocks it joins
    assert placed[0].extent[1] <= ops[0].anchor[0] <= placed[1].extent[0]


def test_expr_preserves_values_through_placement():
    V = np.ones((2, 2))
    _, placed = expr([Block(2, 2, values=V, name="A")])
    assert np.array_equal(placed[0].values, V)


def test_bounds_covers_every_block_with_padding():
    _, placed = expr([Block(2, 2), "=", Block(2, 2)], cell=1.0)
    xlim, ylim = bounds(placed, pad=0.5)
    assert xlim[0] == pytest.approx(placed[0].extent[0] - 0.5)
    assert xlim[1] == pytest.approx(placed[-1].extent[1] + 0.5)
    assert ylim == pytest.approx((-1.5, 1.5))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_matrix.py -v`
Expected: FAIL — `ImportError: cannot import name 'expr' from 'figlib.matrix'`

- [ ] **Step 3: Write the implementation**

Append to `src/figlib/matrix.py`:

```python
# --- the expression row --------------------------------------------------

#: operator token -> the LaTeX drawn for it. An unknown token is passed
#: through verbatim, so a figure can write its own.
OPERATORS: dict[str, str] = {
    "=": "=", "+": "+", "-": "-", "@": r"\cdot", "→": r"\mapsto",
    "≈": r"\approx",
}


def expr(terms: Sequence[Block | str], *, cell: float = 1.0,
         gap: float = 0.5, op_gap: float = 1.2, origin: XY = (0.0, 0.0),
         op_role: Role = Role.ANNOTATION,
         op_size_pt: float | None = None
         ) -> tuple[list[MathLabel], list[Block]]:
    """Blocks and operator tokens laid left to right in ONE coordinate
    system, on a common vertical center. Returns (operator labels, placed
    Blocks).

    The placed Blocks are what the figure draws into and hands to the
    gates: `origin` and `cell` on the inputs are overridden, deliberately.
    One shared `cell` is the load-bearing property — it is what makes A's
    column count and B's row count line up *geometrically*, so a
    non-conformable product looks wrong on the page and not merely in the
    assertion.

    This is why the row is one Scene and not a `Figure` of `Panel`s:
    panels carry independent transforms, which would let a 3x2 and a 2x4
    render at different cell sizes and silently destroy that property.

    `origin` is the left edge at the vertical center of the row.
    """
    ox, cy = float(origin[0]), float(origin[1])
    x = ox
    ops: list[MathLabel] = []
    placed: list[Block] = []
    for k, t in enumerate(terms):
        if isinstance(t, Block):
            b = t.at((x, cy + t.m * cell / 2.0), cell=cell)
            placed.append(b)
            x += b.width
        else:
            ops.append(MathLabel(OPERATORS.get(t, t),
                                 (x + op_gap / 2.0, cy), role=op_role,
                                 ha="center", va="center",
                                 size_pt=op_size_pt))
            x += op_gap
        if k < len(terms) - 1:
            x += gap
    return ops, placed


def bounds(blocks: Sequence[Block], *, pad: float = 0.0
           ) -> tuple[tuple[float, float], tuple[float, float]]:
    """(xlim, ylim) covering every block, padded — for Scene lims."""
    if not blocks:
        raise ValueError("bounds() needs at least one block")
    xs = [v for b in blocks for v in (b.extent[0], b.extent[1])]
    ys = [v for b in blocks for v in (b.extent[2], b.extent[3])]
    return ((min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make test`
Expected: PASS — 33 tests in `tests/test_matrix.py`.

- [ ] **Step 5: Commit**

```bash
git add src/figlib/matrix.py tests/test_matrix.py
git commit -m "matrix: expr — the equation of figures in one coordinate system

Blocks and operator tokens left to right on a common center, sharing one
cell. The shared cell is the point: it makes inner dimensions line up
geometrically, which Panels (independent transforms) could not do."
```

---

### Task 5: The gates

**Files:**
- Modify: `src/figlib/matrix.py` (append)
- Modify: `docs/superpowers/specs/2026-07-29-matrix-layer-design.md` (§4)
- Test: `tests/test_matrix.py` (append)

**Interfaces:**
- Consumes: `Block`, `expr`, `hinton` from Tasks 1–4; `gates.Checks` (`.check(cond, msg)`, `.done()`).
- Produces:
  - `check_conformable(checks, terms) -> None`
  - `check_cell_uniform(checks, blocks) -> None`
  - `check_square_cells(checks, blocks) -> None`
  - `check_no_overlap(checks, blocks) -> None`
  - `check_expr(checks, terms, *, rtol=1e-9, atol=0.0) -> None`
  - `check_hinton_area(checks, b, squares, *, rtol=1e-6) -> None`

**Why there is no `check_shape_faithful`.** The spec originally named one
(drawn aspect ratio equals `m/n`). With `width = n*cell` and
`height = m*cell*aspect`, that is true by construction whenever
`aspect == 1`, so asserting it would be gate theater — precisely what
`docs/skill.md` forbids. It is replaced by three checks that *can* fail:
`check_cell_uniform` (a hand-placed block next to `expr`-placed ones
renders at a different scale, so the picture lies about relative size),
`check_square_cells` (a figure arguing about dimensions drawn with the
gallery's tall cells), and `check_no_overlap` (a `gap`/`op_gap` too small
silently collides two matrices). The spec is already amended to match — no
edit is needed here.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_matrix.py`:

```python
from figlib.gates import Checks
from figlib.matrix import (check_cell_uniform, check_conformable, check_expr,
                           check_hinton_area, check_no_overlap,
                           check_square_cells)


def _fails(fn) -> list[str]:
    c = Checks()
    fn(c)
    return c.failures


def test_check_conformable_is_silent_on_a_valid_product():
    terms = [Block(3, 2, name="A"), "@", Block(2, 4, name="B"), "=",
             Block(3, 4, name="C")]
    assert _fails(lambda c: check_conformable(c, terms)) == []


def test_check_conformable_catches_a_mismatched_inner_dimension():
    terms = [Block(3, 2, name="A"), "@", Block(3, 4, name="B")]
    msgs = _fails(lambda c: check_conformable(c, terms))
    assert any("non-conformable" in m and "inner 2 != 3" in m for m in msgs)


def test_check_conformable_catches_summands_of_different_shape():
    terms = [Block(2, 2), "=", Block(2, 2), "+", Block(2, 3)]
    msgs = _fails(lambda c: check_conformable(c, terms))
    assert any("summands differ" in m for m in msgs)


def test_check_conformable_catches_unequal_sides_of_the_equals():
    terms = [Block(3, 4), "=", Block(3, 5)]
    msgs = _fails(lambda c: check_conformable(c, terms))
    assert any("sides differ" in m for m in msgs)


def test_check_cell_uniform_catches_a_block_placed_at_another_scale():
    _, placed = expr([Block(2, 2), "=", Block(2, 2)], cell=1.0)
    assert _fails(lambda c: check_cell_uniform(c, placed)) == []
    rogue = placed + [Block(2, 2, origin=(20.0, 0.0), cell=0.4, name="R")]
    msgs = _fails(lambda c: check_cell_uniform(c, rogue))
    assert any("cell" in m and "R" in m for m in msgs)


def test_check_square_cells_catches_a_shape_claim_drawn_with_tall_cells():
    assert _fails(lambda c: check_square_cells(c, [Block(2, 3)])) == []
    msgs = _fails(lambda c:
                  check_square_cells(c, [Block(2, 3, aspect=2.5, name="G")]))
    assert any("aspect" in m and "G" in m for m in msgs)


def test_check_cell_uniform_also_compares_aspect():
    blocks = [Block(2, 2), Block(2, 2, origin=(5.0, 0.0), aspect=2.0,
                                 name="tall")]
    msgs = _fails(lambda c: check_cell_uniform(c, blocks))
    assert any("aspect" in m for m in msgs)


def test_check_no_overlap_catches_a_gap_too_small_for_the_operator():
    _, placed = expr([Block(2, 2), "=", Block(2, 2)], cell=1.0)
    assert _fails(lambda c: check_no_overlap(c, placed)) == []
    stacked = [Block(2, 2, name="A"), Block(2, 2, origin=(1.0, 0.0), name="B")]
    msgs = _fails(lambda c: check_no_overlap(c, stacked))
    assert any("overlap" in m for m in msgs)


def test_check_expr_proves_a_true_factorization():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 3))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    terms = [Block(5, 3, values=A, name="A"), "=",
             Block(5, 3, values=U, name="U"), "@",
             Block(3, 3, values=np.diag(s), name="S"), "@",
             Block(3, 3, values=Vt, name="Vt")]
    assert _fails(lambda c: check_expr(c, terms)) == []


def test_check_expr_catches_a_perturbed_factorization():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 3))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s[0] *= 1.01                                  # a 1% lie
    terms = [Block(5, 3, values=A, name="A"), "=",
             Block(5, 3, values=U, name="U"), "@",
             Block(3, 3, values=np.diag(s), name="S"), "@",
             Block(3, 3, values=Vt, name="Vt")]
    msgs = _fails(lambda c: check_expr(c, terms))
    assert any("does not hold" in m for m in msgs)


def test_check_expr_sums_over_plus():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 3))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    terms = [Block(4, 3, values=A, name="A"), "="]
    for k in range(3):
        terms += [Block(4, 3, values=s[k] * np.outer(U[:, k], Vt[k, :]),
                        name=f"s{k}"), "+"]
    terms.pop()                                   # drop the trailing '+'
    assert _fails(lambda c: check_expr(c, terms)) == []


def test_check_expr_refuses_blocks_without_values():
    terms = [Block(2, 2), "=", Block(2, 2)]
    msgs = _fails(lambda c: check_expr(c, terms))
    assert any("carries no values" in m for m in msgs)


def test_check_hinton_area_is_silent_on_a_correct_diagram():
    b = Block(2, 2, values=np.array([[1.0, -4.0], [9.0, 0.0]]))
    sq = hinton(b)
    assert _fails(lambda c: check_hinton_area(c, b, sq)) == []


def test_check_hinton_area_catches_side_proportional_to_magnitude():
    from figlib.scene import FilledCurve as FC

    b = Block(1, 2, values=np.array([[1.0, 4.0]]))
    bad = []
    for j, v in ((0, 1.0), (1, 4.0)):
        cx, cy = b.center(0, j)
        h = 0.1 * v                          # side ∝ |v| — the classic bug
        bad.append(FC(np.array([[cx - h, cy - h], [cx + h, cy - h],
                                [cx + h, cy + h], [cx - h, cy + h]])))
    msgs = _fails(lambda c: check_hinton_area(c, b, bad))
    assert any("area/|v| not constant" in m for m in msgs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_matrix.py -v`
Expected: FAIL — `ImportError: cannot import name 'check_conformable' from 'figlib.matrix'`

- [ ] **Step 3: Write the implementation**

Append to `src/figlib/matrix.py`:

```python
# --- gates ---------------------------------------------------------------
# Called from a figure's assertions() through gates.Checks, so one run
# reports every failure. Each of these can genuinely fail: none restates a
# fact the constructor already guarantees.

def _split(terms: Sequence, op: str) -> list[list]:
    out: list[list] = []
    cur: list = []
    for t in terms:
        if isinstance(t, str) and t == op:
            out.append(cur)
            cur = []
        else:
            cur.append(t)
    out.append(cur)
    return out


def _product_blocks(part: Sequence) -> list[Block]:
    return [t for t in part if isinstance(t, Block)]


def check_conformable(checks, terms: Sequence[Block | str]) -> None:
    """Inner dimensions across the DRAWN term list.

    Every adjacent pair in a product chains (`A.n == B.m`); every summand
    of a `+` has the same shape; both sides of `=` have the same shape.
    Reads the placed Blocks, so it catches a figure that drew a product
    the arrays never justified.
    """
    side_shapes: list[tuple[int, int]] = []
    for si, side in enumerate(_split(terms, "=")):
        shapes: list[tuple[int, int]] = []
        for prod in _split(side, "+"):
            bs = _product_blocks(prod)
            if not bs:
                checks.check(False, f"expr side {si}: a summand has no blocks")
                continue
            for a, b in zip(bs, bs[1:]):
                checks.check(a.n == b.m,
                             f"non-conformable: {_nm(a)}({a.m}x{a.n}) @ "
                             f"{_nm(b)}({b.m}x{b.n}) — inner {a.n} != {b.m}")
            shapes.append((bs[0].m, bs[-1].n))
        for s in shapes[1:]:
            checks.check(s == shapes[0],
                         f"expr side {si}: summands differ in shape "
                         f"{shapes[0]} vs {s}")
        if shapes:
            side_shapes.append(shapes[0])
    for s in side_shapes[1:]:
        checks.check(s == side_shapes[0],
                     f"'=' sides differ in shape {side_shapes[0]} vs {s}")


def check_cell_uniform(checks, blocks: Sequence[Block]) -> None:
    """One (cell, aspect) across the figure.

    Two matrices drawn at different scales make their shapes
    incomparable — the reader cannot see that A's column count matches
    B's row count. `expr` guarantees this within one row; a figure that
    hand-places a block alongside is what this catches.
    """
    if not blocks:
        return
    ref_c, ref_a = blocks[0].cell, blocks[0].aspect
    for b in blocks[1:]:
        checks.check(abs(b.cell - ref_c) < 1e-12
                     and abs(b.aspect - ref_a) < 1e-12,
                     f"{_nm(b)}: (cell {b.cell:g}, aspect {b.aspect:g}) != "
                     f"({ref_c:g}, {ref_a:g}) — blocks at different scales "
                     f"are not comparable")


def check_square_cells(checks, blocks: Sequence[Block]) -> None:
    """Every block has square cells, so its drawn aspect ratio IS its shape.

    Call this from any figure arguing about *dimensions* — a
    factorization, a conformability claim, a block algorithm. Do NOT call
    it on a basis gallery: there, `aspect != 1.0` is a deliberate opt-out
    (the cells hold waveforms and the figure makes no claim about shape),
    and the two cases are told apart by which figures call this.
    """
    for b in blocks:
        checks.check(abs(b.aspect - 1.0) < 1e-12,
                     f"{_nm(b)}: aspect {b.aspect:g} != 1 — the drawn "
                     f"rectangle no longer shows the shape ({b.m}, {b.n})")


def check_no_overlap(checks, blocks: Sequence[Block]) -> None:
    """No two drawn matrices share area. A `gap` or `op_gap` too small for
    the row collides two blocks, and the operator then reads as inside one
    of them."""
    for i, a in enumerate(blocks):
        ax0, ax1, ay0, ay1 = a.extent
        for b in blocks[i + 1:]:
            bx0, bx1, by0, by1 = b.extent
            hit = (min(ax1, bx1) - max(ax0, bx0) > 1e-9
                   and min(ay1, by1) - max(ay0, by0) > 1e-9)
            checks.check(not hit,
                         f"blocks {_nm(a)} and {_nm(b)} overlap — widen gap")


def check_expr(checks, terms: Sequence[Block | str], *, rtol: float = 1e-9,
               atol: float = 0.0) -> None:
    """Evaluate the drawn expression in numpy and assert it holds.

    This is the gate the whole layer exists for. Draw A = U S V^T and this
    proves the picture is of a *true* factorization — not a plausible
    arrangement of rectangles. Every Block must carry `values`.
    """
    sides = _split(terms, "=")
    if len(sides) != 2:
        checks.check(False,
                     f"check_expr needs exactly one '=', got {len(sides) - 1}")
        return
    evaluated: list[np.ndarray] = []
    for side in sides:
        total: np.ndarray | None = None
        for prod in _split(side, "+"):
            bs = _product_blocks(prod)
            if not bs:
                checks.check(False, "check_expr: a summand has no blocks")
                return
            missing = [b for b in bs if b.values is None]
            if missing:
                checks.check(False, "check_expr: block "
                                    f"{_nm(missing[0])} carries no values")
                return
            acc = np.asarray(bs[0].values, dtype=float)
            for b in bs[1:]:
                acc = acc @ np.asarray(b.values, dtype=float)
            total = acc if total is None else total + acc
        assert total is not None
        evaluated.append(total)
    lhs, rhs = evaluated
    if lhs.shape != rhs.shape:
        checks.check(False, "check_expr: sides evaluate to shapes "
                            f"{lhs.shape} vs {rhs.shape}")
        return
    err = float(np.max(np.abs(lhs - rhs)))
    scale = float(np.max(np.abs(lhs))) or 1.0
    checks.check(err <= rtol * scale + atol,
                 f"drawn expression does not hold: max|LHS-RHS| = {err:.3e} "
                 f"> {rtol:.1e} * {scale:.4g}")


def check_hinton_area(checks, b: Block, squares: Sequence[FilledCurve], *,
                      rtol: float = 1e-6) -> None:
    """Square AREA proportional to |v|, over `hinton`'s row-major order.

    Guards the encoding bug that makes side proportional to |v|: that
    squares the scale, so an entry at half the magnitude of its neighbour
    is drawn at a quarter of the area and the diagram understates
    everything small.
    """
    V = _values(b)
    mags = [abs(float(V[i, j])) for i in range(b.m) for j in range(b.n)
            if V[i, j] != 0.0]
    if len(mags) != len(squares):
        checks.check(False, f"hinton emitted {len(squares)} squares for "
                            f"{len(mags)} nonzero entries")
        return
    if not mags:
        return
    ratios = []
    for v, sq in zip(mags, squares):
        side = float(sq.pts[:, 0].max() - sq.pts[:, 0].min())
        ratios.append(side * side / v)
    lo, hi = min(ratios), max(ratios)
    checks.check(hi - lo <= rtol * hi,
                 f"hinton area/|v| not constant: {lo:.6g}..{hi:.6g} — a side "
                 f"proportional to |v| squares the encoding")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make test`
Expected: PASS — 47 tests in `tests/test_matrix.py`, no regressions elsewhere.

- [ ] **Step 5: Commit**

```bash
git add src/figlib/matrix.py tests/test_matrix.py
git commit -m "matrix: the gates — conformable, cell/square, no-overlap, expr, hinton area

check_expr is the load-bearing one: evaluate the drawn factorization in
numpy and prove the picture is true, not merely plausible. The others
each catch a drawn lie: mismatched scales, tall cells under a shape
claim, colliding blocks, a side-proportional Hinton diagram."
```

---

### Task 6: Benchmark figure — `matrix_four_views`

**Files:**
- Create: `figures/matrix_four_views.py`
- Create: `figures/out/matrix_four_views.svg`, `.png`, `.readback.md`

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: the exemplar for the structure grammar, referenced from `docs/skill.md` in Task 8.

- [ ] **Step 1: Write the figure program**

Create `figures/matrix_four_views.py`:

```python
"""One matrix, four readings — and the reading is what makes AB a sum.

Top row: the same 3x2 A drawn four ways (whole / 6 entries / 2 columns /
3 rows). Bottom row: AB as the sum of two rank-1 matrices, one per
(column of A, row of B) pair — which is only visible once you have chosen
the column-and-row reading. Hiranabe's grammar (The Art of Linear
Algebra), gated: check_expr evaluates the drawn sum on the same arrays.
"""

import numpy as np

from figlib.format import WIDE
from figlib.gates import Checks
from figlib.matrix import (Block, bands, bounds, check_cell_uniform,
                           check_conformable, check_expr, check_no_overlap,
                           check_square_cells, expr, lattice, outline, rank1)
from figlib.scene import MathLabel, Scene
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "A matrix has four readings — one whole, mn entries, n columns, m rows "
    "— and choosing the column-and-row reading is what turns the product AB "
    "into a sum of rank-1 matrices, one per column of A paired with the "
    "matching row of B."
)

THEME = RISO
FORMAT = WIDE

PARAMS = {
    "A": [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],   # 3x2
    "B": [[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]],     # 2x3
    "cell": 0.62,
    "gap": 0.45,
    "op_gap": 0.9,
    "band_gap": 0.10,
    "view_row_y": 1.55,
    "sum_row_y": -1.15,
    "caption_drop": 0.62,      # below a block's bottom edge, math units
    "pad": 0.55,
}

VIEW_CAPTIONS = ("1\\ \\mathrm{matrix}", "6\\ \\mathrm{numbers}",
                 "2\\ \\mathrm{columns}", "3\\ \\mathrm{rows}")


def compute(p):
    A = np.array(p["A"], dtype=float)
    B = np.array(p["B"], dtype=float)
    AB = A @ B
    terms = [np.outer(A[:, k], B[k, :]) for k in range(A.shape[1])]
    return {"A": A, "B": B, "AB": AB, "terms": terms, "params": p}


def _rows(g):
    """Both expression rows, placed. Shared by build() and assertions()."""
    p = g["params"]
    A = g["A"]
    kw = dict(cell=p["cell"], gap=p["gap"], op_gap=p["op_gap"])
    views = [Block(*A.shape, values=A, name=f"view{k}") for k in range(4)]
    view_terms = [views[0], "=", views[1], "=", views[2], "=", views[3]]
    view_ops, view_blocks = expr(view_terms, origin=(0.0, p["view_row_y"]), **kw)

    sum_terms = [Block(*g["AB"].shape, values=g["AB"], name="AB"), "="]
    for k, T in enumerate(g["terms"]):
        sum_terms += [Block(*T.shape, values=T, name=f"a{k}b{k}"), "+"]
    sum_terms.pop()
    sum_ops, sum_blocks = expr(sum_terms, origin=(0.0, p["sum_row_y"]), **kw)
    return (view_terms, view_ops, view_blocks,
            sum_terms, sum_ops, sum_blocks)


def build(g):
    p = g["params"]
    (_, view_ops, views, _, sum_ops, sums) = _rows(g)
    col_hue, row_hue, dot_hue = (THEME.categorical(0), THEME.categorical(1),
                                 THEME.categorical(2))

    s = Scene()
    # --- the four readings of one matrix
    s.add(*outline(views[0], wash=0.42))
    s.add(*outline(views[1]), *lattice(views[1], color=dot_hue))
    s.add(*outline(views[2]),
          *bands(views[2], "col", color=col_hue, gap=p["band_gap"]))
    s.add(*outline(views[3]),
          *bands(views[3], "row", color=row_hue, gap=p["band_gap"]))
    s.add(*view_ops)
    for b, cap in zip(views, VIEW_CAPTIONS):
        s.add(MathLabel(cap, (b.extent[0] + b.width / 2,
                              b.extent[2] - p["caption_drop"]),
                        role=Role.ANNOTATION, ha="center", va="center",
                        size_pt=9.0))

    # --- AB as a sum of rank-1 matrices
    s.add(*outline(sums[0], wash=0.42))
    for k, b in enumerate(sums[1:]):
        s.add(*rank1(b, k, k, col_color=col_hue, row_color=row_hue))
        s.add(*outline(b))
        s.add(MathLabel(rf"\boldsymbol{{a}}_{k + 1}\boldsymbol{{b}}_{k + 1}^{{T}}",
                        (b.extent[0] + b.width / 2, b.extent[2] - p["caption_drop"]),
                        role=Role.ANNOTATION, ha="center", va="center",
                        size_pt=9.0))
    s.add(*sum_ops)
    s.add(MathLabel(r"AB", (sums[0].extent[0] + sums[0].width / 2,
                            sums[0].extent[2] - p["caption_drop"]),
                    role=Role.ANNOTATION, ha="center", va="center",
                    size_pt=9.0))

    s.xlim, s.ylim = bounds(views + sums, pad=p["pad"])
    s.ylim = (s.ylim[0] - p["caption_drop"], s.ylim[1])
    return s


def assertions(g):
    (view_terms, _, views, sum_terms, _, sums) = _rows(g)
    # substitute the PLACED blocks back into the term list, positionally:
    # the gates must read what was drawn, not the unplaced inputs
    vi, si = iter(views), iter(sums)
    placed_view = [next(vi) if isinstance(t, Block) else t for t in view_terms]
    placed_sum = [next(si) if isinstance(t, Block) else t for t in sum_terms]
    c = Checks()
    # the drawn sum of rank-1 matrices really is AB
    check_expr(c, placed_sum)
    check_conformable(c, placed_sum)
    check_conformable(c, placed_view)
    # every block on the page at one scale, none colliding
    blocks = [t for t in placed_view + placed_sum if isinstance(t, Block)]
    check_cell_uniform(c, blocks)
    check_no_overlap(c, blocks)
    # this figure argues about dimensions, so its cells must stay square
    check_square_cells(c, blocks)
    # the rank-1 claim itself: each drawn term has rank exactly 1
    for k, T in enumerate(g["terms"]):
        c.check(np.linalg.matrix_rank(T) == 1,
                f"drawn term {k} is not rank 1")
    c.done()
```

- [ ] **Step 2: Run the gates**

Run: `make check F=figures/matrix_four_views.py`
Expected: PASS. If `annotation-load` or `clipped` fires, the diagnostic
carries the fix — apply `offset_px` nudges or widen `pad`/`caption_drop`
as printed. **Never reduce `size_pt`.**

- [ ] **Step 3: Inspect and record the baseline**

Run: `make check F="figures/matrix_four_views.py --report"` to confirm the
two rows are separated and no label box overlaps a block. Then:

```bash
make update F=figures/matrix_four_views.py
```

- [ ] **Step 4: Readback gate**

```bash
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib uv run figcheck figures/matrix_four_views.py --readback-prompt
```

Dispatch a **context-free** agent (no CLAIM, no conversation) on
`figures/out/matrix_four_views.png` with that prompt. Write the result with
`readback.record()` to `figures/out/matrix_four_views.readback.md`. Every
confusion bullet is design review: fix it, or state in the record why it
is accepted.

- [ ] **Step 5: Commit**

```bash
git add figures/matrix_four_views.py figures/out/matrix_four_views.*
git commit -m "figure: matrix_four_views — the four readings, and AB as a sum of rank-1

Exercises bands/lattice/rank1/expr/check_conformable/check_expr. The sum
row is gated numerically: the drawn rank-1 terms really do add to AB."
```

---

### Task 7: Benchmark figure — `svd_low_rank`

**Files:**
- Create: `figures/svd_low_rank.py`
- Create: `figures/out/svd_low_rank.svg`, `.png`, `.readback.md`

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: the exemplar for value encoders on real arrays.

- [ ] **Step 1: Write the figure program**

Create `figures/svd_low_rank.py`:

```python
"""A causal attention map is four rank-1 matrices plus an admitted residual.

The synthetic 32x32 map is seeded and stated in PARAMS — no hidden data.
The drawn equation is EXACT (A_4 is the rank-4 truncation, not A), and the
approximation is admitted as a number on the figure: the Frobenius energy
the truncation keeps. The Hinton inset carries what no ramp can — the SIGN
structure of the top right-singular vectors.
"""

import numpy as np

from figlib.format import WIDE
from figlib.gates import Checks
from figlib.matrix import (Block, bounds, causal, check_cell_uniform,
                           check_conformable, check_expr, check_hinton_area,
                           check_no_overlap, check_square_cells, expr, heat,
                           hinton, mask, outline)
from figlib.scene import MathLabel, Scene
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "A causal attention map is well approximated by four rank-1 matrices: "
    "the drawn sum is exactly the rank-4 truncation, and that truncation "
    "keeps most of the map's Frobenius energy — the rest is admitted as a "
    "number, not hidden."
)

THEME = RISO
FORMAT = WIDE

PARAMS = {
    "n": 32,
    "seed": 7,
    "decay": 0.18,          # positional bias: log-attention falls off with distance
    "noise": 0.7,           # scale of the random score component
    "rank": 4,
    "cell": 0.055,          # a 32-cell block is 1.76 math units wide
    "gap": 0.55,
    "op_gap": 0.85,
    "row_y": 0.0,
    "hinton_rows": 4,       # top-k right singular vectors in the inset
    "hinton_cols": 8,       # first columns (key positions) shown
    "hinton_cell": 0.16,
    "hinton_origin": (-0.35, -1.55),
    "caption_drop": 0.30,
    "pad": 0.40,
}


def _attention(p):
    """Seeded causal-softmax scores. Stated, reproducible, not scraped."""
    n = p["n"]
    rng = np.random.default_rng(p["seed"])
    i, j = np.indices((n, n))
    scores = p["noise"] * rng.standard_normal((n, n)) - p["decay"] * (i - j)
    scores = np.where(j <= i, scores, -np.inf)          # causal
    scores -= scores.max(axis=1, keepdims=True)
    w = np.exp(scores)
    return w / w.sum(axis=1, keepdims=True)


def compute(p):
    A = _attention(p)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    r = p["rank"]
    terms = [s[k] * np.outer(U[:, k], Vt[k, :]) for k in range(r)]
    A_r = np.sum(terms, axis=0)
    energy = float(np.sum(s[:r] ** 2) / np.sum(s ** 2))
    resid = float(np.linalg.norm(A - A_r, "fro") / np.linalg.norm(A, "fro"))
    V_top = Vt[:p["hinton_rows"], :p["hinton_cols"]]
    return {"A": A, "A_r": A_r, "terms": terms, "s": s, "V_top": V_top,
            "energy": energy, "resid": resid, "params": p}


def _row(g):
    p = g["params"]
    terms = [Block(*g["A_r"].shape, values=g["A_r"], name="A_4"), "="]
    for k, T in enumerate(g["terms"]):
        terms += [Block(*T.shape, values=T, name=f"s{k}"), "+"]
    terms.pop()
    ops, placed = expr(terms, cell=p["cell"], gap=p["gap"],
                       op_gap=p["op_gap"], origin=(0.0, p["row_y"]))
    return terms, ops, placed


def build(g):
    p = g["params"]
    _, ops, placed = _row(g)
    vmax = float(np.abs(g["A_r"]).max())

    s = Scene()
    for k, b in enumerate(placed):
        s.add(heat(b, ramp=THEME.ramp, vmin=0.0, vmax=vmax))
        s.add(*outline(b))
        cap = (r"A_4" if k == 0
               else rf"\sigma_{k}\boldsymbol{{u}}_{k}\boldsymbol{{v}}_{k}^{{T}}")
        s.add(MathLabel(cap, (b.extent[0] + b.width / 2,
                              b.extent[3] + p["caption_drop"]),
                        role=Role.ANNOTATION, ha="center", va="bottom",
                        size_pt=9.0))
    s.add(*ops)

    # the causal structure, drawn as a silhouette on the first block
    s.add(*mask(placed[0], ~causal(placed[0]), role=Role.MUTED, opacity=0.55))

    # the sign structure of the top right-singular vectors — no ramp can do this
    hb = Block(p["hinton_rows"], p["hinton_cols"], values=g["V_top"],
               origin=p["hinton_origin"], cell=p["hinton_cell"], name="Vt")
    s.add(*outline(hb), *hinton(hb))
    s.add(MathLabel(r"\boldsymbol{v}_1..\boldsymbol{v}_4\ \mathrm{(sign)}",
                    (hb.extent[0], hb.extent[2] - p["caption_drop"]),
                    role=Role.ANNOTATION, ha="left", va="center",
                    size_pt=9.0))

    # the honesty pass: what the equation leaves out, as a number
    s.add(MathLabel(
        rf"\|A - A_4\|_F / \|A\|_F = {g['resid']:.3f}"
        rf"\quad ({100 * g['energy']:.1f}\%\ \mathrm{{energy\ kept}})",
        (placed[0].extent[0], hb.extent[3] + p["caption_drop"]),
        role=Role.ANNOTATION, ha="left", va="bottom", size_pt=10.0))

    xlim, ylim = bounds(list(placed) + [hb], pad=p["pad"])
    s.xlim, s.ylim = xlim, (ylim[0] - p["caption_drop"], ylim[1] + p["caption_drop"])
    return s


def assertions(g):
    p = g["params"]
    terms, _, placed = _row(g)
    it = iter(placed)
    drawn = [next(it) if isinstance(t, Block) else t for t in terms]
    c = Checks()
    # the drawn equation is exact: A_4 IS the sum of the four rank-1 terms
    check_expr(c, drawn, rtol=1e-12)
    check_conformable(c, drawn)
    blocks = [t for t in drawn if isinstance(t, Block)]
    check_cell_uniform(c, blocks)
    check_no_overlap(c, blocks)
    check_square_cells(c, blocks)
    # each drawn summand is genuinely rank 1
    for k, T in enumerate(g["terms"]):
        c.check(np.linalg.matrix_rank(T) == 1, f"term {k} is not rank 1")
    # the annotated residual matches the arrays that got drawn
    resid = float(np.linalg.norm(g["A"] - g["A_r"], "fro")
                  / np.linalg.norm(g["A"], "fro"))
    c.check(abs(resid - g["resid"]) < 1e-12, "annotated residual is stale")
    # rows of the attention map are normalized and causal
    c.check(np.allclose(g["A"].sum(axis=1), 1.0), "attention rows are not normalized")
    c.check(np.all(np.triu(g["A"], 1) == 0.0), "attention map is not causal")
    # the Hinton inset encodes area, not side
    hb = Block(p["hinton_rows"], p["hinton_cols"], values=g["V_top"],
               origin=p["hinton_origin"], cell=p["hinton_cell"], name="Vt")
    check_hinton_area(c, hb, hinton(hb))
    c.done()
```

- [ ] **Step 2: Run the gates**

Run: `make check F=figures/svd_low_rank.py`
Expected: PASS. Apply printed `offset_px` nudges verbatim if a collision
fires; if `annotation-load` fires, cut a caption — never shrink type.

- [ ] **Step 3: Inspect and record the baseline**

Run: `make check F="figures/svd_low_rank.py --report"`, confirm the Hinton
inset clears the expression row and the residual line, then:

```bash
make update F=figures/svd_low_rank.py
```

- [ ] **Step 4: Readback gate**

```bash
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib uv run figcheck figures/svd_low_rank.py --readback-prompt
```

Dispatch a context-free agent on `figures/out/svd_low_rank.png`; write the
result with `readback.record()` to `figures/out/svd_low_rank.readback.md`.
Pay particular attention to whether the cold reader reads the Hinton inset
as *signed* and the heat blocks as *magnitude* — if they conflate the two
encodings, that is a real defect and the inset needs a legend or more
separation.

- [ ] **Step 5: Commit**

```bash
git add figures/svd_low_rank.py figures/out/svd_low_rank.*
git commit -m "figure: svd_low_rank — a causal attention map as four rank-1 terms

Exercises heat/hinton/mask/expr on real arrays. The drawn equation is
exact (A_4, not A) and gated at rtol 1e-12; the approximation is admitted
on the figure as a residual and an energy fraction, not hidden."
```

---

### Task 8: Documentation and the corpus sweep

**Files:**
- Modify: `src/figlib/__init__.py`
- Modify: `docs/skill.md`
- Modify: `docs/primitive-gaps.md`

**Interfaces:**
- Consumes: the two figures from Tasks 6–7 as named exemplars.
- Produces: nothing importable.

**All three files in this task carry another session's uncommitted
changes** (`docs/architecture.md`, `docs/skill.md`,
`docs/primitive-gaps.md`, `src/figlib/__init__.py` were all dirty when
this plan was written, alongside new untracked `src/figlib/correspond.py`
and `src/figlib/place.py`). Make **targeted Edits after a fresh Read** —
never a Write-rewrite, and never `git add -A`. Stage only the specific
paths listed in Step 5. This is the parallel-agent hygiene rule in
`docs/primitive-gaps.md`.

- [ ] **Step 1: Add `matrix` to the module map**

In `src/figlib/__init__.py`, under `WRITING A FIGURE`, insert after the
`schematic` line:

```
    matrix      matrices as Blocks drawn at their own aspect ratio;
                structure/value encoders, the expression row, shape gates
```

- [ ] **Step 2: Add the exemplar rows**

In `docs/skill.md`, append two rows to the **Device -> exemplar index**
table:

```markdown
| matrix structure: four readings, factorization as a sum of rank-1 | `matrix_four_views.py` |
| matrix values: heatmap + signed Hinton, low-rank truncation | `svd_low_rank.py` |
```

- [ ] **Step 3: Record the gap the VCA survey could not find**

In `docs/primitive-gaps.md`, append a section at the end:

```markdown
## Matrices (landed; invisible to the VCA survey)

The evidence base for this document was Needham, and Needham has no
matrices — so the largest gap for a machine-learning corpus was never
counted. `matrix.py` closes it, sourced from Hiranabe's *The Art of Linear
Algebra* (graphic notes on Strang), whose grammar turns out to be small
and orthogonal in exactly the way the doctrine above predicts: one
rectangle with four readings, and every factorization reducing to a sum of
rank-1 rectangles.

The transferable finding is that **shape is the thing to make geometric**.
Drawing a matrix at its own aspect ratio is not a stylistic choice; it is
what makes a non-conformable product undrawable and lets `check_expr`
evaluate the drawn factorization in numpy and prove the picture is true.
A figure that merely *arranges* rectangles cannot be gated at all.

Still open, deliberately deferred: **einsum / tensor-network diagrams**
(Penrose notation). A tensor has legs, not a 2-D shape to draw to scale,
so it belongs on `schematic.py` as named nodes with contracted edges —
forcing it into `Block` would corrupt the shape-is-geometry invariant.
References: *An introduction to graphical tensor notation for mechanistic
interpretability* (arXiv 2402.01790) and *Named Tensor Notation* (arXiv
2102.13196).
```

- [ ] **Step 4: Full test and corpus sweep**

Run: `make test`
Expected: PASS, all suites.

Run: `make regress`
Expected: clean — **no drift in any pre-existing figure**. `matrix.py` adds
no renderer and touches no shared module, so any diff in another figure's
baseline is a real bug: find it before continuing. The two new figures are
already committed baselines from Tasks 6–7 and must report MATCH.

- [ ] **Step 5: Commit**

```bash
git add src/figlib/__init__.py docs/skill.md docs/primitive-gaps.md
git commit -m "docs: route the matrix layer

Module map, two exemplar rows, and the finding primitive-gaps could not
have reached from a Needham-only evidence base: shape drawn as geometry
is what makes a matrix figure gateable at all."
```

---

## Self-Review

**Spec coverage.** §1 Block → Task 1. §2 structure encoders → Task 2;
value encoders → Task 3 (with `rank1`). §3 expression row, including the
rejected-Panel rationale → Task 4. §4 gates → Task 5, with one documented
substitution (`check_shape_faithful` was a tautology; replaced by
`check_cell_uniform` + `check_no_overlap`, and the spec is amended in the
same task). §5 theming → satisfied structurally: `matrix.py` imports no
theme, and both figures pass `THEME.categorical(0..2)` / `THEME.ramp`.
§6 benchmarks → Tasks 6–7, each with the mandatory readback. §7 tests →
written first in every task. Follow-on → Task 8.

**Placeholders.** None: every code step carries the actual code, every
test step the actual assertions, every command the actual invocation.

**Type consistency.** `Block.extent` returns `(x0, x1, y0, y1)` and is used
in that order by `heat` (RasterField extent), `bounds`, `check_no_overlap`,
`grid_lines`, `brackets`, `edge`, and both figures. `center` (not
`cell_center`) and `extent` (not `bbox`) are the names everywhere, matching
the merged signals spec. `expr` returns `(ops, placed)` in that order at every
call site. `_values`, `_nm`, `_split`, `_product_blocks`, `_inset` are each
defined once and used with matching signatures. Checkers all take `checks`
first and return `None`.
