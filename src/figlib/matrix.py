"""Matrices as drawable, gateable objects.

The organizing idea, taken from Hiranabe's *The Art of Linear Algebra*
(graphic notes on Strang): a matrix is a rectangle whose **aspect ratio is
its shape**, and the interesting choice is which of four readings of that
rectangle the argument needs — the opaque whole, the `mn` entries, the `n`
column bands, or the `m` row bands. Every factorization (CR, LU, QR,
QLQ^T, USV^T) then reduces to the same picture: a sum of rank-1
rectangles.

**Why this is a module and not local arithmetic in each figure.**
`dft_matrix_basis.py` addresses an 8x8 grid with a two-line `center`
closure and needs nothing from here — and it is right not to, because
addressing is cheap to rewrite and the doctrine says absorb verbosity.
The part that is *not* cheap to rewrite is the gating. `check_expr`
evaluates a drawn factorization in numpy; `check_conformable` reads inner
dimensions off the drawn term list. Neither can live inside one figure,
because a gate is the shared oracle — and both need shape and values to
be one object that survives from `compute()` into `assertions()`. That
object is `Block`. Everything else here exists to feed it.

Shape-as-geometry is what makes the layer gateable rather than
decorative: a non-conformable product cannot be drawn to scale, and a
`Block` that carries its `values` makes the picture and the numbers the
same object, so the drawn equation can be *proved* rather than trusted.

This module is a producer of scene items. It imports no theme and names
no color: encoders take a `Role` and an optional `color`, and the figure
program supplies `THEME.categorical(i)` / `THEME.ramp(t)`.

Coordinates: math coords, +y UP, and `Block.origin` is the **top-left**
corner — so row 0 is at the top, matching `RasterField`'s row-0-at-`y1`
rule and ordinary matrix index convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np

from .scene import Curve, FilledCurve, MathLabel, Point, RasterField
from .style import Role

XY = tuple[float, float]
Mask = np.ndarray | Callable[[int, int], bool]


@dataclass(frozen=True)
class Block:
    """A matrix drawn to shape: `n*cell` wide, `m*cell` tall.

    Draws nothing — it answers coordinate questions and the author
    composes items against it. Square cells are not a style choice: they
    are what make the drawn rectangle's aspect ratio EQUAL the shape
    (m, n), which is what every gate below rests on.

    `values` is optional. Supplying it unlocks the value encoders (`heat`,
    `hinton`, `entries`) and the `check_expr` gate; a purely structural
    figure leaves it None.
    """
    m: int
    n: int
    origin: XY = (0.0, 0.0)            # TOP-left, math coords
    cell: float = 1.0
    values: np.ndarray | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if self.m <= 0 or self.n <= 0:
            raise ValueError(
                f"Block shape must be positive, got ({self.m}, {self.n})")
        if self.cell <= 0:
            raise ValueError(f"Block cell must be positive, got {self.cell}")
        if self.values is not None:
            v = np.asarray(self.values)
            if v.shape != (self.m, self.n):
                raise ValueError(
                    f"Block {self.name or '?'}: values shape {v.shape} != "
                    f"({self.m}, {self.n})")

    # --- derived geometry (no scene items) -------------------------------

    @property
    def width(self) -> float:
        return self.n * self.cell

    @property
    def height(self) -> float:
        return self.m * self.cell

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(x0, x1, y0, y1) — named for the RasterField parameter it feeds."""
        ox, oy = self.origin
        return (ox, ox + self.width, oy - self.height, oy)

    def span(self, j0: int, j1: int, i0: int, i1: int) -> np.ndarray:
        """Closed rect (4, 2) over column range [j0, j1) x row range [i0, i1)."""
        ox, oy = self.origin
        x0, x1 = ox + j0 * self.cell, ox + j1 * self.cell
        y1, y0 = oy - i0 * self.cell, oy - i1 * self.cell
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
        return (ox + (j + 0.5) * self.cell, oy - (i + 0.5) * self.cell)

    def sub(self, rows: slice, cols: slice) -> "Block":
        """A sub-block: still a Block, so encoders and gates compose on it."""
        ri = range(*rows.indices(self.m))
        ci = range(*cols.indices(self.n))
        ox, oy = self.origin
        return Block(
            len(ri), len(ci),
            origin=(ox + ci.start * self.cell, oy - ri.start * self.cell),
            cell=self.cell,
            values=None if self.values is None else self.values[rows, cols],
            name=f"{self.name}[{ri.start}:{ri.stop},{ci.start}:{ci.stop}]"
                 if self.name else "")

    def at(self, origin: XY, cell: float | None = None) -> "Block":
        """The same matrix placed elsewhere — what `expr` uses to lay out."""
        return replace(self, origin=(float(origin[0]), float(origin[1])),
                       cell=self.cell if cell is None else float(cell))


def _nm(b: Block) -> str:
    return b.name or f"{b.m}x{b.n}"


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


def rank1(b: Block, j: int, i: int, *, col_color: str | None = None,
          row_color: str | None = None, role: Role = Role.CONTENT,
          ground: float = 0.12) -> list:
    """The outer product mark: the whole rectangle as a faint ground,
    crossed by column `j` and row `i`.

    This is the one mark every factorization reduces to. In
    A = sum_k a_k b_k^T, term k is drawn on a result-shaped block with
    column k of the left factor and row k of the right factor marked, so
    the reader can see *which* column and row produced this summand.
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
    """One cell size across the figure.

    Two matrices drawn at different scales make their shapes
    incomparable — the reader cannot see that A's column count matches
    B's row count. `expr` guarantees this within one row; a figure that
    hand-places a block alongside is what this catches.
    """
    if not blocks:
        return
    ref = blocks[0].cell
    for b in blocks[1:]:
        checks.check(abs(b.cell - ref) < 1e-12,
                     f"{_nm(b)}: cell {b.cell:g} != {ref:g} — blocks at "
                     f"different scales are not comparable")


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
