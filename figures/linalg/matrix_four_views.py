"""One matrix, four readings — and the reading is what makes AB a sum.

Top row: the same 3x2 A drawn four ways (whole / 6 entries / 2 columns /
3 rows). Bottom row: AB as the sum of two rank-1 matrices, one per
(column of A, row of B) pair — which is only visible once you have
chosen the column-and-row reading. Hiranabe's grammar (The Art of Linear
Algebra), gated: check_expr evaluates the drawn sum on the same arrays
that got drawn.
"""

import numpy as np

from figlib.format import WIDE
from figlib.gates import Checks
from figlib.matrix import (Block, bands, bounds, check_cell_uniform,
                           check_conformable, check_expr, check_no_overlap,
                           expr, lattice, outline, rank1)
from figlib.scene import MathLabel, Scene
from figlib.style import WEIGHT_ACTOR, WEIGHT_BG, Role
from figlib.theme import RISO

CLAIM = (
    "A matrix has four readings — one whole, mn entries, n columns, m rows "
    "— and choosing the column-and-row reading is what turns the product AB "
    "into a sum of rank-1 matrices, one per column of A paired with the "
    "matching row of B."
)

EXPOSITION = """
Strang's point is that a matrix product is not one operation with one
correct way to compute it -- it has (at least) four readings, and each
reading makes a different fact about the product obvious. Read AB
entry by entry (row of A dot column of B) and you get a number at a
time, with no structure. Read B's columns as things A acts on, and AB's
columns are combinations of A's columns. Read A's rows as things acting
on B, and AB's rows are combinations of B's rows. But read A by its
columns AND B by its rows together, pairing column k of A with row k of
B, and the product becomes a SUM of rank-1 matrices, one per pairing --
a fact invisible from the entry-by-entry view and not derivable from
either one-sided reading alone. Which reading to reach for is not a
matter of taste: the column-times-row reading is the one that explains
why low-rank matrices decompose the way they do, why outer-product
updates work, and why a matrix product can be built up term by term
instead of computed all at once.
"""

THEME = RISO
FORMAT = WIDE

PARAMS = {
    "A": [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],   # 3x2
    "B": [[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]],     # 2x3
    "cell": 0.62,
    "gap": 0.5,
    "op_gap": 0.95,
    "band_gap": 0.10,
    "view_row_y": 1.75,
    "sum_row_y": -2.15,        # clears the top row's captions AND the b_k^T strip
    "caption_drop": 0.42,      # below a block's bottom edge, math units
    "row_lift": 0.30,          # gap from a summand's top edge to its b_k^T
    "dim_pad": 0.20,           # dimension label offset from an edge
    "pad": 0.5,
}

VIEW_CAPTIONS = (r"1\ \mathrm{matrix}", r"6\ \mathrm{numbers}",
                 r"2\ \mathrm{columns}", r"3\ \mathrm{rows}")


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
    view_ops, view_blocks = expr(view_terms, origin=(0.0, p["view_row_y"]),
                                 **kw)

    sum_terms = [Block(*g["AB"].shape, values=g["AB"], name="AB"), "="]
    for k, T in enumerate(g["terms"]):
        sum_terms += [Block(*T.shape, values=T, name=f"a{k}b{k}"), "+"]
    sum_terms.pop()
    sum_ops, sum_blocks = expr(sum_terms, origin=(0.0, p["sum_row_y"]), **kw)
    return (view_terms, view_ops, view_blocks,
            sum_terms, sum_ops, sum_blocks)


def _dims(b, drop=0.20):
    """m and n drawn ON the block's edges. The figure GATES conformability;
    without this the reader cannot check it, which a cold readback flagged
    ('the dimensions of A and B are never stated')."""
    x0, x1, y0, y1 = b.extent
    return [
        MathLabel(str(b.m), (x0 - drop, (y0 + y1) / 2), role=Role.ANNOTATION,
                  ha="right", va="center", size_pt=9.0),
        MathLabel(str(b.n), ((x0 + x1) / 2, y1 + drop), role=Role.ANNOTATION,
                  ha="center", va="bottom", size_pt=9.0),
    ]


def _caption(b, latex, drop):
    return MathLabel(latex, (b.extent[0] + b.width / 2, b.extent[2] - drop),
                     role=Role.ANNOTATION, ha="center", va="center",
                     size_pt=9.0)


def build(g):
    p = g["params"]
    (_, view_ops, views, _, sum_ops, sums) = _rows(g)
    col_hue = THEME.categorical(0)
    row_hue = THEME.categorical(1)
    dot_hue = THEME.categorical(2)

    s = Scene()
    # --- the four readings of one matrix. The last two ARE the claim's
    # readings (columns x rows is what makes the bottom row work), so they
    # take actor weight and the first two sit back as the family.
    s.add(*outline(views[0], wash=0.42, width_scale=WEIGHT_BG))
    s.add(*outline(views[1], width_scale=WEIGHT_BG),
          *lattice(views[1], color=dot_hue))
    s.add(*outline(views[2], width_scale=WEIGHT_ACTOR),
          *bands(views[2], "col", color=col_hue, gap=p["band_gap"]))
    s.add(*outline(views[3], width_scale=WEIGHT_ACTOR),
          *bands(views[3], "row", color=row_hue, gap=p["band_gap"]))
    s.add(*view_ops)
    s.add(*_dims(views[0], p["dim_pad"]))
    for b, cap in zip(views, VIEW_CAPTIONS):
        s.add(_caption(b, cap, p["caption_drop"]))

    # --- AB as a sum of rank-1 matrices
    # AB is the thing being explained; the summands are the explanation
    s.add(*outline(sums[0], wash=0.42, width_scale=WEIGHT_BG))
    s.add(_caption(sums[0], r"AB", p["caption_drop"]))
    s.add(*_dims(sums[0], p["dim_pad"]))
    for k, b in enumerate(sums[1:]):
        # every column of a_k b_k^T is a_k scaled by b_k[j] — that IS
        # rank 1, and it makes the two summands visibly different
        s.add(*rank1(b, g["A"][:, k], g["B"][k, :], color=col_hue,
                     gap=p["band_gap"]))
        s.add(*outline(b, width_scale=WEIGHT_ACTOR))
        # b_k^T drawn ABOVE, entry by entry, with the SAME opacity law —
        # so the column shading below is decodable instead of decorative
        # (the cold reader could not tell whether the pale band meant
        # anything). This also puts the missing row factor on the page.
        # origin is TOP-left, so lift by the row's own height too, or the
        # block grows downward into the summand
        rowb = Block(1, b.n,
                     origin=(b.extent[0],
                             b.extent[3] + p["row_lift"] + b.cell),
                     cell=b.cell, values=g["B"][k, :][None, :],
                     name=f"b{k}")
        s.add(*rank1(rowb, np.ones(1), g["B"][k, :], color=row_hue,
                     gap=p["band_gap"]))
        s.add(*outline(rowb))
        s.add(MathLabel(rf"\boldsymbol{{b}}_{k + 1}^{{T}}",
                        (rowb.extent[0] - p["dim_pad"],
                         (rowb.extent[2] + rowb.extent[3]) / 2),
                        role=Role.ANNOTATION, ha="right", va="center",
                        size_pt=9.0))
        s.add(_caption(
            b, rf"\boldsymbol{{a}}_{k + 1}\boldsymbol{{b}}_{k + 1}^{{T}}",
            p["caption_drop"]))
    s.add(*sum_ops)

    xlim, ylim = bounds(views + sums, pad=p["pad"])
    s.xlim = xlim
    s.ylim = (ylim[0] - p["caption_drop"],
              ylim[1] + p["row_lift"] + p["cell"])
    return s


def assertions(g):
    (view_terms, _, views, sum_terms, _, sums) = _rows(g)
    # substitute the PLACED blocks back into the term lists, positionally:
    # the gates must read what was drawn, not the unplaced inputs
    vi, si = iter(views), iter(sums)
    placed_view = [next(vi) if isinstance(t, Block) else t for t in view_terms]
    placed_sum = [next(si) if isinstance(t, Block) else t for t in sum_terms]

    c = Checks()
    # the drawn sum of rank-1 matrices really is AB
    check_expr(c, placed_sum)
    check_conformable(c, placed_sum)
    check_conformable(c, placed_view)
    blocks = [t for t in placed_view + placed_sum if isinstance(t, Block)]
    check_cell_uniform(c, blocks)
    check_no_overlap(c, blocks)
    # the rank-1 claim itself: each drawn term has rank exactly 1
    for k, T in enumerate(g["terms"]):
        c.check(np.linalg.matrix_rank(T) == 1,
                f"drawn term {k} is not rank 1")
    # the product the top row's reading justifies: A's columns pair with
    # B's rows, so there are exactly n_cols(A) summands
    c.check(len(g["terms"]) == g["A"].shape[1],
            "summand count does not equal A's column count")
    c.done()
