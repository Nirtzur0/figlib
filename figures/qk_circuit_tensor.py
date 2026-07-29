"""The QK circuit: two tensor diagrams, one array.

Design steps (architecture.md 0-9), recorded because for a Class B figure
the layout IS the content.

0. EARN IT. The prose — "attention scores are X W_Q W_K^T X^T" — is true
   and installs the wrong picture: four matrices in a row reads as four
   stages, and the reader infers that queries and keys are two mechanisms.
   They are not. The two projections meet along the head axis and nothing
   downstream can see where that axis was cut, so W_Q W_K^T is ONE bilinear
   form of rank <= d_head. That is the inference this figure makes visible
   by drawing both networks and gating that they are the same array.
1. CLAIM. Below.
2. REPRESENTATION. Penrose / graphical-tensor notation, because the object
   is which axes are identified with which and nothing here has a shape to
   draw to scale. The head axis h is drawn as an internal wire in row one
   and is simply absent in row two — the collapse is a line disappearing,
   which is the strongest thing the notation can say.
3. SIZE. WIDE. Two rows of four and three nodes, with the derived einsum
   string under each; a COLUMN slot would put the specs on two lines and
   the comparison is between the two rows, so they must be one glance apart.
4. TRAVERSAL. Both rows read left to right along the same s...t spine, and
   the eye lands on the difference: row one has an extra node and an extra
   wire. The equals sign between the rows is the claim.
5. MECHANISM ANNOTATION. Readable off the figure: which axes are summed
   (every internal wire), which survive (the two dangling legs, s and t, on
   both rows), that h is internal to the product and never reaches the
   output, and that the two einsum strings differ while the arrays do not.
6. READER EFFORT. Delegated: nothing. The dimensions are printed in the
   header from `dims(net)`, so no counting is asked for.
7. CHANNELS. Shape = provenance: data is a circle, a learned weight is a
   box. That is the only categorical channel, and it is not hue, so the
   correspondence budget stays free for the ACCENT: W_QK, the merged node,
   which is what the figure argues for.
8. HONESTY. The claim is about arrays, so it is asserted on arrays. Both
   networks contract, and `np.allclose` between them is the figure's
   central check; the rank of W_QK is asserted to be exactly d_head, which
   is what makes "the factorization is a parameterization" true rather than
   a slogan. The einsum strings printed under each row are DERIVED from the
   drawn networks by `spec()`, so a caption cannot drift from its picture.
9. WHAT IS NOT DRAWN. The softmax and the OV circuit. Neither is a
   contraction; drawing them here would need a second notation on one page.
"""

import numpy as np

from figlib.format import WIDE
from figlib.gates import Checks
from figlib.scene import MathLabel, Scene
from figlib.schematic import crossing_count
from figlib.style import Role
from figlib.tensor import (Leg, Network, Tensor, check_einsum,
                           check_index_arity, check_index_dims, check_leg_aim,
                           check_output, contract, dims, edges, extent, free,
                           items, spec)
from figlib.theme import RISO

CLAIM = (
    "Attention's query and key projections are not two mechanisms: they "
    "meet along the head axis, so X W_Q W_K^T X^T and X W_QK X^T are the "
    "same array for W_QK = W_Q W_K^T — a single bilinear form of rank 3, "
    "the head width, inside an 8-dimensional model."
)

THEME = RISO
FORMAT = WIDE

PARAMS = {
    "seq_q": 5,             # query positions (free index s)
    "seq_k": 4,             # key positions   (free index t)
    "d_model": 8,
    "d_head": 3,
    "seed": 20260729,
    "dx": 1.55,             # node spacing along a row
    "y_top": 1.35,
    "y_bot": -1.35,
    "radius": 0.30,
    "box_w": 0.86,          # a learned weight is a box, and wider than tall
    "stub": 0.46,
    "spec_drop": 0.34,      # einsum caption below its OWN row, close enough to own it
    "banner_gap": 0.30,     # between the two header lines
    "pad": 0.42,
}


def compute(p):
    rng = np.random.default_rng(p["seed"])
    X = rng.standard_normal((p["seq_q"], p["d_model"])) / np.sqrt(p["d_model"])
    Y = rng.standard_normal((p["seq_k"], p["d_model"])) / np.sqrt(p["d_model"])
    W_Q = rng.standard_normal((p["d_model"], p["d_head"]))
    W_K = rng.standard_normal((p["d_model"], p["d_head"]))
    W_QK = W_Q @ W_K.T
    return {"X": X, "Y": Y, "W_Q": W_Q, "W_K": W_K, "W_QK": W_QK,
            "scores": X @ W_QK @ Y.T, "params": p}


def _nets(g):
    """Both networks, placed. Shared by build() and assertions() so the
    gates read the diagrams that got drawn."""
    p = g["params"]
    dx, r, bw = p["dx"], p["radius"], p["box_w"]
    data = dict(radius=r, fill=RISO.background, role=Role.CONTENT)
    weight = dict(radius=r, width=bw, shape="box", fill=RISO.background)

    def x(k):
        return k * dx

    # --- row 1: s-[X]-d-[W_Q]-h-[W_K]-e-[Y]-t
    # legs are listed in AXIS order and angled independently, which is why
    # W_K carries (e, h) while its wires leave left and right the other way
    y = p["y_top"]
    factored = Network(
        (Tensor("X", (x(0), y), (Leg("s", 90.0), Leg("d", 0.0)),
                values=g["X"], label="X", **data),
         Tensor("WQ", (x(1), y), (Leg("d", 180.0), Leg("h", 0.0)),
                values=g["W_Q"], label=r"W_Q", role=Role.CONTENT, **weight),
         Tensor("WK", (x(2), y), (Leg("e", 0.0), Leg("h", 180.0)),
                values=g["W_K"], label=r"W_K", role=Role.CONTENT, **weight),
         Tensor("Y", (x(3), y), (Leg("t", 90.0), Leg("e", 180.0)),
                values=g["Y"], label="Y", **data)),
        out=("s", "t"), stub=p["stub"])

    # --- row 2: the same thing with the head axis contracted away
    y = p["y_bot"]
    merged = Network(
        (Tensor("X", (x(0), y), (Leg("s", 90.0), Leg("d", 0.0)),
                values=g["X"], label="X", **data),
         Tensor("WQK", (x(1.5), y), (Leg("d", 180.0), Leg("e", 0.0)),
                values=g["W_QK"], label=r"W_{QK}", role=Role.ACCENT1,
                **weight),
         Tensor("Y", (x(3), y), (Leg("t", 90.0), Leg("e", 180.0)),
                values=g["Y"], label="Y", **data)),
        out=("s", "t"), stub=p["stub"])
    return factored, merged


def _spec_latex(net):
    """The einsum string, typeset. Derived, never retyped: this is the one
    caption in the figure that could contradict the drawing, and it cannot."""
    lhs, rhs = spec(net).split("->")
    return rf"\mathrm{{{lhs}}}\to\mathrm{{{rhs}}}"


def build(g):
    p = g["params"]
    factored, merged = _nets(g)
    d = dims(factored)

    s = Scene()
    s.add(*items(factored))
    s.add(*items(merged))

    # the derived einsum under each row: the two strings differ (one index
    # fewer, one tensor fewer) while the arrays do not, which IS the claim
    spec_y = []
    for net in (factored, merged):
        x0, x1, y0, _ = extent(net)
        spec_y.append(y0 - p["spec_drop"])
        s.add(MathLabel(_spec_latex(net), ((x0 + x1) / 2, spec_y[-1]),
                        role=Role.ANNOTATION, ha="center", va="top",
                        size_pt=10.0))

    # The equals sign IS the argument, so it goes ON the spine between the
    # two rows, not out in the margin: read top to bottom the page is now
    # `diagram / its einsum / = / diagram / its einsum`, one equation. Two
    # cold readers in a row called a left-margin `=` an annotation rather
    # than a relational operator, which is exactly what it looked like.
    mid_x = 0.5 * (extent(factored)[0] + extent(factored)[1])
    mid_y = 0.5 * (p["y_top"] + p["y_bot"])
    s.add(MathLabel("=", (mid_x, mid_y), role=Role.ACCENT1,
                    ha="center", va="center", size_pt=22.0))

    # In this notation a transpose is not a mark, it is which leg you join
    # — the reason row one needs no dagger anywhere. Said on the page,
    # because a reader who knows the algebra reads the chain as W_Q W_K.
    s.add(MathLabel(
        r"\mathrm{no\ transpose\ mark:\ a\ transpose\ IS\ which\ leg\ joins}",
        (factored.tensors[2].center[0], p["y_top"] + 0.34),
        role=Role.ANNOTATION, ha="center", va="bottom", size_pt=8.5))

    # W_QK is defined ON the page, with the fact that makes the whole
    # argument non-trivial. A cold reader had to infer the transpose from
    # the einsum strings and could not tell whether h = 3 was illustrative
    # or load-bearing; it is load-bearing, and it is the punchline.
    # Every number on this line is MEASURED off the arrays that got drawn,
    # including the rank: a printed `rank = h` is a claim, `rank =
    # matrix_rank(W_QK)` that happens to equal h is evidence, and a reader
    # was right to ask which one this was. The residual rides here too,
    # attached to the identity it certifies instead of floating in the
    # margin beside the `=`.
    gap = float(np.max(np.abs(contract(factored) - contract(merged))))
    gap_tex = ("0" if gap == 0.0            # `10^-15`, never `1e-15`: set as
               else rf"10^{{{int(np.floor(np.log10(gap)))}}}")  # math a bare
    rank = int(np.linalg.matrix_rank(g["W_QK"]))               # `e-15` is a
    s.add(MathLabel(                                           # subtraction
        rf"W_{{QK}} = W_Q W_K^{{T}},\quad \mathrm{{rank}}\,W_{{QK}} = "
        rf"{rank} = h < {d['d']} = d_{{model}},\quad "
        rf"\max|\mathrm{{row\,1}} - \mathrm{{row\,2}}| \leq {gap_tex}",
        (0.5 * (extent(merged)[0] + extent(merged)[1]),
         min(spec_y) - 0.34), role=Role.ACCENT1, ha="center", va="top",
        size_pt=10.5))

    # the dimensions, printed from the arrays. d_head is the number the
    # figure is about: it is the one axis that exists only in row one.
    x0, x1, _, y1 = extent(factored)
    s.add(MathLabel(
        # index names set exactly as they are on the legs — upright here
        # and italic on the wire would read as two different alphabets
        rf"s={d['s']},\ t={d['t']},\ "
        rf"d=e={d['d']}\ (d_{{model}}),\ "
        rf"h={d['h']}\ (d_{{head}})",
        ((x0 + x1) / 2, y1 + 0.30), role=Role.ANNOTATION,
        ha="center", va="bottom", size_pt=10.0))
    # what the dangling legs ARE. Without this the figure never says where
    # its output lives, and a reader must guess whether the softmax is
    # missing or merely elsewhere.
    s.add(MathLabel(
        r"X,\ Y:\ \mathrm{query\ and\ key\ inputs}\ \ \cdot\ \ "
        r"\mathrm{the\ dangling\ legs\ }(\mathrm{s},\mathrm{t})"
        r"\mathrm{\ are\ the\ attention\ logits,\ before\ softmax}",
        ((x0 + x1) / 2, y1 + 0.30 + p["banner_gap"]), role=Role.ANNOTATION,
        ha="center", va="bottom", size_pt=9.5))

    xa0, xa1, _, ya1 = extent(factored, pad=p["pad"])
    xb0, xb1, _, yb1 = extent(merged, pad=p["pad"])
    # the bottom is set by the LOWEST spec caption, not by the padded node
    # extent — padding the row and then dropping the caption below it again
    # leaves a band of dead paper the size of the caption drop
    s.xlim = (min(xa0, xb0, mid_x) - 0.30, max(xa1, xb1))
    s.ylim = (min(spec_y) - 0.34 - 0.34,
              max(ya1, yb1) + 0.34 + p["banner_gap"])
    return s


def assertions(g):
    p = g["params"]
    factored, merged = _nets(g)
    c = Checks()

    # both diagrams are drawable networks with conformable wires
    for net in (factored, merged):
        check_index_arity(c, net)
        check_index_dims(c, net)
        check_leg_aim(c, net)
        check_output(c, net, (p["seq_q"], p["seq_k"]))
        c.check(free(net) == ("s", "t"),
                f"dangling legs {free(net)} are not the score axes (s, t)")

    # each drawn network really is the array the figure names
    check_einsum(c, factored, g["scores"], rtol=1e-9, atol=1e-12)
    check_einsum(c, merged, g["scores"], rtol=1e-9, atol=1e-12)

    # THE claim: two different diagrams, one array. Asserted between the
    # drawn networks, not against a third expression written by hand.
    a, b = contract(factored), contract(merged)
    c.check(np.allclose(a, b, rtol=1e-9, atol=1e-12),
            f"the two drawn networks differ by {np.max(np.abs(a - b)):.2e} — "
            f"the figure's whole argument is that they cannot")

    # ...and the diagrams are genuinely different, or the claim is empty
    c.check(spec(factored) != spec(merged),
            "the two rows contract identically — nothing is being argued")
    c.check(len(merged.tensors) == len(factored.tensors) - 1,
            "row two does not have one tensor fewer; the head axis has not "
            "been contracted away")
    c.check("h" not in dims(merged),
            "the head axis survives into row two, so it was never internal")

    # the rank claim that makes 'a parameterization, not a mechanism' true
    r = int(np.linalg.matrix_rank(g["W_QK"]))
    c.check(r == p["d_head"],
            f"rank(W_QK) = {r}, not d_head = {p['d_head']} — the bilinear "
            f"form is not pinned to the head width and the claim is false")
    c.check(p["d_head"] < p["d_model"],
            "d_head is not smaller than d_model, so the low-rank claim is "
            "vacuous for these parameters")
    # a rank-deficient X or Y would make the equality hold for reasons that
    # have nothing to do with the head axis
    for k in ("X", "Y"):
        c.check(np.linalg.matrix_rank(g[k]) == g[k].shape[0],
                f"{k} is rank deficient; the equality would then be an "
                f"accident of the data rather than an identity")

    # routing honesty, borrowed from schematic: no wire crosses another
    for name, net in (("factored", factored), ("merged", merged)):
        n = crossing_count(edges(net))
        c.check(n == 0, f"{name} row has {n} wire crossings; a crossing in a "
                        f"chain this simple is a layout error")
    c.done()
