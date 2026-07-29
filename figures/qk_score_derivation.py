"""The QK attention score of two perturbed vectors, expanded term by term.

Design steps (architecture.md 0-9). This is a Class B figure in the sense
that layout is the content — but the layout is TYPOGRAPHY, not a graph.
There is numerics, and it is the thing being asserted: the four drawn
terms are generated from one spec, and that same spec is summed and
compared against the unexpanded product.

0. EARN IT. The prose version is one line: "expand the bilinear form and
   you get four terms — signal-signal, two cross terms, error-error."
   True, and it fails in the one way that matters: read as prose it is a
   LIST, and the reader has no reason to believe the list is complete or
   that its members differ in size. Both facts are structural. Written as
   an equation with the terms co-located on one line, exhaustiveness is
   perceptual (you can see the row closes), and the size ordering can be
   carried by ink weight on the terms themselves. The specific inference
   the reader must make — "the corruption of an attention score is, to
   leading order, TWO terms, each pairing a full-size signal against an
   error" — becomes a thing you look at rather than a thing you derive.
1. CLAIM. Below.
2. REPRESENTATION. The equation IS the figure. No boxes, no arrows, no
   node/edge furniture: the expert's private picture of this is not a
   diagram, it is the expansion written out with a finger pointing at the
   middle two terms. So the primitive is `derivation.derivation_row` —
   terms at their real typeset widths, operators midway in the whitespace
   — and the only non-typographic mark on the page is one brace. Rung:
   one instance of the identity (a single score, a single W), because the
   claim is algebraic and a family would only repeat it. The 2x2 structure
   (clean/noisy query x clean/noisy key) is what the four glosses name, so
   the reader can check the enumeration is closed by reading the glosses
   alone.
3. SIZE. WIDE. A derivation row is wide and short: 1000 px of slot, ~230
   px of canvas. Layout is stated in READING px (see PARAMS/compute) and
   one math unit IS one reading px, which is the honest frame for a
   typographic object — WIDE is declared at 1000 px but read at
   1000/1.45 = 690, and `size_pt` is reading-size pt, so measuring in
   display px would lay the row out 1.45x too tight. assertions() pins the
   identity.
4. TRAVERSAL. The eye enters on the left at `s = (...)(...)`, the thing it
   already knows, and runs right along one line. The 3-second glance
   yields "an equation, four terms, two of them highlighted and braced".
   The 30-second read yields the 2x2 enumeration from the glosses, the
   name of the braced pair, and the fact that the last term is drawn
   fainter because it is smaller.
5. MECHANISM ANNOTATION. Readable off the figure: which factor is
   corrupted in each term (the glosses); that the two cross terms are the
   first-order error (the brace); that the last term is second order (its
   muted ink, and its gloss); and that nothing was dropped (the sans line
   at the foot — the exhaustiveness claim is the punchline and would
   otherwise ride silently on the reader's trust).
6. READER EFFORT. Delegated: composing "these two are first order" with
   "this one is drawn faint" into the size ordering |t1| > |cross| > |t4|.
   Kept: nothing about decoding — every term carries its own gloss, so no
   subscript has to be held in memory across the row.
7. CHANNELS. Ink weight is the hierarchy, and it encodes ORDER IN eps:
   CONTENT for the signal term, ACCENT1 for the two first-order cross
   terms (they are THE object — the figure's punchline), MUTED for the
   second-order term. Hue is not carrying correspondence here and no
   categorical slot is used; that would be decoration. The glosses are a
   second VOICE, not a second colour: sans register, ANNOTATION ink, 0.8x
   the size, hung on a hairline tick. The brace is the only drawn glyph,
   at annotation weight, so it groups without competing with the type.
8. HONESTY. The identity is exact — bilinearity, no truncation — and that
   is stated on the figure rather than assumed. What the figure does elide,
   and confesses only here: W is drawn as one matrix, where a real
   attention score is x_q^T W_Q W_K^T x_k / sqrt(d) (the decomposition is
   unchanged, the factorization is not the claim); eps is left abstract
   (quantization, an upstream ablation, an adversarial perturbation all
   land here); and the softmax that consumes s is absent, so the figure
   says nothing about how a score error becomes an attention error. The
   accidental assertion worth auditing: the four terms are drawn the same
   SIZE, which is not their magnitude ordering — magnitude rides on ink,
   and assertions() checks that the ordering it claims is the one the
   numbers actually have.
9. GATES. Numerical: the four terms generated from the spec sum to the
   unexpanded product to machine precision (exhaustiveness), and over an
   ensemble at |eps| = delta|x| the mean magnitudes order as
   signal : cross : second = 1 : delta : delta^2, which is the ordering
   the ink weights claim. Metric: the glosses are laid out from runtime
   font metrics and the builder does NOT solve their collisions, so their
   pairwise non-overlap and their clearance below the term boxes are
   asserted, as is the brace covering exactly the two cross terms and the
   whole row fitting inside the declared xlim.
"""

from __future__ import annotations

import numpy as np

from figlib import schematic as sch
from figlib.derivation import Term, derivation_row
from figlib.format import WIDE
from figlib.scene import Brace, MathLabel, Scene
from figlib.style import Role
from figlib.theme import RISO
from figlib.typeset import apply_register, render_math

CLAIM = (
    "A QK attention score computed on perturbed vectors decomposes exactly "
    "into four interaction terms — the intended signal-signal score, two "
    "cross terms that pair a full-size signal against an error, and one "
    "error-error term — so the corruption of an attention score is, to "
    "leading order, the two cross terms and nothing else."
)

THEME = RISO
FORMAT = WIDE

PARAMS = {
    # The expansion, stated once. Each entry is (query factor, key factor)
    # over {signal, error}; the LaTeX drawn on the page, the ink weight, and
    # the numbers assertions() sums are ALL derived from this list, so a
    # term cannot be drawn without being checked.
    "expansion": [
        ("x", "x", "the intended score"),
        ("x", "e", "the key error"),
        ("e", "x", "the query error"),
        ("e", "e", "error on error"),
    ],
    "braced": (1, 2),               # the two cross terms: first order in eps
    "brace_label": r"\text{first order in }\varepsilon",
    # names the object: the subscripts q, k and the matrix W are the only
    # thing on the page that says "attention", and a cold reader should not
    # have to infer the subject from a subscript
    "title": "the QK attention score, with an error on each side",
    "title_y": 30.0,
    "foot": "no truncation: these four terms are all of them",
    # --- typography, in READING px (= math units; see compute) -------------
    "pad_frac": 0.08,               # layout.Transform's uniform padding
    "size_pt": 11.0,                # reading-size pt for the mathematics
    "gloss_scale": 0.8,             # the second voice, quieter
    "gap_pt": 17.0,                 # clearance floor on each side of an operator
    # 3.4 term-heights, not the builder's default 1.8: the gloss tick spans
    # the middle 0.6 of the drop, so at 1.8 the tick's top end (0.36 h below
    # the baseline) lands INSIDE a term box (half-height 0.5 h) and strikes
    # the glyphs. Anything above 2.5 clears; 3.4 leaves ~5 display px.
    "drop_factor": 3.4,
    "brace_y": 17.0,                # above the term boxes (half-height ~9)
    "brace_depth": 11.0,
    "foot_y": -82.0,
    "ylim": (-94.0, 50.0),
    "min_gloss_gap": 8.0,           # reading px of clear air between glosses
    # --- the numerical gate -------------------------------------------------
    "dim": 64,
    "n_samples": 512,
    "delta": 0.1,                   # |eps| / |x|
    "seed": 20260729,
}

_SYM = {("x", "q"): r"x_q", ("e", "q"): r"\varepsilon_q",
        ("x", "k"): r"x_k", ("e", "k"): r"\varepsilon_k"}

LHS = r"s = (x_q + \varepsilon_q)^{\top} W (x_k + \varepsilon_k)"


def _term_latex(left: str, right: str) -> str:
    return rf"{_SYM[(left, 'q')]}^{{\top}} W {_SYM[(right, 'k')]}"


def _role(left: str, right: str) -> Role:
    """Ink weight = order in eps. Signal, first order, second order."""
    order = (left == "e") + (right == "e")
    return (Role.CONTENT, Role.ACCENT1, Role.MUTED)[order]


def _unit_rows(rng, n: int, d: int) -> np.ndarray:
    v = rng.standard_normal((n, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def compute(p):
    # --- the frame: one math unit is one READING px -------------------------
    # WIDE is declared at 1000 px and read near 690 (format.ink_scale 1.45),
    # and MathLabel.size_pt is reading-size pt that the render rescales. So
    # the row is laid out in reading px and the format maps them to the page;
    # measuring in display px would compress the row by 1.45x.
    units_across = WIDE.display_width_px / (WIDE.ink_scale * (1.0 + 2.0 * p["pad_frac"]))
    size_pt = p["size_pt"]
    gloss_pt = p["gloss_scale"] * size_pt

    terms = [Term(_term_latex(l, r), gloss=g, role=_role(l, r), key=f"{l}{r}")
             for l, r, g in p["expansion"]]
    items, anchors = derivation_row(
        terms, lhs=LHS, y=0.0, size_pt=size_pt, gloss_size_pt=gloss_pt,
        px_per_unit=1.0, gap_pt=p["gap_pt"], drop_factor=p["drop_factor"])

    # widths at the SAME metrics the builder and the mechanical gate use
    w_term = [render_math(t.latex, size_pt).width_px for t in terms]
    h_term = [render_math(t.latex, size_pt).height_px for t in terms]
    w_gloss = [render_math(apply_register(t.gloss, "sans"), gloss_pt).width_px
               for t in terms]

    # the row runs from 0 to its own right edge; centre it in the slot
    row_w = anchors[-1][0] + w_term[-1] / 2.0
    x0 = row_w / 2.0 - units_across / 2.0
    xlim = (x0, x0 + units_across)

    i, j = p["braced"]
    brace = ((anchors[i][0] - w_term[i] / 2.0, p["brace_y"]),
             (anchors[j][0] + w_term[j] / 2.0, p["brace_y"]))

    # the foot is a statement about the ENUMERATION, so it centres under the
    # terms, not under the canvas: the left side of the row carries no gloss
    # and a canvas-centred line reads as hanging off the first two terms
    foot_x = 0.5 * (anchors[0][0] - w_term[0] / 2.0
                    + anchors[-1][0] + w_term[-1] / 2.0)

    # --- the numbers the drawn terms claim ----------------------------------
    # Same spec, evaluated: the sum of the four terms against the unexpanded
    # product (exhaustiveness), and their magnitudes at |eps| = delta |x|
    # (the ordering the ink weights assert).
    rng = np.random.default_rng(p["seed"])
    n, d = p["n_samples"], p["dim"]
    W = rng.standard_normal((d, d)) / np.sqrt(d)
    vec = {("x", "q"): _unit_rows(rng, n, d), ("x", "k"): _unit_rows(rng, n, d),
           ("e", "q"): p["delta"] * _unit_rows(rng, n, d),
           ("e", "k"): p["delta"] * _unit_rows(rng, n, d)}
    parts = {(l, r): np.einsum("ni,ij,nj->n", vec[(l, "q")], W, vec[(r, "k")])
             for l, r, _ in p["expansion"]}
    full = np.einsum("ni,ij,nj->n",
                     vec[("x", "q")] + vec[("e", "q")], W,
                     vec[("x", "k")] + vec[("e", "k")])

    return {
        "params": p, "items": items, "anchors": anchors, "terms": terms,
        "w_term": w_term, "h_term": h_term, "w_gloss": w_gloss,
        "gloss_pt": gloss_pt, "row_w": row_w, "units_across": units_across,
        "brace": brace, "foot_x": foot_x, "parts": parts, "full": full,
        "xlim": xlim, "ylim": p["ylim"],
    }


def build(g):
    p = g["params"]
    s = Scene(xlim=g["xlim"], ylim=g["ylim"])
    s.add(*g["items"])
    # the only drawn glyph on the page: the two cross terms are one object
    s.add(Brace(g["brace"][0], g["brace"][1], side=1.0,
                depth=p["brace_depth"], label=p["brace_label"]))
    # the subject, in the second voice, hung on the row's left edge — the
    # quadrant the equation leaves empty
    s.add(MathLabel(p["title"], (0.0, p["title_y"]), role=Role.ANNOTATION,
                    size_pt=g["gloss_pt"], ha="left", va="center",
                    register="sans", pin=True))
    # the punchline that the row cannot draw: the enumeration is closed
    s.add(MathLabel(p["foot"], (g["foot_x"], p["foot_y"]),
                    role=Role.ANNOTATION, size_pt=g["gloss_pt"],
                    ha="center", va="center", register="sans", pin=True))
    return s


def assertions(g):
    from figlib.gates import Checks

    c = Checks()
    p = g["params"]
    anchors, w_term, h_term = g["anchors"], g["w_term"], g["h_term"]
    parts, full = g["parts"], g["full"]

    # --- the mathematics the figure argues ---------------------------------
    # 1. EXHAUSTIVE: the four terms drawn (generated from p["expansion"])
    #    reconstruct the unexpanded product. A missing, duplicated or
    #    mis-paired entry in the spec fails here and changes the picture.
    total = sum(parts.values())
    resid = float(np.max(np.abs(total - full)))
    c.check(len(parts) == 4 and resid < 1e-12,
            f"the drawn terms do not sum to the product: {len(parts)} terms, "
            f"max residual {resid:.3e}")

    # 2. The ink weights claim an ORDER IN eps. Check the magnitudes have it,
    #    and at the right rate: cross ~ delta, second order ~ delta^2.
    mag = {k: float(np.mean(np.abs(v))) for k, v in parts.items()}
    sig, ee = mag[("x", "x")], mag[("e", "e")]
    cross = 0.5 * (mag[("x", "e")] + mag[("e", "x")])
    c.check(sig > cross > ee,
            f"the magnitude ordering the ink claims does not hold: "
            f"signal {sig:.4f}, cross {cross:.4f}, second order {ee:.4f}")
    for name, ratio, want in (("cross", cross / sig, p["delta"]),
                              ("second order", ee / sig, p["delta"] ** 2)):
        c.check(0.5 * want < ratio < 2.0 * want,
                f"{name} / signal = {ratio:.4g}, not O({want:g}) — the term "
                f"is not the order in eps its ink weight says it is")

    # --- the layout, which depends on runtime font metrics ------------------
    # 3. The builder does not solve gloss collisions (its doctrine: report,
    #    don't solve), so the prose length is the figure's problem. These are
    #    the checks that fail when a gloss is reworded.
    for i in range(len(anchors) - 1):
        gap = ((anchors[i + 1][0] - anchors[i][0])
               - (g["w_gloss"][i] + g["w_gloss"][i + 1]) / 2.0)
        c.check(gap >= p["min_gloss_gap"],
                f"glosses {i} and {i + 1} are {gap:.1f} reading px apart, "
                f"under the {p['min_gloss_gap']:.0f} px floor — shorten the prose")

    # 4. The gloss tick must start BELOW the term box, not inside the glyphs.
    #    The builder ties the tick to drop_factor, so this is a live coupling.
    tick_top = (1.0 - 0.6) / 2.0 * p["drop_factor"] * min(h_term)
    c.check(tick_top > 0.5 * max(h_term) + 1.0,
            f"the gloss tick starts {tick_top:.2f} px below the row and the "
            f"term boxes reach {0.5 * max(h_term):.2f} px — it strikes the "
            f"mathematics; raise drop_factor")

    # 5. The brace covers exactly the two cross terms: it must reach their
    #    outer edges and clear both neighbours, or it groups the wrong thing.
    i, j = p["braced"]
    (bx0, _), (bx1, _) = g["brace"]
    c.check(bx1 - bx0 > 0, "the brace has no span")
    for k in range(len(anchors)):
        inside = bx0 - 1e-9 <= anchors[k][0] <= bx1 + 1e-9
        c.check(inside == (i <= k <= j),
                f"term {k} ({g['terms'][k].latex}) is "
                f"{'inside' if inside else 'outside'} the brace, which claims "
                f"exactly the first-order pair")
    for k in (i - 1, j + 1):
        if 0 <= k < len(anchors):
            edge = anchors[k][0] + (1 if k < i else -1) * w_term[k] / 2.0
            clear = (bx0 - edge) if k < i else (edge - bx1)
            c.check(clear > 0.0,
                    f"the brace overlaps term {k}: {clear:.2f} px")

    # 6. The frame identity the whole layout rests on: one math unit is one
    #    READING px, i.e. px_per_unit / ink_scale == 1. If xlim, the format or
    #    the padding drifts, every measured width above becomes a lie.
    scale = sch.px_per_unit(g["xlim"], WIDE.display_width_px, p["pad_frac"])
    c.check(abs(scale / WIDE.ink_scale - 1.0) < 1e-9,
            f"one math unit is {scale / WIDE.ink_scale:.4f} reading px, not 1 — "
            f"the typeset widths no longer match the page")
    c.check(g["row_w"] <= g["units_across"],
            f"the row is {g['row_w']:.0f} reading px wide and the slot carries "
            f"{g['units_across']:.0f} — cut a term or shorten the left side")

    c.done()
