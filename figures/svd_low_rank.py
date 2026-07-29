"""A smooth kernel matrix is four rank-1 modes plus an admitted residual.

The object is the Gaussian Gram matrix K[i,j] = exp(-(i-j)^2/2l^2) on 32
points — the kernel behind Nystrom approximation, RBF regression, and the
kernel reading of attention. Its spectrum decays fast, so "nearly low
rank" is a true statement about it and not a hope.

The drawn equation is EXACT and it shows K itself, because the residual
is drawn as the last term: K = sum_k sigma_k u_k v_k^T + R. So the
approximation claim is not on trust — R is rendered at the SAME symmetric
color scale as everything else, and its near-flatness is the claim,
readable off the page. (An earlier version drew only K_4 = sum, which is
exact by definition and left the interesting quantity invisible; a cold
reader caught it.)

The modes are SIGNED and oscillate at rising frequency, so all blocks
share one symmetric scale with a colorbar; the Hinton inset carries the
sign structure directly, which no monotone ramp can.
"""

import numpy as np

from figlib.format import WIDE
from figlib.gates import Checks
from figlib.matrix import (Block, bounds, check_cell_uniform,
                           check_conformable, check_expr, check_hinton_area,
                           check_no_overlap, expr, heat, hinton, outline)
from figlib.plots import Ticks, colorbar, linear
from figlib.scene import MathLabel, Scene
from figlib.style import WEIGHT_ACTOR, WEIGHT_BG, Role
from figlib.theme import RISO

CLAIM = (
    "A smooth Gaussian kernel matrix is nearly rank 4: drawn exactly as "
    "K = sum_{k<=4} sigma_k u_k v_k^T + R, its four rank-1 modes are "
    "signed and oscillate at rising frequency like a Fourier basis, and "
    "the residual R is visibly flat at the same color scale as K — 5.6% "
    "relative Frobenius error, 99.7% of the energy kept."
)

THEME = RISO
FORMAT = WIDE

PARAMS = {
    "n": 32,
    "ell": 6.0,             # kernel width in samples; sets the spectrum decay
    "rank": 4,
    "cell": 0.045,          # a 32-cell block is 1.44 math units wide
    "gap": 0.42,
    "op_gap": 0.62,
    "hinton_rows": 4,       # top-k right singular vectors in the inset
    "hinton_cols": 10,      # first key positions shown
    "hinton_cell": 0.15,
    "cap_lift": 0.22,       # caption above a block's top edge
    "inset_drop": 0.75,     # inset below the expression row
    "bar_thick": 0.14,
    "pad": 0.30,
}


def _kernel(p):
    """The Gaussian Gram matrix on a uniform grid. Deterministic, fully
    stated by (n, ell) — nothing seeded, nothing scraped."""
    i, j = np.indices((p["n"], p["n"]))
    return np.exp(-((i - j) ** 2) / (2.0 * p["ell"] ** 2))


def compute(p):
    A = _kernel(p)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    r = p["rank"]
    terms = [s[k] * np.outer(U[:, k], Vt[k, :]) for k in range(r)]
    A_r = np.sum(terms, axis=0)
    energy = float(np.sum(s[:r] ** 2) / np.sum(s ** 2))
    resid = float(np.linalg.norm(A - A_r, "fro") / np.linalg.norm(A, "fro"))
    return {"A": A, "A_r": A_r, "R": A - A_r, "terms": terms, "s": s,
            "U": U, "Vt": Vt,
            "V_top": Vt[:p["hinton_rows"], :p["hinton_cols"]],
            "energy": energy, "resid": resid, "params": p}


def _row(g):
    """K = sum_k sigma_k u_k v_k^T + R, drawn and gated as the SAME
    identity. Including R is what puts the approximation on the page:
    the equation stays exact, K is visible, and the residual's flatness
    is the claim rather than a number to be believed."""
    p = g["params"]
    terms = [Block(*g["A"].shape, values=g["A"], name="K"), "="]
    for k, T in enumerate(g["terms"]):
        terms += [Block(*T.shape, values=T, name=f"s{k}"), "+"]
    terms += [Block(*g["R"].shape, values=g["R"], name="R")]
    ops, placed = expr(terms, cell=p["cell"], gap=p["gap"],
                       op_gap=p["op_gap"], origin=(0.0, 0.0))
    return terms, ops, placed


def _hinton_block(g):
    p = g["params"]
    _, _, placed = _row(g)
    return Block(p["hinton_rows"], p["hinton_cols"], values=g["V_top"],
                 origin=(placed[0].extent[0],
                         placed[0].extent[2] - p["inset_drop"]),
                 cell=p["hinton_cell"], name="Vt")


def build(g):
    p = g["params"]
    _, ops, placed = _row(g)
    # ONE symmetric scale across every block. Symmetric because the modes
    # are signed — sigma_k u_k v_k^T oscillates — so 0 must sit at the same
    # place in the ramp everywhere; common because per-block normalization
    # would rescale each summand to itself and make the sum look wrong.
    vmax = max(float(np.abs(T).max())
               for T in [g["A"], *g["terms"], g["R"]])

    s = Scene()
    for k, b in enumerate(placed):
        s.add(heat(b, ramp=THEME.ramp, vmin=-vmax, vmax=vmax))
        s.add(*outline(b, width_scale=WEIGHT_ACTOR if k == 0 else WEIGHT_BG))
        if k == 0:
            cap = r"K"
        elif k <= p["rank"]:
            cap = (rf"\sigma_{k}\boldsymbol{{u}}_{k}"
                   rf"\boldsymbol{{v}}_{k}^{{T}}")
        else:
            cap = r"R = K - K_4"
        s.add(MathLabel(cap, (b.extent[0] + b.width / 2,
                              b.extent[3] + p["cap_lift"]),
                        role=Role.ANNOTATION, ha="center", va="bottom",
                        size_pt=9.0))
    s.add(*ops)

    # sign structure of the top right-singular vectors — no ramp can do this
    hb = _hinton_block(g)
    s.add(*outline(hb, width_scale=WEIGHT_BG), *hinton(hb))
    s.add(MathLabel(
        r"\boldsymbol{v}_1..\boldsymbol{v}_4:\ \mathrm{area}=|v|,\ "
        r"\mathrm{fill}=\mathrm{sign}\ \ (\mathrm{the\ bar\ scales\ "
        r"every\ block\ above})",
        (hb.extent[1] + 0.18, (hb.extent[2] + hb.extent[3]) / 2),
        role=Role.ANNOTATION, ha="left", va="center", size_pt=9.0))

    # the ramp finally gets a scale: without it the reader cannot see
    # where zero sits, and "these modes are SIGNED" is the claim
    bar_scale = linear(-vmax, vmax)
    tv = np.array([-vmax, 0.0, vmax])
    s.add(*colorbar(bar_scale, THEME.ramp,
                    at=(placed[-1].extent[1] + 0.62, placed[-1].extent[2]),
                    length=placed[-1].height, thickness=p["bar_thick"],
                    orient="y",
                    ticks=Ticks(values=tv, positions=bar_scale.fwd(tv),
                                labels=("-", "0", "+")),
                    label=None))

    # the honesty pass: what the drawn equation leaves out, as a number
    s.add(MathLabel(
        rf"\|K - K_4\|_F / \|K\|_F = {g['resid']:.3f}"
        rf"\quad ({100 * g['energy']:.1f}\%\ \mathrm{{energy\ kept}})",
        (placed[0].extent[0], hb.extent[2] - 0.30),
        role=Role.ANNOTATION, ha="left", va="top", size_pt=10.0))

    xlim, ylim = bounds(list(placed) + [hb], pad=p["pad"])
    s.xlim = (xlim[0], xlim[1] + 1.9)          # room for the inset legend
    s.ylim = (ylim[0] - 0.62, ylim[1] + p["cap_lift"] + 0.30)
    return s


def assertions(g):
    p = g["params"]
    terms, _, placed = _row(g)
    it = iter(placed)
    drawn = [next(it) if isinstance(t, Block) else t for t in terms]

    c = Checks()
    # the drawn equation is EXACT: A_4 IS the sum of the four rank-1 terms
    check_expr(c, drawn, rtol=1e-12)
    check_conformable(c, drawn)
    blocks = [t for t in drawn if isinstance(t, Block)]
    check_cell_uniform(c, blocks)
    check_no_overlap(c, blocks)
    # each drawn MODE is genuinely rank 1 (the residual is not, by design)
    for k, T in enumerate(g["terms"]):
        c.check(np.linalg.matrix_rank(T) == 1, f"term {k} is not rank 1")
    c.check(np.linalg.matrix_rank(g["R"]) > 1,
            "the residual is rank 1 — then rank 5 would be exact and the "
            "figure has picked the wrong truncation to argue about")
    # the residual block really is the discarded part, not a redraw
    c.check(np.allclose(g["R"], g["A"] - g["A_r"], atol=1e-15),
            "the drawn residual block is not K - K_4")
    c.check(float(np.abs(g["R"]).max()) < 0.25 * float(np.abs(g["A"]).max()),
            "the residual is not visibly smaller than K at a shared scale — "
            "the figure's central visual claim would be false")
    # the annotated residual and energy match the arrays that got drawn
    resid = float(np.linalg.norm(g["A"] - g["A_r"], "fro")
                  / np.linalg.norm(g["A"], "fro"))
    c.check(abs(resid - g["resid"]) < 1e-12, "annotated residual is stale")
    # energy kept and residual are two views of one number: r^2 + e = 1
    c.check(abs(g["resid"] ** 2 + g["energy"] - 1.0) < 1e-9,
            "residual and energy fraction are inconsistent")
    # the depicted object really is a symmetric PSD kernel
    c.check(np.allclose(g["A"], g["A"].T), "kernel matrix is not symmetric")
    c.check(g["s"].min() > -1e-12, "kernel matrix is not PSD")
    # the low-rank claim is a fact about THIS kernel, not a hope: the
    # rank-4 truncation must clear the energy the CLAIM states
    c.check(g["energy"] > 0.99,
            f"rank-{p['rank']} keeps only {100 * g['energy']:.1f}% — the "
            f"'nearly low rank' claim does not hold for this kernel")
    # the modes really do oscillate at rising frequency: sign changes grow
    signs = [int(np.sum(np.diff(np.sign(g["Vt"][k, :])) != 0))
             for k in range(p["rank"])]
    c.check(signs == sorted(signs) and signs[-1] > signs[0],
            f"mode sign-change counts {signs} are not increasing — the "
            f"'rising frequency' claim is not what the arrays show")
    # the Hinton inset encodes area, not side
    hb = _hinton_block(g)
    check_hinton_area(c, hb, hinton(hb))
    # the inset is worth drawing only if those vectors actually change sign
    c.check(np.any(g["V_top"] > 0) and np.any(g["V_top"] < 0),
            "singular vectors are single-signed — the Hinton inset is idle")
    c.done()
