"""PRML Fig 1.4 — polynomial capacity, swept, as small multiples.

The design step, 0-9.

0. Earn it. The prose is "increasing the polynomial degree fits the
   sample better and the signal worse." That sentence is believed and
   not seen; what it hides is *where* the extra freedom goes. The
   inference the reader must make is a comparison across four fits of
   ONE sample, and comparison across four things is exactly what prose
   cannot co-locate. Drawn, the conclusion is a visual feature: the red
   curve leaves the green one and starts chasing dots.
1. CLAIM below.
2. Representation. Rung: family — four instances of one estimator,
   indexed by M, over a bound concrete case (one 10-point sample). The
   private picture is not four separate plots; it is one plot the eye
   flickers between, which is why every non-M quantity is drawn
   identically and the panels sit in a grid rather than a row.
3. Size and slot. WIDE. Four panels, each carrying two axes with tick
   labels and two in-panel annotations; at COLUMN each slot is 317 px
   and the type would have to shrink, which is not allowed.
4. Traversal. Enter at [a] (flat line, obviously wrong), read left to
   right, top to bottom; the 3-second glance yields "the red curve gets
   wigglier"; the 30-second study yields "at M = 9 it passes through
   every dot and leaves the frame twice." The grid is the traversal:
   [c] sits directly under [a] so the good fit is one saccade from the
   worst.
5. Mechanism annotation. Three quantities have to be readable off the
   page to reconstruct the argument, and each is written on the object
   it describes, in that object's own ink: the estimator and its degree
   on the fit, N on the dots, sin 2*pi*x on the signal. N is there
   because M = 9 = N - 1 is the whole punchline and a reader made to
   count dots is doing the figure's work.
6. Reader effort. Nothing spent on decoding: three hues, each named at
   its referent, identical in every panel. The kept step is the
   punchline — noticing that the M = 9 curve interpolates.
7. Channels. Hue = correspondence (three nouns: data, truth, fit).
   Position carries everything that argues. No ramp, no ordering
   channel: M is not drawn as a quantity, it is drawn as four panels.
7b. Binding. `data` and `truth` are the same object in all four panels,
   pixel-identical; `fit` is the only key that moves. That is the whole
   device, and it is the declaration below rather than a promise.
8. Honesty. Two lies to admit. (i) The M = 9 fit is truncated at
   |t| = 1.5 — it reaches -2.83 — so the frame hides how bad it is; the
   truncation is drawn as a curve running off the frame edge rather
   than a curve that bends back, so the reader sees the exit. (ii) One
   seed. The noise is a single draw and the figure says nothing about
   the sampling distribution of the fits; the claim is about this
   sample, and PARAMS names the seed.
9. Gates. The least-squares certificate (residual orthogonal to every
   design column) at each M; interpolation at M = 9 as an *achieved*
   residual; strictly decreasing RSS along the nested sequence; the
   clipper never emits a point outside the band; the shared lims are one
   object across the four panels.
"""

import numpy as np
from numpy.polynomial import chebyshev

from figlib.correspond import Correspondence, keyed
from figlib.figure import Figure, Panel
from figlib.format import WIDE
from figlib.gates import Checks
from figlib.plots import Ticks, axis, linear, markers, px_units, series, tick_honesty
from figlib.scene import Curve, MathLabel, Scene
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "Least-squares polynomial fits of one 10-point sample of "
    "sin(2*pi*x) + noise go from underfitting to interpolating as the degree M "
    "rises: M = 0 and M = 1 cannot reach the signal, M = 3 recovers it, and "
    "M = 9 — one free coefficient per data point — passes exactly through "
    "every observation and swings off the frame between them, fitting the "
    "noise rather than the function."
)

THEME = RISO
FORMAT = WIDE

PARAMS = {
    "n_points": 10,          # PRML's N; M = 9 is then interpolation
    "noise_sigma": 0.25,
    "seed": 4,               # ONE draw, named — see honesty note (ii)
    "degrees": (0, 1, 3, 9),
    "n_curve": 1200,         # dense enough for the M = 9 oscillation
    "t_view": 1.5,           # curves are truncated at |t| = this
    "gap_px": 46.0,
}

# Hue is a referential noun; these three slots are fixed for the whole
# figure. Not the palette prefix (0,1,2): indigo/teal/coral is the
# best-separated triple in RISO's fixed order (all-pairs dE 13.9 under
# the worst simulated deficiency, vs 13.6 for the prefix), and it lands
# on PRML's own blue / green / red without anyone choosing a hex.
HUE_DATA = RISO.categorical(0)      # indigo
HUE_TRUTH = RISO.categorical(3)     # teal
HUE_FIT = RISO.categorical(5)       # coral

# Panel frame, in math coords. Identical in all four panels by construction:
# these are module constants, not four literals, because a binding written
# twice is the trap correspond.py exists to catch.
XLIM = (-0.30, 1.14)
YLIM = (-2.12, 1.88)
# The spine sits a clear 0.24 below the truncation band: at M = 9 the fit
# leaves through t = -1.5 twice, and a readback read the exit stroke as
# touching the axis arrow when the two were 0.12 apart.
X_AXIS_T = -1.74          # x spine, below the drawn band
Y_AXIS_X = -0.09          # t spine, left of x = 0
TICK_PX = 6.0
MARKER_PX = 3.4
ANNOT_PT = 10.0
# Nothing is drawn above t = 1.5 in any panel (the band truncates there),
# so this strip is free for the panel's caption line — the estimator and
# its degree, in the fit's own ink.
CAPTION_T = 1.86


def _truth(x):
    return np.sin(2 * np.pi * np.asarray(x, dtype=float))


def _edge(x0, y0, x1, y1, lo, hi):
    """Where the segment from the INSIDE point (x0, y0) to the outside
    point (x1, y1) crosses the band boundary."""
    b = hi if y1 > hi else lo
    s = (b - y0) / (y1 - y0)
    return (x0 + s * (x1 - x0), b)


def _clip_band(x, y, lo, hi):
    """Contiguous runs of the polyline inside [lo, hi], cut at the exact
    crossings.

    Not `Scene(clip="frame")`: that clips the whole scene group, tick
    labels and axis names included, so the frame would have to be pushed
    out past every piece of furniture and the truncation would stop
    coinciding with the band the reader sees. Clipping in compute keeps
    the truncation a property of the geometry, which is also what lets
    assertions() check it.
    """
    inside = (y >= lo) & (y <= hi)
    segs, cur = [], []
    for i in range(x.size):
        if inside[i]:
            if not cur and i > 0:
                cur.append(_edge(x[i], y[i], x[i - 1], y[i - 1], lo, hi))
            cur.append((x[i], y[i]))
        elif cur:
            cur.append(_edge(x[i - 1], y[i - 1], x[i], y[i], lo, hi))
            segs.append(np.array(cur))
            cur = []
    if cur:
        segs.append(np.array(cur))
    return [s for s in segs if s.shape[0] >= 2]


def compute(p):
    x = np.linspace(0.0, 1.0, p["n_points"])
    t = _truth(x) + np.random.default_rng(p["seed"]).normal(
        0.0, p["noise_sigma"], p["n_points"])
    xx = np.linspace(0.0, 1.0, p["n_curve"])
    v = p["t_view"]

    fits = []
    for m in p["degrees"]:
        # Chebyshev basis on the fitted domain, not the monomial one: the
        # 10x10 monomial Vandermonde on [0, 1] has condition ~1e7, and at
        # M = 9 the whole claim is a residual that must be zero to machine
        # precision. Same subspace, same least-squares solution, ~1e-15
        # instead of ~1e-9.
        c = chebyshev.Chebyshev.fit(x, t, m)
        yy = c(xx)
        r = t - c(x)
        fits.append({
            "M": m,
            "resid": r,
            "rss": float(r @ r),
            "y_curve": yy,
            "segments": _clip_band(xx, yy, -v, v),
            "excursion": float(np.max(np.abs(yy))),
        })

    return {
        "params": p, "x": x, "t": t, "xx": xx,
        "truth_curve": _truth(xx), "fits": fits,
    }


def _panel(g, fit) -> Scene:
    p = g["params"]
    s = Scene(xlim=XLIM, ylim=YLIM, height_px=310.0)
    # exact math-units-per-pixel for this panel's slot, so tick lengths and
    # marker radii are px quantities on an axis pair with unequal scales
    ux, uy = px_units(s, (WIDE.display_width_px - p["gap_px"]) / 2.0)

    s.items += axis(linear(0.0, 1.0), orient="x", at=X_AXIS_T, side=-1,
                    ticks=Ticks(values=np.array([0.0, 1.0]),
                                positions=np.array([0.0, 1.0]),
                                labels=("0", "1")),
                    tick_len=TICK_PX * uy, extend=(0.03, 0.11),
                    label="x", size_pt=ANNOT_PT)
    s.items += axis(linear(-p["t_view"], p["t_view"]), orient="y",
                    at=Y_AXIS_X, side=-1,
                    ticks=Ticks(values=np.array([-1.0, 0.0, 1.0]),
                                positions=np.array([-1.0, 0.0, 1.0]),
                                labels=("-1", "0", "1")),
                    tick_len=TICK_PX * ux, extend=(0.0, 0.13),
                    label="t", size_pt=ANNOT_PT)

    # A dotted FRAME rule at t = +-t_view was tried here, to say "the M = 9
    # fit is truncated, not mis-rendered". A cold reader looking at that
    # render still asked whether the break was a clipping artifact and did
    # not mention the rules at all: at FRAME weight they are invisible, at
    # any heavier weight they read as a second pair of content lines. Cut —
    # the truncation is admitted in the docstring and every reader did read
    # "the curve leaves the plotted range", which is the half that argues.

    # the function being estimated
    s.add(*keyed("truth", *series(g["xx"], g["truth_curve"], role=Role.CONTENT,
                                  color=HUE_TRUTH, width_scale=0.9)))
    # Anchored where the maximum of all four fits over x in [0.40, 0.74] is
    # 0.55 — a strip above the signal that no panel's fit enters, so
    # auto-place never nudges it and the label lands on the same pixel in
    # all four. A label the reader sees move is a second difference.
    s.add(*keyed("truth", MathLabel(r"\sin 2\pi x", (0.40, 1.19),
                                    role=Role.CONTENT, color=HUE_TRUTH,
                                    ha="left", va="bottom", size_pt=ANNOT_PT)))
    # the sample. Filled = attained (these are observations, not endpoints
    # of an open interval); drawn UNDER the fit so that at M = 9 the reader
    # can see the curve pass through each centre rather than take it on trust
    s.add(*keyed("data", *markers(g["x"], g["t"], "circle",
                                  size=(MARKER_PX * ux, MARKER_PX * uy),
                                  filled=True, role=Role.CONTENT,
                                  color=HUE_DATA)))
    # N lives with the dots, not in the caption: it is the count of THESE
    # marks, and the punchline (M = 9 with N = 10 is exactly determined) is
    # only available to a reader who has been told the count rather than
    # made to do the counting. The strip t in [-1.5, -1.15] left of x = 0.5
    # carries no ink in any of the four panels.
    s.add(*keyed("data", MathLabel(rf"N = {p['n_points']}\ \text{{observations}}",
                                   (0.02, -1.22), role=Role.CONTENT,
                                   color=HUE_DATA, ha="left", va="top",
                                   size_pt=ANNOT_PT)))
    # the one thing that varies
    for seg in fit["segments"]:
        # the heaviest stroke on the page: the fit is THE object, the signal
        # behind it is the reference it is judged against
        s.add(*keyed("fit", Curve(seg, role=Role.CONTENT, color=HUE_FIT,
                                  width_scale=1.7)))
    # The estimator, named on the page and not left to the caption. Three
    # cold readers in a row inferred "polynomial degree" correctly from the
    # shapes alone, but each flagged that they were inferring it: what fits
    # the curve, and what M indexes, are exactly the two things the figure
    # is arguing you should not trust on faith.
    s.add(*keyed("fit", MathLabel(
        rf"\text{{least squares polynomial, degree }} M = {fit['M']}",
        (0.99, CAPTION_T), role=Role.CONTENT, color=HUE_FIT,
        ha="right", va="top", size_pt=10.0)))
    return s


def build(g):
    return Figure(
        panels=[Panel(_panel(g, f), tag=rf"[{c}]")
                for f, c in zip(g["fits"], "abcd")],
        grid=(2, 2), gap_px=g["params"]["gap_px"],
    )


CORRESPONDENCE = [
    Correspondence(
        parts=(0, 1, 2, 3),
        varies="the degree M of the least-squares polynomial",
        # 'data' and 'truth' are absent on purpose: they are the fixed set,
        # and the figure fails the moment either stops matching.
        # NOTE: changes=("fit",) is currently satisfied by the per-panel
        # caption's TEXT ("... degree M = <k>"), not by the fit curve's
        # geometry — correspond._facet's fingerprint is (type, role, color,
        # dash, filled, text) and excludes position, so a swept-geometry
        # family whose instances carry no differing label (e.g. curve
        # position/shape alone) would false-fire stale-change here. Library
        # follow-up pending; see task-10-report.md friction item 1.
        changes=("fit",),
    ),
]


def assertions(g):
    p = g["params"]
    ck = Checks()
    x, t = g["x"], g["t"]

    v = p["t_view"]
    for f in g["fits"]:
        m, r = f["M"], f["resid"]
        # The least-squares certificate: the residual is orthogonal to every
        # column of the design matrix. Basis-independent, so it checks the
        # solve rather than restating how it was written.
        for k in range(m + 1):
            proj = float(r @ (2 * x - 1) ** k)
            ck.check(abs(proj) < 1e-9,
                     f"M={m}: residual not orthogonal to the degree-{k} "
                     f"column (<r, u^{k}> = {proj:.3e}) — not a least-squares fit")
        # Not "every segment stays inside [-v, v]" — _clip_band emits only
        # points inside the band by construction, so that would be true by
        # construction and could never fail. The real fragility is upstream
        # of the clipper: if two ADJACENT samples of the dense curve land on
        # opposite sides of the band (one below -v, the next above +v, or
        # the mirror), _clip_band never sees an inside point to anchor a
        # segment on and silently drops the whole crossing — a visible
        # excursion through the band would vanish from the page with no
        # signal. Sampling xx densely enough that consecutive points can't
        # jump the full band height is the actual invariant this figure
        # relies on; assert it directly rather than the tautology.
        yy = f["y_curve"]
        ck.check(not ((yy[:-1] < -v) & (yy[1:] > v)).any(),
                 f"M={m}: an adjacent sample pair jumps from below -{v} to "
                 f"above {v} — _clip_band would drop that crossing entirely")
        ck.check(not ((yy[:-1] > v) & (yy[1:] < -v)).any(),
                 f"M={m}: an adjacent sample pair jumps from above {v} to "
                 f"below -{v} — _clip_band would drop that crossing entirely")

    by_m = {f["M"]: f for f in g["fits"]}

    # The punchline, as an achieved number: N = 10 points, M = 9 has ten free
    # coefficients, so the fit must interpolate. Data-dependent and
    # conditioning-dependent — this is what fails if the basis goes bad.
    top = by_m[9]
    worst = float(np.max(np.abs(top["resid"])))
    ck.check(worst < 1e-10,
             f"M=9 does not interpolate the sample: max |t_n - y(x_n)| = "
             f"{worst:.3e} (N = {p['n_points']} points, 10 free coefficients)")
    # and it is drawn interpolating: every observation sits on a drawn segment
    for xn, tn in zip(x, t):
        on = min((float(np.min(np.abs(np.interp(xn, s[:, 0], s[:, 1]) - tn)))
                  for s in top["segments"]
                  if s[0, 0] <= xn <= s[-1, 0]), default=np.inf)
        ck.check(on < 5e-3,
                 f"M=9: the drawn curve misses the observation at x = {xn:.3f} "
                 f"by {on:.3g}")
    # ... and the wild interpolant really does leave the frame, which is the
    # half of the claim the truncation would otherwise hide
    ck.check(top["excursion"] > p["t_view"],
             f"M=9 peaks at |t| = {top['excursion']:.3f}, inside the "
             f"{p['t_view']} band — the figure shows no overshoot to truncate")

    # nested models: RSS is non-increasing in M, and strictly decreasing here
    rss = [by_m[m]["rss"] for m in p["degrees"]]
    for m0, m1, a, b in zip(p["degrees"], p["degrees"][1:], rss, rss[1:]):
        ck.check(b < a,
                 f"RSS did not fall from M={m0} ({a:.4g}) to M={m1} ({b:.4g}) — "
                 f"the degree-{m0} fit lies in the degree-{m1} model space")

    # the axes are honest about the data they carry
    tick_honesty(ck, linear(0.0, 1.0), x, name="x axis")
    tick_honesty(ck, linear(-p["t_view"], p["t_view"]), t, name="t axis")

    # One frame, four panels: asserted once, because a stray edit to one
    # panel's lims is exactly how a small-multiples family stops being one.
    scenes = [pa.scene for pa in build(g).panels]
    ck.check(len({sc.xlim for sc in scenes}) == 1
             and len({sc.ylim for sc in scenes}) == 1
             and len({sc.height_px for sc in scenes}) == 1,
             f"panels do not share one frame: "
             f"{[(sc.xlim, sc.ylim, sc.height_px) for sc in scenes]}")
    ck.done()
