"""The typical set: probability mass lives where multiplicity and
per-sequence probability trade off, not at the most probable sequence.

Design notes (exposition.md steps 0-9):

0. EARN IT. The prose version -- "the all-zeros sequence is the single
   most probable outcome, yet almost all the probability sits on
   sequences with about pN ones" -- states the paradox but leaves the
   reader to take the mechanism on faith. Drawn in log space, the
   mechanism becomes additive and hence visible: log(count) is a rising-
   then-falling hump, log(per-sequence probability) is a falling LINE,
   and their SUM -- the actual mass curve -- is what the eye can watch
   peel away from both parents and concentrate where the hump's downward
   slope matches the line's slope. That balancing point is a place the
   eye finds; "differentiate and set to zero" is not.
1. CLAIM: below.
2. REPRESENTATION. r (count of ones) is the one live coordinate; for
   each r, three log10 quantities are drawn as functions of r rather than
   as one bar chart over 2^N sequences (which cannot be drawn at all).
   This fixes the abstraction rung at "the r-macrostate", which is
   exactly the rung the asymptotic-equipartition argument needs.
3. SIZE. WIDE: the dynamic range (fifty-plus decades) needs width to
   keep the y axis legible, and the annotation (band, brace, two
   callouts) needs room.
4. TRAVERSAL. Enter at the accent (total mass) curve's single visible
   hump -- it is the only one of the three that looks like a normal,
   bounded density. Read left to the r=0 marker and its callout (the
   provocation: this point is higher on the OTHER curve). Read the wash
   band under the hump; the brace states its width; the dashed rule
   states its center.
5. MECHANISM. Readable off the figure: the three curves are one
   equation, log(total) = log(C) + log(p_seq), so multiplicity and
   per-sequence probability visibly cancel to leave an O(1) peak; the
   band's mass and the r=0 point's mass are both stated in the callouts.
6. READER EFFORT. Delegated: verifying that the accent curve peaks where
   the other two visibly cross in slope (no arithmetic needed to see
   it). Kept: the leap from "narrow band of r" to "almost all sequences
   with this many ones," which is exactly what "typical set" means.
7. CHANNELS. Hue distinguishes the two FACTORS (multiplicity, per-
   sequence probability) via categorical(0)/(1); the accent role (no
   hue) marks their product, the object the whole claim is about;
   ACCENT2 marks the single distinguished sequence r=0. Position (r,
   log-value) carries everything; dash marks construction (the r=pN
   guide) vs content.
8. HONESTY. The y axis is explicitly log10 -- tick labels read 10^k, and
   the axis title says so; a silent log axis on a plot spanning fifty
   decades would be indefensible. Nothing is clipped: y limits are
   derived from the actual min/max of the three drawn arrays plus a
   stated pad, never from a guessed round number.
9. GATES. assertions() checks: p < 1/2 (so r=0 really is the argmax of
   per-sequence probability); the per-r masses sum to 1; the drawn band
   mass exceeds the claimed floor; r=0 lies OUTSIDE that band while its
   per-sequence probability EXCEEDS every per-sequence probability drawn
   inside the band; and the lgamma-based multiplicity/mass arrays agree
   with independent math.comb (exact big-int) evaluations at a probe
   set of r values.
"""

import math

import numpy as np
from scipy.special import gammaln

from figlib import plots
from figlib.format import WIDE
from figlib.gates import Checks
from figlib.scene import Brace, Callout, Curve, FilledCurve, MathLabel, Point, Scene
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "For N i.i.d. Bernoulli(p) draws with p < 1/2, the all-zeros sequence "
    "(r = 0 ones) is individually the most probable single sequence, but it "
    "lies outside the typical set -- the band of width O(sqrt(N)) centered "
    "at r = pN that carries essentially all the probability mass -- because "
    "the multiplicity C(N,r), which rises toward r = N/2, and the "
    "per-sequence probability p^r(1-p)^(N-r), which falls monotonically, "
    "trade off there and nowhere else."
)

EXPOSITION = """
MacKay's asymptotic equipartition argument (Information Theory, Inference,
and Learning Algorithms, ch. 4) is the sharpest counterexample to the
instinct that "most probable" and "typical" are the same idea. Take N
tosses of a p-biased coin. The single most probable OUTCOME -- the exact
sequence with the highest probability of all 2^N sequences -- is all
zeros when p < 1/2, since every 1 costs a factor p/(1-q) < 1 relative to
a 0. But almost none of the PROBABILITY lives there. Group sequences by
r, the number of ones: there are C(N,r) sequences in that group, each
individually carrying probability p^r(1-p)^(N-r). The group's total mass
is their product, and as r moves away from 0 the multiplicity explodes
combinatorially (a distinct fact, driven by log2 C(N,r) approximately
N*H(r/N)) far faster than the per-sequence probability decays, so the
mass climbs even as each individual sequence in the group gets less
likely. The two trends only balance -- log(count) rising with slope
log((N-r)/r), log(probability) falling with constant slope log(p/(1-p))
-- at r near pN, and it is there, in a band of width O(sqrt(N)), that
essentially all of the probability concentrates. The all-zeros string
sits at r = 0: the highest point on the probability-per-sequence curve,
and by the time the group sizes are weighted in, a vanishingly small
contributor to the total. This is the seed of the source coding theorem:
a random N-bit message from the source lands, with overwhelming
probability, in this typical set, so log2(size of the typical set) --
not log2(2^N) -- is the number of bits actually needed to describe it.
"""

THEME = RISO
FORMAT = WIDE

PARAMS = {
    "N": 60,                  # number of Bernoulli draws
    "p": 0.30,                # bias; must be < 1/2 so r=0 is the argmax
    "band_sigma": 3.0,        # typical-band half-width, in units of sqrt(Np(1-p))
    "mass_floor": 0.99,       # claimed lower bound on the band's probability mass
    "y_pad_decades": 1.5,     # headroom around the data's own log10 range
    "r_pad_left": 19.0,       # x-axis headroom left of 0, in r units (room for the callout)
    "r_pad_right": 4.0,       # x-axis headroom right of N, in r units
    "r_tick_step": 10.0,      # r-axis tick spacing
    "y_tick_step": 10,        # y-axis tick spacing, in log10 decades
    "axis_margin_decades": 4.0,  # room below the r-axis spine for its ticks/labels
    "top_pad_decades": 5.0,   # room above the data ceiling for the brace + callout
    "height_px": 620.0,
    "probe_count": 9,         # independent math.comb cross-check points
}


def compute(p):
    N, prob = p["N"], p["p"]
    r = np.arange(0, N + 1, dtype=float)
    log10e = 1.0 / math.log(10.0)

    log10_pseq = r * math.log10(prob) + (N - r) * math.log10(1.0 - prob)
    log10_C = (gammaln(N + 1.0) - gammaln(r + 1.0) - gammaln(N - r + 1.0)) * log10e
    log10_tot = log10_C + log10_pseq

    pseq = np.power(10.0, log10_pseq)
    Cnr = np.power(10.0, log10_C)
    total = np.power(10.0, log10_tot)

    mu = N * prob
    sigma = math.sqrt(N * prob * (1.0 - prob))
    half = p["band_sigma"] * sigma
    r_lo = max(0, int(math.floor(mu - half)))
    r_hi = min(N, int(math.ceil(mu + half)))

    band_mass = float(total[r_lo:r_hi + 1].sum())
    total_mass = float(total.sum())

    r_star = 0 if prob < 0.5 else N
    pseq_star = float(pseq[r_star])
    total_star = float(total[r_star])
    pseq_band_max = float(pseq[r_lo:r_hi + 1].max())

    pad = p["y_pad_decades"]
    all_log = np.concatenate([log10_C, log10_pseq, log10_tot])
    y_data_lo = 10.0 ** (math.floor(float(all_log.min())) - pad)
    y_data_hi = 10.0 ** (math.ceil(float(all_log.max())) + pad)

    x_scale = plots.linear(-p["r_pad_left"], N + p["r_pad_right"])
    y_scale = plots.log10(y_data_lo, y_data_hi)

    return {
        "p": p, "N": N, "prob": prob, "r": r,
        "pseq": pseq, "Cnr": Cnr, "total": total,
        "mu": mu, "sigma": sigma, "r_lo": r_lo, "r_hi": r_hi,
        "band_mass": band_mass, "total_mass": total_mass,
        "r_star": r_star, "pseq_star": pseq_star, "total_star": total_star,
        "pseq_band_max": pseq_band_max,
        "x_scale": x_scale, "y_scale": y_scale,
        "y_data_lo": y_data_lo, "y_data_hi": y_data_hi,
    }


# --- build -------------------------------------------------------------------

def _sparse_log_ticks(y_scale: plots.Log10, step: int) -> plots.Ticks:
    """Decade ticks, but only every `step` decades -- a plain Log10.ticks()
    would place one label per decade, and this axis spans fifty-plus."""
    k0, k1 = y_scale.range
    k0i, k1i = int(math.ceil(k0 - 1e-9)), int(math.floor(k1 + 1e-9))
    start = step * math.ceil(k0i / step)
    ks = np.arange(start, k1i + 1, step, dtype=float)
    return plots.Ticks(values=np.power(10.0, ks), positions=ks,
                       labels=tuple(f"10^{{{int(k)}}}" for k in ks))


def build(g):
    p = g["p"]
    x_scale, y_scale = g["x_scale"], g["y_scale"]
    x0, x1 = x_scale.range
    u_lo, u_hi = y_scale.range   # log10(y_data_lo), log10(y_data_hi)

    ylim = (u_lo - p["axis_margin_decades"], u_hi + p["top_pad_decades"])
    s = Scene(xlim=(x0, x1), ylim=ylim, height_px=p["height_px"])

    # the typical-set wash: r in [r_lo, r_hi], the full data band in y
    s.add(FilledCurve(
        np.array([[g["r_lo"], u_lo], [g["r_hi"], u_lo],
                  [g["r_hi"], u_hi], [g["r_lo"], u_hi]]),
        role=Role.MUTED, opacity=0.23, outline=False))

    # axes: r along the bottom (a frame axis, not tied to a data value),
    # value on the left, explicitly log10 -- the tick labels say 10^k and
    # the axis title says "log10" a second time, so the scale cannot be
    # mistaken for linear
    s.add(*plots.axis(x_scale, orient="x", at=u_lo, side=-1,
                      ticks=x_scale.ticks(step=p["r_tick_step"])))
    s.add(MathLabel(r"r\ (\text{number of ones})", (0.5 * g["N"], u_lo),
                    role=Role.ANNOTATION, ha="center", va="top",
                    offset_px=(0.0, 34.0)))
    s.add(*plots.axis(y_scale, orient="y", at=x0, side=-1,
                      ticks=_sparse_log_ticks(y_scale, p["y_tick_step"]),
                      extend=(p["axis_margin_decades"], p["top_pad_decades"] - 1.0),
                      label=r"\text{value }(\log_{10}\text{ scale})"))

    # the two factors, hue = correspondence between them
    s.add(*plots.series(g["r"], g["Cnr"], yscale=y_scale,
                        role=Role.CONTENT, color=RISO.categorical(0)))
    s.add(*plots.series(g["r"], g["pseq"], yscale=y_scale,
                        role=Role.CONTENT, color=RISO.categorical(1)))
    # their product -- the object the whole claim is about -- as the accent,
    # no hue override
    s.add(*plots.series(g["r"], g["total"], yscale=y_scale,
                        role=Role.ACCENT1, width_scale=1.6))

    r_peak = float(g["r"][int(np.argmax(g["total"]))])
    y_peak = float(y_scale.fwd([float(g["total"].max())])[0])
    s.add(Curve(np.array([[g["mu"], u_lo], [g["mu"], y_peak]]),
                role=Role.CONSTRUCTION, dash="dashed", width_scale=0.8))
    s.add(MathLabel(r"r = pN", (g["mu"], y_peak + 1.2), role=Role.CONSTRUCTION,
                    ha="center", va="bottom", halo=True))

    # curve labels, colored to match, placed where each curve has room
    r_c_peak = float(g["r"][int(np.argmax(g["Cnr"]))])
    y_c_peak = float(y_scale.fwd([float(g["Cnr"].max())])[0])
    s.add(MathLabel(r"C(N,r)\ \text{multiplicity}", (r_c_peak, y_c_peak),
                    role=Role.CONTENT, color=RISO.categorical(0),
                    ha="center", va="bottom", offset_px=(0.0, -10.0)))
    r_lbl = 0.78 * g["N"]
    y_lbl = float(y_scale.fwd([float(np.interp(r_lbl, g["r"], g["pseq"]))])[0])
    s.add(MathLabel(r"p^{r}(1-p)^{N-r}\ \text{per-sequence probability}",
                    (r_lbl, y_lbl), role=Role.CONTENT, color=RISO.categorical(1),
                    ha="right", va="bottom", offset_px=(0.0, 8.0)))
    s.add(MathLabel(r"\text{total mass } C(N,r)\,p^{r}(1-p)^{N-r}",
                    (32.2, 1.05), role=Role.ACCENT1,
                    ha="center", va="center"))

    # the distinguished sequence: r = 0, the highest point on the
    # per-sequence-probability curve, and its callout
    y_star = float(y_scale.fwd([g["pseq_star"]])[0])
    s.add(Point((float(g["r_star"]), y_star), role=Role.ACCENT2, radius_scale=1.3))
    s.add(Callout(
        r"\text{most probable single sequence: } P = (1{-}p)^N",
        anchor=(-0.30 * p["r_pad_left"], 0.5 * (y_star + u_hi)),
        target=(float(g["r_star"]), y_star), role=Role.ACCENT2))

    # the typical-set width, measured, above the hump
    brace_y = u_hi + 1.0
    s.add(Brace((float(g["r_lo"]), brace_y), (float(g["r_hi"]), brace_y),
                side=1.0, depth=0.7,
                label=rf"\text{{typical set: width}} = 2\times{p['band_sigma']:.0f}\sigma"
                      rf"\;=\;O(\sqrt N)\;\;(r\in[{g['r_lo']},{g['r_hi']}])",
                role=Role.ANNOTATION))

    # the numbers behind the letters, so the figure is checkable without
    # a caption: N and p pin down every curve; sigma pins down the band
    s.add(MathLabel(
        rf"N={g['N']:.0f},\ \ p={g['prob']:.2f},\ \ "
        rf"\sigma=\sqrt{{Np(1-p)}}={g['sigma']:.2f}",
        (42.5, 27.1), role=Role.ANNOTATION, ha="center", va="center"))
    return s


# --- the numerical gate -------------------------------------------------------

def assertions(g):
    c = Checks()
    p, prob, N = g["p"], g["prob"], g["N"]

    c.check(0.0 < prob < 0.5,
            f"p={prob} must be < 1/2 for the all-zeros sequence to be the "
            f"single most probable one")

    # the per-r masses are a probability distribution
    c.check(abs(g["total_mass"] - 1.0) < 1e-9,
            f"the per-r masses sum to {g['total_mass']:.12f}, not 1")

    # the marked typical band really carries at least the claimed mass
    c.check(g["band_mass"] >= p["mass_floor"],
            f"typical band [{g['r_lo']},{g['r_hi']}] carries "
            f"{g['band_mass']:.6f} of the mass, short of the claimed floor "
            f"{p['mass_floor']}")

    # the whole point: r* is OUTSIDE the drawn band ...
    c.check(g["r_star"] < g["r_lo"] or g["r_star"] > g["r_hi"],
            f"r*={g['r_star']} falls inside the drawn typical band "
            f"[{g['r_lo']},{g['r_hi']}] -- the figure has nothing to argue")
    # ... while individually MORE probable than any sequence drawn inside it
    c.check(g["pseq_star"] > g["pseq_band_max"],
            f"P_seq at r*={g['r_star']} ({g['pseq_star']:.3e}) is not larger "
            f"than the largest per-sequence probability drawn in the band "
            f"({g['pseq_band_max']:.3e})")
    # and its total contribution to the mass is negligible next to the band's
    c.check(g["total_star"] < 0.01 * g["band_mass"],
            f"the most-probable-sequence group's own mass "
            f"({g['total_star']:.3e}) is not negligible next to the band's "
            f"({g['band_mass']:.3e})")

    # independent cross-check: gammaln-derived multiplicity and total mass
    # against exact math.comb (arbitrary-precision integers), at a probe set
    probe = np.unique(np.linspace(0, N, p["probe_count"]).round().astype(int))
    for rv in probe:
        rv = int(rv)
        exact_C = float(math.comb(N, rv))
        c.check(abs(g["Cnr"][rv] - exact_C) < 1e-6 * max(exact_C, 1.0),
                f"C(N,{rv}): gammaln gives {g['Cnr'][rv]:.6e}, "
                f"math.comb gives {exact_C:.6e}")
        exact_mass = exact_C * prob ** rv * (1.0 - prob) ** (N - rv)
        c.check(abs(g["total"][rv] - exact_mass) < 1e-6 * max(exact_mass, 1e-300),
                f"total mass at r={rv}: drawn {g['total'][rv]:.3e}, "
                f"exact {exact_mass:.3e}")

    # axis honesty: nothing clipped, nothing padded into unreadability
    plots.tick_honesty(c, g["x_scale"], g["r"], name="r axis")
    plots.tick_honesty(c, g["y_scale"],
                       np.concatenate([g["Cnr"], g["pseq"], g["total"]]),
                       name="value axis",
                       ticks=_sparse_log_ticks(g["y_scale"], p["y_tick_step"]))
    c.done()
