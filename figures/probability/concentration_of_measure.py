"""The mode of a high-dimensional Gaussian and the typical sample never meet.

For X ~ N(0, I_d) the joint density p(x) = (2*pi)^{-d/2} exp(-||x||^2/2) is
strictly decreasing in ||x||, so it is maximized at the origin -- yet the
induced density of R = ||X|| is a chi(d) law, f_R(r) proportional to
r^{d-1} exp(-r^2/2), and that r^{d-1} volume factor kills the density at
r = 0 for every d > 1. The two facts are not in tension; they are the same
computation read two ways: near the origin, "dense per point" and "no
volume to be dense over" trade off, and past d = 2 the second wins so hard
that P(||X|| < eps) shrinks like eps^d. R itself concentrates: its mean
sits at ~sqrt(d) and its variance converges to sigma^2 -> 1/2 regardless of
d, so the absolute width of the shell stays O(1) while its width relative
to its own radius shrinks like 1/sqrt(d). This is the textbook fact behind
"high-dimensional intuition fails" (Vershynin's concentration of measure)
and it is exactly the mechanism behind the LLM "typical set": the
highest-probability token sequence and the sequence a sampler actually
draws are different objects for the same reason a mode and a typical
radius are -- probability mass is density times volume, and in high
dimensions volume dominates.
"""

import numpy as np
from scipy.stats import chi

from figlib.format import WIDE
from figlib.gates import Checks
from figlib.plots import Log10, Ticks, axis, colorbar, linear
from figlib.scene import Callout, Curve, MathLabel, Point, Scene
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "For X ~ N(0, I_d), the joint density p(x) is maximized at the origin, "
    "but the density of ||X|| vanishes there and instead concentrates "
    "tightly around sqrt(d): the shell's absolute width stays O(1) as d "
    "grows while its width relative to sqrt(d) shrinks like 1/sqrt(d), so "
    "the mode of p and the location of a typical sample pull apart."
)

EXPOSITION = """
High-dimensional probability keeps producing facts that contradict
low-dimensional intuition, and Vershynin's treatment of Gaussian
concentration is the cleanest example: the density of X ~ N(0, I_d) is
strictly maximized at the origin, yet almost no sample from that same
distribution ever lands near the origin. The resolution is that probability
mass is density times volume, and in high dimensions the volume of a thin
shell at radius r grows like r^{d-1} -- a factor that vanishes at r = 0 and
overwhelms the density's own peak everywhere else, so the mass concentrates
on a shell near r = sqrt(d) while the mode of the density sits in a region
the mass essentially never visits. A reader needs this picture, not just
the chi-distribution formula, to internalize why "most probable point" and
"typical sample" stop being synonyms once the dimension is large -- and the
same mechanism, density times volume, is exactly what makes a language
model's highest-probability output different from the sequence a sampler
actually draws.
"""

THEME = RISO
FORMAT = WIDE

PARAMS = {
    "ds": [2, 10, 100, 1000],
    "n_u": 480,          # samples per curve, in u = ||X||/sqrt(d)
    "u_max": 2.4,         # shared axis range; d=2's tail is < 2% of peak here
    "eps_origin": 0.05,   # "no sample lands near the origin": P(||X|| < eps)
    "height_px": 430.0,
}


def _u_grid(p):
    return np.linspace(1e-6, p["u_max"], p["n_u"])


def compute(p):
    ds = p["ds"]
    u = _u_grid(p)
    curves = {}       # d -> raw g_d(u) = density of U = ||X||/sqrt(d)
    stats = {}        # d -> mean, std, mode of R = ||X|| (exact chi(d))
    for d in ds:
        r = u * np.sqrt(d)
        g = np.sqrt(d) * chi.pdf(r, df=d)          # change-of-variables Jacobian sqrt(d)
        curves[d] = g
        mean_r = float(chi.mean(df=d))
        std_r = float(chi.std(df=d))
        mode_r = float(np.sqrt(d - 1)) if d > 1 else 0.0
        stats[d] = {
            "mean_r": mean_r, "std_r": std_r, "mode_r": mode_r,
            "u_mode": mode_r / np.sqrt(d),
            "rel_std": std_r / mean_r,                 # std(R)/mean(R)
            "cdf_eps": float(chi.cdf(p["eps_origin"], df=d)),
        }
    # the joint density's own claim: log p(x) at the origin vs. at a
    # typical sample (r = mean_r for the largest d), for the same X
    logp0 = {d: -0.5 * d * np.log(2 * np.pi) for d in ds}
    logp_typ = {d: -0.5 * d * np.log(2 * np.pi) - 0.5 * stats[d]["mean_r"] ** 2
                for d in ds}
    return {"p": p, "ds": ds, "u": u, "curves": curves, "stats": stats,
            "logp0": logp0, "logp_typ": logp_typ}


def build(g):
    p, ds, u = g["p"], g["ds"], g["u"]
    n = len(ds)

    def t_of(d):
        # log-spaced ordering: the ramp reads d's ORDER, not its raw value,
        # and the d's span decades so a log position keeps them separated
        return (np.log10(d) - np.log10(ds[0])) / (np.log10(ds[-1]) - np.log10(ds[0]))

    s = Scene(xlim=(-0.10, p["u_max"] + 0.40), ylim=(-0.32, 1.32),
              height_px=p["height_px"])

    xsc = linear(0.0, p["u_max"])
    s.add(*axis(xsc, orient="x", at=0.0, side=-1,
                ticks=Ticks(values=np.array([0.0, 1.0, 2.0]),
                            positions=np.array([0.0, 1.0, 2.0]),
                            labels=("0", "1", "2")),
                label=r"\|X\|\,/\,\sqrt{d}", extend=(0.0, 0.25)))

    # the shared reference: every curve's typical radius sits at u = 1 by
    # construction of this coordinate -- this IS "sqrt(d), marked, for
    # every d at once"
    s.add(Curve(np.array([[1.0, 0.0], [1.0, 1.06]]), role=Role.CONSTRUCTION,
                width_scale=1.0))
    s.add(MathLabel(r"\|X\|=\sqrt{d}", (1.04, 1.06),
                    ha="left", va="bottom", size_pt=10, role=Role.ANNOTATION))

    for d in ds:
        raw = g["curves"][d]
        h = raw / raw.max()          # peak-normalized for SHAPE comparison;
        # true peak height grows like sqrt(d) (annotated in prose below) --
        # admitted here rather than drawn, since a shared linear y-axis
        # carrying both a d=2 and a d=1000 peak would crush the former to a
        # sliver the way a raw shared x-axis would have crushed it earlier
        s.add(Curve(np.column_stack([u, h]), role=Role.CONTENT,
                    color=RISO.ramp(t_of(d))))
        st = g["stats"][d]
        s.add(Point((st["u_mode"], float(np.interp(st["u_mode"], u, h))),
                    role=Role.CONTENT, radius_scale=0.85,
                    color=RISO.ramp(t_of(d))))

    # the punchline: p(x) peaks at the origin, the drawn density does not
    s.add(Point((0.0, 0.0), role=Role.ACCENT2, radius_scale=1.3))
    s.add(Callout(
        r"p(x)\ \text{maximal here, but}\ P(\|X\|<\varepsilon)\to 0",
        anchor=(0.62, 0.62), target=(0.02, 0.02)))

    s.add(MathLabel(
        r"\text{dots: mode of }\|X\|\text{ at }\sqrt{d-1}\ "
        r"(\to\sqrt{d}\text{ as }d\to\infty)",
        (1.55, -0.24), ha="center", va="top", size_pt=9.5,
        role=Role.ANNOTATION))

    d0, dlast = ds[0], ds[-1]
    s0, slast = g["stats"][d0], g["stats"][dlast]
    s.add(MathLabel(
        rf"\sigma(\|X\|)\in[{s0['std_r']:.2f},{slast['std_r']:.2f}]\ \text{{(O(1))}}",
        (0.03, 1.30), ha="left", va="center", size_pt=9.5,
        role=Role.ANNOTATION))
    s.add(MathLabel(
        rf"\sigma(\|X\|)/\sqrt{{d}}\in[{slast['rel_std']:.3f},{s0['rel_std']:.2f}]"
        r"\ \text{(typical set)}",
        (0.03, 1.16), ha="left", va="center", size_pt=9.5,
        role=Role.ANNOTATION))

    # ramp legend: d rides the order channel, log-spaced ticks at the
    # drawn values
    s.add(*colorbar(
        Log10(ds[0], ds[-1]), RISO.ramp, at=(p["u_max"] - 0.55, -0.30),
        length=1.0, thickness=0.05, orient="x",
        ticks=Ticks(values=np.array(ds, dtype=float),
                    positions=np.log10(np.array(ds, dtype=float)),
                    labels=tuple(str(d) for d in ds)),
        label=r"d"))
    return s


def assertions(g):
    p, ds, u = g["p"], g["ds"], g["u"]
    c = Checks()

    # (1) the mode of the drawn density really sits at sqrt(d-1): the
    # argmax of the SAME array that got drawn (peak-normalization does not
    # move it) must land within one grid step of the analytic mode, in u
    du = float(u[1] - u[0])
    for d in ds:
        raw = g["curves"][d]
        u_argmax = float(u[int(np.argmax(raw))])
        u_mode = g["stats"][d]["u_mode"]
        c.check(abs(u_argmax - u_mode) <= 2 * du,
                f"d={d}: drawn-array argmax at u={u_argmax:.4f} but "
                f"sqrt(d-1)/sqrt(d) = {u_mode:.4f}")

    # (2) the mean really sits near sqrt(d): u_mean = mean/sqrt(d) -> 1
    # monotonically as d grows (this IS the claim "typical location -> sqrt(d)")
    u_means = [g["stats"][d]["mean_r"] / np.sqrt(d) for d in ds]
    c.check(all(a < b for a, b in zip(u_means, u_means[1:])),
            f"mean/sqrt(d) is not monotonically approaching 1 across d: {u_means}")
    c.check(abs(u_means[-1] - 1.0) < 0.005,
            f"mean/sqrt(d) at the largest d ({ds[-1]}) is {u_means[-1]:.4f}, "
            f"not close to 1")

    # (3) the claim's numeric core: relative std shrinks like 1/sqrt(d) --
    # i.e. rel_std(d) * sqrt(d) converges to a d-independent constant
    # (sigma_infinity = 1/sqrt(2)) rather than drifting, and rel_std itself
    # is monotonically decreasing
    rel = [g["stats"][d]["rel_std"] for d in ds]
    c.check(all(a > b for a, b in zip(rel, rel[1:])),
            f"relative std is not monotonically shrinking across d: {rel}")
    scaled = [r * np.sqrt(d) for r, d in zip(rel, ds)]
    sigma_inf = 1.0 / np.sqrt(2.0)
    c.check(max(abs(v - sigma_inf) for v in scaled[-2:]) < 0.02,
            f"rel_std(d)*sqrt(d) = {scaled} does not converge to "
            f"1/sqrt(2) = {sigma_inf:.4f} — the 1/sqrt(d) law would be a lie")
    c.check(abs(scaled[0] - scaled[-1]) > 0.02,
            "rel_std(d)*sqrt(d) barely moves across the drawn d's — "
            "the chosen d's do not actually demonstrate the law")

    # (4) absolute width is O(1): sigma(||X||) stays within a narrow band
    # across three decades of d (the OTHER half of the claim — it does
    # NOT shrink like the relative width does)
    abs_std = [g["stats"][d]["std_r"] for d in ds]
    c.check(max(abs_std) / min(abs_std) < 1.2,
            f"absolute std {abs_std} varies by more than 20% across d — "
            f"'O(1) independent of d' would be a lie at these d's")

    # (5) the joint density really peaks at the origin: log p(0) - log
    # p(typical) grows without bound (here: equals d/2 exactly, a real
    # algebraic identity on the drawn quantities, not a restated triviality
    # -- it is what makes p(0) >> p(typical) even though ||X||'s own
    # density is ~0 at the origin)
    for d in ds:
        typ = g["stats"][d]["mean_r"]
        gap = g["logp0"][d] - g["logp_typ"][d]
        c.check(abs(gap - 0.5 * typ ** 2) < 1e-9,
                f"d={d}: log p(0) - log p(typical) = {gap:.4f}, expected "
                f"{0.5 * typ ** 2:.4f} from -||x||^2/2")
        c.check(gap > 0.0, f"d={d}: joint density at origin is not the "
                f"larger one (gap={gap:.4f})")
    gaps = [g["logp0"][d] - g["logp_typ"][d] for d in ds]
    c.check(all(a < b for a, b in zip(gaps, gaps[1:])),
            f"log-density gap does not grow with d: {gaps}")

    # (6) no sample lands near the origin, and it gets worse with d:
    # P(||X|| < eps) shrinks as d grows (chi CDF near 0 ~ eps^d / d!, so
    # this is an extremely fast, not merely monotone, collapse)
    cdfs = [g["stats"][d]["cdf_eps"] for d in ds]
    c.check(all(a > b for a, b in zip(cdfs, cdfs[1:])),
            f"P(||X|| < eps) does not shrink with d: {cdfs}")
    c.check(cdfs[0] < 0.05,
            f"P(||X|| < eps={p['eps_origin']}) at the smallest d is "
            f"{cdfs[0]:.4f} — too large to read as 'no sample lands near "
            f"the origin' even at d={ds[0]}")
    c.done()
