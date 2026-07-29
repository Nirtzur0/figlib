"""Sampling replicates the spectrum; aliasing is a visible overlap.

The O&S replica stack: an iconic triangle stands for any bandlimited
spectrum — bandwidth and symmetry are its only live features — and three
frames sharing one omega axis carry the deformation "lower the sampling
rate". Impulses use the Bracewell convention (arrow height = weight);
the aliased band is drawn as the pointwise minimum of the two replicas
that own it, filled in accent ink: the defect IS a region you can see.
"""

import numpy as np

from figlib.format import COLUMN
from figlib.gates import Checks
from figlib.plots import Ticks, axis, band, impulses, linear
from figlib.scene import Callout, Curve, MathLabel, Scene
from figlib.style import Role
from figlib.theme import RISO

THEME = RISO
FORMAT = COLUMN

CLAIM = (
    "Sampling at rate omega_s convolves a spectrum with an impulse comb of "
    "spacing omega_s, replicating the triangle at every multiple of omega_s; "
    "the replicas stay separated exactly when omega_s > 2*omega_m, and when "
    "omega_s < 2*omega_m adjacent replicas overlap in a band of width "
    "2*omega_m - omega_s where the spectrum is unrecoverable."
)

EXPOSITION = """
This is Oppenheim & Schafer's picture for why sampling has a rate at
all: sampling a continuous signal at rate omega_s does not distort its
spectrum, it replicates it -- convolving with an impulse comb places a
full copy of X(j omega) at every integer multiple of omega_s. Nothing
about a single replica changes; what changes is whether replicas
collide. The Nyquist condition omega_s > 2 omega_m is not an assumption
about the signal imposed for its own sake, it is the exact statement
that consecutive copies of a triangle of half-width omega_m stay clear
of each other. Read the frames in sequence: at omega_s > 2 omega_m the
gap between copies is a real gap, and an ideal lowpass filter recovers
the base replica intact. Push omega_s below 2 omega_m and the gap goes
negative -- the replicas overlap in a band of width 2 omega_m - omega_s,
and inside that band the sampled spectrum is the SUM of two triangles
that no filter can separate back into one, because the aliased
frequency components are now indistinguishable from ones that were
never there. That band, not an inequality, is what aliasing is.
"""

PARAMS = {
    "omega_m": 1.0,
    "omega_s_ok": 2.5,       # panel b: > 2 omega_m, gap = 0.5
    "omega_s_bad": 1.4,      # panel c: < 2 omega_m, overlap = 0.6
    "window": 4.0,           # drawn omega range is [-window, window]
    "comb_h": 1.15,          # drawn impulse height (weight 1/T, admitted on figure)
    "lpf_h": 1.05,           # ideal-filter gain line (T), above the unit peak
    "n_band": 64,            # samples per overlap band polygon
    "panel_h_px": 128.0,
}


def _tri(x: np.ndarray, center: float, om: float) -> np.ndarray:
    """The iconic triangle: peak 1 at `center`, support center +- om."""
    return np.maximum(0.0, 1.0 - np.abs(np.asarray(x, float) - center) / om)


def _tri_pts(center: float, om: float, w: float) -> np.ndarray | None:
    """Triangle polyline clipped to [-w, w]; None when fully outside."""
    knots = sorted({-w, w, *(v for v in (center - om, center, center + om)
                             if -w <= v <= w)})
    xs = np.array([k for k in knots if -w <= k <= w])
    ys = _tri(xs, center, om)
    if np.all(ys <= 1e-12):
        return None
    return np.column_stack([xs, ys])


def _centers(omega_s: float, om: float, w: float) -> list[float]:
    kmax = int(np.floor((w + om) / omega_s))
    return [k * omega_s for k in range(-kmax, kmax + 1)]


def _overlaps(omega_s: float, om: float, w: float,
              n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(x, min of the two adjacent replicas) per overlapping pair, clipped."""
    out = []
    if omega_s >= 2 * om:
        return out
    cs = _centers(omega_s, om, w)
    for c0, c1 in zip(cs[:-1], cs[1:]):
        lo, hi = max(c1 - om, -w), min(c0 + om, w)
        if hi <= lo:
            continue
        x = np.linspace(lo, hi, n)
        out.append((x, np.minimum(_tri(x, c0, om), _tri(x, c1, om))))
    return out


def compute(p):
    om, w = p["omega_m"], p["window"]
    return {
        "params": p,
        "centers_ok": _centers(p["omega_s_ok"], om, w),
        "centers_bad": _centers(p["omega_s_bad"], om, w),
        "overlaps_ok": _overlaps(p["omega_s_ok"], om, w, p["n_band"]),
        "overlaps_bad": _overlaps(p["omega_s_bad"], om, w, p["n_band"]),
        "gap_ok": p["omega_s_ok"] - 2 * om,
        "gap_bad": p["omega_s_bad"] - 2 * om,
    }


def _omega_axis(om: float, w: float) -> list:
    tk = Ticks(values=np.array([-om, om]), positions=np.array([-om, om]),
               labels=(r"-\omega_m", r"\omega_m"))
    return axis(linear(-w, w), orient="x", at=0.0, side=-1, ticks=tk,
                tick_len=0.07, extend=(0.15, 0.3), label=r"\omega",
                arrow=True)


def _panel(g, omega_s: float | None) -> Scene:
    """One frame of the stack; omega_s=None is the unsampled spectrum."""
    p = g["params"]
    om, w = p["omega_m"], p["window"]
    s = Scene(xlim=(-w - 0.55, w + 0.55), ylim=(-0.34, 1.52),
              height_px=p["panel_h_px"])
    s.items += _omega_axis(om, w)

    if omega_s is None:
        s.add(Curve(_tri_pts(0.0, om, w), role=Role.CONTENT, width_scale=1.2))
        s.add(MathLabel(r"X(j\omega)", (0.0, 1.0), role=Role.CONTENT,
                        ha="left", va="bottom", offset_px=(8, -4)))
        s.add(MathLabel(r"1", (0.0, 1.0), role=Role.ANNOTATION,
                        ha="right", va="bottom", offset_px=(-6, -2)))
        return s

    for c in _centers(omega_s, om, w):
        pts = _tri_pts(c, om, w)
        if pts is not None:
            s.add(Curve(pts, role=Role.CONTENT,
                        width_scale=1.2 if c == 0.0 else 0.9))
    comb = [c for c in _centers(omega_s, 0.0, w)]
    s.items += impulses(comb, [p["comb_h"]] * len(comb), role=Role.ACCENT2,
                        width_scale=0.9)
    s.add(MathLabel(r"S(j\omega)", (omega_s, p["comb_h"]),
                    role=Role.ACCENT2, ha="center", va="bottom",
                    offset_px=(0, -4)))
    s.add(MathLabel(r"\omega_s", (omega_s, 0.0), role=Role.ANNOTATION,
                    ha="center", va="top", offset_px=(14, 5)))

    if omega_s >= 2 * om:
        # recoverable: the ideal lowpass (gain T, cutoff omega_s/2) passes
        # the base replica and nothing else — drawn dashed, construction
        wc = omega_s / 2
        h = p["lpf_h"]
        s.add(Curve(np.array([[-wc, 0], [-wc, h], [wc, h], [wc, 0]]),
                    role=Role.CONSTRUCTION, dash="dashed"))
        s.add(MathLabel(r"H_r(j\omega)", (0.0, h), role=Role.CONSTRUCTION,
                        ha="center", va="bottom", offset_px=(0, -3)))
    else:
        for x, y in _overlaps(omega_s, om, w, p["n_band"]):
            s.add(band(x, np.zeros_like(x), y, role=Role.ACCENT1,
                       opacity=0.5, outline=False))
        x0, y0 = _overlaps(omega_s, om, w, p["n_band"])[len(g["centers_bad"]) // 2]
        cx = float(x0.mean())
        s.add(Callout(r"\mathrm{alias}", (cx + 1.7, 1.3),
                      (cx + 0.05, float(y0.max()) * 0.45)))
    return s


def build(g):
    from figlib.figure import Figure, Panel
    p = g["params"]
    return Figure(
        panels=[Panel(_panel(g, None), tag=r"[a]"),
                Panel(_panel(g, p["omega_s_ok"]), tag=r"[b]"),
                Panel(_panel(g, p["omega_s_bad"]), tag=r"[c]")],
        grid=(3, 1), gap_px=26.0,
    )


def assertions(g):
    p = g["params"]
    om, w = p["omega_m"], p["window"]
    ck = Checks()

    # every replica is an exact shift of the base triangle (checked on a
    # probe grid, against the same _tri the drawing sampled)
    x = np.linspace(-w, w, 801)
    for c in g["centers_bad"]:
        ck.check(bool(np.allclose(_tri(x, c, om), _tri(x - c, 0.0, om))),
                 f"replica at {c} is not a pure shift")

    # separation criterion, both regimes, from the drawn arrays
    ck.check(g["gap_ok"] > 0 and not g["overlaps_ok"],
             "panel b: omega_s > 2 omega_m must leave replicas separated")
    ck.check(g["gap_bad"] < 0 and len(g["overlaps_bad"]) > 0,
             "panel c: omega_s < 2 omega_m must produce overlap bands")

    # the central overlap band is exactly [omega_s - omega_m, omega_m]
    mid = [ov for ov in g["overlaps_bad"]
           if ov[0][0] < om and ov[0][-1] > 0][0]
    ck.check(abs(mid[0][0] - (p["omega_s_bad"] - om)) < 1e-9 and
             abs(mid[0][-1] - om) < 1e-9,
             "central alias band endpoints are not [omega_s-omega_m, omega_m]")
    # its width is the textbook 2 omega_m - omega_s
    ck.check(abs((mid[0][-1] - mid[0][0]) - (2 * om - p["omega_s_bad"])) < 1e-9,
             "alias band width is not 2 omega_m - omega_s")

    # comb spacing = omega_s (the drawn impulse positions)
    cs = np.array(g["centers_bad"])
    ck.check(bool(np.allclose(np.diff(cs), p["omega_s_bad"])),
             "comb spacing is not omega_s")
    ck.done()
