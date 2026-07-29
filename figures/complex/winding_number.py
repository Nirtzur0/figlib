"""The argument principle, Needham-style: count zeros by watching the image
curve turn.

Needham's argument (Visual Complex Analysis, ch. 7) replaces a contour
integral with something the eye can tally directly: as z walks once
counterclockwise around a closed contour Gamma that misses every zero and
pole of f, the image point f(z) walks a closed curve in the w-plane, and
that image curve winds around the origin exactly as many times as Gamma
encloses zeros minus poles (multiplicity counted). The mechanism is
angle bookkeeping, not calculus — arg f(z) = sum_i arg(z - z_i) -
sum_j arg(z - p_j), and only the terms for enclosed roots return a net
2*pi as z completes its loop; everything outside contributes zero net
turn because its "vantage point" never gets circled.

This figure earns its keep by making that bookkeeping visible rather than
asserted: a double zero and a simple zero sit inside a circular Gamma,
drawn with direction ticks; the image f(Gamma) is drawn with the same
tick parametrization, so the reader watches it coil around w = 0 three
times while Gamma goes around once — the double root pulling twice the
turn of the simple one is the whole content of "multiplicity."
"""

import numpy as np

from figlib.figure import Connector, Figure, Panel
from figlib.format import WIDE
from figlib.gates import Checks
from figlib.scene import Callout, Curve, MathLabel, Point, Scene
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "As z traces Gamma once counterclockwise, f(z) = (z-a)^2 (z-b) traces a "
    "closed curve that winds around w = 0 exactly three times — twice for "
    "the double zero at a, once for the simple zero at b — because the "
    "winding number of f(Gamma) equals the enclosed zero count with "
    "multiplicity."
)

EXPOSITION = """
Chapter 7, "Winding Numbers and Topology," builds toward the Argument
Principle by first making the winding number itself a physical,
countable thing — how many times a loop wraps the origin, read off a
picture rather than computed as an integral. Needham then asks the
question the topology exists to answer: if f is analytic and Gamma is a
loop that avoids every zero and pole of f, how many times does the
image curve f(Gamma) wind around w = 0? His answer, argued directly
from arg f(z) = sum_i arg(z - z_i) - sum_j arg(z - p_j), is that only
the zeros and poles Gamma actually encloses contribute net turning as z
completes its loop — everything outside contributes nothing, because
its vantage point is never circled. A zero of multiplicity two pulls
twice the turning of a simple zero, for the same reason a rope wound
twice around a post takes two full turns to unwind.

This figure makes that bookkeeping visible: a double zero and a simple
zero sit inside a circular Gamma, and the reader watches f(Gamma) coil
around the origin three times for one circuit of Gamma — two turns
traceable to the double root, one to the simple one — so multiplicity
is read off the image curve's own winding, not asserted from the
formula.
"""

THEME = RISO
FORMAT = WIDE

PARAMS = {
    "a": (0.40, 0.15),          # double zero, inside Gamma
    "b": (-0.55, -0.35),        # simple zero, inside Gamma
    "radius": 1.15,             # |Gamma|: encloses both a and b
    "n_samples": 720,
    "arrow_fracs": (0.05, 0.22, 0.40, 0.56, 0.73, 0.90),
    "z_margin": 0.55,           # z-plane axis overrun past the circle
}


def compute(p):
    a = complex(*p["a"])
    b = complex(*p["b"])
    R = p["radius"]

    t = np.linspace(0.0, 2 * np.pi, p["n_samples"], endpoint=True)
    z = R * np.exp(1j * t)
    w = (z - a) ** 2 * (z - b)

    # the winding number as it is actually measured: unwrap arg f(z) along
    # the sampled contour and read off the net turn in units of 2*pi
    arg = np.unwrap(np.angle(w))
    winding = (arg[-1] - arg[0]) / (2.0 * np.pi)

    # the independent check: the SAME cubic's roots, found by companion-
    # matrix eigenvalues (numpy.roots), not by restating "a, a, b"
    roots = np.roots(np.poly([a, a, b]))
    enclosed = int(np.round(np.sum(np.abs(roots) < R)))
    clearance = float(np.min(np.abs(z[:, None] - roots[None, :])))

    return {
        "a": a, "b": b, "R": R, "t": t, "z": z, "w": w,
        "winding": winding, "roots": roots, "enclosed": enclosed,
        "clearance": clearance,
        "arrow_fracs": p["arrow_fracs"],
        "z_margin": p["z_margin"],
    }


def _z_plane(g) -> Scene:
    R, m = g["R"], g["z_margin"]
    s = Scene(xlim=(-R - m, R - m + 1.0), ylim=(-R - m, R - m + 1.0))

    # bare Re/Im axes, frame furniture
    lim = R + m
    s.add(Curve(np.array([[-lim, 0.0], [lim, 0.0]]), role=Role.FRAME))
    s.add(Curve(np.array([[0.0, -lim], [0.0, lim]]), role=Role.FRAME))

    # Gamma itself, with direction ticks
    s.add(Curve(np.column_stack([g["z"].real, g["z"].imag]),
                role=Role.CONTENT, closed=True,
                arrows=g["arrow_fracs"]))
    s.add(MathLabel(r"\Gamma", (R * np.cos(2.4), R * np.sin(2.4)),
                    role=Role.CONTENT, ha="left", va="bottom",
                    offset_px=(6, -4)))

    # the simple zero b: one filled dot
    bp = (g["b"].real, g["b"].imag)
    s.add(Point(bp, role=Role.ACCENT2))
    s.add(MathLabel(r"b", bp, role=Role.ACCENT2, ha="left", va="top",
                    offset_px=(7, 7)))

    # the double zero a: a plain filled dot — multiplicity is carried by the
    # "(double)" annotation and, mechanically, by the w-plane panel showing
    # two of the three turns collapse onto it (a hollow ring would misuse
    # the excluded/attained convention that "open dot" already owns)
    ap = (g["a"].real, g["a"].imag)
    s.add(Point(ap, role=Role.ACCENT1, radius_scale=1.15))
    s.add(MathLabel(r"a", ap, role=Role.ACCENT1, ha="left", va="bottom",
                    offset_px=(9, -7)))
    s.add(MathLabel(r"\text{(double)}", ap, role=Role.ANNOTATION, size_pt=9,
                    ha="left", va="top", offset_px=(9, 9)))

    return s


def _w_plane(g) -> Scene:
    w = g["w"]
    reach = float(np.max(np.abs(w))) * 1.12
    s = Scene(xlim=(-reach, reach), ylim=(-reach, reach))

    lim = reach * 0.98
    s.add(Curve(np.array([[-lim, 0.0], [lim, 0.0]]), role=Role.FRAME))
    s.add(Curve(np.array([[0.0, -lim], [0.0, lim]]), role=Role.FRAME))

    s.add(Curve(np.column_stack([w.real, w.imag]),
                role=Role.CONTENT, closed=True,
                arrows=g["arrow_fracs"]))
    s.add(MathLabel(r"f(\Gamma)", (w[0].real, w[0].imag), role=Role.CONTENT,
                    ha="left", va="bottom", offset_px=(39, 6)))

    s.add(Point((0.0, 0.0), role=Role.ANNOTATION, radius_scale=0.8))
    s.add(Callout(
        latex=rf"n = {g['enclosed']}",
        anchor=(0.62 * reach, 0.72 * reach),
        target=(0.0, 0.0),
    ))

    return s


def build(g):
    return Figure(
        panels=[Panel(_z_plane(g), tag=r"[a]"),
                Panel(_w_plane(g), tag=r"[b]")],
        connectors=[Connector(0, 1, kind="map",
                              label=r"f(z) = (z-a)^2 (z-b)")],
    )


def assertions(g):
    ck = Checks()

    # Gamma passes nowhere near a zero — the argument principle's hypothesis
    ck.check(g["clearance"] > 0.15,
             f"Gamma comes within {g['clearance']:.4f} of a root — too close "
             "to trust the discretized argument")

    # the load-bearing check: the winding number READ OFF the sampled
    # contour (unwrapped argument of the drawn f(Gamma)) must match the
    # zero count found independently, by root-finding the same cubic
    ck.check(abs(g["winding"] - g["enclosed"]) < 1e-3,
             f"measured winding {g['winding']:.6f} != enclosed zero count "
             f"{g['enclosed']} (roots at {np.round(g['roots'], 4)})")

    # the winding number is (very nearly) an integer — a real check on a
    # discretized curve, not a restatement of the theorem
    ck.check(abs(g["winding"] - round(g["winding"])) < 1e-3,
             f"measured winding {g['winding']:.6f} is not close to an integer")

    # the double zero a really does account for two of the three turns:
    # drop it from the product and the remaining curve (b alone) should
    # wind once, not three times — the multiplicity is doing real work
    z, b = g["z"], g["b"]
    w_b_only = z - b
    arg_b = np.unwrap(np.angle(w_b_only))
    winding_b = (arg_b[-1] - arg_b[0]) / (2.0 * np.pi)
    ck.check(abs(winding_b - 1.0) < 1e-3,
             f"b alone should contribute winding 1, measured {winding_b:.6f}")

    ck.done()
