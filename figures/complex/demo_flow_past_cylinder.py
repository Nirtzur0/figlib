"""Flow past a cylinder with circulation (VCA Ch12).

Complex potential Omega(z) = U(z + 1/z) - i Gamma/(2pi) log z; the
streamlines are the level sets of Psi = Im Omega, drawn at equal steps
so line density encodes speed. Circulation Gamma = -2piU puts the two
stagnation points at sin(theta) = Gamma/4piU = -1/2 — visibly slid
below the cylinder — and crowds the flux tubes over the top.
"""

import numpy as np

from figlib.builders import stagnation_points, stream_function_lines
from figlib.format import WIDE
from figlib.scene import Curve, FilledCurve, MathLabel, Point, Scene, Vector
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "Circulation slides both stagnation points down the cylinder to "
    "sin(theta_s) = Gamma/4piU and crowds the equally spaced Psi-levels over "
    "the top: the flux-tube density is the speedup that generates lift."
)

EXPOSITION = """
Needham reaches this picture two figures after establishing the totally
irrotational flow round the disc (VCA, "Flows and Harmonic Functions,"
Fig [12]): Omega = U(z + 1/z), no circulation, stagnation points sitting
symmetrically at +-1. He then superposes a vortex term
-i*Gamma/(2pi) log z -- Fig [13] -- and reports, without derivation, that
the two stagnation points slide toward each other along the circle as
|Gamma| grows, coalescing at i for one critical value and then leaving
the circle altogether (Fig [14], a qualitatively different flow). The
reader is left to take the sliding on faith. This figure is the picture
that makes sin(theta_s) = Gamma/(4 pi U) checkable: circulation at a
value short of coalescence, the two stagnation points visibly displaced
but still on the cylinder, and the equally spaced Psi-levels crowding
over the top -- the same equal-increment k-cells from the complex-
potential construction, now compressed on one side, which is the
geometric content of lift before any mention of the Kutta-Joukowski
theorem.
"""

FORMAT = WIDE
THEME = RISO

PARAMS = {
    "U": 1.0,
    "Gamma": -2.0 * np.pi,      # sin(theta_s) = Gamma/(4 pi U) = -1/2
    "xlim": (-3.6, 3.6),
    "ylim": (-2.3, 2.3),
    "dpsi": 0.4,                # equal Psi steps: density encodes speed
    "k_lo": -2, "k_hi": 8,      # levels k*dpsi, k != 0 (0 is the separatrix)
    "n": 520,                   # contour grid resolution per axis
}


def compute(p):
    U, G = p["U"], p["Gamma"]

    def psi(x, y):
        r2 = x * x + y * y
        return U * y * (1.0 - 1.0 / r2) - G / (4.0 * np.pi) * np.log(r2)

    def vel(P):
        z = P[:, 0] + 1j * P[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            W = U * (1.0 - 1.0 / (z * z)) - 1j * G / (2.0 * np.pi * z)
        return np.column_stack([W.real, -W.imag])

    inside = lambda X, Y: X * X + Y * Y < 1.0
    levels = [k * p["dpsi"] for k in range(p["k_lo"], p["k_hi"] + 1) if k != 0]
    streams = stream_function_lines(
        psi, p["xlim"], p["ylim"], levels, n=p["n"], mask=inside,
        arrows=(0.5,), role=Role.CONTENT, width_scale=0.8, arrow_scale=0.8)
    # Psi = 0 = Psi(stagnation): the separatrix, THE distinguished object
    sep = stream_function_lines(
        psi, p["xlim"], p["ylim"], [0.0], n=p["n"], mask=inside,
        arrows=(0.3, 0.85), role=Role.ACCENT2, width_scale=1.4)
    stag = stagnation_points(vel, p["xlim"], p["ylim"], grid_n=96)
    return {"streams": streams, "sep": sep, "stag": sorted(stag),
            "levels": levels, "psi": psi, "vel": vel, "p": p}


def build(g):
    p = g["p"]
    s = Scene(xlim=p["xlim"], ylim=p["ylim"])
    s.add(*g["streams"])
    s.add(*g["sep"])

    # the obstacle: a filled disc over the contour ends at its boundary
    th = np.linspace(0.0, 2.0 * np.pi, 240)
    circle = np.column_stack([np.cos(th), np.sin(th)])
    s.add(FilledCurve(circle, role=Role.MUTED, opacity=0.85, outline=False))
    s.add(Curve(circle, closed=True, role=Role.CONTENT, width_scale=1.0))

    # circulation dial: clockwise arc (Gamma < 0), filled head = motion
    arc_t = np.linspace(2.0, -2.4, 80)
    s.add(Curve(0.5 * np.column_stack([np.cos(arc_t), np.sin(arc_t)]),
                role=Role.ANNOTATION, arrows=(1.0,), arrow_scale=0.85))
    s.add(MathLabel(r"\Gamma", (0.0, 0.0), ha="center", va="center", size_pt=13))

    # the free stream, far upstream, centered in its flux tube
    s.add(Vector((-3.38, 1.54), (-2.88, 1.54), role=Role.CONTENT))
    s.add(MathLabel(r"U", (-3.13, 1.54), ha="center", va="bottom",
                    offset_px=(0, -9)))

    # stagnation points and the theorem that places them
    for x, y in g["stag"]:
        s.add(Point((x, y), filled=True, role=Role.ACCENT2, radius_scale=0.9))
    s.add(MathLabel(r"\sin\theta_s = \Gamma/4\pi U", (0.0, -1.25),
                    ha="center", va="center", role=Role.ANNOTATION))
    s.add(MathLabel(r"\Psi = \Psi_s", (2.5, -1.04), ha="left",
                    va="bottom", role=Role.ACCENT2, size_pt=10))

    # the honesty device: the levels are an equal ladder
    s.add(MathLabel(r"\text{equal steps }\Delta\Psi", (-3.45, -2.1),
                    ha="left", va="bottom", size_pt=9, role=Role.ANNOTATION))
    return s


def assertions(g):
    p, psi, vel = g["p"], g["psi"], g["vel"]
    ladder = np.sort(np.append(np.array(g["levels"]), 0.0))
    span = ladder[-1] - ladder[0]
    # every emitted streamline point lies on its Psi-level
    for c in g["streams"] + g["sep"]:
        vals = psi(c.pts[:, 0], c.pts[:, 1])
        err = np.min(np.abs(vals[:, None] - ladder[None, :]), axis=1)
        assert err.max() < 2e-3 * span, f"streamline off its level by {err.max():.2e}"
    # two stagnation points, on the cylinder, where the theorem puts them
    assert len(g["stag"]) == 2, f"expected 2 stagnation points, got {len(g['stag'])}"
    sin_ts = p["Gamma"] / (4.0 * np.pi * p["U"])
    for x, y in g["stag"]:
        assert np.hypot(*vel(np.array([[x, y]]))[0]) < 1e-8, "not a field zero"
        assert abs(np.hypot(x, y) - 1.0) < 1e-6, "stagnation point off |z|=1"
        assert abs(y - sin_ts) < 1e-6, "stagnation angle violates sin = Gamma/4piU"
    # the separatrix level is Psi at the stagnation points
    sx, sy = g["stag"][0]
    assert abs(float(psi(np.float64(sx), np.float64(sy)))) < 1e-9
    # equal ladder
    d = np.diff(ladder)
    assert np.allclose(d, d[0]), "Psi-levels not equally spaced"
