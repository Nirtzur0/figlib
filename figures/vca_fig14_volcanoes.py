"""VCA Fig [14] (p.66): the modular surface of 1/(1+z^2).

The mystery of the real function 1/(1+x^2) — smooth and bounded on all
of R, yet its Taylor series at 0 dies at |x| = 1 — dissolves when the
graph is seen as the real-axis slice of |f(z)| over C: the surface
erupts at the invisible poles z = +-i, and the series can only converge
up to the nearest eruption.
"""

import numpy as np

from figlib.scene import Scene
from figlib.style import Role
from figlib.surface3d import (Camera, as_floor, compose, label3,
                              polyline_items, surface_items, vector3)
from figlib.format import WIDE
from figlib.theme import RISO

THEME = RISO
FORMAT = WIDE

CLAIM = (
    "The graph of 1/(1+x^2) is the tranquil real-axis slice of the modular "
    "surface of 1/(1+z^2), which erupts to infinity at the poles z = +-i; "
    "the Taylor series of the real function at 0 therefore converges only "
    "up to distance 1 — the distance to the invisible complex poles."
)

PARAMS = {
    "half_width": 2.3,
    "grid_n": 116,
    "cap": 2.9,
    "knee": 1.6,
    "plane_margin": 0.55,
    "azim": -38.0,
    "elev": 30.0,
}


def compute(p):
    h = p["half_width"]
    xs = np.linspace(-h, h, p["grid_n"])
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    Zc = X + 1j * Y
    F_raw = 1.0 / np.abs(1.0 + Zc * Zc)
    # exact below the knee; smoothly compressed toward cap above it, so the
    # infinite chimneys render as clean spires with no clipping artifacts
    knee, cap = p["knee"], p["cap"]

    def compress(v):
        return np.where(v <= knee, v,
                        knee + (cap - knee) * np.tanh((v - knee) / (cap - knee)))

    F = compress(F_raw)
    # the |z| = 1 circle draped on the surface: it passes through both poles
    th = np.linspace(0, 2 * np.pi, 500)
    zc = np.exp(1j * th)
    circ_h = compress(1.0 / np.abs(1.0 + zc * zc))
    circle = np.column_stack([zc.real, zc.imag, circ_h])
    xt = np.linspace(-h, h, 220)
    trace = np.column_stack([xt, np.zeros_like(xt), 1.0 / (1.0 + xt * xt)])
    return {"X": X, "Y": Y, "F": F, "F_raw": F_raw, "knee": knee,
            "trace": trace, "circle": circle, "cap": p["cap"], "h": h}


def build(g):
    cam = Camera(PARAMS["azim"], PARAMS["elev"])
    h, m = g["h"], PARAMS["plane_margin"]
    s = Scene()

    # base plane C: paper-adjacent ground, theme-routed so it restyles
    plane3 = np.array([[-h - m, -h - m, 0], [h + m, -h - m, 0],
                       [h + m, h + m, 0], [-h - m, h + m, 0]], dtype=float)
    plane_items = as_floor(surface_items(
        plane3[:, 0].reshape(2, 2), plane3[:, 1].reshape(2, 2),
        np.zeros((2, 2)), cam, dark=THEME.paper[1], lite=THEME.paper[0],
        edge=THEME.ink(Role.MUTED).color, edge_width=0.6))

    surf = surface_items(g["X"], g["Y"], g["F"], cam,
                         shade=THEME.surface_shade, edge=THEME.surface_edge)
    trace = polyline_items(g["trace"], cam, depth_bias=0.05,
                           role=Role.ACCENT2, width_scale=1.5)
    circle = polyline_items(g["circle"], cam, depth_bias=0.06,
                            role=Role.ACCENT1, width_scale=1.1)

    # dashed verticals through the pole chimneys
    poles = []
    for yy in (1.0, -1.0):
        line3 = np.column_stack([np.zeros(40), np.full(40, yy),
                                 np.linspace(0.0, g["cap"] + 0.35, 40)])
        poles += polyline_items(line3, cam, depth_bias=0.02,
                                role=Role.CONSTRUCTION, width_scale=0.9)

    s.items.extend(compose(plane_items, surf, trace, circle, poles))

    # axis arrows on the plane, outside the surface footprint
    for tail3, tip3, latex in (
        ((h + 0.12, 0.0, 0.0), (h + m + 0.85, 0.0, 0.0), r"x = \mathrm{Re}\, z"),
        ((h + m + 0.28, -2.0, 0.0), (h + m + 0.28, -0.4, 0.0), r"\mathrm{Im}\, z"),
    ):
        s.add(vector3(tail3, tip3, cam, role=Role.ANNOTATION, width_scale=0.9))
        s.add(label3(latex, tip3, cam, ha="left", va="center",
                     offset_px=(8, 4), role=Role.ANNOTATION))

    # pole labels above the chimney tops
    for yy, latex in ((1.0, r"z = i"), (-1.0, r"z = -i")):
        s.add(label3(latex, (0.0, yy, g["cap"] + 0.42), cam,
                     ha="center", va="bottom", offset_px=(0, -3)))

    # the tranquil slice, named at its front end
    slice_anchor = (1.45, 0.0, 0.0)
    s.add(label3(r"y = \frac{1}{1+x^2}", slice_anchor, cam, ha="center",
                 va="top", offset_px=(-10, 34), role=Role.ACCENT2))

    # the mechanism: R = 1 = distance from 0 to the poles
    s.add(label3(r"R = \mathrm{dist}(0, \pm i) = 1", slice_anchor, cam,
                 ha="center", va="top", offset_px=(-10, 81), size_pt=10,
                 role=Role.ANNOTATION))

    # the convergence circle, labeled where it crosses the front
    s.add(label3(r"|z| = 1", (0.72, -0.72, 0.75), cam, ha="left", va="top",
                 offset_px=(16, 20), role=Role.ACCENT1, halo=True))
    # name the surface and admit the truncation
    s.add(label3(r"\left|\tfrac{1}{1+z^2}\right|", (-2.0, 0.9, 0.55), cam,
                 ha="right", va="center", offset_px=(-78, -46), role=Role.ANNOTATION))
    s.add(label3(r"(\text{spires truncated; poles are infinite})",
                 (0.0, 1.0, g["cap"] + 0.42), cam, ha="left", va="center",
                 offset_px=(60, -4), size_pt=9, role=Role.ANNOTATION))

    # plane label
    s.add(label3(r"\mathbb{C}", (h + m - 0.25, -h + 0.28, 0.0), cam,
                 ha="center", va="center", role=Role.ANNOTATION))
    return s


def assertions(g):
    # surface really is |1/(1+z^2)|, checked via the independent identity
    # |1/(1+z^2)| * |z-i| * |z+i| = 1
    Zc = g["X"] + 1j * g["Y"]
    prod = g["F_raw"] * np.abs(Zc - 1j) * np.abs(Zc + 1j)
    assert np.max(np.abs(prod - 1.0)) < 1e-9, "surface heights are not |1/(1+z^2)|"
    # plotted heights are exact wherever below the knee (covers the trace)
    exact = g["F_raw"] <= g["knee"]
    assert np.array_equal(g["F"][exact], g["F_raw"][exact]), "compression leaked below knee"
    # the trace is the real restriction
    xt, _, yt = g["trace"].T
    assert np.max(np.abs(yt - 1.0 / (1.0 + xt * xt))) < 1e-12, "trace is not 1/(1+x^2)"
    # the masked chimneys sit exactly at the poles
    for yy in (1.0, -1.0):
        near = g["F"] > 0.95 * g["cap"]
        d = np.hypot(g["X"][near], g["Y"][near] - yy)
        assert d.min() < 0.12, f"no spire at z = {yy}i"
