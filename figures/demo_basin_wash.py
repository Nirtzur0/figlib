"""Exemplar: regions are content — wash the basins, don't leave them white.

Gradient flow of the double well V(x, y) = (x^2 - 1)^2 / 4 + y^2 / 2:

    xdot = x - x^3,    ydot = -y

Two sinks at (+-1, 0), a saddle at the origin. The saddle's stable
manifold W^s — here exactly the y-axis, a single curve of measure zero —
partitions the entire plane into the two basins of attraction.

The worked exemplar for: the wash channel (two basins as opaque
paper-blended fills, slot-matched so each basin could carry the same
correspondence hue as marks that refer to it), a streamline family at
WEIGHT_BG under a WEIGHT_ACTOR separatrix, hollow direction markers on
the family vs filled on the manifolds, and gate assertions that certify
the washes' claim — every drawn trajectory ends in the sink of the basin
it was seeded in.
"""

import numpy as np

from figlib.builders import integrate_streamlines, stagnation_points
from figlib.gates import Checks
from figlib.scene import Curve, FilledCurve, MathLabel, Point, Scene
from figlib.style import WEIGHT_ACTOR, WEIGHT_BG, WEIGHT_CONTENT, Role
from figlib.theme import RISO

CLAIM = (
    "One curve of measure zero — the saddle's stable manifold, here the "
    "y-axis — splits the whole plane into the two basins of the double-well "
    "gradient flow; every trajectory off it ends at the sink of its side."
)

THEME = RISO

PARAMS = {
    "xlim": (-2.25, 2.25),
    "ylim": (-1.42, 1.42),
    "seed_x": (-1.95, -1.35, -0.55, -0.18, 0.18, 0.55, 1.35, 1.95),
    "seed_y": (-1.18, -0.62, 0.62, 1.18),
    "h": 0.015,
    "max_len": 14.0,
    "sink_tol": 0.02,          # drawn endpoint must land this close to a sink
}


def _field(P):
    x, y = P[:, 0], P[:, 1]
    return np.column_stack([x - x ** 3, -y])


def compute(p):
    seeds = [(sx, sy) for sx in p["seed_x"] for sy in p["seed_y"]]
    family = integrate_streamlines(
        _field, seeds, p["xlim"], p["ylim"], h=p["h"], max_len=p["max_len"],
        arrows=(0.42,), role=Role.CONTENT, width_scale=WEIGHT_BG,
        arrow_scale=0.7, opacity=0.85)
    fixed = sorted(stagnation_points(_field, p["xlim"], p["ylim"], grid_n=64))
    return {"family": family, "seeds": seeds, "fixed": fixed, "p": p}


def build(g):
    p = g["p"]
    (x0, x1), (y0, y1) = p["xlim"], p["ylim"]
    s = Scene(xlim=p["xlim"], ylim=p["ylim"], clip="frame")

    # layer 1 — the claim itself as area: the two basins, wash-filled.
    # Slot-matched hues (wash(2)=sky, wash(1)=ochre): region identity on
    # the correspondence channel, at paper-blended strength so a third of
    # the page of ink stays quieter than any stroke on top of it.
    for i, (xa, xb) in ((2, (x0, 0.0)), (1, (0.0, x1))):
        rect = np.array([[xa, y0], [xb, y0], [xb, y1], [xa, y1]])
        s.add(FilledCurve(rect, color=RISO.wash(i), opacity=1.0,
                          outline=False))

    # layer 2 — the population: one streamline family, background weight,
    # hollow markers (the streamline convention)
    s.add(*g["family"])

    # layer 3 — the unstable manifold: saddle -> each sink along the
    # x-axis, filled markers (motion), content weight
    for xa, xb in ((0.0, -1.0), (0.0, 1.0)):
        xs = np.linspace(xa, xb, 120)
        s.add(Curve(np.column_stack([xs, np.zeros_like(xs)]),
                    role=Role.ACCENT1, width_scale=WEIGHT_CONTENT,
                    arrows=(0.55,), arrow_style="filled"))

    # layer 4 — THE object: W^s(0), the one curve the claim is about.
    # Heaviest ink on the page, cased to read over the family; each half
    # flows INTO the saddle (it is the stable manifold).
    for ya in (y1, y0):
        ys = np.linspace(ya, 0.0, 120)
        s.add(Curve(np.column_stack([np.zeros_like(ys), ys]),
                    role=Role.ACCENT2, width_scale=WEIGHT_ACTOR,
                    arrows=(0.45, 0.9), arrow_style="filled", casing=True))

    # fixed points: sinks filled, saddle hollow (the standard vocabulary)
    for (fx, fy) in g["fixed"]:
        s.add(Point((fx, fy), filled=not np.isclose(fx, 0.0), radius_scale=1.2))

    s.add(MathLabel(r"\dot x = x - x^3,\ \ \dot y = -y", (-2.1, 1.22),
                    ha="left", va="center", size_pt=10.5,
                    role=Role.ANNOTATION, halo=True))
    s.add(MathLabel(r"W^s(0)", (0.0, -1.02), ha="left", va="center",
                    offset_px=(7, 0), size_pt=11, halo=True,
                    color=RISO.ink(Role.ACCENT2).color))
    s.add(MathLabel(r"x_-^* = -1", (-1.0, 0.0), ha="center", va="top",
                    offset_px=(0, 10), size_pt=10, role=Role.ANNOTATION,
                    halo=True))
    s.add(MathLabel(r"x_+^* = +1", (1.0, 0.0), ha="center", va="top",
                    offset_px=(0, 10), size_pt=10, role=Role.ANNOTATION,
                    halo=True))
    s.add(MathLabel(r"\mathrm{basin\ of\ } x_-^*", (-1.62, -1.16),
                    ha="center", va="center", size_pt=10,
                    role=Role.ANNOTATION, halo=True))
    s.add(MathLabel(r"\mathrm{basin\ of\ } x_+^*", (1.62, 1.16),
                    ha="center", va="center", size_pt=10,
                    role=Role.ANNOTATION, halo=True))
    return s


def assertions(g):
    p = g["p"]
    c = Checks()

    # (1) the fixed-point census: exactly (+-1, 0) and (0, 0)
    c.check(len(g["fixed"]) == 3,
            f"expected 3 fixed points, found {len(g['fixed'])}")
    for (fx, fy), tx in zip(g["fixed"], (-1.0, 0.0, 1.0)):
        c.check(np.hypot(fx - tx, fy) < 1e-8,
                f"fixed point at ({fx:.6f}, {fy:.6f}), not ({tx:+.0f}, 0)")

    # (2) their types, from the (diagonal) Jacobian [1 - 3x^2, -1]:
    # sinks at +-1 (both eigenvalues < 0), saddle at 0 (opposite signs)
    for xs, kind in ((-1.0, "sink"), (1.0, "sink"), (0.0, "saddle")):
        lam = (1.0 - 3.0 * xs * xs, -1.0)
        if kind == "sink":
            c.check(lam[0] < 0 and lam[1] < 0, f"({xs:+.0f},0) is not a sink")
        else:
            c.check(lam[0] > 0 > lam[1], "(0,0) is not a saddle")

    # (3) both drawn manifolds are invariant lines of the field: no flow
    # crosses x = 0 (xdot = 0 there) and none leaves y = 0 (ydot = 0)
    ys = np.linspace(p["ylim"][0], p["ylim"][1], 101)
    c.check(float(np.max(np.abs(_field(np.column_stack(
        [np.zeros_like(ys), ys]))[:, 0]))) == 0.0,
        "flow crosses the drawn separatrix")
    xs = np.linspace(p["xlim"][0], p["xlim"][1], 101)
    c.check(float(np.max(np.abs(_field(np.column_stack(
        [xs, np.zeros_like(xs)]))[:, 1]))) == 0.0,
        "flow leaves the drawn unstable manifold")
    # and the flow ON each manifold runs the way the markers claim:
    # into the saddle along x = 0, outward toward each sink along y = 0
    c.check(bool(np.all(_field(np.column_stack(
        [np.zeros_like(ys[ys > 0]), ys[ys > 0]]))[:, 1] < 0)),
        "stable manifold's upper half does not flow into the saddle")
    mid = xs[(0 < np.abs(xs)) & (np.abs(xs) < 1)]
    c.check(bool(np.all(np.sign(_field(np.column_stack(
        [mid, np.zeros_like(mid)]))[:, 0]) == np.sign(mid))),
        "unstable manifold does not flow from the saddle toward the sinks")

    # (4) the washes tell the truth about the DRAWN family: every
    # trajectory stays on its seed's side and its forward end lands at
    # that side's sink
    for cv, (sx, sy) in zip(g["family"], g["seeds"]):
        side = np.sign(sx)
        c.check(bool(np.all(np.sign(cv.pts[:, 0]) == side)),
                f"trajectory seeded at ({sx:.2f}, {sy:.2f}) crosses the separatrix")
        end = cv.pts[-1]
        c.check(np.hypot(end[0] - side, end[1]) < p["sink_tol"],
                f"trajectory from ({sx:.2f}, {sy:.2f}) ends at "
                f"({end[0]:.3f}, {end[1]:.3f}), not its basin's sink")
    c.done()
