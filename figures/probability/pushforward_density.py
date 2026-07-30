"""Change of variables, made visible: mass is conserved, area is not.

When a map T pushes a density forward, the new density is not "the old
density carried along" — it is divided by |det J_T|, because probability
is a MEASURE (mass on a set), not a function value, and T distorts the
area a unit of mass occupies. This is the change-of-variables formula
p_Y(T(x)) = p_X(x) / |det J_T(x)|, read off a grid instead of derived: a
uniform density rho_0 on a square is pushed through a nonlinear shear T,
the cells stretch where the map expands and shrink where it contracts,
and the drawn density on each image cell is exactly rho_0 times the
inverse of that cell's area ratio — nothing else is free to vary once
mass is held fixed.

The same identity is the one line every normalizing-flow paper writes
down and every implementation must get right: log p_Y(y) = log p_X(x) -
log|det J_T(x)|, x = T^{-1}(y). The log-det term is not a regularizer or
a correction — it IS the bookkeeping that makes p_Y integrate to 1 after
the map has moved mass into a smaller or larger region. A flow that gets
the sign or the cell wrong silently mis-normalizes; this figure draws the
identity on the actual geometry so there is nothing left to take on
faith.
"""

import numpy as np

from figlib.correspond import Correspondence
from figlib.figure import Connector, Figure, Panel
from figlib.format import WIDE
from figlib.gates import Checks
from figlib.layout import geometry_extents
from figlib.plots import Ticks, colorbar, linear
from figlib.scene import Callout, Curve, FilledCurve, MathLabel, Scene
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "Mass is conserved under a pushforward, area is not: pushing a uniform "
    "density through a nonlinear map T, each cell's image density equals "
    "the source density divided by that cell's own |det J_T| — the cell "
    "that grows gets visibly lighter, the cell that shrinks visibly darker, "
    "read straight off the drawn polygons' shoelace areas."
)

EXPOSITION = """
Every normalizing-flow paper writes down log p_Y(y) = log p_X(x) -
log|det J_T(x)| as a single line of bookkeeping, and it is easy to read
past it as a correction term rather than see what it is actually doing.
Probability is a measure -- mass assigned to a set -- not a function value
carried along by a map, so when a transformation T pushes a density
forward, the density at the image point has to be divided by how much the
map locally stretched or compressed area, because the SAME mass now
occupies a different amount of space. A reader who has only manipulated
the change-of-variables formula algebraically needs to see it happen on an
actual grid: a uniform density pushed through a nonlinear shear, where the
cells that expand visibly lighten and the cells that contract visibly
darken, and where that lightening or darkening is exactly the reciprocal
of the cell's own area ratio -- not an approximation of it. Get the sign or
the determinant wrong in an implementation and the model silently
mis-normalizes; this figure is the geometry that makes the correct
bookkeeping visible instead of assumed.
"""

THEME = RISO
FORMAT = WIDE

PARAMS = {
    "n": 5,                 # grid cells per axis
    "L": 1.0,               # domain half-width: x, y in [-L, L]
    "k": 0.35,               # shear strength of T; det J = 1 + k*(y - x)
    "rho0": 1.0,             # uniform source density
    "bar_thick": 0.16,       # colorbar strip thickness, math units
    "bar_gap": 0.55,         # gap between image panel and colorbar
    "colorbar_frac": 0.11,   # page width fraction reserved for the colorbar
}


def _T(pts: np.ndarray, k: float) -> np.ndarray:
    """The pushforward map: Tx = x(1+ky), Ty = y(1-kx).

    Each component is linear in x for fixed y and linear in y for fixed x,
    so a grid edge (constant x or constant y) maps to a straight segment —
    every cell's image is an exact quadrilateral, not a curve to sample.
    """
    x, y = pts[:, 0], pts[:, 1]
    return np.column_stack([x * (1.0 + k * y), y * (1.0 - k * x)])


def _det_j(x: np.ndarray, y: np.ndarray, k: float) -> np.ndarray:
    """det J_T(x, y) = (1+ky)(1-kx) + k^2 xy = 1 + k(y - x), exactly."""
    return 1.0 + k * (y - x)


def _shoelace(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def compute(p):
    n, L, k, rho0 = p["n"], p["L"], p["k"], p["rho0"]
    edges = np.linspace(-L, L, n + 1)

    src_cells, img_cells, centers, img_density, jac = [], [], [], [], []
    for i in range(n):
        for j in range(n):
            x0, x1 = edges[i], edges[i + 1]
            y0, y1 = edges[j], edges[j + 1]
            src = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            img = _T(src, k)
            xc, yc = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            d = float(_det_j(np.array(xc), np.array(yc), k))
            src_cells.append(src)
            img_cells.append(img)
            centers.append((xc, yc))
            jac.append(d)
            img_density.append(rho0 / abs(d))

    src_cells = np.array(src_cells)          # (n*n, 4, 2)
    img_cells = np.array(img_cells)
    centers = np.array(centers)
    jac = np.array(jac)
    img_density = np.array(img_density)
    src_area = np.array([_shoelace(c) for c in src_cells])
    img_area = np.array([_shoelace(c) for c in img_cells])
    src_density = np.full(len(src_cells), rho0)

    i_expand = int(np.argmax(jac))     # area grows most -> density drops most
    i_compress = int(np.argmin(jac))   # area shrinks most -> density rises most

    # a fine sample of det J over the whole domain, to certify T never
    # folds (orientation-preserving, hence genuinely invertible there)
    xs = np.linspace(-L, L, 60)
    xx, yy = np.meshgrid(xs, xs)
    jac_field = _det_j(xx, yy, k)

    dmin = float(min(rho0, img_density.min()))
    dmax = float(max(rho0, img_density.max()))
    img_x, img_y = img_cells[..., 0].ravel(), img_cells[..., 1].ravel()

    return {
        "p": p,
        "src_cells": src_cells, "img_cells": img_cells,
        "src_area": src_area, "img_area": img_area,
        "src_density": src_density, "img_density": img_density,
        "jac": jac, "jac_field": jac_field, "centers": centers,
        "i_expand": i_expand, "i_compress": i_compress,
        "dmin": dmin, "dmax": dmax,
        "img_ylim": (float(img_y.min()), float(img_y.max())),
    }


def _t(density: float, g) -> float:
    span = g["dmax"] - g["dmin"]
    return 0.0 if span <= 0 else float(np.clip((density - g["dmin"]) / span, 0.0, 1.0))


def _panel(g, mapped: bool) -> Scene:
    cells = g["img_cells"] if mapped else g["src_cells"]
    density = g["img_density"] if mapped else g["src_density"]
    s = Scene()
    for cell, rho in zip(cells, density):
        # UNKEYED, deliberately: a fill here is coloured by DENSITY, so its ink
        # rides the order ramp. Naming it 'compressed-cell' would put two
        # channels under one name -- the ramp's #b15319 and the accent's
        # #c8402f -- and a hue that names an object cannot also encode a
        # magnitude. The identity of the two annotated cells is carried by the
        # accent outlines below, which is the only mark whose hue means "which".
        s.add(FilledCurve(cell, role=Role.CONTENT, opacity=1.0, outline=True,
                          color=THEME.ramp(_t(rho, g))))
    # accent outlines pick the two annotated cells out of the wash, tracked
    # by the SAME hue across both panels so "this is the same cell" is a
    # correspondence read, not a caption
    s.add(Curve(cells[g["i_expand"]], closed=True, role=Role.ACCENT1,
                width_scale=1.5, key="expanded-cell"))
    s.add(Curve(cells[g["i_compress"]], closed=True, role=Role.ACCENT2,
                width_scale=1.5, key="compressed-cell"))

    if not mapped:
        x0, x1 = g["src_cells"][..., 0].min(), g["src_cells"][..., 0].max()
        y1 = g["src_cells"][..., 1].max()
        s.add(MathLabel(rf"\rho_X = {g['p']['rho0']:.2f}\ \text{{(uniform)}}",
                        (float((x0 + x1) / 2.0), float(y1)), role=Role.ANNOTATION,
                        ha="center", va="bottom", offset_px=(0, -10), size_pt=10.0))

    if mapped:
        y_lo, y_hi = g["img_ylim"]
        span = y_hi - y_lo
        for idx, label_key, role, above in (
            (g["i_expand"], "expanded-cell", Role.ACCENT1, True),
            (g["i_compress"], "compressed-cell", Role.ACCENT2, False),
        ):
            cx, cy = cells[idx].mean(axis=0)
            jac = g["jac"][idx]
            rho = g["img_density"][idx]
            anchor_y = y_hi + 0.30 * span if above else y_lo - 0.30 * span
            s.add(Callout(
                rf"\det J_T = {jac:.2f},\ \ \rho = {rho:.2f}",
                anchor=(float(cx), float(anchor_y)), target=(float(cx), float(cy)),
                role=role, size_pt=10.0, key=label_key))
    return s


CORRESPONDENCE = [
    Correspondence(
        parts=(0, 1),
        varies=("the pushforward map T applied to every cell: the tracked "
                "cell's area and hence its density change by the inverse "
                "of |det J_T| at that cell"),
        changes=("expanded-cell", "compressed-cell"),
    ),
]


def build(g):
    p = g["p"]
    a, b = _panel(g, mapped=False), _panel(g, mapped=True)
    dx_a = np.ptp(geometry_extents(a)[0])
    dx_b = np.ptp(geometry_extents(b)[0])
    remain = 1.0 - p["colorbar_frac"]
    frac_b = remain * dx_b / (dx_a + dx_b)
    frac_a = remain - frac_b

    # colorbar: its own scene sharing the ramp, so the reader can read a
    # cell's color as a number instead of a rank
    bar_scale = linear(g["dmin"], g["dmax"])
    tv = np.array([g["dmin"], p["rho0"], g["dmax"]])
    bar = Scene()
    bar.add(*colorbar(bar_scale, THEME.ramp, at=(0.0, 0.0), length=2.0,
                      thickness=p["bar_thick"], orient="y",
                      ticks=Ticks(values=tv, positions=bar_scale.fwd(tv),
                                  labels=tuple(f"{v:.2f}" for v in tv)),
                      label=r"\rho"))
    # room for the tick-label glyphs, which geometry_extents does not
    # measure (it sees the anchor point, not the typeset width)
    bar.xlim = (-0.1, p["bar_thick"] + 0.85)
    bar.ylim = (-0.2, 2.3)

    return Figure(
        panels=[Panel(a, tag=r"[a]\ \rho_X", width_frac=frac_a),
                Panel(b, tag=r"[b]\ \rho_Y", width_frac=frac_b),
                Panel(bar, tag=None, frame=False,
                      width_frac=p["colorbar_frac"])],
        connectors=[Connector(0, 1, kind="map", label=r"T")],
    )


def assertions(g):
    c = Checks()
    rho0 = g["p"]["rho0"]

    # mass is conserved cell-by-cell: image_density * image_area (both
    # measured by the shoelace formula on the ACTUAL drawn vertices) must
    # equal source_density * source_area — this is the change-of-variables
    # identity holding on the geometry, not restated from the Jacobian
    src_mass = g["src_density"] * g["src_area"]
    img_mass = g["img_density"] * g["img_area"]
    c.check(np.max(np.abs(img_mass - src_mass)) < 1e-9,
            f"per-cell mass not conserved: max |img_mass - src_mass| = "
            f"{np.max(np.abs(img_mass - src_mass)):.3e}")

    # total mass across the whole grid agrees between the two panels
    c.check(abs(src_mass.sum() - img_mass.sum()) < 1e-8,
            f"total mass drifted: {src_mass.sum():.6f} vs {img_mass.sum():.6f}")

    # the drawn density really is the source density divided by the
    # cell's OWN area ratio (shoelace on the drawn polygons), not a
    # restatement of the analytic Jacobian used to color it
    area_ratio = g["img_area"] / g["src_area"]
    c.check(np.max(np.abs(g["img_density"] * area_ratio - rho0)) < 1e-9,
            "image_density * (image_area / source_area) != source_density "
            "on the drawn polygons")

    # T never folds over the drawn domain: det J stays one sign, so the
    # pushforward is genuinely invertible everywhere it is depicted
    c.check(g["jac_field"].min() > 0.05,
            f"det J_T drops to {g['jac_field'].min():.3f} on the sampled "
            f"domain — T is not safely invertible where it is drawn")

    # the two annotated cells really are the extremes: no undrawn cell
    # expands or compresses more than the ones singled out for labels
    c.check(g["jac"][g['i_expand']] >= g["jac"].max() - 1e-12,
            "the labeled 'expanded' cell is not the cell of maximum |det J|")
    c.check(g["jac"][g['i_compress']] <= g["jac"].min() + 1e-12,
            "the labeled 'compressed' cell is not the cell of minimum |det J|")

    # the two annotated cells are visibly different regimes, not a wash
    c.check(g["jac"][g["i_expand"]] > 1.15 and g["jac"][g["i_compress"]] < 0.85,
            "expanded/compressed cells are not distorted enough to read by "
            "eye: |det J| too close to 1 for the claim to be visible")

    c.done()
