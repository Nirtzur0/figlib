"""The annulus 1 <= |z| <= 2 read off the modulus field.

Level-set reading: the raster is |z| itself, the two circles are its 1-
and 2-level sets, and the stippled band between them is the annulus. The
brace measures the radial width, the callout names the excluded disc, and
the frame deliberately crops the outer circle at the top (Scene.clip).
"""

import numpy as np

from figlib.format import COLUMN
from figlib.scene import (Brace, Callout, Curve, FilledCurve, MathLabel,
                          RasterField, Scene)
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "The annulus 1 <= |z| <= 2 is the band between the level sets |z| = 1 "
    "and |z| = 2 of the modulus field: its radial width is delta = 1, its "
    "area is pi(2^2 - 1^2) = 3 pi, and the open unit disc is excluded."
)

THEME = RISO
FORMAT = COLUMN

PARAMS = {
    "r_in": 1.0,
    "r_out": 2.0,
    "n_pts": 720,                       # polygon area error ~1e-4 << gate tol
    "raster_n": 48,
    "raster_extent": (-2.4, 2.4, -2.4, 2.4),
    "xlim": (-2.4, 3.35),
    "ylim": (-2.4, 1.7),                # crops the outer circle's top
    "brace_angle_deg": 225.0,
    "band_label_angle_deg": 30.0,
}


def _circle(r: float, n: int) -> np.ndarray:
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return r * np.column_stack([np.cos(th), np.sin(th)])


def compute(p):
    outer = _circle(p["r_out"], p["n_pts"])
    inner = _circle(p["r_in"], p["n_pts"])
    x0, x1, y0, y1 = p["raster_extent"]
    n = p["raster_n"]
    xs = x0 + (x1 - x0) * (np.arange(n) + 0.5) / n
    ys = y1 - (y1 - y0) * (np.arange(n) + 0.5) / n   # row 0 at the top (y1)
    field = np.hypot(xs[None, :], ys[:, None])       # |z| at pixel centers
    a = np.deg2rad(p["brace_angle_deg"])
    u = np.array([np.cos(a), np.sin(a)])
    return {
        "outer": outer, "inner": inner, "field": field,
        "r_in": p["r_in"], "r_out": p["r_out"],
        "brace_p1": p["r_in"] * u, "brace_p2": p["r_out"] * u,
        "params": p,
    }


def build(g):
    p = g["params"]
    s = Scene(xlim=p["xlim"], ylim=p["ylim"], clip="frame")
    # the field itself, beneath everything: the object the circles are level sets OF
    s.add(RasterField(g["field"], extent=p["raster_extent"],
                      vmin=0.0, vmax=float(g["field"].max()), opacity=0.42))
    # the annulus: one region, hole excluded, texture not tint
    s.add(FilledCurve(g["outer"], holes=(g["inner"],), pattern="stipple",
                      role=Role.ACCENT1, opacity=1.0, outline=False))
    # its boundary: the two level sets
    s.add(Curve(g["outer"], closed=True, role=Role.CONTENT),
          Curve(g["inner"], closed=True, role=Role.CONTENT))
    # radial width, measured on the figure
    s.add(Brace(tuple(g["brace_p1"]), tuple(g["brace_p2"]),
                side=1.0, depth=0.14, label=r"\delta"))
    # the excluded open disc, named from outside the band; anchor low
    # enough that the leader clears the rotated band label
    s.add(Callout(r"\mathrm{excluded}", anchor=(2.85, 0.5), target=(0.3, 0.12)))
    # band label reading along the 30-degree radius
    ang = np.deg2rad(p["band_label_angle_deg"])
    mid_r = (g["r_in"] + g["r_out"]) / 2
    s.add(MathLabel(r"1 \le |z| \le 2",
                    (mid_r * np.cos(ang), mid_r * np.sin(ang)),
                    ha="center", va="center",
                    angle_deg=p["band_label_angle_deg"], role=Role.CONTENT))
    return s


def _shoelace(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def assertions(g):
    # annulus area from the drawn polygons matches pi (R^2 - r^2)
    area = _shoelace(g["outer"]) - _shoelace(g["inner"])
    target = np.pi * (g["r_out"] ** 2 - g["r_in"] ** 2)
    assert abs(area - target) / target < 1e-3, \
        f"annulus polygon area {area:.6f} != {target:.6f}"
    # the brace spans exactly the radial width delta = R - r
    delta = float(np.hypot(*(g["brace_p2"] - g["brace_p1"])))
    assert abs(delta - (g["r_out"] - g["r_in"])) < 1e-12, "brace span != delta"
    # the field is |z| on pixel centers (spot check at the grid center)
    n = g["field"].shape[0]
    assert abs(g["field"][n // 2, n // 2]
               - np.hypot(0.05, 0.05)) < 0.2, "field is not |z|"
    # raster row 0 renders at the TOP: emitted image y == transform of y1
    from figlib.render import to_svg_tree
    root, t = to_svg_tree(build(g), THEME, width_px=680)
    img = next(e for e in root.iter()
               if e.tag == "image" and e.get("preserveAspectRatio") == "none")
    x0, x1, y0, y1 = g["params"]["raster_extent"]
    assert abs(float(img.get("y")) - t.to_canvas((x0, y1))[1]) < 0.01, \
        "raster top edge is not at y1"
