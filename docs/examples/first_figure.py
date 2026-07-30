"""The smallest complete figure program — the README's quickstart.

Kept outside `figures/` on purpose: it is documentation, not corpus, so it
does not carry a readback record and `make regress` does not gate it. It is
still a real program — `tests/test_first_figure.py` runs it through the same
runner every other figure uses, so the README cannot drift into fiction.
"""

import numpy as np

from figlib.format import COLUMN
from figlib.scene import Brace, Curve, MathLabel, Point, Scene
from figlib.style import Role
from figlib.theme import RISO

CLAIM = ("The tangent to y = x^2 at x = a meets the axis at a/2 — the "
         "subtangent is half the abscissa, for every a.")

EXPOSITION = """
Before limits were available, the tangent to a parabola was found by this
construction: drop the ordinate at x = a, halve it, and join. The figure is
here to make the halving visible as a length rather than as an algebraic
step, so that the reader who has just met the derivative 2a can see what the
number 2 buys geometrically — a subtangent that is always a/2, independent of
where on the curve you stand. That independence is the whole content, and it
is exactly the part a formula states without showing.
"""

FORMAT = COLUMN
THEME = RISO

PARAMS = {"a": 1.5, "xlim": (-0.9, 2.2), "ylim": (-1.4, 3.85),
          "curve_x": (-0.85, 1.94), "reach": (0.95, 0.42), "height_px": 400,
          "brace_y": -0.6}


def compute(p):
    a, lo_r, hi_r = p["a"], *p["reach"]
    x = np.linspace(*p["curve_x"], 400)
    # the drawn tangent stops just past the touch point: past that it runs
    # alongside the parabola and the eye reads two curves where there is one
    # line and one curve
    t = np.array([a - lo_r, a + hi_r])
    return {
        "parabola": np.column_stack([x, x * x]),
        "tangent": np.column_stack([t, 2.0 * a * (t - a) + a * a]),
        "foot": (a, a * a),
        "p": p,
    }


def build(g):
    a, p = g["p"]["a"], g["p"]
    s = Scene(xlim=p["xlim"], ylim=p["ylim"], height_px=p["height_px"])
    s.add(Curve(np.array([[p["xlim"][0], 0.0], [p["xlim"][1], 0.0]]),
                role=Role.FRAME))
    s.add(Curve(g["parabola"], role=Role.CONTENT))
    s.add(Curve(g["tangent"], role=Role.ACCENT1))
    s.add(Curve(np.array([[a, 0.0], list(g["foot"])]), role=Role.CONSTRUCTION))
    s.add(Point(g["foot"], filled=True, role=Role.ACCENT1))
    s.add(Point((a / 2.0, 0.0), filled=False, role=Role.ACCENT1))

    # the touch point names itself into the open upper-left, away from both
    # the tangent and the parabola, which run together to its lower left
    s.add(MathLabel(r"(a,\,a^2)", g["foot"], ha="right", va="bottom",
                    offset_px=(-10, -7)))
    # both abscissae read BELOW the axis, clear of the tangent crossing it --
    # and a/2 to the RIGHT of its crossing, since below-left is where the
    # tangent actually runs
    s.add(MathLabel(r"\tfrac{a}{2}", (a / 2.0, 0.0), ha="left", va="top",
                    offset_px=(7, 6), role=Role.ACCENT1))
    s.add(MathLabel(r"a", (a, 0.0), ha="center", va="top", offset_px=(0, 7)))
    # the claim as a LENGTH, not a coordinate: what the figure is for
    s.add(Brace((a / 2.0, p["brace_y"]), (a, p["brace_y"]), side=-1.0,
                label=r"\text{subtangent} = \tfrac{a}{2}",
                role=Role.ACCENT1))
    s.add(MathLabel(r"y = x^2", (g["parabola"][0][0], g["parabola"][0][1]),
                    ha="left", va="center", offset_px=(11, -7)))
    return s


def assertions(g):
    a = g["p"]["a"]
    (x0, y0), (x1, y1) = g["tangent"]
    # where the DRAWN segment crosses y = 0 — not where theory says it does
    root = x0 - y0 * (x1 - x0) / (y1 - y0)
    assert abs(root - a / 2.0) < 1e-12, f"subtangent hits {root}, not {a / 2}"
    assert abs(g["foot"][1] - a * a) < 1e-12, "foot is off the parabola"
