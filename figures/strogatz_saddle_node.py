"""Saddle-node bifurcation of xdot = r + x^2: the collision of two fixed points.

Design notes (exposition.md steps 0-9; the figure exists because step 0
failed):

0. EARN IT. The prose version — "for r < 0 there are two fixed points at
   ±sqrt(-r), the left stable and the right unstable; they approach as r
   rises, merge at r = 0, and vanish for r > 0" — states the result but
   makes the reader run the algebra to believe the stabilities and to see
   why the merge is forced. Drawn, the inference is perceptual: the
   parabola rides UP as r rises, and where a parabola crosses a
   horizontal line is a thing eyes compute for free. Two roots, one
   root, none is then a fact about a moving curve, not about a
   discriminant.
1. CLAIM: below.
2. REPRESENTATION. The primitive is the graph of f over the state line,
   not the phase line alone: with only dots and arrows the reader must
   trust that the arrows follow sign f. Co-locating the parabola with the
   line it crosses makes "flow direction = sign of f" checkable by eye —
   the hypothesis becomes a visible property. Rungs: (a)-(c) are three
   bound instances, (d) is the family; the correspondence between rungs
   is drawn (the [a] and [c] slices are ruled on the diagram, with the
   instance's own dots sitting on them) rather than left implicit.
3. SIZE. WIDE: three small multiples across, the family map beneath.
4. TRAVERSAL. Attention enters at (a) — two dots, arrows pointing at one
   and away from the other — sweeps right through the merge to the empty
   line in (c), then drops to (d), which is the same three pictures
   compiled into one. The vertical rules are the hinge.
5. MECHANISM. Readable off the figure: the vertex height IS r (labelled
   at the vertex, not in a caption); root positions as ±sqrt(-r), so the
   label states the theorem and the panel states the instance; stability
   as filled/hollow; the flow as arrows whose direction the parabola
   justifies.
6. READER EFFORT. Delegated: check any arrow against the parabola's sign
   above it — the answer is already on the page. Kept: the leap that the
   three panels are one motion, which (d) then confirms.
7. CHANNELS. Stability rides DASH and FILL, not hue, so the figure is
   faithful in monochrome; one accent (the flow) and one distinguished
   object (the saddle-node point). Position carries the claim.
8. HONESTY. The parabolas are truncated at the frame, identically in all
   three panels (same xlim/ylim — a small-multiple whose panels are
   scaled differently is a lie). The r axis in (d) runs well past 0 into
   the region where nothing exists: that empty half is content, so
   tick_honesty runs there with a stated 50% floor rather than the
   default.
9. GATES. assertions() re-derives the roots, their stabilities from
   f'(x*), and every drawn arrow's direction from sign f at its own
   midpoint, then runs the tick-honesty checks on all three axes.

Judgment call: no trajectory-fan panel. x(t) for several x0 would argue a
DIFFERENT claim (approach to the stable point, and blow-up in finite time
for r > 0) — true, interesting, and off-message here; "one claim per
figure", and deletion outranks signalling.
"""

import numpy as np

from figlib import plots
from figlib.figure import Figure, Panel, layout_figure
from figlib.format import WIDE
from figlib.gates import Checks
from figlib.scene import Curve, MathLabel, Scene
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "In xdot = r + x^2 the two fixed points x* = -sqrt(-r) (stable) and "
    "x* = +sqrt(-r) (unstable) are the two crossings of a parabola that rides "
    "upward with r: as r -> 0 from below they approach, at r = 0 they merge "
    "into a single half-stable point, and for r > 0 the parabola has cleared "
    "the axis, so no fixed point exists and the flow is rightward everywhere."
)

THEME = RISO
FORMAT = WIDE

PARAMS = {
    "panel_r": [-1.0, 0.0, 1.0],
    "state_lim": [-2.1, 2.1],          # the drawn phase line
    "phase_xlim": [-2.25, 2.35],       # panel frame, a touch wider
    "phase_ylim": [-1.72, 2.05],
    "arrow_frac": 0.55,
    "r_range": [-1.5, 1.0],            # bifurcation diagram
    "xstar_range": [-1.35, 1.35],
    "d_xlim": [-1.62, 1.20],
    "d_ylim": [-1.45, 1.62],
    "d_height_px": 450.0,
    "samples": 400,
}

# Panel (d) is deliberately wider than tall; the branches still carry the
# claim on position, and both axes are ticked, so the aspect asserts nothing.


def _f(r: float):
    """The vector field, as the one callable everything else derives from."""
    return lambda x: r + np.asarray(x, dtype=float) ** 2


def _fixed_points(r: float) -> tuple[np.ndarray, list[bool | None]]:
    """Roots of r + x^2 and their stability from the sign of f'(x*) = 2x*.

    Nothing here is tabulated: the count comes from the discriminant, the
    stability from the derivative, so assertions() can re-derive both.
    """
    if r > 0.0:
        xs = np.empty(0)
    elif r == 0.0:
        xs = np.zeros(1)
    else:
        s = float(np.sqrt(-r))
        xs = np.array([-s, s])
    stab: list[bool | None] = []
    for x in xs:
        d = 2.0 * float(x)
        stab.append(None if abs(d) < 1e-12 else bool(d < 0.0))
    return xs, stab


def compute(p):
    x0, x1 = p["state_lim"]
    ytop = p["phase_ylim"][1]
    panels = []
    for r in p["panel_r"]:
        f = _f(r)
        xs, stab = _fixed_points(r)
        arrows = plots.flow_intervals(f, xs, (x0, x1), frac=p["arrow_frac"])
        # the parabola is drawn only where it is inside the frame; the
        # truncation is computed, not clipped away by the renderer
        half = min(abs(x0), abs(x1), float(np.sqrt(max(ytop - r, 0.0))))
        px = np.linspace(-half, half, p["samples"])
        panels.append({"r": float(r), "fixed": xs, "stable": stab,
                       "arrows": arrows, "par_x": px, "par_y": f(px)})

    rb = np.linspace(p["r_range"][0], 0.0, p["samples"])
    return {
        "panels": panels,
        "rb": rb,
        "x_stable": -np.sqrt(-rb),
        "x_unstable": np.sqrt(-rb),
        "r_scale": plots.linear(*p["r_range"]),
        "x_scale": plots.linear(*p["xstar_range"]),
        "px_scale": plots.linear(*p["phase_xlim"]),
        "py_scale": plots.linear(*p["phase_ylim"]),
        "state_lim": (float(x0), float(x1)),
        "p": p,
    }


# --- build ------------------------------------------------------------------

DOT_PX = 6.5          # fixed-point dot radius, canvas px
TICK_PX = 6.0


def _phase_panel(s: Scene, pan: dict, g: dict, upx: float, upy: float,
                 first: bool) -> None:
    r = pan["r"]
    x0, x1 = g["state_lim"]
    ylo, yhi = g["p"]["phase_ylim"]

    # the xdot axis: a bare arrow, no ticks — these panels are qualitative,
    # the quantities they carry are the two labelled roots and the vertex
    empty = plots.Ticks(np.empty(0), np.empty(0), ())
    s.add(*plots.axis(plots.linear(ylo + 0.05, yhi - 0.12), orient="y", at=0.0,
                      ticks=empty, label=r"\dot{x}" if first else None,
                      role=Role.ANNOTATION))

    # the graph of f, and the phase line it crosses, at one location
    s.add(Curve(np.column_stack([pan["par_x"], pan["par_y"]]), role=Role.CONTENT))
    fixed = [plots.FixedPoint(float(x), st)
             for x, st in zip(pan["fixed"], pan["stable"])]
    s.add(*plots.phase_line(fixed, pan["arrows"], xlim=(x0, x1), y=0.0,
                            dot=(DOT_PX * upx, DOT_PX * upy), spine_arrow=True,
                            spine_role=Role.ANNOTATION, dot_role=Role.CONTENT,
                            arrow_role=Role.ACCENT1, arrow_width=1.1))
    if first:
        s.add(MathLabel("x", (x1, 0.0), role=Role.ANNOTATION, ha="left",
                        va="center", offset_px=(9.0, 0.0)))

    # r labelled where r IS something: the height of the vertex. Always
    # directly under it, haloed over the xdot axis it straddles — the
    # parameter reads as a measured height, not as a panel title.
    s.add(MathLabel(f"r = {r:.0f}", (0.0, r), role=Role.ANNOTATION, ha="center",
                    va="top", offset_px=(0.0, 10.0), halo=True))

    # roots named by the theorem, not by their values; panel (b) inherits
    # the naming from (a) — the merged root needs no second label to say
    # that one dot is one dot
    if r < 0:
        s.add(MathLabel(r"-\sqrt{-r}", (float(pan["fixed"][0]), 0.0),
                        role=Role.CONTENT, ha="right", va="top",
                        offset_px=(-4.0, 7.0)))
        s.add(MathLabel(r"+\sqrt{-r}", (float(pan["fixed"][1]), 0.0),
                        role=Role.CONTENT, ha="left", va="top",
                        offset_px=(4.0, 7.0)))


def _bifurcation_panel(s: Scene, g: dict, upx: float, upy: float) -> None:
    r_sc, x_sc = g["r_scale"], g["x_scale"]
    rx, ry = DOT_PX * upx, DOT_PX * upy
    guide_top, guide_bot = x_sc.hi * 0.96, x_sc.lo * 0.96

    for rv in (-1.0, 1.0):
        s.add(Curve(np.array([[rv, guide_bot], [rv, guide_top]]), role=Role.FRAME))
    s.add(*plots.axis(r_sc, orient="x", at=0.0, ticks=r_sc.ticks(step=0.5),
                      tick_len=TICK_PX * upy, skip=(0.0,), label="r",
                      extend=(0.05, 0.12), role=Role.ANNOTATION))
    s.add(*plots.axis(x_sc, orient="y", at=0.0, ticks=x_sc.ticks(step=1.0),
                      tick_len=TICK_PX * upx, skip=(0.0,),
                      extend=(0.06, 0.10), role=Role.ANNOTATION))
    # the x* name beside the arrow, not above it: vertical room here is the
    # scarce quantity, and the tags need the space over the rules
    s.add(MathLabel(r"x^{*}", (0.0, x_sc.hi + 0.10), role=Role.ANNOTATION,
                    ha="left", va="center", offset_px=(18.0, 0.0)))

    s.add(*plots.series(g["rb"], g["x_stable"], role=Role.CONTENT,
                        width_scale=1.15))
    s.add(*plots.series(g["rb"], g["x_unstable"], role=Role.CONTENT,
                        width_scale=1.15, dash="dashed"))

    # the instances, sitting on their own slices: panel (a)'s two dots
    s.add(*plots.markers([-1.0], [-1.0], size=(rx, ry), filled=True,
                         role=Role.CONTENT))
    s.add(*plots.markers([-1.0], [1.0], size=(rx, ry), filled=False,
                         role=Role.CONTENT))
    # panel (b)'s half-stable point, the same glyph, in the accent
    s.add(*plots.phase_line([plots.FixedPoint(0.0, None)], np.empty((0, 2)),
                            xlim=(0.0, 0.0), y=0.0, dot=(rx, ry), spine=False,
                            dot_role=Role.ACCENT2))

    s.add(MathLabel(r"[a]", (-1.0, guide_top), role=Role.ANNOTATION,
                    ha="center", va="bottom", offset_px=(0.0, -5.0)))
    s.add(MathLabel(r"[b]", (0.0, guide_top), role=Role.ANNOTATION,
                    ha="right", va="bottom", offset_px=(-12.0, -5.0)))
    s.add(MathLabel(r"[c]", (1.0, guide_top), role=Role.ANNOTATION,
                    ha="center", va="bottom", offset_px=(0.0, -5.0)))
    # branch names sit just outside their own branch, near enough that the
    # nearest curve is unambiguous (contiguity, Mayer d ~ 1.1)
    s.add(MathLabel(r"-\sqrt{-r}\ \ \text{stable}", (-0.86, -1.18),
                    role=Role.CONTENT, ha="left", va="center"))
    s.add(MathLabel(r"+\sqrt{-r}\ \ \text{unstable}", (-0.86, 1.18),
                    role=Role.CONTENT, ha="left", va="center"))
    s.add(MathLabel(r"\text{no fixed points}", (0.20, 0.62), role=Role.ANNOTATION,
                    ha="left", va="center"))
    s.add(MathLabel(r"\text{saddle-node}", (0.10, -0.42), role=Role.ACCENT2,
                    ha="left", va="center"))


def build(g):
    p = g["p"]
    scenes = [Scene(xlim=tuple(p["phase_xlim"]), ylim=tuple(p["phase_ylim"]))
              for _ in g["panels"]]
    d_scene = Scene(xlim=tuple(p["d_xlim"]), ylim=tuple(p["d_ylim"]),
                    height_px=p["d_height_px"])
    fig = Figure(
        panels=[Panel(sc, tag=t) for sc, t in zip(scenes, ("[a]", "[b]", "[c]"))]
        + [Panel(d_scene, tag="[d]")],
        grid=(2, 3))

    # marker and tick sizes are px quantities, so they are measured through
    # the very transforms the renderer will use
    lay = layout_figure(fig, FORMAT.display_width_px)
    for sc, pan, slot in zip(scenes, g["panels"], lay.slots):
        _phase_panel(sc, pan, g, 1.0 / slot.transform.scale_x,
                     1.0 / slot.transform.scale_y, first=pan is g["panels"][0])
    t = lay.slots[3].transform
    _bifurcation_panel(d_scene, g, 1.0 / t.scale_x, 1.0 / t.scale_y)
    return fig


# --- the numerical gate -----------------------------------------------------


def assertions(g):
    c = Checks()
    p = g["p"]

    for pan in g["panels"]:
        r = pan["r"]
        f = _f(r)
        xs, stab = pan["fixed"], pan["stable"]
        tag = f"r={r:+.0f}"

        n_expected = 2 if r < 0 else (1 if r == 0 else 0)
        c.check(len(xs) == n_expected,
                f"{tag}: {len(xs)} fixed points, expected {n_expected}")
        for x in xs:
            c.check(abs(float(f(x))) < 1e-12,
                    f"{tag}: x*={x:.6f} is not a root, f = {float(f(x)):.2e}")
        if r < 0:
            s = float(np.sqrt(-r))
            c.check(abs(xs[0] + s) < 1e-12 and abs(xs[1] - s) < 1e-12,
                    f"{tag}: roots {xs} are not -+sqrt(-r) = -+{s:.6f}")
            c.check(stab == [True, False],
                    f"{tag}: expected left stable / right unstable, got {stab}")
        if r == 0:
            c.check(stab == [None], f"{tag}: origin must be half-stable, got {stab}")
        if r > 0:
            grid = np.linspace(-6.0, 6.0, 20001)
            c.check(float(f(grid).min()) > 0.0,
                    f"{tag}: f attains {float(f(grid).min()):.3f} <= 0 somewhere")

        # stability is read from f'(x*), never tabulated
        for x, st in zip(xs, stab):
            d = 2.0 * float(x)
            if st is True:
                c.check(d < -1e-9, f"{tag}: x*={x:.3f} marked stable but f'={d:.2e}")
            elif st is False:
                c.check(d > 1e-9, f"{tag}: x*={x:.3f} marked unstable but f'={d:.2e}")
            else:
                c.check(abs(d) < 1e-12,
                        f"{tag}: x*={x:.3f} marked half-stable but f'={d:.2e}")

        # every DRAWN arrow: direction from sign f at its own midpoint, and
        # no arrow may straddle a fixed point (that would draw flow through it)
        n_int = len(xs) + 1
        c.check(len(pan["arrows"]) == n_int,
                f"{tag}: {len(pan['arrows'])} flow arrows for {n_int} intervals")
        for a, b in pan["arrows"]:
            mid = 0.5 * (a + b)
            want = float(np.sign(f(mid)))
            got = float(np.sign(b - a))
            c.check(got == want,
                    f"{tag}: arrow at x={mid:.3f} points {got:+.0f} but "
                    f"f={float(f(mid)):+.3f}")
            for x in xs:
                c.check(not (min(a, b) < x < max(a, b)),
                        f"{tag}: arrow ({a:.2f},{b:.2f}) straddles x*={x:.3f}")
            c.check(p["state_lim"][0] - 1e-9 <= min(a, b)
                    and max(a, b) <= p["state_lim"][1] + 1e-9,
                    f"{tag}: arrow ({a:.2f},{b:.2f}) leaves the phase line")

        # the drawn parabola is the graph of f, truncated at the frame
        c.check(float(np.max(np.abs(pan["par_y"] - f(pan["par_x"])))) < 1e-12,
                f"{tag}: drawn curve is not the graph of f")
        c.check(float(pan["par_y"].max()) <= p["phase_ylim"][1] + 1e-9,
                f"{tag}: parabola drawn outside the frame")

    # the branches lie on the defining equation, with the drawn stabilities
    for name, xb, sign in (("stable", g["x_stable"], -1.0),
                           ("unstable", g["x_unstable"], +1.0)):
        res = float(np.max(np.abs(g["rb"] + xb ** 2)))
        c.check(res < 1e-12, f"{name} branch violates r + x^2 = 0 by {res:.2e}")
        c.check(float(np.max(sign * xb)) >= 0.0 >= float(np.min(sign * xb)) - 1e-15,
                f"{name} branch has the wrong sign")
        interior = 2.0 * xb[:-1]        # f'(x*), excluding the merge point
        c.check(bool(np.all(sign * interior > 0.0)),
                f"{name} branch: f'(x*) does not keep the sign its dash claims")
    c.check(abs(g["x_stable"][-1]) < 1e-12 and abs(g["x_unstable"][-1]) < 1e-12,
            "the branches do not meet at the origin")

    # tick honesty on every axis that carries data
    px = np.concatenate([np.array(g["state_lim"])]
                        + [pan["par_x"] for pan in g["panels"]])
    py = np.concatenate([np.zeros(1)] + [pan["par_y"] for pan in g["panels"]])
    plots.tick_honesty(c, g["px_scale"], px, name="phase-line x axis",
                       ticks=g["px_scale"].ticks())
    plots.tick_honesty(c, g["py_scale"], py, name="phase-line xdot axis",
                       ticks=g["py_scale"].ticks())
    # 50%, not the default 40%: the r > 0 half of this axis is empty ON
    # PURPOSE — it is where the claim says nothing exists — so the check has
    # to be stated deliberately rather than relaxed away
    plots.tick_honesty(c, g["r_scale"], g["rb"], name="bifurcation r axis",
                       min_span_frac=0.5, ticks=g["r_scale"].ticks(step=0.5))
    plots.tick_honesty(c, g["x_scale"],
                       np.concatenate([g["x_stable"], g["x_unstable"]]),
                       name="bifurcation x* axis", ticks=g["x_scale"].ticks(step=1.0))
    c.done()
