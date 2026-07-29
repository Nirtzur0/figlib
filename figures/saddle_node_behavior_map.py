"""Saddle-node behavior map: the bifurcation diagram with the systems inside it.

Victor's behavior-map device (corpus-study runner-up): the diagram is not a
plot OF the system, it IS an atlas of systems — every vertical slice of the
(r, x*) plane is a one-dimensional dynamical system, and the branches are
where those systems keep their fixed points. Word-scale phase-line
thumbnails, embedded at three chosen r and pinned to the axis by leaders,
make the atlas reading literal.

Design notes (steps 0-9):

0. EARN IT. The prose — "each point of the r axis carries a phase line;
   the branch curves record its fixed points" — states the atlas reading
   but leaves the reader to IMAGINE a phase line per r. strogatz_saddle_node
   already draws three phase panels beside the diagram; what that layout
   cannot make perceptual is that the systems live INSIDE the parameter
   plane. Embedding the phase lines at their own r turns "slice = system"
   from a caption into a spatial fact: the thumbnail sits over the very r
   it depicts, its dots wear the branch hues, and the leader is the slice.
1. CLAIM: below.
2. REPRESENTATION. Rung: behavior map — the family is the subject, the
   three thumbnails are its bound instances. Primitive: the (r, x*) plane
   with hue-keyed branches, plus word-scale phase lines (dots + flow
   arrows, no internal annotation: at thumbnail scale the convention IS
   the content). Hue binds instance to family: the stable branch and every
   stable dot share one categorical hue; likewise unstable; the saddle
   node is a third, accent-role object.
3. SIZE. WIDE: the thumbnails must stay legible at reading size (type and
   ink never scale), so each needs ~250 px of width INSIDE the diagram —
   only WIDE has three such vacancies plus a diagram that still dominates.
4. TRAVERSAL. Enter at the branches (the largest ink, center-left), catch
   the matching hue in the r<0 thumbnail sitting inside the parabola, read
   its leader down to the axis: slice = system. Then right along the axis:
   the merge thumbnail pinned to r=0, the empty-line thumbnail pinned into
   the empty half-plane — the vacancy at r>0 is itself content, and the
   thumbnail says what lives there: flow, no dots.
5. MECHANISM. Readable off the figure: branch equations at the branches;
   stability as dash (branch) and fill (dot) redundantly with hue;
   r-values on each thumbnail frame; the leader to the exact r; flow
   direction in every thumbnail checkable against where its dots are.
6. READER EFFORT. Delegated: matching any dot to its branch (hue does it),
   checking any arrow against the dots it runs between. Kept: the atlas
   inference itself — that BETWEEN the sampled r nothing qualitative
   changes except at 0 — which the continuous branches then confirm.
7. CHANNELS. Correspondence hue = object identity across host and insets
   (stable / unstable), the figure's one load-bearing hue use; ACCENT2 =
   the saddle-node (THE distinguished object); ACCENT1 = flow; dash and
   fill carry stability redundantly so monochrome keeps the claim;
   FRAME/ANNOTATION ink for thumbnail frames and leaders — quiet, the
   diagram stays the subject.
8. HONESTY. The host plane is anisotropic (height_px), so thumbnail dot
   radii are pre-compensated through the actual Transform to render
   circular; thumbnail frames are proportioned in px, not math units, so
   no thumbnail silently asserts a region of the plane. Insets are NOT
   clipped by embed: containment inside each frame is asserted, not
   assumed. The r > 0 half is empty of branches on purpose — the r-axis
   honesty check runs with the leader-pinned r included as data.
9. GATES. assertions() re-derives roots and stabilities per thumbnail,
   checks every drawn arrow against sign f, the branches against
   r + x^2 = 0, the saddle-node location against the branch merge, inset
   containment, frame/branch and frame/frame clearance, and leader
   targets on the axis. The hue-binding gate holds one-hue-one-object
   across host and insets via shared keys.
"""

import numpy as np

from figlib import plots
from figlib.correspond import keyed
from figlib.format import WIDE
from figlib.gates import Checks
from figlib.inset import embed
from figlib.scene import Curve, MathLabel, Scene
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "The bifurcation diagram of xdot = r + x^2 is a map of behaviors: every "
    "vertical slice IS a one-dimensional phase line, shown literally by three "
    "word-scale phase-line thumbnails pinned to their r on the axis — two "
    "fixed points for r < 0 whose dots wear the hues of the stable (solid) "
    "and unstable (dashed) branches they live on, one half-stable point at "
    "the saddle-node r = 0 where the branches merge, and bare rightward flow "
    "for r > 0 where the diagram is empty because nothing exists there."
)

THEME = RISO
FORMAT = WIDE

PARAMS = {
    # host plane
    "r_range": [-1.5, 1.0],
    "xstar_range": [-1.35, 1.35],
    "host_xlim": [-1.72, 1.30],
    "host_ylim": [-1.56, 1.68],
    "host_height_px": 470.0,
    # thumbnails: state line and frame proportions
    "inset_r": [-1.0, 0.0, 1.0],
    "inset_center": [[-1.02, 0.46], [0.60, 1.18], [0.72, -0.78]],
    "state_lim": [-2.0, 2.0],
    "inset_pad": 0.30,            # small-scene x margin beyond the state line
    "inset_w": 0.85,              # host x-units; ~250 px at WIDE
    "inset_h_px": 64.0,           # rendered thumbnail height
    "arrow_frac": 0.55,
    "samples": 400,
}

DOT_PX = 5.5                      # fixed-point dot radius, canvas px


def _f(r: float):
    return lambda x: r + np.asarray(x, dtype=float) ** 2


def _fixed_points(r: float) -> tuple[np.ndarray, list[bool | None]]:
    """Roots of r + x^2 with stability from sign f'(x*) = 2x* — derived,
    never tabulated, so assertions() can re-derive both independently."""
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
    # px bridge through the ACTUAL host transform: the host plane is
    # anisotropic, so every px-true quantity (dot radii in host and small
    # coords, the small scene's y-extent) is derived here, once
    host = Scene(xlim=tuple(p["host_xlim"]), ylim=tuple(p["host_ylim"]),
                 height_px=p["host_height_px"])
    upx, upy = plots.px_units(host, FORMAT.display_width_px)

    x0, x1 = p["state_lim"]
    hx = x1 + p["inset_pad"]                      # small-scene half-width
    s_embed = p["inset_w"] / (2.0 * hx)           # embed's uniform scale
    # small-scene half-height st. the rendered thumbnail is inset_h_px tall:
    # rendered_h = (2*hy) * s_embed / upy
    hy = 0.5 * p["inset_h_px"] * upy / s_embed
    # dot radii in SMALL-scene units st. after s_embed and the host
    # transform every dot renders as a DOT_PX-radius circle
    rx_small = DOT_PX * upx / s_embed
    ry_small = DOT_PX * upy / s_embed

    insets = []
    for r, c in zip(p["inset_r"], p["inset_center"]):
        f = _f(r)
        xs, stab = _fixed_points(r)
        insets.append({
            "r": float(r), "center": (float(c[0]), float(c[1])),
            "fixed": xs, "stable": stab,
            "arrows": plots.flow_intervals(f, xs, (x0, x1),
                                           frac=p["arrow_frac"]),
        })

    rb = np.linspace(p["r_range"][0], 0.0, p["samples"])
    return {
        "insets": insets,
        "rb": rb,
        "x_stable": -np.sqrt(-rb),
        "x_unstable": np.sqrt(-rb),
        "saddle": (0.0, 0.0),      # asserted against the branch merge below
        "r_scale": plots.linear(*p["r_range"]),
        "x_scale": plots.linear(*p["xstar_range"]),
        "upx": upx, "upy": upy,
        "small_hx": hx, "small_hy": hy,
        "rx_small": rx_small, "ry_small": ry_small,
        "state_lim": (float(x0), float(x1)),
        "p": p,
    }


# --- build ------------------------------------------------------------------

KEY_STABLE = "stable-branch"
KEY_UNSTABLE = "unstable-branch"
KEY_SADDLE = "saddle-node"
TICK_PX = 6.0

# Tick values whose LABELS the drawn axes skip (r: the origin, and 1.0
# where the pinned leader marks that r more strongly than a label would;
# x*: the origin). ONE source for both the axis calls and tick_honesty —
# the gate must certify the tick set the reader actually gets, not the
# unskipped one.
R_TICK_SKIP = (0.0, 1.0)
X_TICK_SKIP = (0.0,)


def _drawn_ticks(tk, skip):
    """`tk` minus the skipped values — the labeled ticks the page shows."""
    keep = [i for i, v in enumerate(tk.values)
            if not any(abs(float(v) - s) <= 1e-9 * max(1.0, abs(s))
                       for s in skip)]
    return plots.Ticks(values=tk.values[keep], positions=tk.positions[keep],
                       labels=tuple(tk.labels[i] for i in keep),
                       minor=tk.minor)


def _thumbnail(ins: dict, g: dict) -> Scene:
    """One word-scale phase line as its own Scene, ready for embed().

    Dots wear the branch hues (stable -> categorical 0, unstable -> 1,
    the merge -> ACCENT2) under the same keys as the host branches — the
    binding the hue gate then holds across the whole figure. phase_line
    has no per-dot color channel, so its dots are dropped and re-drawn
    with plots.markers at the same radii.
    """
    sm = Scene(xlim=(-g["small_hx"], g["small_hx"]),
               ylim=(-g["small_hy"], g["small_hy"]))
    fixed = [plots.FixedPoint(float(x), st)
             for x, st in zip(ins["fixed"], ins["stable"])]
    saddle = any(st is None for st in ins["stable"])
    dot_role = Role.ACCENT2 if saddle else Role.CONTENT
    rxy = (g["rx_small"], g["ry_small"])
    line = plots.phase_line(fixed, ins["arrows"], xlim=g["state_lim"], y=0.0,
                            dot=rxy, spine_role=Role.ANNOTATION,
                            dot_role=dot_role, arrow_role=Role.ACCENT1,
                            arrow_width=1.1)
    if saddle:                     # the merged point keeps phase_line's glyph
        sm.add(*(i for i in line if i.role is not Role.ACCENT2))
        sm.add(*keyed(KEY_SADDLE, *(i for i in line if i.role is Role.ACCENT2)))
    else:
        sm.add(*(i for i in line if i.role is not Role.CONTENT))
        for x, st in zip(ins["fixed"], ins["stable"]):
            key, hue = ((KEY_STABLE, THEME.categorical(0)) if st
                        else (KEY_UNSTABLE, THEME.categorical(1)))
            sm.add(*keyed(key, *plots.markers([float(x)], [0.0], size=rxy,
                                              filled=bool(st), color=hue)))
    return sm


def _diagram(s: Scene, g: dict) -> None:
    p = g["p"]
    r_sc, x_sc = g["r_scale"], g["x_scale"]
    rx, ry = DOT_PX * g["upx"], DOT_PX * g["upy"]

    # the 1.0 tick is skipped: the r = +1 leader lands exactly there, and
    # its arrowhead through the tick label read as a collision — the pinned
    # leader marks that r more strongly than the tick it replaces
    s.add(*plots.axis(r_sc, orient="x", at=0.0, ticks=r_sc.ticks(step=0.5),
                      tick_len=TICK_PX * g["upy"], skip=R_TICK_SKIP, label="r",
                      extend=(0.05, 0.10), role=Role.ANNOTATION))
    s.add(*plots.axis(x_sc, orient="y", at=0.0, ticks=x_sc.ticks(step=1.0),
                      tick_len=TICK_PX * g["upx"], skip=X_TICK_SKIP,
                      extend=(0.05, 0.09), role=Role.ANNOTATION))
    s.add(MathLabel(r"x^{*}", (0.0, x_sc.hi + 0.08), role=Role.ANNOTATION,
                    ha="left", va="center", offset_px=(16.0, 0.0)))

    # the branches: hue = identity, dash = stability (redundant channels)
    # actor weight: the branches are the family the whole figure is about,
    # so they dominate the spines/axes weight level
    s.add(*keyed(KEY_STABLE,
                 *plots.series(g["rb"], g["x_stable"], role=Role.CONTENT,
                               width_scale=1.6, color=THEME.categorical(0))))
    s.add(*keyed(KEY_UNSTABLE,
                 *plots.series(g["rb"], g["x_unstable"], role=Role.CONTENT,
                               width_scale=1.6, dash="dashed",
                               color=THEME.categorical(1))))
    s.add(MathLabel(r"x^{*}=-\sqrt{-r}\ \ \text{stable}", (-1.50, -1.34),
                    role=Role.CONTENT, color=THEME.categorical(0),
                    ha="left", va="center", key=KEY_STABLE))
    s.add(MathLabel(r"x^{*}=+\sqrt{-r}\ \ \text{unstable}", (-1.50, 1.34),
                    role=Role.CONTENT, color=THEME.categorical(1),
                    ha="left", va="center", key=KEY_UNSTABLE))

    # the r < 0 thumbnail's dots, replotted ON their branches at that r —
    # the instance sits on the family, in the instance's own glyphs
    for ins in g["insets"]:
        rv = ins["r"]
        for xv, st in zip(ins["fixed"], ins["stable"]):
            if st is None:         # the saddle-node: half-stable glyph
                s.add(*keyed(KEY_SADDLE, *plots.phase_line(
                    [plots.FixedPoint(float(rv), None)], np.empty((0, 2)),
                    xlim=(rv, rv), y=float(xv), dot=(rx, ry), spine=False,
                    dot_role=Role.ACCENT2)))
            else:
                key, hue = ((KEY_STABLE, THEME.categorical(0)) if st
                            else (KEY_UNSTABLE, THEME.categorical(1)))
                s.add(*keyed(key, *plots.markers([rv], [float(xv)],
                                                 size=(rx, ry),
                                                 filled=bool(st), color=hue)))
    s.add(MathLabel(r"\text{saddle-node}", (0.12, 0.30), role=Role.ACCENT2,
                    ha="left", va="center", key=KEY_SADDLE))
    # the system itself, stated once — the readback showed a cold reader
    # otherwise has to reverse-engineer the ODE from the branch equations
    s.add(MathLabel(r"\dot{x} = r + x^{2}", (-0.62, 1.52), role=Role.CONTENT,
                    ha="center", va="center"))


def build(g):
    p = g["p"]
    s = Scene(xlim=tuple(p["host_xlim"]), ylim=tuple(p["host_ylim"]),
              height_px=p["host_height_px"])
    _diagram(s, g)
    half_h = 0.5 * p["inset_h_px"] * g["upy"]
    for ins in g["insets"]:
        cx, cy = ins["center"]
        s.add(*embed(_thumbnail(ins, g), at=(cx, cy), width=p["inset_w"],
                     leader_to=(ins["r"], 0.0)))
        s.add(MathLabel(f"r = {ins['r']:+.0f}".replace("+0", "0"),
                        (cx, cy + half_h), role=Role.ANNOTATION, ha="center",
                        va="bottom", offset_px=(0.0, -5.0)))
    return s


# --- the numerical gate -----------------------------------------------------


def assertions(g):
    c = Checks()
    p = g["p"]

    for ins in g["insets"]:
        r = ins["r"]
        f = _f(r)
        xs, stab = ins["fixed"], ins["stable"]
        tag = f"r={r:+.0f}"

        n_expected = 2 if r < 0 else (1 if r == 0 else 0)
        c.check(len(xs) == n_expected,
                f"{tag}: {len(xs)} fixed points, expected {n_expected}")
        for x in xs:
            c.check(abs(float(f(x))) < 1e-12,
                    f"{tag}: x*={x:.6f} is not a root, f = {float(f(x)):.2e}")
        for x, st in zip(xs, stab):
            d = 2.0 * float(x)
            if st is True:
                c.check(d < -1e-9,
                        f"{tag}: x*={x:.3f} marked stable but f'={d:.2e}")
            elif st is False:
                c.check(d > 1e-9,
                        f"{tag}: x*={x:.3f} marked unstable but f'={d:.2e}")
            else:
                c.check(abs(d) < 1e-12,
                        f"{tag}: x*={x:.3f} marked half-stable but f'={d:.2e}")

        # every drawn arrow: direction from sign f, no straddling, on the line
        c.check(len(ins["arrows"]) == len(xs) + 1,
                f"{tag}: {len(ins['arrows'])} arrows for {len(xs) + 1} intervals")
        for a, b in ins["arrows"]:
            mid = 0.5 * (a + b)
            c.check(float(np.sign(b - a)) == float(np.sign(f(mid))),
                    f"{tag}: arrow at x={mid:.3f} contradicts f={float(f(mid)):+.3f}")
            for x in xs:
                c.check(not (min(a, b) < x < max(a, b)),
                        f"{tag}: arrow ({a:.2f},{b:.2f}) straddles x*={x:.3f}")
            c.check(g["state_lim"][0] - 1e-9 <= min(a, b)
                    and max(a, b) <= g["state_lim"][1] + 1e-9,
                    f"{tag}: arrow ({a:.2f},{b:.2f}) leaves the state line")

        # embed does not clip: everything drawn in the small scene must fit
        # inside its own frame (dots including their radii)
        for x in xs:
            c.check(abs(float(x)) + g["rx_small"] <= g["small_hx"] + 1e-9,
                    f"{tag}: dot at x*={x:.3f} leaks past the inset frame")
        c.check(g["ry_small"] <= g["small_hy"] + 1e-9,
                f"{tag}: dot radius {g['ry_small']:.3f} exceeds inset half-"
                f"height {g['small_hy']:.3f}")
        c.check(max(abs(v) for v in g["state_lim"]) <= g["small_hx"] + 1e-9,
                f"{tag}: state line leaks past the inset frame")

        # the leader lands on the r axis at this thumbnail's own r
        c.check(g["r_scale"].lo <= r <= g["r_scale"].hi,
                f"{tag}: leader target r={r} is off the axis")

    # branches satisfy the defining equation, keep their claimed signs and
    # stabilities, and merge at the computed saddle-node
    for name, xb, sign in (("stable", g["x_stable"], -1.0),
                           ("unstable", g["x_unstable"], +1.0)):
        res = float(np.max(np.abs(g["rb"] + xb ** 2)))
        c.check(res < 1e-12, f"{name} branch violates r + x^2 = 0 by {res:.2e}")
        interior = 2.0 * xb[:-1]
        c.check(bool(np.all(sign * interior > 0.0)),
                f"{name} branch: f'(x*) contradicts the dash it wears")
    sr, sx = g["saddle"]
    c.check(abs(g["rb"][-1] - sr) < 1e-12
            and abs(g["x_stable"][-1] - sx) < 1e-12
            and abs(g["x_unstable"][-1] - sx) < 1e-12,
            f"branches do not merge at the drawn saddle-node ({sr}, {sx})")

    # thumbnail frames: inside the plane, clear of both branches and of
    # each other — a frame overlapping a branch would occlude the family
    # the thumbnails exist to instantiate
    w = p["inset_w"]
    h = p["inset_h_px"] * g["upy"]
    rects = []
    for ins in g["insets"]:
        cx, cy = ins["center"]
        rects.append((ins["r"], cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    branch_pts = np.vstack([
        np.column_stack([g["rb"], g["x_stable"]]),
        np.column_stack([g["rb"], g["x_unstable"]])])
    for r, ax, ay, bx, by in rects:
        tag = f"inset r={r:+.0f}"
        c.check(p["host_xlim"][0] <= ax and bx <= p["host_xlim"][1]
                and p["host_ylim"][0] <= ay and by <= p["host_ylim"][1],
                f"{tag}: frame [{ax:.2f},{bx:.2f}]x[{ay:.2f},{by:.2f}] "
                f"leaves the host plane")
        inside = ((branch_pts[:, 0] >= ax) & (branch_pts[:, 0] <= bx)
                  & (branch_pts[:, 1] >= ay) & (branch_pts[:, 1] <= by))
        c.check(not bool(inside.any()),
                f"{tag}: frame overlaps a branch curve at "
                f"r={float(branch_pts[inside][0][0]) if inside.any() else 0:.2f}")
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            _, ax, ay, bx, by = rects[i]
            _, cx2, cy2, dx2, dy2 = rects[j]
            c.check(bx <= cx2 or dx2 <= ax or by <= cy2 or dy2 <= ay,
                    f"inset frames {i} and {j} overlap")

    # axis honesty: the r axis carries the branches AND the pinned r of
    # every thumbnail (the leaders make those r data on this axis)
    r_data = np.concatenate([g["rb"], np.asarray(p["inset_r"], dtype=float)])
    plots.tick_honesty(c, g["r_scale"], r_data, name="r axis",
                       ticks=_drawn_ticks(g["r_scale"].ticks(step=0.5),
                                          R_TICK_SKIP))
    plots.tick_honesty(c, g["x_scale"],
                       np.concatenate([g["x_stable"], g["x_unstable"]]),
                       name="x* axis",
                       ticks=_drawn_ticks(g["x_scale"].ticks(step=1.0),
                                          X_TICK_SKIP))
    c.done()
