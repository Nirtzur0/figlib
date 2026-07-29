"""Cauchy's theorem as homotopy: the contour integral is a property of the
loop's class, not of its shape.

Design notes (exposition.md steps 0-9):

0. EARN IT. The prose statement — "the integral of an analytic function
   around a closed loop is unchanged as the loop is continuously deformed,
   provided it never sweeps across a singularity" — is exactly Cauchy's
   theorem plus its topological reading, and a reader can recite it without
   ever having watched it happen. What the sentence does not make
   perceptual is the asymmetry: nothing special happens along almost the
   entire deformation, and then one single instant is different in kind,
   not degree. Drawn as a family of visibly different loops carrying the
   identical printed value, followed by one loop dragged across the pole
   instead of around it, the invariance and its one failure mode become
   things the eye checks rather than premises taken on faith.
1. CLAIM: below.
2. REPRESENTATION. Three bound instances of one continuously-deforming
   loop (rung: instance-in-family) establish "shape changes, value does
   not"; the fourth is the same rung with the family's one rule broken on
   purpose, so the reader sees the exception against the pattern just
   established rather than in isolation.
3. SIZE. WIDE: three small multiples across (the homotopy), one wide panel
   beneath (the crossing) — the strogatz-filmstrip grammar, reused because
   the shape of the argument is the same: instances, then the map of what
   the instances are instances OF.
4. TRAVERSAL. Enter at [a] (biggest, wobbliest loop), sweep right as the
   loop tightens toward the pole in [c], each panel repeating the same
   printed value — then drop to [d], where a ghost of exactly panel [c]'s
   loop is dragged bodily across z0 instead of shrunk around it, and the
   printed value changes.
5. MECHANISM. Readable off the figure: the value itself, typeset
   identically under [a]-[c] (not merely asserted equal — printed
   character-for-character equal, so the reader's eye does the compare);
   the dotted transitional loop in [d] passing exactly through z0, marking
   the one configuration where "inside" and "outside" are undefined; the
   before/after values in [d] and their signed difference.
6. READER EFFORT. Delegated: whether [a], [b], [c] carry the same digits
   (printed, not summarized). Kept: registering that [d] is the same
   family member as [c], relocated rather than reshaped — the ghost outline
   is [c]'s exact loop, so that leap costs nothing extra.
7. CHANNELS. Position carries the claim (loop shape/location). Role carries
   history in [d]: CONSTRUCTION (dashed) = the state before the drag,
   CONTENT (solid) = the state after, dotted CONSTRUCTION = the instant
   between them where the boundary sits on the pole. Direction markers on
   every loop (Needham's filled-head convention) fix orientation, since the
   theorem's sign depends on it.
7b. BINDING. Parts [a],[b],[c]: the pole (key "pole") and the printed value
   (key "value") are the fixed set; the only thing that changes is the loop
   itself (key "contour"). [d] is deliberately outside this correspondence:
   it is not another instance of the family, it is the family's rule
   broken, so forcing it into the same binding would misstate what the
   panel is for.
8. HONESTY. All of [a]-[d] share one page scale for the geometry that must
   compare truthfully (the loop family); [d] is allowed its own crop
   because it draws a different, subsequent event, not a fourth instance of
   the same one — declaring no correspondence over it says exactly that.
   The wobble amplitude is chosen small enough that no drawn loop
   self-intersects; assertions() checks this rather than trusting the
   parameter.
9. GATES. Complex trapezoidal quadrature on the exact drawn point arrays,
   cross-checked at 4x resolution for convergence, must agree with 2 pi i
   times the residue for every loop enclosing the pole, agree pairwise
   across [a]-[c], and disagree with the excluded-pole loop in [d] by
   exactly that amount; the transitional loop in [d] must pass through the
   pole to within its own sampling resolution.
"""

import numpy as np

from figlib.correspond import Correspondence, keyed
from figlib.figure import Figure, Panel
from figlib.format import WIDE
from figlib.gates import Checks
from figlib.plots import markers
from figlib.scene import Curve, MathLabel, Scene, Vector
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "The integral of f(z) = Res/(z - z0) around a closed loop depends only "
    "on the loop's homotopy class relative to z0, not on its shape: three "
    "visibly different loops enclosing z0 all carry the identical value "
    "2 pi i * Res, while dragging that same loop bodily across z0 instead "
    "of shrinking it around z0 changes the value discontinuously, by "
    "exactly 2 pi i * Res, at the one instant the boundary sweeps over the "
    "pole."
)

THEME = RISO
FORMAT = WIDE

PARAMS = {
    "pole": (0.55, 0.35),        # z0 in the complex plane
    "residue": 1.0,
    "R_big": 1.30,               # loop radius at t=0 (panel [a])
    "R_tight": 0.42,             # loop radius at t=1 (panel [c])
    "wobble_amp0": 0.30,         # radial wobble amplitude at t=0, ->0 at t=1
    "wobble_k": 3,               # wobble's angular frequency
    "wobble_phase": 0.35,
    "panel_t": (0.0, 0.5, 1.0),
    "n_theta": 6000,
    "n_theta_fine": 24000,       # independent resolution for the convergence check
    "row_xlim": (-1.4, 2.5),
    "row_ylim": (-1.85, 2.25),
    "shift": (0.95, -0.65),      # rigid translation carrying [c]'s loop across z0
    "d_xlim": (-0.6, 3.15),
    "d_ylim": (-1.05, 1.05),
    "arrow_fracs": (0.08, 0.33, 0.58, 0.83),
    "marker_size": 0.065,
}


def _loop(center: complex, R: float, amp: float, k: float, phase: float,
          n: int) -> tuple[np.ndarray, np.ndarray]:
    """(theta, z) for a star-shaped loop: r(theta) = R(1 + amp cos(k theta + phase)),
    z(theta) = center + r(theta) e^{i theta}. amp=0 is a plain circle."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = R * (1.0 + amp * np.cos(k * theta + phase))
    z = center + r * np.exp(1j * theta)
    return theta, r, z


def _contour_integral(z: np.ndarray, f) -> complex:
    """Complex trapezoidal quadrature on a closed, arc-sampled loop: the
    SAME points that get drawn, closed by wrapping to the first."""
    zc = np.concatenate([z, z[:1]])
    fz = f(zc)
    return complex(np.sum(0.5 * (fz[:-1] + fz[1:]) * (zc[1:] - zc[:-1])))


def compute(p):
    z0 = complex(*p["pole"])
    res = p["residue"]
    f = lambda z: res / (z - z0)

    panels = []
    for t in p["panel_t"]:
        R = p["R_big"] + (p["R_tight"] - p["R_big"]) * t
        amp = p["wobble_amp0"] * (1.0 - t)
        _, r, z = _loop(z0, R, amp, p["wobble_k"], p["wobble_phase"], p["n_theta"])
        _, _, z_fine = _loop(z0, R, amp, p["wobble_k"], p["wobble_phase"],
                             p["n_theta_fine"])
        panels.append({
            "t": float(t), "R": R, "amp": amp, "r": r, "z": z,
            "I": _contour_integral(z, f),
            "I_fine": _contour_integral(z_fine, f),
        })

    # [d]: exactly panel [c]'s loop (t=1, a plain circle of radius R_tight),
    # rigidly translated by `shift`. Reusing panel[-1]["z"] rather than
    # recomputing means "before" IS [c]'s drawn loop, not a look-alike.
    R1 = p["R_tight"]
    z_before = panels[-1]["z"]
    shift = complex(*p["shift"])
    z_after = z_before + shift

    s_cross = R1 / abs(shift)          # drag fraction where the boundary meets z0
    c_center = z0 + s_cross * shift
    _, _, z_cross = _loop(c_center, R1, 0.0, p["wobble_k"], p["wobble_phase"],
                          p["n_theta"])

    return {
        "p": p, "z0": z0, "residue": res, "f": f,
        "panels": panels,
        "z_before": z_before, "z_after": z_after, "z_cross": z_cross,
        "I_before": _contour_integral(z_before, f),
        "I_after": _contour_integral(z_after, f),
        "shift": shift, "s_cross": s_cross, "c_center": c_center,
        "R1": R1,
    }


# --- build --------------------------------------------------------------

PANEL_TAGS = ("[a]", "[b]", "[c]", "[d]")
VALUE_LABEL = r"\oint_\gamma \dfrac{dz}{z - z_0} = 2\pi i"


def _row_panel(g, pan: dict, first: bool) -> Scene:
    p = g["p"]
    s = Scene(xlim=tuple(p["row_xlim"]), ylim=tuple(p["row_ylim"]))
    z0 = g["z0"]

    s.add(*keyed("contour", Curve(
        np.column_stack([pan["z"].real, pan["z"].imag]), role=Role.CONTENT,
        closed=True, arrows=p["arrow_fracs"], arrow_style="filled")))
    s.add(*keyed("pole", *markers([z0.real], [z0.imag], "cross", filled=False,
                                  size=p["marker_size"], role=Role.ACCENT2,
                                  width_scale=1.3)))
    if first:
        # named once, at the panel whose ring is farthest from the marker;
        # the cross glyph alone identifies it in [b] and [c], where the
        # tight loop leaves no clearance for a second label beside it
        s.add(MathLabel(r"z_0", (z0.real, z0.imag), role=Role.ACCENT2,
                        ha="center", va="bottom", offset_px=(0.0, -24.0)))
    s.add(*keyed("value", MathLabel(
        VALUE_LABEL, (0.5 * (p["row_xlim"][0] + p["row_xlim"][1]),
                     p["row_ylim"][0] + 0.14),
        role=Role.ANNOTATION, ha="center", va="bottom", halo=True,
        offset_px=(0.0, 22.0))))
    return s


def _crossing_panel(g) -> Scene:
    p = g["p"]
    s = Scene(xlim=tuple(p["d_xlim"]), ylim=tuple(p["d_ylim"]))
    z0, R1 = g["z0"], g["R1"]
    zb, za, zx = g["z_before"], g["z_after"], g["z_cross"]

    c0, c1 = z0, z0 + g["shift"]
    xc = g["c_center"]

    s.add(Curve(np.column_stack([zb.real, zb.imag]), role=Role.CONSTRUCTION,
                closed=True, dash="dashed"))
    s.add(Curve(np.column_stack([zx.real, zx.imag]), role=Role.CONSTRUCTION,
                closed=True, dash="dotted", width_scale=0.85))
    lowest = min(z0.imag, xc.imag, c1.imag) - R1 - 0.28
    s.add(MathLabel(r"\text{mid-drag: boundary sits ON } z_0",
                    (xc.real, lowest), role=Role.CONSTRUCTION,
                    ha="center", va="top"))
    s.add(Curve(np.column_stack([za.real, za.imag]), role=Role.CONTENT,
                closed=True, arrows=p["arrow_fracs"], arrow_style="filled"))

    s.add(*markers([z0.real], [z0.imag], "cross", filled=False,
                   size=p["marker_size"], role=Role.ACCENT2, width_scale=1.3))
    s.add(MathLabel(r"z_0", (z0.real, z0.imag), role=Role.ACCENT2,
                    ha="right", va="bottom", offset_px=(-7.0, -6.0)))

    s.add(Vector((c0.real, c0.imag), (c1.real, c1.imag), role=Role.ACCENT1,
                 width_scale=1.1))
    # anchored above both circles (clear of all drawn ink), not at the
    # arrow's own midpoint, which sits inside the loops it connects
    top = max(c0.imag, c1.imag) + R1 + 0.15
    s.add(MathLabel(r"\text{dragged across } z_0",
                    (0.5 * (c0.real + c1.real), top),
                    role=Role.ACCENT1, ha="center", va="bottom"))

    s.add(MathLabel(r"\text{before: } 2\pi i", (c0.real, c0.imag - R1),
                    role=Role.CONSTRUCTION, ha="center", va="top",
                    offset_px=(0.0, 8.0), halo=True))
    s.add(MathLabel(r"\text{after: } 0", (c1.real, c1.imag - R1),
                    role=Role.CONTENT, ha="center", va="top",
                    offset_px=(0.0, 8.0), halo=True))
    s.add(MathLabel(
        r"\Delta = -2\pi i = -2\pi i\,\mathrm{Res}(f, z_0)",
        (p["d_xlim"][1] - 0.05, p["d_ylim"][1] - 0.05),
        role=Role.ANNOTATION, ha="right", va="top", halo=True))
    return s


CORRESPONDENCE = [
    Correspondence(
        parts=(0, 1, 2),
        varies=("the loop's shape and size, continuously deformed from a "
                "wobbly loop toward a tight circle, always enclosing z0"),
        # nothing goes in changes=: position/shape is exactly what the
        # fingerprint excludes (correspond.py — "moving is usually the
        # claim"), so the loop's deformation is not a tracked facet at
        # all. What IS tracked — the pole's role/style and the value
        # label's own text — must come out identical, which is the claim.
    ),
]


def build(g):
    panels = [Panel(_row_panel(g, pan, first=(i == 0)), tag=tag)
              for i, (pan, tag) in enumerate(zip(g["panels"], PANEL_TAGS[:3]))]
    panels.append(Panel(_crossing_panel(g), tag="[d]"))
    return Figure(panels=panels, grid=(2, 3))


# --- the numerical gate ---------------------------------------------------


def assertions(g):
    c = Checks()
    res = g["residue"]
    target = 2j * np.pi * res
    # trapezoidal quadrature at n_theta=6000 is empirically ~1e-6 accurate
    # here (measured, not assumed): tolerances carry a >x2 margin over the
    # worst observed per-panel error, not an arbitrarily loose slack
    tol = 2e-6

    values = []
    for pan, tag in zip(g["panels"], PANEL_TAGS[:3]):
        c.check(float(np.min(pan["r"])) > 0.0,
                f"{tag}: loop radius goes non-positive (r_min={pan['r'].min():.4f}); "
                f"the drawn curve self-intersects")
        c.check(abs(pan["I"] - target) < tol,
                f"{tag}: drawn-point quadrature gives {pan['I']!r}, "
                f"expected 2 pi i * Res = {target!r}")
        c.check(abs(pan["I"] - pan["I_fine"]) < tol,
                f"{tag}: quadrature at drawn resolution ({pan['I']!r}) does not "
                f"match the 4x-finer resolution ({pan['I_fine']!r}); the drawn "
                f"point density under-resolves the integral")
        values.append(pan["I"])

    for (ta, ia), (tb, ib) in zip(zip(PANEL_TAGS[:3], values),
                                   zip(PANEL_TAGS[1:3], values[1:])):
        c.check(abs(ia - ib) < tol,
                f"{ta} and {tb} are homotopic loops enclosing z0 but "
                f"disagree: {ia!r} vs {ib!r}")

    # [d]: "before" must equal [c] exactly (same array, so this is a wiring
    # check, not a numerical one) and equal the shared homotopy-class value;
    # "after" must be the excluded-pole value; the two must differ by
    # exactly 2 pi i * Res, the residue-theorem jump.
    c.check(np.array_equal(g["z_before"], g["panels"][-1]["z"]),
            "[d]'s 'before' loop is not literally [c]'s drawn loop")
    c.check(abs(g["I_before"] - target) < tol,
            f"[d] before-crossing: {g['I_before']!r}, expected {target!r}")
    c.check(abs(g["I_after"]) < tol,
            f"[d] after-crossing: {g['I_after']!r}, expected 0 (z0 excluded)")
    c.check(abs((g["I_before"] - g["I_after"]) - target) < 2 * tol,
            f"[d]: jump is {g['I_before'] - g['I_after']!r}, "
            f"expected 2 pi i * Res = {target!r}")

    # the translation must actually exclude the pole afterward, and the
    # drag must genuinely pass through the enclosing state at some point
    # strictly between start and end (0 < s_cross < 1) for "forced across"
    # to be the honest description of what happened
    c.check(abs(g["shift"]) > g["R1"],
            "translation is shorter than the loop radius; z0 stays enclosed")
    c.check(0.0 < g["s_cross"] < 1.0,
            f"crossing fraction s={g['s_cross']:.4f} is not strictly between "
            f"the endpoints of the drag")
    # the transitional loop's closest sampled point must land essentially ON
    # z0 (a broken s_cross formula would miss by O(R1), not by sampling
    # noise, so this is a real check on the geometry, not the resolution)
    dmin = float(np.min(np.abs(g["z_cross"] - g["z0"])))
    c.check(dmin < 1e-3,
            f"transitional loop's closest approach to z0 is {dmin:.3e} — too "
            f"far to read as 'the boundary sweeps over the pole'")

    c.done()
