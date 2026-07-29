"""The 2D Ising model's continuous phase transition: |m(T)| and the three
configurations behind it.

Design notes (exposition.md steps 0-9):

0. EARN IT. Prose: "the magnetisation falls continuously to zero at
   T_c = 2/ln(1+sqrt2), and near T_c the spin configuration has clusters
   at every scale" states two claims but proves neither. The reader must
   trust that "continuous" really means C^0-but-not-differentiable (not a
   jump dressed up in words), and "every scale" is a phrase, not a
   picture of a phrase. Drawn, the first becomes a curve an eye can see
   touch zero without a kink or a jump; the second becomes a raster the
   eye can compare against its two neighbours.
1. CLAIM: below.
2. REPRESENTATION. Order parameter vs control parameter (Strogatz's
   bifurcation-diagram rung) for the analytic claim; three bound
   instances (deep-ordered, critical, deep-disordered) for the "scale-free
   only at T_c" claim — a raster field, since a 64x64 spin lattice is a
   dense grid with no vector substitute at this density, and spin is a
   two-level CATEGORICAL quantity (hue), never an ordered one.
3. SIZE. WIDE: three lattice small multiples across the top row, the
   order-parameter curve spanning the full width beneath — the theorem
   gets the bigger panel, the instances that ground it sit above it,
   exactly the small-multiples-then-family grammar of the saddle-node
   exemplar, just read top-to-bottom instead of bottom-to-top.
4. TRAVERSAL. Enter at the curve: the eye follows |m| down to the axis at
   T_c and asks "why zero, not a jump". The guide rails at the three
   sampled T rise to the tags [a][b][c], sending the reader up to the
   lattices that answer "what does the system look like there". [b] sits
   AT the rail that touches zero — the critical point is where the curve
   loses its footing.
5. MECHANISM. Readable off the figure: T_c's value, drawn as a labelled
   vertical rail, not stated in a caption; the sampled |m| at each T
   marked ON the curve (a point the reader can compare against the curve
   itself); the three lattices are the ONLY drawn evidence for "clusters
   at every scale" — no annotation asserts scale-freeness, the raster at
   [b] has to earn that reading against [a] and [c] sitting next to it.
6. READER EFFORT. Delegated: whether the marked sample sits ON the
   analytic curve (drawn, not asserted in prose) — the answer is already
   on the page. Kept: judging visually that [b]'s clusters span many
   sizes while [a] is one domain and [c] is salt-and-pepper — that
   judgement IS the figure's claim, so it is not resolved for the reader.
7. CHANNELS. Spin rides CATEGORICAL hue (a two-level identity, never a
   ramp — the grammar.md rule this figure exists to test). |m| rides
   position on the curve (the order channel). Rails ride FRAME dash, tags
   ride ANNOTATION. The curve itself is the one CONTENT object; the
   sampled points are ACCENT2 so they read as "this observation", not as
   a fourth data series.
8. HONESTY. The curve is drawn to T well past T_c on both sides so the
   reader sees the "already zero" plateau, not a truncated approach; the
   x axis states its floor via tick_honesty rather than a silently
   generous range. The three lattices are single snapshots after burn-in,
   not idealised cartoons — admitted by seeding and sweep count living in
   PARAMS, gated by the |m| ordering check below.
9. GATES. T_c matches 2/ln(1+sqrt2); the analytic curve is exactly zero
   above T_c and strictly positive below it; and — the check that can
   actually fail, catching an under-equilibrated sampler silently
   producing noise at every temperature — the sampled |m| is ordered
   ordered > critical > disordered with a real margin.
"""

import numpy as np

from figlib import plots
from figlib.figure import Figure, Panel, layout_figure
from figlib.format import WIDE
from figlib.gates import Checks
from figlib.scene import Curve, MathLabel, RasterField, Scene
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "The 2D Ising magnetisation |m(T)| vanishes CONTINUOUSLY at "
    "T_c = 2/ln(1+sqrt2) via the exact Onsager law, and only exactly at "
    "T_c does a sampled spin configuration show clusters at every scale — "
    "the ordered and disordered configurations either side do not."
)

EXPOSITION = """
The statistical-mechanics account of a continuous phase transition rests
on two claims that are easy to state and hard to make convincing in prose:
the order parameter falls to zero smoothly rather than jumping, and the
system's spatial structure changes qualitatively as the transition is
approached, developing correlations at every length scale exactly at the
critical point. For the 2D Ising model, Onsager's exact solution gives the
magnetisation |m(T)| as a curve that touches zero at T_c with a genuine
critical exponent rather than a kink or a discontinuity, and a reader
needs to see that touch happen on the page to believe "continuous" means
something more specific than "not a jump". The second claim needs its own
evidence entirely separate from the curve: sampled spin configurations well
below T_c form one dominant domain, well above T_c look like salt-and-pepper
noise, and only at T_c does the configuration show clusters of aligned
spins at every scale simultaneously -- the visual signature of scale
invariance at a critical point, and the reason phase transitions are the
bridge between statistical mechanics and renormalization-group theory.
"""

THEME = RISO
FORMAT = WIDE

_TC = 2.0 / np.log(1.0 + np.sqrt(2.0))

PARAMS = {
    "L": 64,                       # lattice side
    "seed": 20260729,              # single seed drives every sweep below
    "burn_sweeps": 600,            # equilibration sweeps before sampling
    "sample_sweeps": 60,           # sweeps averaged for the reported |m|
    "T_ordered": 1.6,
    "T_critical": _TC,
    "T_disordered": 3.6,
    "T_curve_lo": 0.9,
    "T_curve_hi": 4.0,
    "T_curve_n": 600,
    "xlim": (0.85, 4.05),
    "ylim": (-0.06, 1.10),
    "curve_height_px": 360.0,
    "lattice_pad_frac": 0.06,      # equal-aspect frame padding round the grid
}

PANEL_TAGS = ("[a]", "[b]", "[c]")
PANEL_KEYS = ("ordered", "critical", "disordered")
PANEL_NAMES = {"ordered": r"\text{deep ordered}", "critical": r"T \approx T_c",
              "disordered": r"\text{deep disordered}"}


def _onsager_m(T: np.ndarray) -> np.ndarray:
    """Exact Onsager order parameter, zero identically above T_c (not just
    numerically small — the branch is chosen from the sign of T_c - T)."""
    T = np.asarray(T, dtype=float)
    m = np.zeros_like(T)
    below = T < _TC
    s = np.sinh(2.0 / T[below])
    m[below] = (1.0 - s ** -4) ** (1.0 / 8.0)
    return m


def _ising_sweep(spins: np.ndarray, T: float, rng: np.random.Generator) -> None:
    """One checkerboard-parity Metropolis sweep, in place.

    Vectorized over each of the two sublattices so a 64x64 lattice runs
    fast enough for hundreds of sweeps at render time, deterministic
    given `rng`: every accept/reject draw comes from the same generator
    the caller seeded from PARAMS.
    """
    L = spins.shape[0]
    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    parity_masks = ((ii + jj) % 2 == 0, (ii + jj) % 2 == 1)
    for mask in parity_masks:
        nbr_sum = (np.roll(spins, 1, axis=0) + np.roll(spins, -1, axis=0)
                  + np.roll(spins, 1, axis=1) + np.roll(spins, -1, axis=1))
        dE = 2.0 * spins * nbr_sum
        accept = (dE <= 0.0) | (rng.random(spins.shape) < np.exp(-dE / T))
        flip = mask & accept
        spins[flip] *= -1


def _sample_configuration(p: dict, T: float, rng: np.random.Generator):
    """Burn in, then report the LAST sweep's configuration and the mean
    |m| over the sampling window — the snapshot drawn is one member of
    the window whose average is what the assertion checks."""
    L = p["L"]
    spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
    for _ in range(p["burn_sweeps"]):
        _ising_sweep(spins, T, rng)
    m_samples = np.empty(p["sample_sweeps"])
    for k in range(p["sample_sweeps"]):
        _ising_sweep(spins, T, rng)
        m_samples[k] = spins.mean()
    return spins.copy(), float(np.mean(np.abs(m_samples)))


def compute(p):
    rng = np.random.default_rng(p["seed"])
    panels = {}
    for key, T in zip(PANEL_KEYS,
                      (p["T_ordered"], p["T_critical"], p["T_disordered"])):
        spins, m_sim = _sample_configuration(p, T, rng)
        panels[key] = {"T": T, "spins": spins, "m_sim": m_sim}

    T_curve = np.linspace(p["T_curve_lo"], p["T_curve_hi"], p["T_curve_n"])
    m_curve = _onsager_m(T_curve)

    return {
        "p": p,
        "Tc": _TC,
        "T_curve": T_curve,
        "m_curve": m_curve,
        "panels": panels,
        "T_scale": plots.linear(*p["xlim"]),
        "m_scale": plots.linear(*p["ylim"]),
    }


# --- build -------------------------------------------------------------


def _lattice_rgb(spins: np.ndarray) -> np.ndarray:
    """Spin +-1 -> an (H, W, 3) sRGB image through the theme's CATEGORICAL
    channel — spin is a two-level identity, never an ordered ramp."""
    from figlib.color import to_rgb

    up = np.array(to_rgb(RISO.categorical(0)))
    down = np.array(to_rgb(RISO.categorical(1)))
    up_mask = (spins > 0)[:, :, None]
    return np.where(up_mask, up, down).astype(float)


def _lattice_scene(p: dict) -> Scene:
    L = p["L"]
    pad = p["lattice_pad_frac"] * L
    return Scene(xlim=(-pad, L + pad), ylim=(-pad, L + pad))


def _fill_lattice_panel(s: Scene, key: str, dat: dict, p: dict) -> None:
    L = p["L"]
    pad = p["lattice_pad_frac"] * L
    s.add(RasterField(_lattice_rgb(dat["spins"]), extent=(0.0, L, 0.0, L)))
    s.add(Curve(np.array([[0.0, 0.0], [L, 0.0], [L, L], [0.0, L]]),
                closed=True, role=Role.FRAME))
    s.add(MathLabel(PANEL_NAMES[key], (L / 2.0, -pad * 0.35), role=Role.ANNOTATION,
                    ha="center", va="top"))


DOT_PX = 6.0   # sampled-point marker radius, canvas px


def _curve_scene(p: dict) -> Scene:
    return Scene(xlim=tuple(p["xlim"]), ylim=tuple(p["ylim"]),
                height_px=p["curve_height_px"])


def _fill_curve_panel(s: Scene, g: dict, upx: float, upy: float) -> None:
    p = g["p"]
    T_sc, m_sc = g["T_scale"], g["m_scale"]

    s.add(*plots.axis(T_sc, orient="x", at=0.0, ticks=T_sc.ticks(step=0.5),
                      label="T", extend=(0.02, 0.10), role=Role.ANNOTATION))
    s.add(*plots.axis(m_sc, orient="y", at=p["xlim"][0], ticks=m_sc.ticks(step=0.25),
                      label=r"|m|", extend=(0.02, 0.08), role=Role.ANNOTATION))

    # the theorem: one CONTENT curve, drawn well past T_c on both sides so
    # the "already zero" plateau above T_c is seen, not merely implied
    s.add(*plots.series(g["T_curve"], g["m_curve"], role=Role.CONTENT,
                        width_scale=1.15))

    # T_c: a rail with no analogue in the saddle-node figure's r=0 case
    # (there the bifurcation IS the axis; here it is an interior point) —
    # so it is drawn and labelled explicitly, at the height it names. The
    # label sits to the RIGHT of the rail, above the T > T_c plateau —
    # the one region of the panel that is provably ink-free (m == 0
    # identically there), rather than guessed clear of the curve.
    s.add(Curve(np.array([[g["Tc"], p["ylim"][0]], [g["Tc"], 0.85]]),
                role=Role.FRAME, dash="dashed"))
    s.add(MathLabel(r"T_c = 2/\ln(1+\sqrt{2})", (g["Tc"], 0.80), role=Role.ANNOTATION,
                    ha="left", va="center", offset_px=(8.0, 0.0)))

    # the three sampled points, each on its own rail up to the tag that
    # names the lattice panel above it — the rail is what ties [a][b][c]
    # to a location on THIS curve, mirroring the saddle-node family panel
    guide_top = p["ylim"][1] * 0.94
    for tag, key in zip(PANEL_TAGS, PANEL_KEYS):
        dat = g["panels"][key]
        T = dat["T"]
        if abs(T - g["Tc"]) > 1e-9:
            s.add(Curve(np.array([[T, p["ylim"][0]], [T, guide_top]]),
                        role=Role.FRAME))
        s.add(MathLabel(tag, (T, guide_top), role=Role.ANNOTATION,
                        ha="center", va="bottom"))
        s.add(*plots.markers([T], [dat["m_sim"]], size=(DOT_PX * upx, DOT_PX * upy),
                             role=Role.ACCENT2))


# No CORRESPONDENCE: the three lattice panels are independent draws (each
# its own random configuration at its own T), not the same object put
# through a transformation — there is no shared key for correspond.py to
# bind, and forcing one would claim a relation the figure does not draw.


def build(g):
    p = g["p"]
    scenes = {key: _lattice_scene(p) for key in PANEL_KEYS}
    curve_scene = _curve_scene(p)
    fig = Figure(
        panels=[Panel(scenes[key], tag=tag)
               for key, tag in zip(PANEL_KEYS, PANEL_TAGS)]
        + [Panel(curve_scene, tag=None, frame=False)],
        grid=(2, 3))

    # marker/lattice sizes are px quantities, so they are measured through
    # the very transforms the renderer will use — scenes are populated
    # only after the layout that fixes those transforms exists
    lay = layout_figure(fig, FORMAT.display_width_px)
    for key in PANEL_KEYS:
        _fill_lattice_panel(scenes[key], key, g["panels"][key], p)
    t = lay.slots[3].transform
    _fill_curve_panel(curve_scene, g, 1.0 / t.scale_x, 1.0 / t.scale_y)
    return fig


# --- the numerical gate --------------------------------------------------


def assertions(g):
    c = Checks()
    p = g["p"]

    c.check(abs(g["Tc"] - 2.0 / np.log(1.0 + np.sqrt(2.0))) < 1e-12,
            f"T_c = {g['Tc']:.6f} does not match 2/ln(1+sqrt2)")

    above = g["T_curve"] > g["Tc"]
    below = g["T_curve"] < g["Tc"]
    c.check(bool(np.all(g["m_curve"][above] == 0.0)),
            f"analytic m is nonzero above T_c: max "
            f"{float(g['m_curve'][above].max() if above.any() else 0.0):.3e}")
    c.check(bool(np.all(g["m_curve"][below] > 0.0)),
            "analytic m is not strictly positive below T_c")
    c.check(bool(np.all(np.diff(g["m_curve"][below]) < 1e-9)),
            "analytic m is not monotone decreasing toward T_c")
    # continuity at T_c: critical exponent beta = 1/8 makes the approach to
    # zero genuinely slow (m ~ (T_c-T)^{1/8}, so it is NOT small even a
    # thousandth of a degree below T_c) — the honest continuity check is
    # therefore "closer to T_c is strictly smaller", already the monotone
    # check above, plus that the true limit as epsilon -> 0 keeps falling
    # rather than saturating at some floor away from zero.
    eps = np.array([1e-2, 1e-4, 1e-6, 1e-8])
    m_eps = _onsager_m(g["Tc"] - eps)
    c.check(bool(np.all(np.diff(m_eps) < 0.0)),
            f"m(T_c - eps) does not keep decreasing as eps -> 0: {m_eps}")
    c.check(float(m_eps[-1]) < float(m_eps[0]),
            "m does not approach 0 in the eps -> 0 limit at T_c")

    # the one check that can genuinely fail: an under-equilibrated sampler
    # would produce indistinguishable noise at every temperature instead of
    # this ordering
    m_o = g["panels"]["ordered"]["m_sim"]
    m_cr = g["panels"]["critical"]["m_sim"]
    m_d = g["panels"]["disordered"]["m_sim"]
    c.check(m_o > 0.75, f"ordered |m| = {m_o:.3f} is not strongly ordered")
    c.check(m_d < 0.30, f"disordered |m| = {m_d:.3f} is not weakly ordered")
    c.check(m_o > m_cr > m_d,
            f"sampled |m| not ordered ordered>critical>disordered: "
            f"{m_o:.3f}, {m_cr:.3f}, {m_d:.3f}")

    for key in PANEL_KEYS:
        spins = g["panels"][key]["spins"]
        c.check(bool(np.all(np.isin(spins, (-1, 1)))),
                f"{key}: lattice has values other than +-1")
        c.check(spins.shape == (p["L"], p["L"]),
                f"{key}: lattice shape {spins.shape} != ({p['L']}, {p['L']})")

    plots.tick_honesty(c, g["T_scale"], g["T_curve"], name="T axis",
                       ticks=g["T_scale"].ticks(step=0.5))
    plots.tick_honesty(c, g["m_scale"], g["m_curve"], name="|m| axis",
                       ticks=g["m_scale"].ticks(step=0.25), min_span_frac=0.35)

    c.done()
