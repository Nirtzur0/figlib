"""Needham Fig [9], restyled: the exponential series at iθ vs θ.

The geometry IS the argument: right-angle turns force the spiral to
close in on a point; straight continuation forces escape along the axis.
"""

import numpy as np

from figlib.geometry import exp_partial_sums
from figlib.scene import (AngleMark, Curve, MathLabel, Point,
                          RightAngleMark, Scene, Vector)
from figlib.style import Role
from figlib.theme import RISO

CLAIM = (
    "The partial sums of the power series for e^z, evaluated at z = iθ, "
    "turn a right angle at every step and spiral into the point e^{iθ} on "
    "the unit circle, while the same series at real z = θ moves straight "
    "along the real axis toward e^θ."
)

EXPOSITION = """
This is Needham's Fig [9] (VCA ch. 1, "Euler's Formula"), the second of
two independent arguments he gives for e^{i theta} = cos theta +
i sin theta. The first argument is kinematic: a particle at Z(t) = e^{it}
has velocity i*Z(t), its position rotated a right angle, so it must trace
the unit circle at unit speed. The second, drawn here, instead takes the
defining property f' = f as a power series and evaluates it at
z = i*theta: because multiplying by i is a right-angle turn, each term of
the partial sum turns 90 degrees from the last, and the partial sums
spiral rather than march. Needham's point is that ordinary convergence of
the real series for e^theta -- terms all pointing the same way, summing
straight down the real axis -- already guarantees the spiral converges to
some point; what the picture has to additionally make undeniable is that
the point it converges to is the very same e^{i theta} the kinematic
argument produced, two arguments meeting at one point by two different
routes.
"""

THEME = RISO

PARAMS = {
    "theta": 2.0,
    "n_vector_steps": 5,   # spiral steps drawn as arrows
    "n_tail": 24,          # further terms drawn as a thin polyline
    "n_real_steps": 5,
}


def compute(p):
    th = p["theta"]
    spiral = exp_partial_sums(1j * th, p["n_tail"])
    real = exp_partial_sums(th + 0j, p["n_tail"])
    return {
        "theta": th,
        "spiral": spiral,
        "real": real,
        "e_itheta": np.exp(1j * th),
        "e_theta": float(np.exp(th)),
    }


def build(g):
    th = g["theta"]
    sp = g["spiral"]
    re = g["real"]
    z = g["e_itheta"]
    et = g["e_theta"]
    nv = PARAMS["n_vector_steps"]
    nr = PARAMS["n_real_steps"]

    s = Scene()

    # horizontal axis through 0, running past e^theta
    s.add(Curve(np.array([[-1.6, 0.0], [et + 0.9, 0.0]]), role=Role.CONSTRUCTION, width_scale=0.8))

    # unit circle (construction) + radius to e^{i theta} + angle theta
    tt = np.linspace(0, 2 * np.pi, 200)
    s.add(Curve(np.column_stack([np.cos(tt), np.sin(tt)]), role=Role.CONSTRUCTION))
    s.add(Curve(np.array([[0.0, 0.0], [z.real, z.imag]]), role=Role.CONSTRUCTION))
    s.add(AngleMark((0, 0), (1, 0), (z.real, z.imag), radius=0.34))
    s.add(MathLabel(r"\theta", (0.42 * np.cos(th / 2), 0.42 * np.sin(th / 2)),
                    ha="center", va="center", role=Role.ANNOTATION))

    # shared first term: 0 -> 1, in plain ink (belongs to both series)
    s.add(Vector((0, 0), (1, 0), role=Role.CONTENT))
    s.add(MathLabel(r"1", (0.5, 0.0), ha="center", va="top", offset_px=(0, 8), role=Role.CONTENT))

    # spiral chain (accent1): steps 1..nv as arrows
    for k in range(1, nv):
        s.add(Vector(tuple(sp[k]), tuple(sp[k + 1]), role=Role.ACCENT1))
    # remaining terms as a thin polyline into the limit point, with the
    # partial sums themselves marked so the accumulation is visible
    s.add(Curve(sp[nv:], role=Role.ACCENT1, width_scale=0.6))
    for k in range(nv, nv + 4):
        s.add(Point(tuple(sp[k]), role=Role.ACCENT1, radius_scale=0.55))
    s.add(Point((z.real, z.imag), role=Role.ACCENT1))
    s.add(MathLabel(r"e^{i\theta}", (z.real, z.imag), ha="right", va="bottom",
                    offset_px=(-8, -6), role=Role.ACCENT1))

    # right-angle turns at the first two corners of the spiral
    s.add(RightAngleMark(tuple(sp[1]), (-1, 0), (0, 1), size=0.13))
    s.add(RightAngleMark(tuple(sp[2]), (0, -1), (-1, 0), size=0.13))

    # spiral step labels
    s.add(MathLabel(r"i\theta", (1.0, th / 2), ha="left", va="center",
                    offset_px=(9, 0), role=Role.ACCENT1))
    s.add(MathLabel(r"\frac{(i\theta)^2}{2!}", ((sp[2, 0] + sp[3, 0]) / 2, sp[2, 1]),
                    ha="center", va="top", offset_px=(0, 10), role=Role.ACCENT1))
    s.add(MathLabel(r"\frac{(i\theta)^3}{3!}", (sp[3, 0], (sp[3, 1] + sp[4, 1]) / 2),
                    ha="right", va="center", offset_px=(-9, 0), role=Role.ACCENT1))

    # real chain (accent2): drawn just below the axis so collinear arrows stay legible
    drop = -0.0
    for k in range(1, nr):
        s.add(Vector((re[k, 0], drop), (re[k + 1, 0], drop), role=Role.ACCENT2))
    s.add(Point((et, 0.0), role=Role.ACCENT2, filled=False))
    s.add(MathLabel(r"e^{\theta}", (et, 0.0), ha="center", va="bottom",
                    offset_px=(0, -10), role=Role.ACCENT2))
    # real step labels below axis
    s.add(MathLabel(r"\theta", ((re[1, 0] + re[2, 0]) / 2, 0), ha="center", va="top",
                    offset_px=(0, 8), role=Role.ACCENT2))
    s.add(MathLabel(r"\frac{\theta^2}{2!}", ((re[2, 0] + re[3, 0]) / 2, 0), ha="center",
                    va="top", offset_px=(0, 8), role=Role.ACCENT2))
    s.add(MathLabel(r"\frac{\theta^3}{3!}", ((re[3, 0] + re[4, 0]) / 2, 0), ha="center",
                    va="top", offset_px=(0, 8), role=Role.ACCENT2))

    # origin
    s.add(Point((0, 0), role=Role.CONTENT, filled=False))
    s.add(MathLabel(r"0", (0, 0), ha="right", va="top", offset_px=(-6, 6), role=Role.CONTENT))

    return s


def assertions(g):
    th = g["theta"]
    sp = g["spiral"]
    re = g["real"]
    # the plotted chain converges to e^{i theta}
    end = sp[-1, 0] + 1j * sp[-1, 1]
    assert abs(end - np.exp(1j * th)) < 1e-9, "spiral endpoint is not e^{i theta}"
    # every consecutive pair of steps turns by exactly 90 degrees
    steps = np.diff(sp, axis=0)
    zs = steps[:, 0] + 1j * steps[:, 1]
    turns = np.angle(zs[1:] / zs[:-1])
    assert np.allclose(np.abs(turns), np.pi / 2, atol=1e-9), "spiral turns are not right angles"
    # the marked limit point lies on the unit circle
    assert abs(abs(np.exp(1j * th)) - 1) < 1e-12
    # real chain is collinear with the axis and approaches e^theta
    assert np.max(np.abs(re[:, 1])) == 0.0, "real chain left the real axis"
    assert abs(re[-1, 0] - np.exp(th)) < 1e-6, "real chain does not approach e^theta"
