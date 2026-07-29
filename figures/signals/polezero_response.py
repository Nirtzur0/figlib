"""Frequency response read geometrically off the z-plane (JOS idiom).

|H(e^{j omega})| = prod |chords to zeros| / prod |chords to poles|: the
response curve is not a second fact about H but the SAME fact seen while
e^{j omega} walks the unit circle. The figure draws one generic walk
point with its four chords, and the response panel carries the two
consequences a reader should check by eye: the peak sits at the pole
angle, the nulls sit where zeros pin the circle.
"""

import numpy as np

from figlib.figure import Connector, Figure, Panel
from figlib.format import WIDE
from figlib.gates import Checks
from figlib.plots import Ticks, axis, linear, markers, series
from figlib.scene import AngleMark, Curve, MathLabel, Point, Scene
from figlib.style import Role
from figlib.theme import RISO

THEME = RISO
FORMAT = WIDE

CLAIM = (
    "The magnitude response of H(z) is read geometrically off the z-plane: "
    "at each omega, |H(e^{j omega})| is the product of the distances from "
    "e^{j omega} to the zeros divided by the product of the distances to the "
    "poles — so the conjugate pole pair at radius 0.85 and angle pi/3 carves "
    "a resonance peak at omega = pi/3, and the zeros at z = +-1 pin nulls at "
    "omega = 0 and pi."
)

EXPOSITION = """
Julius Smith's way of reading a transfer function: don't compute
|H(e^{j omega})| from algebra, read it off the z-plane geometrically.
Every zero and pole is a fixed point in the plane; e^{j omega} is a
point walking the unit circle as omega sweeps from 0 to pi. At any
instant the magnitude response is the product of the distances from
that walking point to each zero, divided by the product of distances to
each pole -- a ratio of chord lengths, recomputed fresh at every omega,
and the response curve is just that ratio plotted against the walk's
angle. The shape follows without any transform: a pole close to the
unit circle means some angle where the chord to it gets very short, so
the denominator collapses and the response peaks near that pole's
angle; a zero sitting ON the circle means the chord to it hits zero
exactly there, so the response nulls at that zero's angle exactly.
Nothing about the curve is a separate fact from the picture -- it is
the picture, read as e^{j omega} completes its walk.
"""

PARAMS = {
    "pole_r": 0.85,
    "pole_theta": np.pi / 3,
    "zeros": [1.0, -1.0],
    "omega0": 0.55 * np.pi,     # the generic walk point: not the peak, not a null
    "n_omega": 512,
    "n_circle": 256,
    "marker_size": 0.055,
    "resp_h_px": 300.0,
}


def _mag(w: np.ndarray, zeros: np.ndarray, poles: np.ndarray) -> np.ndarray:
    q = np.exp(1j * np.atleast_1d(w))[:, None]
    num = np.abs(q - zeros[None, :]).prod(axis=1)
    den = np.abs(q - poles[None, :]).prod(axis=1)
    return num / den


def compute(p):
    poles = np.array([p["pole_r"] * np.exp(1j * p["pole_theta"]),
                      p["pole_r"] * np.exp(-1j * p["pole_theta"])])
    zeros = np.array(p["zeros"], dtype=complex)
    w = np.linspace(0.0, np.pi, p["n_omega"])
    mag = _mag(w, zeros, poles)
    scale = float(mag.max())
    q0 = np.exp(1j * p["omega0"])
    return {
        "params": p, "poles": poles, "zeros": zeros,
        "w": w, "mag_n": mag / scale, "scale": scale,
        "q0": q0,
        "h0_n": float(_mag(np.array([p["omega0"]]), zeros, poles)[0]) / scale,
        "w_peak": float(w[int(np.argmax(mag))]),
    }


def _plane(g) -> Scene:
    p = g["params"]
    s = Scene(xlim=(-1.42, 1.42), ylim=(-1.42, 1.42))
    th = np.linspace(0.0, 2 * np.pi, p["n_circle"])
    s.add(Curve(np.column_stack([np.cos(th), np.sin(th)]),
                role=Role.CONSTRUCTION, closed=True))
    # bare Re/Im axes, frame furniture
    s.add(Curve(np.array([[-1.3, 0.0], [1.3, 0.0]]), role=Role.FRAME))
    s.add(Curve(np.array([[0.0, -1.3], [0.0, 1.3]]), role=Role.FRAME))

    q0 = g["q0"]
    # chords first (under the markers): to zeros ACCENT2, to poles ACCENT1
    for z in g["zeros"]:
        s.add(Curve(np.array([[q0.real, q0.imag], [z.real, z.imag]]),
                    role=Role.ACCENT2, width_scale=0.85))
    for pl in g["poles"]:
        s.add(Curve(np.array([[q0.real, q0.imag], [pl.real, pl.imag]]),
                    role=Role.ACCENT1, width_scale=0.85))
    # the radius that sets the resonance sharpness
    pu = g["poles"][0]
    s.add(Curve(np.array([[0.0, 0.0], [pu.real, pu.imag]]),
                role=Role.CONSTRUCTION, dash="dashed"))
    s.add(MathLabel(r"r", (pu.real / 2, pu.imag / 2), role=Role.CONSTRUCTION,
                    ha="right", va="bottom", offset_px=(-3, -2)))

    s.items += markers(g["poles"].real, g["poles"].imag, "cross",
                       filled=False, size=p["marker_size"], width_scale=1.3)
    s.items += markers(g["zeros"].real, g["zeros"].imag, "circle",
                       filled=False, size=p["marker_size"], width_scale=1.3)
    s.add(Point((q0.real, q0.imag), role=Role.ACCENT1))
    s.add(MathLabel(r"e^{j\omega_0}", (q0.real, q0.imag), role=Role.ACCENT1,
                    ha="right", va="bottom", offset_px=(-6, -6)))
    s.add(MathLabel(r"p", (g["poles"][0].real, g["poles"][0].imag),
                    ha="left", va="bottom", offset_px=(7, -3)))
    s.add(MathLabel(r"\bar p", (g["poles"][1].real, g["poles"][1].imag),
                    ha="left", va="top", offset_px=(7, 3)))
    s.add(MathLabel(r"1", (1.0, 0.0), role=Role.ANNOTATION,
                    ha="left", va="top", offset_px=(6, 8)))
    s.add(MathLabel(r"-1", (-1.0, 0.0), role=Role.ANNOTATION,
                    ha="right", va="top", offset_px=(-6, 8)))
    # omega_0 is an ANGLE: draw the radius it opens to, then the arc,
    # so the arc visibly ends on e^{j omega_0}, not on the pole
    s.add(Curve(np.array([[0.0, 0.0], [q0.real, q0.imag]]),
                role=Role.CONSTRUCTION, dash="dashed", width_scale=0.7))
    s.add(AngleMark((0.0, 0.0), (1.0, 0.0), (q0.real, q0.imag), radius=0.3))
    s.add(MathLabel(r"\omega_0", (0.44 * np.cos(0.72 * p["omega0"]),
                                  0.44 * np.sin(0.72 * p["omega0"])),
                    role=Role.ANNOTATION, ha="left", va="bottom",
                    offset_px=(1, -1)))
    # the mechanism, on the figure, in the space under the circle
    s.add(MathLabel(
        r"|H(e^{j\omega_0})| = \textstyle\prod_i |e^{j\omega_0}\!-z_i| \,/\, "
        r"\prod_i |e^{j\omega_0}\!-p_i|",
        (0.0, -1.24), role=Role.ANNOTATION, ha="center", va="center"))
    return s


def _response(g) -> Scene:
    p = g["params"]
    th_p, w0 = p["pole_theta"], p["omega0"]
    s = Scene(xlim=(-0.35, np.pi + 0.55), ylim=(-0.3, 1.32),
              height_px=p["resp_h_px"])
    tk = Ticks(values=np.array([0.0, th_p, np.pi]),
               positions=np.array([0.0, th_p, np.pi]),
               labels=("0", r"\tfrac{\pi}{3}", r"\pi"))
    s.items += axis(linear(0.0, np.pi), orient="x", at=0.0, side=-1,
                    ticks=tk, tick_len=0.045, extend=(0.0, 0.25),
                    label=r"\omega", arrow=True)
    s.items += axis(linear(0.0, 1.0), orient="y", at=0.0, side=-1,
                    ticks=Ticks(values=np.array([1.0]),
                                positions=np.array([1.0]), labels=(r"1",)),
                    tick_len=0.05, arrow=False)
    # the alignment construction: peak abscissa = pole angle
    s.add(Curve(np.array([[th_p, 0.0], [th_p, 1.05]]),
                role=Role.CONSTRUCTION, dash="dashed"))
    s.items += series(g["w"], g["mag_n"], role=Role.CONTENT, width_scale=1.4)
    # the walk point, tied to the plane by hue: marker + drop line
    s.add(Curve(np.array([[w0, 0.0], [w0, g["h0_n"]]]),
                role=Role.ACCENT1, dash="dotted", width_scale=0.8))
    s.items += markers([w0], [g["h0_n"]], "circle", size=0.035,
                       role=Role.ACCENT1)
    s.add(MathLabel(r"\omega_0", (w0, 0.0), role=Role.ACCENT1,
                    ha="center", va="top", offset_px=(0, 6)))
    s.add(MathLabel(r"\frac{|H(e^{j\omega})|}{|H|_{\max}}",
                    (0.0, 1.22), role=Role.ANNOTATION,
                    ha="left", va="center", offset_px=(10, 0)))
    return s


def build(g):
    return Figure(
        panels=[Panel(_plane(g), tag=r"[a]"),
                Panel(_response(g), tag=r"[b]", width_frac=0.52)],
        connectors=[Connector(0, 1, kind="map",
                              label=r"\omega \mapsto |H|")],
    )


def assertions(g):
    p = g["params"]
    ck = Checks()
    # the drawn chords recompute the drawn curve value: product of the
    # very segment lengths in the plane panel vs the response ordinate
    q0 = g["q0"]
    num = np.prod([abs(q0 - z) for z in g["zeros"]])
    den = np.prod([abs(q0 - pl) for pl in g["poles"]])
    ck.check(abs(num / den / g["scale"] - g["h0_n"]) < 1e-12,
             "chord product does not reproduce the marked |H(omega_0)|")
    # stability: poles strictly inside the unit circle
    ck.check(bool(np.all(np.abs(g["poles"]) < 1.0)), "pole on/outside unit circle")
    # zeros on the circle pin the nulls
    ck.check(g["mag_n"][0] < 1e-9 and g["mag_n"][-1] < 1e-9,
             "zeros at +-1 fail to null omega = 0, pi")
    # the resonance peak sits at the pole angle (within the r-dependent skew)
    ck.check(abs(g["w_peak"] - p["pole_theta"]) < 0.1,
             f"peak at {g['w_peak']:.3f}, pole angle {p['pole_theta']:.3f}")
    # omega_0 is a generic point: away from peak and both nulls
    ck.check(min(abs(p["omega0"] - v)
                 for v in (0.0, p["pole_theta"], np.pi)) > 0.3,
             "omega_0 is not a generic walk point")
    ck.done()
