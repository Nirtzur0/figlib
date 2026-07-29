"""Three solids under one light: the chromatic shading capability demo.

A flat face under a directional light has constant Lambert shade; the
within-face drift here is deliberate style — each face's gradient runs
along the light direction projected into the face plane, drifting a
fixed amplitude around the face's true (half-)Lambert tone. Shadow ends
rotate cool, lit ends rotate warm (OKLCh), so light->shadow travels
through hue, not just lightness.
"""

import numpy as np

from figlib.format import COLUMN
from figlib.scene import Scene
from figlib.shading import chroma_ramp
from figlib.solids import box_items, cylinder_items
from figlib.surface3d import Camera, compose, drop_shadow
from figlib.theme import RISO

THEME = RISO
FORMAT = COLUMN

CLAIM = (
    "Three solids under a single light: each visible face carries its "
    "Lambert tone drifted along the projected light direction, with cool "
    "shadows and warm lights (OKLCh); the banded cylinder posterizes the "
    "same ramp across its visible sweep; contact shadows tie each solid "
    "to the floor."
)

PARAMS = {
    "azim": -35.0, "elev": 32.0,
    "box_lo": {"center": (0.0, 0.0, 0.55), "size": (1.5, 1.5, 1.1)},
    "box_hi": {"center": (0.15, -0.10, 2.35), "size": (0.9, 0.9, 0.9)},
    "cyl": {"center": (1.55, 1.15, 0.45), "radius": 0.55, "height": 0.9},
    "bands": 6,
    "facets": 64,
    "grain": 0.35,
    "grad_amp": 0.16,
}


def _box_verts(center, size):
    (cx, cy, cz), (sx, sy, sz) = center, size
    return np.array([[cx + dx * sx / 2, cy + dy * sy / 2, cz + dz * sz / 2]
                     for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)])


def compute(p):
    cyl = p["cyl"]
    cx, cy, cz = cyl["center"]
    th = np.linspace(0.0, 2 * np.pi, p["facets"], endpoint=False)
    rim = np.column_stack([cx + cyl["radius"] * np.cos(th),
                           cy + cyl["radius"] * np.sin(th),
                           np.full(p["facets"], cz - cyl["height"] / 2)])
    return {
        "params": p,
        "cyl_rim": rim,
        "lo_verts": _box_verts(**p["box_lo"]),
        "hi_verts": _box_verts(**p["box_hi"]),
        "lo_top": p["box_lo"]["center"][2] + p["box_lo"]["size"][2] / 2,
    }


def build(g):
    p = g["params"]
    cam = Camera(p["azim"], p["elev"])
    amp = p["grad_amp"]
    # ramps anchored on the theme's accent hues; hue rotation signs chosen
    # so shadows head toward violet/navy and lit faces toward yellow
    ramp_brick = chroma_ramp("#c8402f", l_range=(0.34, 0.88),
                             hue_cool=-55.0, hue_warm=45.0,
                             c_scale=(1.15, 1.0))
    ramp_indigo = chroma_ramp("#3467a3", l_range=(0.32, 0.90),
                              hue_cool=-35.0, hue_warm=-120.0,
                              c_scale=(1.2, 0.9))
    ramp_ochre = chroma_ramp("#b08018", l_range=(0.36, 0.90),
                             hue_cool=-70.0, hue_warm=25.0,
                             c_scale=(1.2, 1.05))

    # no drawn floor: the paper itself is the ground (the corpus idiom);
    # each contact shadow is tinted from its own solid's shadow end, so
    # shadows read as colored shade, not gray smudges
    lo, hi, cyl = p["box_lo"], p["box_hi"], p["cyl"]
    items = compose(
        drop_shadow(g["lo_verts"], cam, color=ramp_brick(0.0), opacity=0.22),
        # the floating box shades the box below it, not the floor: cast
        # onto the lower box's top plane, biased to paint over that face
        drop_shadow(g["hi_verts"], cam, z0=g["lo_top"],
                    color=ramp_brick(0.0), opacity=0.14, depth_bias=0.05),
        drop_shadow(g["cyl_rim"], cam, color=ramp_ochre(0.0), opacity=0.22),
        box_items(lo["center"], lo["size"], cam, ramp_brick,
                  side_grad_amp=amp, cap_grad_amp=amp, grain=p["grain"]),
        box_items(hi["center"], hi["size"], cam, ramp_indigo,
                  side_grad_amp=amp, cap_grad_amp=amp, grain=p["grain"]),
        cylinder_items(cyl["center"], cyl["radius"], cyl["height"], cam,
                       ramp_ochre, facets=p["facets"], bands=p["bands"],
                       cap_grad_amp=amp, grain=p["grain"]),
    )
    s = Scene()
    s.items.extend(items)
    return s


def assertions(g):
    p = g["params"]
    # the floating box really floats: its lowest vertex clears the floor
    assert g["hi_verts"][:, 2].min() > 0.5, "upper box does not float"
    # solids do not interpenetrate: box_lo top is below box_hi bottom
    lo_top = p["box_lo"]["center"][2] + p["box_lo"]["size"][2] / 2
    hi_bot = p["box_hi"]["center"][2] - p["box_hi"]["size"][2] / 2
    assert lo_top < hi_bot, "stacked boxes interpenetrate"
