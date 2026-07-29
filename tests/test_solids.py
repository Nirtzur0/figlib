"""Solids: backface culling, winding, gradients, banding."""

import numpy as np

from figlib.scene import FilledCurve
from figlib.shading import chroma_ramp
from figlib.solids import box_items, cylinder_items, extrude_items, face_item
from figlib.surface3d import Camera

CAM = Camera()                       # azim -35, elev 32
RAMP = chroma_ramp("#c0504d")

_TOP = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)


def _lambert_of(poly3):
    from figlib.solids import _lambert
    n = np.cross(poly3[1] - poly3[0], poly3[2] - poly3[0])
    return _lambert(n / np.linalg.norm(n))


def test_face_item_backface_culled():
    # +z-facing square is visible from the default camera (elev > 0)
    assert face_item(_TOP, CAM, RAMP) is not None
    # reversed winding -> -z normal -> backface
    assert face_item(_TOP[::-1], CAM, RAMP) is None


def test_face_item_carries_gradient():
    depth, fc = face_item(_TOP, CAM, RAMP)
    assert isinstance(fc, FilledCurve)
    assert fc.gradient is not None
    assert fc.opacity == 1.0
    # grad_amp=0 -> flat fill, no gradient
    _, flat = face_item(_TOP, CAM, RAMP, grad_amp=0.0)
    assert flat.gradient is None and flat.color == RAMP(_lambert_of(_TOP))


def test_box_visible_faces():
    items = box_items((0, 0, 0), (1, 1, 1), CAM, RAMP)
    # a box shows exactly 3 faces from a generic camera
    assert len(items) == 3
    assert all(isinstance(d, float) for d, _ in items)


def test_box_faces_differ_in_tone():
    items = box_items((0, 0, 0), (1, 1, 1), CAM, RAMP, side_grad_amp=0.0,
                      cap_grad_amp=0.0)
    colors = {fc.color for _, fc in items}
    assert len(colors) == 3          # three faces, three Lambert values


def test_cylinder_banding():
    items = cylinder_items((0, 0, 0), 1.0, 2.0, CAM, RAMP, facets=64, bands=4)
    sides = [fc for _, fc in items if len(fc.pts) == 4]
    caps = [fc for _, fc in items if len(fc.pts) > 4]
    assert len(caps) == 1            # top cap visible, bottom culled
    side_colors = {fc.color for fc in sides}
    assert 2 <= len(side_colors) <= 4


def test_extrude_nonconvex_runs():
    # L-shaped CCW polygon: painter's algorithm input stays well-formed
    L = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
    items = extrude_items(L, 0.0, 1.0, CAM, RAMP)
    assert len(items) >= 4
    for d, fc in items:
        assert fc.pts.shape[1] == 2
