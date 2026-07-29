"""drop_shadow: vertical projection of 3D geometry onto a ground plane,
emitted as a depth-tagged FilledCurve that composes behind the object."""

import numpy as np

from figlib.scene import FilledCurve
from figlib.surface3d import Camera, compose, drop_shadow, project, surface_items


def _paraboloid(n: int = 9):
    u = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(u, u)
    Z = 1.5 - 0.5 * (X ** 2 + Y ** 2)   # bowl floating above z=0
    return X, Y, Z


class TestDropShadow:
    def test_shadow_outline_lies_on_ground_plane(self):
        X, Y, Z = _paraboloid()
        cam = Camera()
        pts3 = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        [(depth, fc)] = drop_shadow(pts3, cam, z0=0.25)
        assert isinstance(fc, FilledCurve)
        # every outline vertex is the projection of some (x, y, z0) point:
        # reproject the footprint corners and check containment
        corners = np.array([[x, y, 0.25] for x in (-1, 1) for y in (-1, 1)])
        c2, _ = project(corners, cam)
        for corner in c2:
            d = np.hypot(fc.pts[:, 0] - corner[0], fc.pts[:, 1] - corner[1])
            assert d.min() < 1e-9, "footprint corner missing from shadow hull"

    def test_shadow_composes_behind_the_object(self):
        X, Y, Z = _paraboloid()
        cam = Camera()
        pts3 = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        shadow = drop_shadow(pts3, cam, z0=0.0)
        surf = surface_items(X, Y, Z, cam)
        ordered = compose(shadow, surf)
        assert isinstance(ordered[0], FilledCurve)
        assert ordered[0] is shadow[0][1], "shadow must paint first"

    def test_shadow_with_depth_bias_paints_over_its_carrier_plane(self):
        # a floating object's shadow cast onto another object's top face
        # must paint AFTER that face (else the face hides it) but BEFORE
        # the floating object itself
        cam = Camera()
        top = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
                       dtype=float)
        floater = np.array([[x, y, z] for x in (-0.4, 0.4)
                            for y in (-0.4, 0.4) for z in (2.0, 2.8)])
        from figlib.solids import face_item
        carrier = face_item(top, cam, lambda t: "#888888")
        shadow = drop_shadow(floater, cam, z0=1.0, depth_bias=0.05)
        d_face = carrier[0]
        d_shadow = shadow[0][0]
        assert d_face < d_shadow, "shadow must paint after its carrier face"

    def test_shadow_is_a_soft_flat_patch(self):
        X, Y, Z = _paraboloid()
        pts3 = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        [(_, fc)] = drop_shadow(pts3, Camera(), z0=0.0)
        assert fc.outline is False
        assert fc.opacity <= 0.3
