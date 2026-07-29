"""Sphere visibility, occlusion, and wireframe hidden-line removal."""

import numpy as np
import pytest

from figlib.sphere3d import (Sphere, anchor, circle_on_sphere, circle_points,
                             disc, drape, occlude, point_on, silhouette,
                             visibility_runs)
from figlib.surface3d import Camera, _depth_buffer, project, wireframe_items

CAM = Camera(azim_deg=-35.0, elev_deg=30.0)
UNIT = Sphere(np.zeros(3), 1.0)


def run_length(run: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(run, axis=0), axis=1).sum())


def visible_fraction(sphere: Sphere, offset: float, cam: Camera) -> float:
    pts = circle_points(sphere, np.array([0.0, 0.0, 1.0]), offset, n=4096)
    runs = visibility_runs(pts, sphere, cam)
    total = sum(run_length(r) for _, r in runs)
    vis = sum(run_length(r) for v, r in runs if v)
    return vis / total


class TestVisibleFraction:
    def test_equator_is_half_visible(self):
        # a great circle is bisected by the limb regardless of elevation
        assert visible_fraction(UNIT, 0.0, CAM) == pytest.approx(0.5, rel=0.02)

    def test_latitude_matches_exact_formula(self):
        # circle at height h: visible iff cos(theta - azim) > -h tan(e)/rho
        h, e = 0.5, np.radians(30.0)
        rho = np.sqrt(1.0 - h * h)
        exact = np.arccos(-h * np.tan(e) / rho) / np.pi
        assert visible_fraction(UNIT, h, CAM) == pytest.approx(exact, rel=0.02)


class TestDotProductInvariant:
    def test_runs_respect_signed_distance(self):
        # crossings are interpolated on a linear signed field: exact zeros
        _, _, toward = CAM.axes()
        pts = circle_points(UNIT, np.array([1.0, 1.0, 1.0]), 0.3, n=512)
        for visible, run in visibility_runs(pts, UNIT, CAM):
            signed = (run - UNIT.center) @ toward
            if visible:
                assert signed.min() > -1e-9
            else:
                assert signed.max() < 1e-9


class TestOcclude:
    def _behind_center_segment(self) -> np.ndarray:
        right, _, toward = CAM.axes()
        return np.array([-2.0 * toward - 3.0 * right, -2.0 * toward + 3.0 * right])

    def test_segment_behind_sphere_splits_2_visible_1_hidden(self):
        items = occlude(self._behind_center_segment(), UNIT, CAM, hidden="dashed")
        solid = [c for _, c in items if c.dash is None]
        dashed = [c for _, c in items if c.dash == "dashed"]
        assert len(solid) == 2 and len(dashed) == 1

    def test_hidden_none_drops_the_middle(self):
        items = occlude(self._behind_center_segment(), UNIT, CAM, hidden=None)
        assert len(items) == 2
        assert all(c.dash is None for _, c in items)

    def test_segment_in_front_stays_whole(self):
        right, _, toward = CAM.axes()
        seg = np.array([2.0 * toward - 3.0 * right, 2.0 * toward + 3.0 * right])
        items = occlude(seg, UNIT, CAM, hidden="dashed")
        assert len(items) == 1 and items[0][1].dash is None


class TestWireframe:
    def _paraboloid(self):
        xs = np.linspace(-1.0, 1.0, 41)
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        return X, Y, X**2 + Y**2

    def test_visible_segments_respect_zbuffer(self):
        X, Y, Z = self._paraboloid()
        tol, bias = 0.05, 0.04
        items = wireframe_items(X, Y, Z, CAM, stride=(4, 4), hidden=None,
                                tol=tol, depth_bias=bias)
        db = _depth_buffer(X, Y, Z, CAM)
        mids = np.array([c.pts.mean(axis=0) for _, c in items])
        depths = np.array([tag - bias for tag, _ in items])
        assert np.all(depths >= db.lookup(mids) - tol - 1e-9)

    def test_bowl_rim_hides_far_interior(self):
        # max slope 2 exceeds tan(30 deg): the near rim must occlude something
        X, Y, Z = self._paraboloid()
        items = wireframe_items(X, Y, Z, CAM, stride=(4, 4), hidden="dashed")
        assert any(c.dash == "dashed" for _, c in items)
        assert any(c.dash is None for _, c in items)


class TestDrapeCircleEquivalence:
    def test_drape_matches_circle_on_sphere_great_circle(self):
        normal = np.array([0.3, -0.5, 0.8])
        a = circle_on_sphere(UNIT, normal, 0.0, CAM, n=256)
        b = drape(circle_points(UNIT, normal, 0.0, n=256), UNIT, CAM)
        assert len(a) == len(b)
        for (da, ca), (db_, cb) in zip(a, b):
            assert da == pytest.approx(db_)
            assert np.allclose(ca.pts, cb.pts)
            assert ca.dash == cb.dash


class TestSilhouetteDiscAnchors:
    def test_limb_is_a_circle_of_the_sphere_radius(self):
        (tag, curve), = silhouette(UNIT, CAM)
        c2, cd = project(UNIT.center[None, :], CAM)
        r = np.linalg.norm(curve.pts - c2[0], axis=1)
        assert np.allclose(r, 1.0)
        assert tag > float(cd[0]) + 1.0  # nearer than the sphere front

    def test_disc_sits_behind_hidden_arcs(self):
        (disc_tag, _), = disc(UNIT, CAM)
        items = circle_on_sphere(UNIT, np.array([0.0, 0.0, 1.0]), 0.0, CAM)
        hidden_tags = [t for t, c in items if c.dash is not None]
        assert hidden_tags and min(hidden_tags) > disc_tag

    def test_point_on_and_anchor_agree_with_project(self):
        p = point_on(UNIT, np.array([1.0, 2.0, 2.0]))
        assert np.linalg.norm(p) == pytest.approx(1.0)
        p2, _ = project(p[None, :], CAM)
        assert anchor(p, CAM) == pytest.approx(tuple(p2[0]))


class TestCustomVisibilityScalar:
    """`drape(signed=...)` is the public seam for hiddenness that is not
    just facing-away — a plane cut, a second occluder — replacing figure
    programs reaching into the private split/emit helpers."""

    def test_default_signed_reproduces_the_dot_test(self):
        pts = circle_points(UNIT, np.array([0.0, 0.0, 1.0]), 0.3, n=257)
        _, _, toward = CAM.axes()
        signed = (pts - UNIT.center) @ toward
        a = drape(pts, UNIT, CAM, tol=0.0)
        b = drape(pts, UNIT, CAM, tol=0.0, signed=signed)
        assert len(a) == len(b)
        for (da, ca), (db, cb) in zip(a, b):
            assert da == pytest.approx(db)
            assert np.allclose(ca.pts, cb.pts)
            assert ca.dash == cb.dash

    def test_a_second_test_hides_more(self):
        # equator, additionally cut by the plane z >= 0 -- which the whole
        # equator sits ON, so min() with z leaves it entirely on the boundary
        # and a strict-positive cut hides everything the dot test kept.
        pts = circle_points(UNIT, np.array([0.0, 1.0, 0.0]), 0.0, n=257)
        _, _, toward = CAM.axes()
        facing = (pts - UNIT.center) @ toward
        both = np.minimum(facing, pts[:, 2])
        vis_dot = [c for _, c in drape(pts, UNIT, CAM, tol=0.0) if c.dash is None]
        vis_cut = [c for _, c in drape(pts, UNIT, CAM, tol=0.0, signed=both)
                   if c.dash is None]
        length = lambda cs: sum(  # noqa: E731
            float(np.linalg.norm(np.diff(c.pts, axis=0), axis=1).sum()) for c in cs)
        assert length(vis_cut) < length(vis_dot)
