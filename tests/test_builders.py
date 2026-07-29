"""Builders vs analytic answers: grids that must tile, contours that must
lie on their level sets, integral curves that must close."""

import numpy as np
import pytest

from figlib.builders import (integrate_streamlines, mapped_grid,
                             stagnation_points, stream_function_lines)
from figlib.scene import FilledCurve
from figlib.style import Role


class TestMappedGrid:
    def test_identity_reproduces_grid(self):
        g = mapped_grid(lambda z: z, u=[-1.0, 0.0, 1.0], v=[-2.0, 0.5, 2.0],
                        samples_per_line=50)
        for ui, c in zip([-1.0, 0.0, 1.0], g.u_curves):
            assert np.allclose(c.pts[:, 0], ui)
            assert c.pts[0, 1] == pytest.approx(-2.0)
            assert c.pts[-1, 1] == pytest.approx(2.0)
        for vj, c in zip([-2.0, 0.5, 2.0], g.v_curves):
            assert np.allclose(c.pts[:, 1], vj)
        assert all(c.role is Role.CONSTRUCTION for c in g.u_curves + g.v_curves)

    def test_z_squared_cell_contains_corner_images(self):
        u = np.linspace(0.5, 1.5, 4)
        v = np.linspace(0.2, 1.0, 4)
        g = mapped_grid(lambda z: z**2, u, v, samples_per_line=80)
        for i in range(3):
            for j in range(3):
                pts = g.cell(i, j)
                for zc in (u[i] + 1j * v[j], u[i + 1] + 1j * v[j],
                           u[i + 1] + 1j * v[j + 1], u[i] + 1j * v[j + 1]):
                    w = zc * zc
                    d = np.hypot(pts[:, 0] - w.real, pts[:, 1] - w.imag)
                    assert d.min() < 1e-12, f"corner image missing from cell ({i},{j})"

    def test_adjacent_cells_share_edge_samples_exactly(self):
        u = np.linspace(0.5, 1.5, 4)
        v = np.linspace(0.2, 1.0, 4)
        g = mapped_grid(lambda z: z**2, u, v, samples_per_line=80)
        m = 80 // 4
        left = {tuple(p) for p in g.cell(0, 0)}
        right = {tuple(p) for p in g.cell(1, 0)}
        # the shared edge u = u[1] is sampled with identical floats in both
        assert len(left & right) >= m

    def test_parity_checkerboard_count_and_items(self):
        g = mapped_grid(lambda z: z, np.arange(4.0), np.arange(4.0),
                        samples_per_line=20)
        black = g.cells(lambda i, j: (i + j) % 2 == 0)
        assert len(black) == 5   # 3x3 cells, checkerboard
        items = g.cell_items(lambda i, j: (i + j) % 2 == 0,
                             role=Role.CONTENT, opacity=0.3)
        assert len(items) == 5
        assert all(isinstance(it, FilledCurve) and it.opacity == 0.3 for it in items)


class TestStreamFunctionLines:
    def test_uniform_flow_levels_are_horizontal_and_point_downstream(self):
        # Psi = y: v = (dPsi/dy, -dPsi/dx) = (1, 0), streamlines y = const
        curves = stream_function_lines(lambda X, Y: Y, (-2, 2), (-1, 1),
                                       levels=[-0.5, 0.0, 0.5], n=101)
        assert len(curves) == 3
        for c, level in zip(curves, [-0.5, 0.0, 0.5]):
            assert np.max(np.abs(c.pts[:, 1] - level)) < 1e-9
            assert c.pts[-1, 0] > c.pts[0, 0], "not oriented with the flow"
            assert c.arrow_style == "hollow"

    def test_circular_level_set_is_closed_with_correct_radius(self):
        curves = stream_function_lines(lambda X, Y: X**2 + Y**2, (-2, 2), (-2, 2),
                                       levels=[1.0], n=201)
        assert len(curves) == 1
        c = curves[0]
        assert c.closed
        r = np.hypot(c.pts[:, 0], c.pts[:, 1])
        assert np.max(np.abs(r - 1.0)) < 1e-3
        # v = (2y, -2x) is clockwise: signed area must be negative
        x, y = c.pts[:, 0], c.pts[:, 1]
        area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        assert area < 0

    def test_mask_drops_interior_segments(self):
        curves = stream_function_lines(lambda X, Y: Y, (-2, 2), (-1, 1),
                                       levels=[0.0], n=201,
                                       mask=lambda X, Y: X**2 + Y**2 < 1.0)
        assert len(curves) == 2   # the y = 0 line broken by the disc
        for c in curves:
            assert np.max(np.abs(c.pts[:, 1])) < 1e-9
            assert np.min(np.hypot(c.pts[:, 0], c.pts[:, 1])) > 0.9


class TestIntegrateStreamlines:
    def test_rotation_field_closes_the_circle(self):
        field = lambda P: np.column_stack([-P[:, 1], P[:, 0]])
        [c] = integrate_streamlines(field, [(1.0, 0.0)], (-2, 2), (-2, 2),
                                    h=0.005, max_len=2 * np.pi)
        r = np.hypot(c.pts[:, 0], c.pts[:, 1])
        assert np.max(np.abs(r - 1.0)) < 1e-6
        # forward endpoint returns to the seed after arc length 2*pi
        assert np.hypot(c.pts[-1, 0] - 1.0, c.pts[-1, 1]) < 1e-2

    def test_stops_at_domain_exit(self):
        field = lambda P: np.column_stack([np.ones(len(P)), np.zeros(len(P))])
        [c] = integrate_streamlines(field, [(0.0, 0.0)], (-1, 1), (-1, 1),
                                    h=0.05, max_len=100.0)
        assert c.pts[-1, 0] <= 1.0 + 0.05
        assert c.pts[0, 0] >= -1.0 - 0.05


class TestStagnationPoints:
    def test_cylinder_flow_without_circulation(self):
        # W(z) = 1 - 1/z^2, v = conj(W): zeros exactly at z = +-1
        def field(P):
            z = P[:, 0] + 1j * P[:, 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                W = 1.0 - 1.0 / (z * z)
            return np.column_stack([W.real, -W.imag])

        pts = sorted(stagnation_points(field, (-2.5, 2.5), (-1.5, 1.5)))
        assert len(pts) == 2
        assert np.hypot(pts[0][0] + 1.0, pts[0][1]) < 1e-8
        assert np.hypot(pts[1][0] - 1.0, pts[1][1]) < 1e-8
