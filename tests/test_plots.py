"""plots.py: axes and series ON scene primitives — transforms, tick
locators, frame furniture, series emitters. No second rendering path."""

import numpy as np
import pytest

from figlib.plots import Axes, nice_ticks
from figlib.scene import Curve, FilledCurve, MathLabel, Point
from figlib.style import Role


class TestTransforms:
    def test_linear_is_identity(self):
        ax = Axes(xlim=(0.0, 10.0), ylim=(-1.0, 1.0))
        pts = ax.to_scene(np.array([0.0, 5.0]), np.array([-1.0, 1.0]))
        np.testing.assert_allclose(pts, [[0.0, -1.0], [5.0, 1.0]])

    def test_log_maps_decades_to_equal_steps(self):
        ax = Axes(xlim=(1.0, 1e3), ylim=(0.0, 1.0), xscale="log")
        pts = ax.to_scene(np.array([1.0, 10.0, 100.0, 1000.0]), np.zeros(4))
        steps = np.diff(pts[:, 0])
        np.testing.assert_allclose(steps, steps[0])

    def test_scene_lims_are_transformed(self):
        ax = Axes(xlim=(1.0, 1e4), ylim=(0.0, 2.0), xscale="log")
        assert ax.scene_xlim == pytest.approx((0.0, 4.0))
        assert ax.scene_ylim == (0.0, 2.0)


class TestTickLocators:
    def test_nice_ticks_use_125_steps_and_cover_range(self):
        ticks = nice_ticks(0.0, 7.3)
        step = ticks[1] - ticks[0]
        mantissa = step / 10 ** np.floor(np.log10(step))
        assert mantissa in (1.0, 2.0, 5.0)
        assert 3 <= len(ticks) <= 9
        assert ticks[0] >= 0.0 and ticks[-1] <= 7.3

    def test_nice_ticks_small_range(self):
        ticks = nice_ticks(0.02, 0.11)
        assert 3 <= len(ticks) <= 9
        assert all(0.02 <= t <= 0.11 for t in ticks)

    def test_log_axis_ticks_are_decades(self):
        ax = Axes(xlim=(1.0, 1e3), ylim=(0.0, 1.0), xscale="log")
        np.testing.assert_allclose(ax.xticks(), [1.0, 10.0, 100.0, 1000.0])


class TestFrame:
    def _ax(self) -> Axes:
        return Axes(xlim=(0.0, 4.0), ylim=(0.0, 2.0))

    def test_frame_emits_spines_ticks_and_labels(self):
        items = self._ax().frame()
        curves = [i for i in items if isinstance(i, Curve)]
        labels = [i for i in items if isinstance(i, MathLabel)]
        n_ticks = len(self._ax().xticks()) + len(self._ax().yticks())
        assert len(curves) == 2 + n_ticks          # 2 spines + one per tick
        assert len(labels) == n_ticks
        assert all(c.role in (Role.ANNOTATION, Role.FRAME) for c in curves)

    def test_log_tick_labels_are_powers_of_ten(self):
        ax = Axes(xlim=(1.0, 100.0), ylim=(0.0, 1.0), xscale="log")
        latexes = [i.latex for i in ax.frame() if isinstance(i, MathLabel)]
        assert "10^{0}" in latexes and "10^{2}" in latexes

    def test_gridlines_optional_and_frame_role(self):
        base = self._ax().frame()
        with_grid = self._ax().frame(grid=True)
        extra = [i for i in with_grid if isinstance(i, Curve)
                 and i.role == Role.FRAME]
        assert len(with_grid) > len(base)
        assert extra, "gridlines must carry the FRAME role"


class TestSeries:
    def _ax(self) -> Axes:
        return Axes(xlim=(1.0, 1e2), ylim=(0.0, 4.0), xscale="log")

    def test_line_transforms_data(self):
        x = np.array([1.0, 10.0, 100.0])
        y = np.array([0.0, 1.0, 2.0])
        c = self._ax().line(x, y)
        assert isinstance(c, Curve)
        np.testing.assert_allclose(c.pts[:, 0], [0.0, 1.0, 2.0])

    def test_band_is_closed_loop_of_both_edges(self):
        x = np.linspace(1.0, 100.0, 7)
        f = self._ax().band(x, np.zeros(7), np.ones(7))
        assert isinstance(f, FilledCurve)
        assert len(f.pts) == 14
        # top edge forward, bottom edge reversed
        np.testing.assert_allclose(f.pts[:7, 0], f.pts[7:, 0][::-1])

    def test_scatter_emits_points(self):
        pts = self._ax().scatter(np.array([1.0, 10.0]), np.array([1.0, 2.0]))
        assert all(isinstance(p, Point) for p in pts)
        assert len(pts) == 2

    def test_hist_step_outline_covers_bins(self):
        ax = Axes(xlim=(0.0, 1.0), ylim=(0.0, 3.0))
        rng = np.random.default_rng(0)
        f = ax.hist(rng.random(500), bins=10, density=True)
        assert isinstance(f, FilledCurve)
        assert f.pts[:, 1].max() > 0.5
        assert f.pts[:, 0].min() >= 0.0 and f.pts[:, 0].max() <= 1.0


class TestAxisTitles:
    def test_xlabel_centered_below_axis(self):
        ax = Axes(xlim=(0.0, 4.0), ylim=(0.0, 2.0))
        lab = ax.xlabel(r"N\ \text{params}")
        assert isinstance(lab, MathLabel)
        assert lab.anchor[0] == pytest.approx(2.0)
        assert lab.anchor[1] == pytest.approx(0.0)
        assert lab.va == "top" and lab.offset_px[1] > 10.0

    def test_ylabel_rotated_along_axis(self):
        ax = Axes(xlim=(0.0, 4.0), ylim=(0.0, 2.0))
        lab = ax.ylabel(r"\text{loss}")
        assert lab.angle_deg == 90.0
        assert lab.anchor[1] == pytest.approx(1.0)
        assert lab.offset_px[0] < -10.0


class TestEndToEnd:
    def test_renders_and_passes_mechanical_gate(self):
        from figlib.gates import mechanical
        from figlib.render import to_svg
        from figlib.scene import Scene
        from figlib.style import DEFAULT_STYLE

        ax = Axes(xlim=(1.0, 1e3), ylim=(0.0, 2.0), xscale="log")
        x = np.geomspace(1.0, 1e3, 50)
        y = 2.0 * x ** -0.1
        sc = Scene(xlim=ax.scene_xlim, ylim=ax.scene_ylim, height_px=420)
        sc.add(*ax.frame())
        sc.add(ax.band(x, y * 0.9, y * 1.1, role=Role.ACCENT1))
        sc.add(ax.line(x, y, role=Role.ACCENT1))
        svg = to_svg(sc, DEFAULT_STYLE, width_px=680)
        assert "<svg" in svg
        diags = mechanical(sc, DEFAULT_STYLE, width_px=680)
        assert [d for d in diags if d.kind == "label-collision"] == []
