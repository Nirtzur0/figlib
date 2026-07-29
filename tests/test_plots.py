"""plots.py: scales, axis furniture, series, phase lines, honesty checks.

The recurring shape of these tests: an emitter is checked against the
analytic answer for the geometry it claims to draw, and every guard is
checked by making it fire.
"""

import numpy as np
import pytest

from figlib.gates import Checks, mechanical
from figlib.plots import (FixedPoint, Ticks, axis, band, flow_intervals,
                          histogram, linear, log10, markers, nice_step,
                          phase_line, px_units, series, tick_honesty)
from figlib.scene import Curve, FilledCurve, MathLabel, Scene, Vector
from figlib.style import DEFAULT_STYLE, Role


class TestScales:
    def test_linear_is_the_identity(self):
        sc = linear(-2.0, 3.0)
        np.testing.assert_allclose(sc.fwd([-2.0, 0.0, 3.0]), [-2.0, 0.0, 3.0])
        np.testing.assert_allclose(sc.inv(sc.fwd([1.5])), [1.5])
        assert sc.range == (-2.0, 3.0)
        assert sc.span == pytest.approx(5.0)

    def test_log10_sends_decades_to_equal_steps(self):
        sc = log10(1.0, 1e4)
        u = sc.fwd([1.0, 10.0, 100.0, 1000.0, 1e4])
        np.testing.assert_allclose(np.diff(u), 1.0)
        assert sc.range == pytest.approx((0.0, 4.0))
        np.testing.assert_allclose(sc.inv(u), [1.0, 10.0, 100.0, 1000.0, 1e4])

    def test_log10_refuses_nonpositive_bounds_and_data(self):
        with pytest.raises(ValueError, match="lo > 0"):
            log10(0.0, 10.0)
        with pytest.raises(ValueError, match="nonpositive"):
            log10(1.0, 10.0).fwd([1.0, 0.0, 5.0])

    def test_degenerate_range_refused(self):
        with pytest.raises(ValueError, match="lo < hi"):
            linear(1.0, 1.0)


class TestTicks:
    @pytest.mark.parametrize("span", [7.3, 0.09, 1234.0, 2.5, 1e-4])
    def test_nice_step_mantissa_and_coverage(self, span):
        h = nice_step(span, target=5)
        mant = h / 10 ** np.floor(np.log10(h))
        assert mant == pytest.approx(1.0) or mant == pytest.approx(2.0) \
            or mant == pytest.approx(5.0)
        # rounding the step up gives at most `target` intervals, and the
        # widest gap between consecutive clean steps is 2.5x
        assert 5.0 / 2.5 - 1e-9 <= span / h <= 5.0 + 1e-9

    def test_nice_step_rejects_nonpositive_span(self):
        with pytest.raises(ValueError):
            nice_step(0.0)

    def test_linear_ticks_are_inside_and_evenly_spaced(self):
        tk = linear(0.0, 7.3).ticks(target=5)
        assert tk.values.min() >= 0.0 and tk.values.max() <= 7.3
        d = np.diff(tk.values)
        np.testing.assert_allclose(d, d[0])
        assert 3 <= len(tk.values) <= 9
        np.testing.assert_allclose(tk.positions, tk.values)

    def test_linear_tick_labels_carry_the_step_s_decimals(self):
        assert linear(-1.5, 1.0).ticks(step=0.5).labels == (
            "-1.5", "-1.0", "-0.5", "0.0", "0.5", "1.0")
        assert linear(-2.0, 2.0).ticks(step=1.0).labels == (
            "-2", "-1", "0", "1", "2")

    def test_explicit_step_overrides_the_locator(self):
        tk = linear(0.0, 1.0).ticks(step=0.25)
        np.testing.assert_allclose(tk.values, [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_linear_minor_ticks_sit_between_majors(self):
        tk = linear(0.0, 2.0).ticks(step=0.5, minor=True)
        assert tk.minor.size and np.all((tk.minor >= 0.0) & (tk.minor <= 2.0))
        for m in tk.minor:
            assert np.min(np.abs(tk.values - m)) == pytest.approx(0.25)

    def test_log_ticks_are_decades_with_power_labels(self):
        tk = log10(1.0, 1000.0).ticks()
        np.testing.assert_allclose(tk.values, [1.0, 10.0, 100.0, 1000.0])
        np.testing.assert_allclose(tk.positions, [0.0, 1.0, 2.0, 3.0])
        assert tk.labels == ("10^{0}", "10^{1}", "10^{2}", "10^{3}")

    def test_log_ticks_ignore_partial_decades(self):
        tk = log10(3.0, 400.0).ticks()
        np.testing.assert_allclose(tk.values, [10.0, 100.0])

    def test_log_minor_ticks_are_the_2_to_9_multiples(self):
        tk = log10(1.0, 100.0).ticks(minor=True)
        assert tk.minor.size == 16                       # 2..9 in two decades
        np.testing.assert_allclose(tk.minor[0], np.log10(2.0))
        assert np.all((tk.minor >= 0.0) & (tk.minor <= 2.0))

    def test_log_ticks_refuse_a_step(self):
        with pytest.raises(ValueError, match="decades"):
            log10(1.0, 100.0).ticks(step=0.5)


class TestAxis:
    def test_emits_spine_ticks_and_labels(self):
        sc = linear(0.0, 4.0)
        tk = sc.ticks(step=1.0)
        items = axis(sc, orient="x", at=0.0, ticks=tk, tick_len=0.1)
        curves = [i for i in items if isinstance(i, Curve)]
        labels = [i for i in items if isinstance(i, MathLabel)]
        assert len(curves) == 1 + len(tk.values)          # spine + one per tick
        assert len(labels) == len(tk.values)
        np.testing.assert_allclose(curves[0].pts, [[0.0, 0.0], [4.0, 0.0]])
        assert curves[0].arrows == (1.0,)

    def test_axis_is_placeable_off_the_frame(self):
        items = axis(linear(-1.0, 1.0), orient="x", at=0.7, ticks=Ticks(
            np.array([0.0]), np.array([0.0]), ("0",)), tick_len=0.2)
        spine = items[0]
        assert np.allclose(spine.pts[:, 1], 0.7)
        tick = [i for i in items[1:] if isinstance(i, Curve)][0]
        np.testing.assert_allclose(tick.pts, [[0.0, 0.7], [0.0, 0.5]])

    def test_side_flips_ticks_and_label_anchoring(self):
        tk = Ticks(np.array([0.0]), np.array([0.0]), ("0",))
        up = axis(linear(-1.0, 1.0), orient="x", side=1, ticks=tk, tick_len=0.2)
        tick = [i for i in up[1:] if isinstance(i, Curve)][0]
        assert tick.pts[1, 1] == pytest.approx(0.2)
        lab = [i for i in up if isinstance(i, MathLabel)][0]
        assert lab.va == "bottom" and lab.offset_px[1] < 0

    def test_y_orientation_swaps_the_coordinates(self):
        tk = Ticks(np.array([1.0]), np.array([1.0]), ("1",))
        items = axis(linear(0.0, 2.0), orient="y", at=0.5, ticks=tk, tick_len=0.3)
        np.testing.assert_allclose(items[0].pts, [[0.5, 0.0], [0.5, 2.0]])
        tick = [i for i in items[1:] if isinstance(i, Curve)][0]
        np.testing.assert_allclose(tick.pts, [[0.5, 1.0], [0.2, 1.0]])
        lab = [i for i in items if isinstance(i, MathLabel)][0]
        assert lab.ha == "right" and lab.va == "center"

    def test_skip_suppresses_a_tick_label_but_not_its_tick(self):
        sc = linear(-1.0, 1.0)
        tk = sc.ticks(step=1.0)
        items = axis(sc, ticks=tk, tick_len=0.1, skip=(0.0,))
        labels = [i.latex for i in items if isinstance(i, MathLabel)]
        assert labels == ["-1", "1"]
        assert len([i for i in items if isinstance(i, Curve)]) == 1 + 3

    def test_extend_and_arrow_shape_the_spine(self):
        items = axis(linear(0.0, 1.0), ticks=Ticks(np.empty(0), np.empty(0), ()),
                     extend=(0.1, 0.2), arrow=False)
        np.testing.assert_allclose(items[0].pts, [[-0.1, 0.0], [1.2, 0.0]])
        assert items[0].arrows == ()

    def test_grid_lines_are_frame_ink(self):
        sc = linear(0.0, 2.0)
        items = axis(sc, ticks=sc.ticks(step=1.0), tick_len=0.1, grid_to=5.0)
        grid = [i for i in items if isinstance(i, Curve) and i.role is Role.FRAME]
        assert len(grid) == 3
        assert all(g.pts[1, 1] == 5.0 for g in grid)

    def test_end_and_centered_axis_labels(self):
        empty = Ticks(np.empty(0), np.empty(0), ())
        end = [i for i in axis(linear(0.0, 4.0), ticks=empty, label="t")
               if isinstance(i, MathLabel)][0]
        assert end.anchor == (4.0, 0.0) and end.ha == "left"
        mid = [i for i in axis(linear(0.0, 4.0), orient="y", ticks=empty,
                               label="L", label_at="center")
               if isinstance(i, MathLabel)][0]
        assert mid.anchor == (0.0, 2.0) and mid.angle_deg == 90.0

    @pytest.mark.parametrize("kw", [{"orient": "z"}, {"side": 0},
                                    {"label_at": "middle"}])
    def test_bad_arguments_refused(self, kw):
        with pytest.raises(ValueError):
            axis(linear(0.0, 1.0), **kw)


class TestSeriesAndMarkers:
    def test_series_applies_both_scales_once(self):
        x = np.array([1.0, 10.0, 100.0])
        y = np.array([2.0, 4.0, 8.0])
        c = series(x, y, xscale=log10(1.0, 100.0), yscale=log10(1.0, 10.0))[0]
        np.testing.assert_allclose(c.pts[:, 0], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(c.pts[:, 1], np.log10(y))

    def test_markers_are_an_identity_channel_orthogonal_to_hue(self):
        for shape, n in (("circle", 24), ("square", 4), ("diamond", 4),
                         ("triangle", 3)):
            it = markers([0.0], [0.0], shape, size=1.0)[0]
            assert isinstance(it, FilledCurve) and len(it.pts) == n

    def test_marker_shapes_carry_equal_area(self):
        def area(p):
            x, y = p[:, 0], p[:, 1]
            return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        areas = [area(markers([0.0], [0.0], s, size=1.0)[0].pts)
                 for s in ("circle", "square", "diamond", "triangle")]
        assert max(areas) / min(areas) < 1.06

    def test_hollow_markers_are_closed_curves_not_fills(self):
        it = markers([0.0], [0.0], "square", size=1.0, filled=False)[0]
        assert isinstance(it, Curve) and it.closed

    def test_marker_radii_may_differ_per_axis(self):
        pts = markers([2.0], [5.0], "diamond", size=(0.5, 3.0))[0].pts
        assert pts[:, 0].max() == pytest.approx(2.0 + 0.5 * 1.253)
        assert pts[:, 1].max() == pytest.approx(5.0 + 3.0 * 1.253)

    def test_series_markers_subsample_and_ride_a_role(self):
        x = np.arange(9.0)
        items = series(x, x, marker="square", marker_every=4, marker_size=0.1,
                       marker_role=Role.ACCENT1)
        marks = items[1:]
        assert len(marks) == 3
        assert all(m.role is Role.ACCENT1 for m in marks)

    @pytest.mark.parametrize("call", [
        lambda: series([1.0, 2.0], [1.0]),
        lambda: series([1.0], [1.0]),
        lambda: markers([0.0, 1.0], [0.0], "circle"),
        lambda: markers([0.0], [0.0], "hexagon"),
    ])
    def test_bad_series_input_refused(self, call):
        with pytest.raises(ValueError):
            call()


class TestBand:
    def test_band_walks_out_and_back(self):
        x = np.linspace(0.0, 1.0, 5)
        f = band(x, np.zeros(5), np.ones(5))
        assert isinstance(f, FilledCurve) and len(f.pts) == 10
        np.testing.assert_allclose(f.pts[:5, 0], f.pts[5:, 0][::-1])
        np.testing.assert_allclose(f.pts[:5, 1], 0.0)
        np.testing.assert_allclose(f.pts[5:, 1], 1.0)

    def test_band_applies_scales(self):
        x = np.array([1.0, 10.0, 100.0])
        f = band(x, np.ones(3), 2 * np.ones(3), xscale=log10(1.0, 100.0))
        np.testing.assert_allclose(f.pts[:3, 0], [0.0, 1.0, 2.0])

    def test_band_length_mismatch_refused(self):
        with pytest.raises(ValueError):
            band([0.0, 1.0], [0.0, 1.0], [0.0])


class TestHistogram:
    def test_step_outline_closes_to_the_baseline(self):
        counts = np.array([2.0, 5.0, 1.0])
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        f = histogram(counts, edges)[0]
        assert f.pts[0].tolist() == [0.0, 0.0] and f.pts[-1].tolist() == [3.0, 0.0]
        assert len(f.pts) == 2 * len(counts) + 2
        # each bin height appears twice: the step is flat across the bin
        for c in counts:
            assert np.sum(f.pts[:, 1] == c) == 2
        assert f.pts[:, 0].min() == 0.0 and f.pts[:, 0].max() == 3.0

    def test_per_bin_gives_one_polygon_each(self):
        bins = histogram([1.0, 2.0], [0.0, 1.0, 2.0], per_bin=True)
        assert len(bins) == 2
        assert bins[1].pts[:, 1].max() == 2.0

    def test_baseline_shifts_the_floor(self):
        f = histogram([3.0], [0.0, 1.0], baseline=-1.0)[0]
        assert f.pts[:, 1].min() == -1.0

    @pytest.mark.parametrize("call", [
        lambda: histogram([1.0, 2.0], [0.0, 1.0]),
        lambda: histogram([1.0, 2.0], [0.0, 1.0, 1.0]),
    ])
    def test_bad_binning_refused(self, call):
        with pytest.raises(ValueError):
            call()


def _f(r):
    return lambda x: r + np.asarray(x, dtype=float) ** 2


class TestPhaseLine:
    def test_flow_intervals_follow_the_sign_of_f(self):
        f = _f(-1.0)
        arrows = flow_intervals(f, [-1.0, 1.0], (-2.0, 2.0))
        assert len(arrows) == 3
        for a, b in arrows:
            mid = 0.5 * (a + b)
            assert np.sign(b - a) == np.sign(f(mid))

    def test_flow_arrows_never_straddle_a_fixed_point(self):
        arrows = flow_intervals(_f(-1.0), [-1.0, 1.0], (-2.0, 2.0), frac=1.0)
        for a, b in arrows:
            for x in (-1.0, 1.0):
                assert not (min(a, b) < x < max(a, b))

    def test_no_fixed_points_gives_one_arrow_over_the_whole_line(self):
        arrows = flow_intervals(_f(1.0), [], (-2.0, 2.0), frac=0.5)
        assert arrows.shape == (1, 2)
        assert arrows[0, 1] > arrows[0, 0]

    def test_fill_encodes_stability_and_the_spine_is_broken(self):
        f = _f(-1.0)
        fixed = [FixedPoint(-1.0, True), FixedPoint(1.0, False)]
        arrows = flow_intervals(f, [-1.0, 1.0], (-2.0, 2.0))
        items = phase_line(fixed, arrows, xlim=(-2.0, 2.0), dot=0.1)
        spine = [i for i in items if isinstance(i, Curve) and not i.closed]
        assert len(spine) == 3                 # one gap per fixed point
        assert all(np.allclose(s.pts[:, 1], 0.0) for s in spine)
        assert len([i for i in items if isinstance(i, Vector)]) == 3
        assert len([i for i in items if isinstance(i, FilledCurve)]) == 1  # stable
        assert len([i for i in items if isinstance(i, Curve) and i.closed]) == 1

    def test_vectors_point_the_way_the_intervals_do(self):
        arrows = flow_intervals(_f(-1.0), [-1.0, 1.0], (-2.0, 2.0))
        vecs = [i for i in phase_line([FixedPoint(-1.0, True),
                                       FixedPoint(1.0, False)], arrows,
                                      xlim=(-2.0, 2.0), dot=0.1)
                if isinstance(i, Vector)]
        for v, (a, b) in zip(vecs, arrows):
            assert np.sign(v.tip[0] - v.tail[0]) == np.sign(b - a)
        # the middle interval of xdot = x^2 - 1 flows LEFT, toward the sink
        assert vecs[1].tip[0] < vecs[1].tail[0]

    def test_half_stable_point_inks_the_attracting_half(self):
        arrows = flow_intervals(_f(0.0), [0.0], (-2.0, 2.0))
        items = phase_line([FixedPoint(0.0, None)], arrows, xlim=(-2.0, 2.0),
                           dot=0.1)
        half = [i for i in items if isinstance(i, FilledCurve)][0]
        # flow is rightward on both sides, so the left half attracts
        assert half.pts[:, 0].min() == pytest.approx(-0.1)
        assert half.pts[:, 0].max() == pytest.approx(0.0, abs=1e-12)
        assert any(isinstance(i, Curve) and i.closed for i in items), \
            "the half-stable dot still needs its full outline"

    def test_half_stable_flips_when_the_flow_does(self):
        items = phase_line([FixedPoint(0.0, None)],
                           flow_intervals(lambda x: -np.asarray(x) ** 2, [0.0],
                                          (-2.0, 2.0)),
                           xlim=(-2.0, 2.0), dot=0.1)
        half = [i for i in items if isinstance(i, FilledCurve)][0]
        assert half.pts[:, 0].max() == pytest.approx(0.1)


class TestTickHonesty:
    def _fails(self, **kw):
        c = Checks()
        tick_honesty(c, **kw)
        return c.failures

    def test_an_honest_axis_reports_nothing(self):
        sc = linear(-1.0, 1.0)
        assert self._fails(scale=sc, data=np.linspace(-0.95, 1.0, 50)) == []

    def test_data_outside_the_range_fires(self):
        out = self._fails(scale=linear(0.0, 1.0), data=[-0.2, 0.5, 1.4])
        assert len(out) == 2
        assert "below axis lo" in out[0] and "above axis hi" in out[1]

    def test_over_padded_axis_fires(self):
        out = self._fails(scale=linear(-10.0, 10.0), data=np.linspace(0.0, 1.0, 5))
        assert any("padding" in m for m in out)

    def test_stated_floor_can_admit_a_deliberately_empty_region(self):
        sc = linear(-1.5, 1.0)
        assert self._fails(scale=sc, data=np.linspace(-1.5, 0.0, 20),
                           min_span_frac=0.5) == []

    def test_log_axis_over_nonpositive_data_fires_without_raising(self):
        out = self._fails(scale=log10(1.0, 100.0), data=[0.0, 10.0])
        assert any("nonpositive" in m or "below axis lo" in m for m in out)

    def test_non_finite_data_fires(self):
        out = self._fails(scale=linear(0.0, 1.0), data=[0.0, np.nan, 1.0])
        assert any("non-finite" in m for m in out)

    def test_ticks_outside_the_range_and_too_few_ticks_fire(self):
        sc = linear(0.0, 1.0)
        bad = Ticks(np.array([0.0, 2.0]), np.array([0.0, 2.0]), ("0", "2"))
        assert any("outside" in m for m in
                   self._fails(scale=sc, data=[0.0, 1.0], ticks=bad))
        lone = Ticks(np.array([0.5]), np.array([0.5]), ("0.5",))
        assert any("tick" in m for m in
                   self._fails(scale=sc, data=[0.0, 1.0], ticks=lone))


class TestPixelBridge:
    def test_px_units_match_the_renderer_transform(self):
        from figlib.layout import Transform

        sc = Scene(xlim=(-2.0, 2.0), ylim=(-1.0, 1.0), height_px=200.0)
        upx, upy = px_units(sc, 400.0)
        t = Transform(sc, width_px=400.0)
        assert upx == pytest.approx(1.0 / t.scale_x)
        assert upy == pytest.approx(1.0 / t.scale_y)
        # a 10 px marker really is 10 px wide on the canvas
        m = markers([0.0], [0.0], "square", size=(10 * upx, 10 * upy))[0]
        w = t.to_canvas_arr(m.pts)[:, 0]
        assert (w.max() - w.min()) == pytest.approx(2 * 10 * 0.886)


class TestEndToEnd:
    def test_log_plot_renders_and_clears_the_mechanical_gate(self):
        from figlib.render import to_svg

        xs, ys = log10(1.0, 1e3), linear(0.0, 2.5)
        x = np.geomspace(1.0, 1e3, 60)
        y = 2.0 * x ** -0.05
        sc = Scene(xlim=(-0.35, 3.05), ylim=(-0.55, 2.6), height_px=380.0)
        sc.add(*axis(xs, orient="x", at=0.0, ticks=xs.ticks(minor=True),
                     tick_len=0.09, label="N"))
        sc.add(*axis(ys, orient="y", at=0.0, ticks=ys.ticks(step=1.0),
                     tick_len=0.09, skip=(0.0,), label=r"\text{loss}"))
        sc.add(band(x, y * 0.9, y * 1.1, xscale=xs, yscale=ys, role=Role.ACCENT1))
        sc.add(*series(x, y, xscale=xs, yscale=ys, role=Role.ACCENT1,
                       marker="triangle", marker_every=15, marker_size=0.05))
        assert "<svg" in to_svg(sc, DEFAULT_STYLE, width_px=680)
        assert mechanical(sc, DEFAULT_STYLE, width_px=680) == []

    def test_benchmark_figure_passes_every_deterministic_gate(self, tmp_path):
        from pathlib import Path

        from figlib.program import run

        prog = (Path(__file__).resolve().parents[1] / "figures"
                / "strogatz_saddle_node.py")
        report = run(prog, out_dir=tmp_path)
        assert report.passed, report.summary()
