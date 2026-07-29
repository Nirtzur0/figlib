"""stems and impulses: the two discrete-sequence types (sample vs measure),
plus the cross marker that unlocks pole-zero planes."""

import numpy as np
import pytest

from figlib.plots import impulses, linear, log10, markers, stems
from figlib.scene import Curve, FilledCurve, Vector


class TestStems:
    def test_one_stem_and_one_marker_per_sample(self):
        items = stems([0, 1, 2], [1.0, 2.0, 0.5])
        curves = [it for it in items if isinstance(it, Curve) and not it.closed]
        heads = [it for it in items if isinstance(it, FilledCurve)]
        assert len(curves) == 3 and len(heads) == 3

    def test_stem_runs_baseline_to_sample(self):
        [stem] = [it for it in stems([2], [1.5]) if isinstance(it, Curve)]
        assert stem.pts[0] == pytest.approx([2.0, 0.0])
        assert stem.pts[-1] == pytest.approx([2.0, 1.5])

    def test_hollow_marker_stem_stops_short_of_the_rim(self):
        items = stems([0], [1.0], filled=False, size=0.1)
        [stem] = [it for it in items if isinstance(it, Curve) and not it.closed]
        assert stem.pts[-1][1] == pytest.approx(0.9)
        # and the head is drawn hollow: a closed Curve, not a FilledCurve
        assert any(isinstance(it, Curve) and it.closed for it in items)

    def test_negative_sample_stems_downward(self):
        items = stems([0], [-1.0], filled=False, size=0.1)
        [stem] = [it for it in items if isinstance(it, Curve) and not it.closed]
        assert stem.pts[-1][1] == pytest.approx(-0.9)

    def test_sample_shorter_than_the_gap_keeps_marker_drops_stem(self):
        items = stems([0], [0.05], filled=False, size=0.1)
        assert not any(isinstance(it, Curve) and not it.closed for it in items)
        assert any(isinstance(it, Curve) and it.closed for it in items)

    def test_marker_none_gives_bare_stems(self):
        items = stems([0, 1], [1.0, 2.0], marker=None)
        assert len(items) == 2
        assert all(isinstance(it, Curve) for it in items)

    def test_scales_applied_once_including_baseline(self):
        items = stems([10.0], [100.0], baseline=1.0, xscale=log10(1, 100),
                      yscale=log10(0.1, 1000), marker=None)
        [stem] = items
        assert stem.pts[0] == pytest.approx([1.0, 0.0])   # log10 of x, baseline
        assert stem.pts[-1] == pytest.approx([1.0, 2.0])  # log10 of y

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            stems([0, 1], [1.0])


class TestImpulses:
    def test_one_vector_per_impulse_height_is_weight(self):
        vs = impulses([0.0, 1.0], [2.0, 0.5])
        assert all(isinstance(v, Vector) for v in vs)
        assert vs[0].tail == pytest.approx((0.0, 0.0))
        assert vs[0].tip == pytest.approx((0.0, 2.0))
        assert vs[1].tip == pytest.approx((1.0, 0.5))

    def test_negative_weight_points_down(self):
        [v] = impulses([0.0], [-1.5], baseline=0.5)
        assert v.tail == pytest.approx((0.0, 0.5))
        assert v.tip == pytest.approx((0.0, -1.0))

    def test_scales_apply(self):
        [v] = impulses([10.0], [3.0], xscale=log10(1, 100))
        assert v.tail[0] == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            impulses([0.0], [1.0, 2.0])


class TestCrossMarker:
    def test_cross_is_two_open_segments_per_point(self):
        items = markers([0.0, 1.0], [0.0, 0.0], "cross", filled=False, size=0.1)
        assert len(items) == 4
        assert all(isinstance(it, Curve) and not it.closed for it in items)
        assert all(it.pts.shape == (2, 2) for it in items)

    def test_cross_spans_its_size(self):
        a, b = markers([0.0], [0.0], "cross", filled=False, size=0.1)
        for seg in (a, b):
            assert np.hypot(*(seg.pts[1] - seg.pts[0])) == pytest.approx(
                2 * 0.1, rel=0.3)

    def test_cross_cannot_be_filled(self):
        with pytest.raises(ValueError):
            markers([0.0], [0.0], "cross", filled=True)

    def test_stems_accept_cross_heads(self):
        items = stems([0], [1.0], marker="cross", filled=False, size=0.08)
        assert sum(isinstance(it, Curve) and not it.closed for it in items) >= 2
