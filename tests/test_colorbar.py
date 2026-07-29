"""colorbar: a ramp channel finally gets a scale. Pure producer — slabs,
box, ticks, labels as ordinary items the call site places."""

import numpy as np
import pytest

from figlib.plots import colorbar, linear, log10
from figlib.scene import Curve, FilledCurve, MathLabel


def _gray(t: float) -> str:
    v = int(round(255 * t))
    return f"#{v:02x}{v:02x}{v:02x}"


def _slabs(items):
    return [it for it in items if isinstance(it, FilledCurve)]


class TestColorbar:
    def test_n_slabs_tile_the_length_exactly(self):
        items = colorbar(linear(0, 1), _gray, at=(2.0, 3.0), length=1.0,
                         thickness=0.1, n=8)
        slabs = _slabs(items)
        assert len(slabs) == 8
        lo = min(s.pts[:, 1].min() for s in slabs)
        hi = max(s.pts[:, 1].max() for s in slabs)
        assert (lo, hi) == pytest.approx((3.0, 4.0))
        for s in slabs:
            assert s.pts[:, 1].max() - s.pts[:, 1].min() == pytest.approx(1 / 8)
            assert (s.pts[:, 0].min(), s.pts[:, 0].max()) == pytest.approx((2.0, 2.1))

    def test_slab_colors_sample_ramp_centers_low_to_high(self):
        items = colorbar(linear(0, 1), _gray, at=(0, 0), length=1.0,
                         thickness=0.1, n=4)
        slabs = sorted(_slabs(items), key=lambda s: s.pts[:, 1].min())
        assert [s.color for s in slabs] == [_gray((k + 0.5) / 4) for k in range(4)]

    def test_slabs_are_opaque_seamless_ink(self):
        for s in _slabs(colorbar(linear(0, 1), _gray, at=(0, 0), length=1,
                                 thickness=0.1)):
            assert s.opacity == 1.0 and s.outline is False

    def test_ticks_land_at_scale_positions_mapped_into_the_strip(self):
        items = colorbar(log10(1, 100), _gray, at=(0.0, 0.0), length=2.0,
                         thickness=0.2)
        ticks = [it for it in items if isinstance(it, Curve) and not it.closed
                 and it.pts.shape == (2, 2)]
        ys = sorted({round(float(t.pts[0, 1]), 9) for t in ticks})
        assert ys == pytest.approx([0.0, 1.0, 2.0])   # decades 1, 10, 100
        labels = [it for it in items if isinstance(it, MathLabel)]
        assert len(labels) >= 3

    def test_horizontal_orient_transposes(self):
        items = colorbar(linear(0, 1), _gray, at=(1.0, 5.0), length=1.0,
                         thickness=0.1, orient="x", n=4)
        slabs = _slabs(items)
        lo = min(s.pts[:, 0].min() for s in slabs)
        hi = max(s.pts[:, 0].max() for s in slabs)
        assert (lo, hi) == pytest.approx((1.0, 2.0))
        assert all((s.pts[:, 1].min(), s.pts[:, 1].max())
                   == pytest.approx((5.0, 5.1)) for s in slabs)

    def test_label_emitted_when_given(self):
        items = colorbar(linear(0, 1), _gray, at=(0, 0), length=1,
                         thickness=0.1, label=r"|X(\omega)|")
        assert any(isinstance(it, MathLabel) and it.latex == r"|X(\omega)|"
                   for it in items)

    def test_bad_orient_raises(self):
        with pytest.raises(ValueError):
            colorbar(linear(0, 1), _gray, at=(0, 0), length=1, thickness=0.1,
                     orient="z")
