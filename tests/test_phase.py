"""The cyclic phase channel, RGB rasters, and domain coloring.

A complex-valued function has no scalar to put on a ramp: its phase is
periodic (arg = -pi and arg = +pi are the same direction) and its modulus
runs over decades. `theme.phase(arg, mag)` is the channel for exactly that
pair, and `builders.domain_color` is the producer that turns f into pixels.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest

from figlib.builders import domain_color
from figlib.color import chroma, lightness, rgb_to_hex_arr, to_oklab
from figlib.render import to_svg_tree
from figlib.scene import RasterField, Scene
from figlib.style import DEFAULT_STYLE
from figlib.theme import CLEAN, PHASE_PERIOD_OCTAVES, RISO

THEMES = [("CLEAN", CLEAN), ("RISO", RISO)]
# one 8-bit sRGB code step is ~0.003 in OKLab L
L_QUANT = 0.004


def _hue_deg(rgb_row: np.ndarray) -> np.ndarray:
    """OKLab hue angle in degrees of each color in an (N, 3) sRGB row."""
    lab = np.array([to_oklab(h) for h in rgb_to_hex_arr(rgb_row)])
    return np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))


class TestPhaseChannelShape:
    @pytest.mark.parametrize("name,theme", THEMES)
    def test_it_is_vectorized_and_stays_inside_the_cube(self, name, theme):
        arg = np.linspace(-np.pi, np.pi, 17)[:, None] * np.ones((1, 11))
        mag = np.logspace(-3, 3, 11)[None, :] * np.ones((17, 1))
        rgb = theme.phase(arg, mag)
        assert rgb.shape == (17, 11, 3)
        assert np.isfinite(rgb).all()
        assert rgb.min() >= 0.0 and rgb.max() <= 1.0

    @pytest.mark.parametrize("name,theme", THEMES)
    def test_a_pole_pixel_gets_a_color_rather_than_a_hole(self, name, theme):
        """f(z) overflows to inf (and 0/0 to nan) exactly at the points a
        portrait is drawn to show, so those must not come back as NaN."""
        rgb = theme.phase(np.array([0.0, 0.0, np.nan]),
                          np.array([np.inf, 0.0, np.nan]))
        assert np.isfinite(rgb).all()
        assert rgb.min() >= 0.0 and rgb.max() <= 1.0


class TestPhaseIsCyclicAndEven:
    @pytest.mark.parametrize("name,theme", THEMES)
    def test_the_wrap_at_pi_is_seamless(self, name, theme):
        a = theme.phase(np.array([-np.pi]), np.array([1.0]))
        b = theme.phase(np.array([np.pi]), np.array([1.0]))
        assert np.abs(a - b).max() < 1.5 / 255

    @pytest.mark.parametrize("name,theme", THEMES)
    def test_equal_phase_steps_are_equal_perceived_hue_steps(self, name, theme):
        """The point of running the circle in OKLCh rather than HSV: a
        constant d(arg) must produce a constant d(hue). Under HSV it does
        not, and the portrait grows structure the function does not have."""
        arg = np.linspace(-np.pi, np.pi, 49)[:-1]
        h = np.unwrap(np.radians(_hue_deg(theme.phase(arg, np.ones_like(arg)))))
        step = np.degrees(np.diff(h))
        assert np.abs(step - step.mean()).max() < 1.0, np.round(step, 2)
        assert abs(abs(step.mean()) - 360 / 48) < 0.05

    @pytest.mark.parametrize("name,theme", THEMES)
    def test_lightness_is_flat_around_the_circle(self, name, theme):
        """Lightness is the modulus channel, so it must carry no phase
        signal: at fixed |f| the whole circle sits at one L."""
        arg = np.linspace(-np.pi, np.pi, 49)[:-1]
        L = np.array([lightness(h)
                      for h in rgb_to_hex_arr(theme.phase(arg, np.ones_like(arg)))])
        assert np.ptp(L) < 2 * L_QUANT, f"{name} L varies {np.ptp(L):.4f} with phase"

    @pytest.mark.parametrize("name,theme", THEMES)
    def test_the_chosen_chroma_is_in_gamut_across_the_whole_band(self, name, theme):
        """Gamut mapping reduces chroma hue-preservingly — graceful, but it
        would make chroma vary with modulus, i.e. put a second signal into a
        channel that is meant to carry none. Each theme's C is picked to
        clear the sRGB boundary everywhere the per-octave sawtooth reaches;
        only the trend excursions at zeros and poles may desaturate."""
        arg = np.linspace(-np.pi, np.pi, 37)[:-1][None, :]
        mag = (2.0 ** np.linspace(0.0, 1.0, 9))[:, None]      # one full octave
        rgb = theme.phase(arg * np.ones_like(mag), mag * np.ones_like(arg))
        C = np.array([chroma(h) for h in rgb_to_hex_arr(rgb).ravel()])
        assert C.min() > theme.phase_chroma - 0.005, (
            f"{name} chroma dips to {C.min():.3f} vs {theme.phase_chroma}")


class TestModulusRidesLightness:
    @pytest.mark.parametrize("name,theme", THEMES)
    def test_one_sawtooth_period_is_one_octave(self, name, theme):
        """Count the lightness drops between |f| = 1 and |f| = 2^6: one per
        doubling, i.e. the modulus contours are the level sets |f| = 2^k."""
        octaves = 6
        mag = np.logspace(0, octaves * np.log10(2.0), 4001)
        L = np.array([lightness(h)
                      for h in rgb_to_hex_arr(theme.phase(np.zeros_like(mag), mag))])
        drops = int(np.sum(np.diff(L) < -theme.phase_band / 3))
        assert drops == int(octaves / PHASE_PERIOD_OCTAVES), (
            f"{name}: {drops} contours over {octaves} octaves")

    @pytest.mark.parametrize("name,theme", THEMES)
    def test_zeros_and_poles_sit_at_opposite_ends_of_lightness(self, name, theme):
        """The sawtooth alone is periodic, so a zero and a pole would look
        the same up close. The saturating trend is what separates them."""
        L = [lightness(h) for h in
             rgb_to_hex_arr(theme.phase(np.zeros(3), np.array([1e-9, 1.0, 1e9])))]
        assert L[0] < L[1] < L[2]
        assert L[2] - L[0] > 1.2 * theme.phase_trend

    def test_the_two_themes_do_not_share_a_phase_palette(self):
        arg = np.linspace(-np.pi, np.pi, 13)
        a = _hue_deg(CLEAN.phase(arg, np.ones_like(arg)))
        b = _hue_deg(RISO.phase(arg, np.ones_like(arg)))
        assert np.abs(((a - b + 180) % 360) - 180).min() > 60


class TestRgbRasterField:
    """RasterField gains an (H, W, 3) mode: a field whose value IS a color
    (the phase channel), so there is no scalar for a ramp to consume."""

    def _pixels(self, fld: RasterField) -> np.ndarray:
        from PIL import Image
        scene = Scene(items=[fld], xlim=(0.0, 4.0), ylim=(0.0, 4.0))
        root, _ = to_svg_tree(scene, DEFAULT_STYLE, width_px=400)
        img = next(e for e in root.iter()
                   if e.tag == "image" and e.get("preserveAspectRatio") == "none")
        data = base64.b64decode(img.get("href").split(",", 1)[1])
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))

    def test_rgb_values_reach_the_png_unchanged(self):
        rgb = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        [[0.0, 0.0, 1.0], [0.2, 0.4, 0.6]]])
        px = self._pixels(RasterField(rgb, extent=(0.0, 2.0, 0.0, 2.0)))
        k = px.shape[0] // 2
        assert px.shape == (2 * k, 2 * k, 4)
        assert tuple(px[0, 0, :3]) == (255, 0, 0)
        assert tuple(px[0, -1, :3]) == (0, 255, 0)
        assert tuple(px[-1, 0, :3]) == (0, 0, 255)
        assert tuple(px[-1, -1, :3]) == (51, 102, 153)
        assert px[0, 0, 3] == 255

    def test_out_of_range_rgb_is_clipped_not_wrapped(self):
        rgb = np.array([[[1.4, -0.2, 0.5]]])
        px = self._pixels(RasterField(rgb, extent=(0.0, 1.0, 0.0, 1.0)))
        assert tuple(px[0, 0, :3]) == (255, 0, 128)

    def test_the_scalar_mode_still_goes_through_the_ramp(self):
        fld = RasterField(np.array([[0.0, 1.0]]), extent=(0.0, 2.0, 0.0, 1.0),
                          ramp=lambda u: f"#{round(255 * u):02x}0000")
        px = self._pixels(fld)
        assert px[0, 0, 0] == 0 and px[0, -1, 0] == 255


def _f(z):
    """The benchmark function: one zero at z = 1, two poles at the primitive
    cube roots of unity."""
    with np.errstate(all="ignore"):
        return (z - 1.0) / (z * z + z + 1.0)


class TestDomainColor:
    def test_it_produces_an_rgb_raster_over_the_requested_window(self):
        fld = domain_color(_f, (-2.0, 2.0), (-2.0, 2.0), 64, RISO)
        assert isinstance(fld, RasterField)
        assert fld.values.shape == (64, 64, 3)
        assert fld.extent == (-2.0, 2.0, -2.0, 2.0)
        assert np.isfinite(fld.values).all()

    def test_row_zero_is_the_top_of_the_window(self):
        """RasterField draws row 0 at y = y1, so the grid must be built with
        y descending or the portrait renders mirrored."""
        fld = domain_color(lambda z: z, (-2.0, 2.0), (-2.0, 2.0), 5, CLEAN)
        want = CLEAN.phase(np.angle(-2.0 + 2.0j), abs(-2.0 + 2.0j))
        assert np.allclose(fld.values[0, 0], want)

    def test_it_is_exactly_the_theme_channel_applied_to_f(self):
        """A pure producer: no styling of its own, so the pixels are
        reproducible from the theme alone."""
        n = 17
        fld = domain_color(_f, (-2.0, 2.0), (-1.0, 1.0), n, RISO)
        xs = np.linspace(-2.0, 2.0, n)
        ys = np.linspace(1.0, -1.0, n)
        w = _f(xs[None, :] + 1j * ys[:, None])
        assert np.allclose(fld.values, RISO.phase(np.angle(w), np.abs(w)))

    def test_the_phase_winds_once_around_the_zero_and_once_around_a_pole(self):
        """The claim a domain-coloring figure makes: a full hue wheel around
        z = 1 (winding +1) and around each pole (winding -1)."""
        for center, want in ((1.0 + 0.0j, +1), (np.exp(2j * np.pi / 3), -1)):
            th = np.linspace(0.0, 2 * np.pi, 400, endpoint=False)
            w = _f(center + 0.05 * np.exp(1j * th))
            turns = np.sum(np.diff(np.unwrap(np.angle(w)))) / (2 * np.pi)
            assert round(turns) == want, (center, turns)
            h = np.unwrap(np.radians(_hue_deg(RISO.phase(np.angle(w), np.abs(w)))))
            assert round((h[-1] - h[0]) / (2 * np.pi)) == want

    def test_the_zero_is_dark_and_the_poles_are_light(self):
        fld = domain_color(_f, (-2.0, 2.0), (-2.0, 2.0), 401, CLEAN)
        L = np.array([lightness(h) for h in rgb_to_hex_arr(fld.values).ravel()
                      ]).reshape(401, 401)

        def at(z):    # nearest grid cell to a complex point
            col = int(round((z.real + 2.0) / 4.0 * 400))
            row = int(round((2.0 - z.imag) / 4.0 * 400))
            return L[row, col]

        assert at(1.0 + 0j) < CLEAN.phase_lightness - CLEAN.phase_trend / 2
        for pole in (np.exp(2j * np.pi / 3), np.exp(-2j * np.pi / 3)):
            assert at(pole) > CLEAN.phase_lightness + CLEAN.phase_trend / 2
