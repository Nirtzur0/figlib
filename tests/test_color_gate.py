"""The color channel is measurable, so it is gated rather than eyeballed.

Two levels: the themes' palettes must be sound in themselves, and a scene
must not put invisible ink or colliding correspondence hues on the page.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from figlib.color import (chroma, composite, contrast, delta_e, is_monotone,
                          lightness, simulate, worst_delta_e)
from figlib.gates import (MIN_CORRESPONDENCE_DELTA_E, MIN_STROKE_CONTRAST,
                          color_gate)
from figlib.scene import Curve, Scene
from figlib.style import Role
from figlib.theme import CLEAN, RISO, CORRESPONDENCE, Hue

THEMES = [("CLEAN", CLEAN), ("RISO", RISO)]


def _line(color=None, role=Role.CONTENT, opacity=1.0):
    pts = np.column_stack([np.linspace(0.0, 1.0, 5), np.zeros(5)])
    return Curve(pts, role=role, color=color, opacity=opacity)


def _scene(*items):
    return Scene(items=list(items), xlim=(0.0, 1.0), ylim=(-0.5, 0.5))


class TestColorimetry:
    def test_contrast_matches_wcag_anchors(self):
        assert contrast("#000000", "#ffffff") == pytest.approx(21.0, abs=1e-6)
        assert contrast("#ffffff", "#ffffff") == pytest.approx(1.0, abs=1e-6)
        # symmetric in its arguments
        assert contrast("#3467a3", "#ffffff") == pytest.approx(
            contrast("#ffffff", "#3467a3"))

    def test_composite_moves_ink_toward_the_paper(self):
        assert composite("#000000", 1.0, "#ffffff") == "#000000"
        assert composite("#000000", 0.0, "#ffffff") == "#ffffff"
        half = composite("#000000", 0.5, "#ffffff")
        assert contrast(half, "#ffffff") < contrast("#000000", "#ffffff")

    def test_deuteranopia_collapses_the_red_green_pair(self):
        """The simulation is load-bearing, so assert it actually does something:
        red and green separate under normal vision and merge under deutan."""
        assert delta_e("#c8402f", "#3f8a3f") > 20
        assert delta_e("#c8402f", "#3f8a3f", "deutan") < 12
        assert simulate("#c8402f", "deutan") != "#c8402f"

    def test_monotonicity_detects_a_reversal(self):
        assert is_monotone(["#eeeeee", "#888888", "#111111"])[0]
        ok, why = is_monotone(["#eeeeee", "#111111", "#888888"])
        assert not ok and "reverses" in why


class TestThemePalettes:
    @pytest.mark.parametrize("name,theme", THEMES)
    def test_every_categorical_slot_is_visible_on_its_paper(self, name, theme):
        for i, c in enumerate(theme.categorical_stops):
            worst = min(contrast(c, p) for p in theme.paper_stops())
            assert worst >= MIN_STROKE_CONTRAST, (
                f"{name} slot {i} ({c}) is {worst:.2f}:1 on paper")

    @pytest.mark.parametrize("name,theme", THEMES)
    def test_slots_within_the_cap_are_pairwise_separable(self, name, theme):
        """All-pairs, not adjacent: any two curves in a plane figure can end
        up side by side, so neighbours-only would hide a real collapse."""
        cols = theme.categorical_stops[:theme.correspondence_cap]
        assert len(cols) >= 2
        for a, b in itertools.combinations(cols, 2):
            d, kind = worst_delta_e(a, b)
            assert d >= MIN_CORRESPONDENCE_DELTA_E, (
                f"{name}: {a} vs {b} collapse to ΔE {d:.1f} under {kind}")

    @pytest.mark.parametrize("name,theme", THEMES)
    def test_the_cap_is_honest(self, name, theme):
        """One slot past the cap must actually fail — otherwise the cap is
        set too low and is costing hues the palette really has."""
        cols = theme.categorical_stops[:theme.correspondence_cap + 1]
        if len(cols) <= theme.correspondence_cap:
            pytest.skip("no slot beyond the cap")
        worst = min(worst_delta_e(a, b)[0] for a, b in itertools.combinations(cols, 2))
        assert worst < MIN_CORRESPONDENCE_DELTA_E, (
            f"{name} cap {theme.correspondence_cap} is too strict: "
            f"one more slot still clears at ΔE {worst:.1f}")

    @pytest.mark.parametrize("name,theme", THEMES)
    def test_the_order_ramp_is_monotone_and_drawable(self, name, theme):
        ok, why = is_monotone(theme.ramp_stops)
        assert ok, f"{name} ramp is not ordered: {why}"
        # it is drawn as line work, so every stop must clear the stroke floor
        for c in theme.ramp_stops:
            worst = min(contrast(c, p) for p in theme.paper_stops())
            assert worst >= MIN_STROKE_CONTRAST, f"{name} ramp stop {c} at {worst:.2f}:1"

    @pytest.mark.parametrize("name,theme", THEMES)
    def test_the_shading_ramp_is_monotone(self, name, theme):
        """Facet shading is exempt from contrast — its light end is meant to
        approach the paper — so monotonicity is the only thing holding it up."""
        ok, why = is_monotone(theme.surface_stops)
        assert ok, f"{name} surface ramp is not ordered: {why}"


class TestCategoricalSlots:
    def test_a_slot_does_not_depend_on_how_many_series_share_the_figure(self):
        """The regression that motivated fixed slots: hue used to be
        interpolated at i/(n-1), so the same object took different colors in
        two panels with different series counts."""
        assert RISO.categorical(1) == RISO.categorical(1)
        assert RISO.categorical(0) != RISO.categorical(1)

    def test_slots_are_tagged_as_correspondence(self):
        c = RISO.categorical(0)
        assert isinstance(c, Hue) and c.channel == CORRESPONDENCE
        assert RISO.ramp(0.5).channel == "order"

    def test_past_the_last_slot_raises_rather_than_generating_a_hue(self):
        with pytest.raises(IndexError, match="never a generated hue"):
            RISO.categorical(len(RISO.categorical_stops))


class TestSceneGate:
    def test_a_clean_scene_passes(self):
        assert color_gate(_scene(_line(), _line(RISO.categorical(0))), RISO) == []

    def test_invisible_content_ink_is_caught(self):
        d = color_gate(_scene(_line("#f4efe3")), RISO)
        assert [x.kind for x in d] == ["faint-ink"]

    def test_scaffolding_may_recede(self):
        """grammar.md's ink hierarchy makes construction/frame ink quiet on
        purpose; holding it to the content floor would flatten the figure."""
        faint = RISO.ink(Role.FRAME).color
        assert min(contrast(faint, p) for p in RISO.paper_stops()) < MIN_STROKE_CONTRAST
        assert color_gate(_scene(_line(role=Role.FRAME)), RISO) == []

    def test_opacity_is_composited_before_measuring(self):
        """A stroke at low opacity is fainter than the hex it declares."""
        solid = RISO.ink(Role.ACCENT1).color
        assert color_gate(_scene(_line(solid)), RISO) == []
        assert color_gate(_scene(_line(solid, opacity=0.25)), RISO) != []

    def test_a_path_bundle_is_judged_as_one_object(self):
        """A distributional claim is *shown* as an ensemble, where visibility
        comes from accumulation — one member need not carry the content floor."""
        faint = composite(RISO.ink(Role.ACCENT1).color, 0.6, RISO.paper_stops()[0])
        few = _scene(*[_line(faint) for _ in range(3)])
        many = _scene(*[_line(faint) for _ in range(40)])
        assert color_gate(few, RISO) != []
        assert color_gate(many, RISO) == []

    def test_colliding_correspondence_hues_are_caught(self):
        a = Hue("#c8402f", CORRESPONDENCE)
        b = Hue("#c9412f", CORRESPONDENCE)   # a different object, the same colour
        d = color_gate(_scene(_line(a), _line(b)), RISO)
        assert any(x.kind == "hue-collapse" for x in d)

    def test_untagged_lookalikes_are_not_treated_as_correspondence(self):
        """Two near-identical hues are only a defect if they claim to encode
        identity; a shading or order ramp legitimately has close neighbours."""
        d = color_gate(_scene(_line("#c8402f"), _line("#c9412f")), RISO)
        assert not any(x.kind == "hue-collapse" for x in d)

    def test_using_more_hues_than_the_cap_is_caught(self):
        items = [_line(RISO.categorical(i))
                 for i in range(RISO.correspondence_cap + 1)]
        d = color_gate(_scene(*items), RISO)
        assert any(x.kind == "hue-collapse" and "cap" in x.detail for x in d)


# One 8-bit sRGB code step is ~0.003 in OKLab L across the mid range, so a
# reversal or deviation below this is quantization, not a design defect.
L_QUANT = 0.004
RAMP_SAMPLES = 129


class TestPerceptualInterpolation:
    """A ramp's semantic contract is ORDER, and the coordinate being ordered
    is OKLab L — so the stops are interpolated there, not in raw sRGB hex.

    Lerping hex channels lets the transfer curve decide what happens between
    two stops: the RISO ramp's sampled L wandered up to 0.010 off the
    piecewise-linear path its stops promise, and reversed by ~0.001 in the
    mustard->brick leg. Both are below the "can you see it" line one step at
    a time and both are visible as a stall in a level-set family.
    """

    HEXES = ["#000000", "#ffffff", "#3467a3", "#e9c46a", "#a17918", "#3d405b"]

    def test_the_inverse_transform_inverts_the_forward_one(self):
        from figlib.color import from_oklab, to_oklab
        for c in self.HEXES:
            assert from_oklab(*to_oklab(c)) == c, c

    def test_mixing_reproduces_the_endpoints(self):
        from figlib.color import mix_oklab
        assert mix_oklab("#3467a3", "#e9c46a", 0.0) == "#3467a3"
        assert mix_oklab("#3467a3", "#e9c46a", 1.0) == "#e9c46a"

    def test_mixing_is_linear_in_perceptual_lightness(self):
        from figlib.color import mix_oklab
        a, b = "#3d405b", "#a17918"
        for f in (0.25, 0.5, 0.75):
            want = (1 - f) * lightness(a) + f * lightness(b)
            assert lightness(mix_oklab(a, b, f)) == pytest.approx(want, abs=L_QUANT)

    def test_out_of_gamut_lab_clips_instead_of_wrapping(self):
        """Clipping is in linear RGB, so an unrepresentable Lab triple lands
        on the gamut boundary rather than wrapping to a nonsense hex."""
        from figlib.color import from_oklab
        assert from_oklab(1.4, 0.0, 0.0) == "#ffffff"
        assert from_oklab(-0.3, 0.0, 0.0) == "#000000"
        # a chroma no display can show lands ON the boundary (some channel
        # saturated), not somewhere arbitrary inside the cube
        out = from_oklab(0.5, 0.4, 0.0)
        assert len(out) == 7 and out.startswith("#")
        assert any(out[k:k + 2] in ("00", "ff") for k in (1, 3, 5)), out


def _sampled(channel) -> list[str]:
    return [channel(float(t)) for t in np.linspace(0.0, 1.0, RAMP_SAMPLES)]


def _stop_profile(stops: list[str], n: int) -> np.ndarray:
    ls = [lightness(c) for c in stops]
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(ls)), ls)


class TestSampledRamps:
    """The theme-level gate the stop list alone cannot give: `ramp(t)` is what
    figures actually call, so monotonicity is asserted on the sampled channel,
    not only on its stops."""

    CHANNELS = [(nm, th, ch) for nm, th in THEMES
                for ch in ("ramp", "surface_shade")]

    @pytest.mark.parametrize("name,theme,chan", CHANNELS)
    def test_the_sampled_channel_is_monotone_in_lightness(self, name, theme, chan):
        ok, why = is_monotone(_sampled(getattr(theme, chan)), tol=L_QUANT)
        assert ok, f"{name}.{chan}(t) is not ordered: {why}"

    @pytest.mark.parametrize("name,theme,chan", CHANNELS)
    def test_the_sampled_channel_walks_lightness_at_a_constant_rate(
            self, name, theme, chan):
        """Monotone is necessary, even is what makes a ramp readable: equal
        steps in t must be equal steps in L, i.e. L(t) is the piecewise-linear
        interpolation of the stops' L. This is exactly what OKLab
        interpolation gives and what hex-channel interpolation does not."""
        stops = theme.ramp_stops if chan == "ramp" else theme.surface_stops
        got = np.array([lightness(c) for c in _sampled(getattr(theme, chan))])
        want = _stop_profile(stops, RAMP_SAMPLES)
        dev = float(np.max(np.abs(got - want)))
        assert dev < L_QUANT, (
            f"{name}.{chan}(t) deviates {dev:.4f} in L from the straight "
            f"perceptual path between its stops (tol {L_QUANT})")
