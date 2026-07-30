"""Typographic REGISTER: typeface as a semantic channel.

mono = literal model input / data, sans = human interpretation, default
(None) = mathematics. The channel is only honest if the mechanical gate
measures the SAME typeset string the renderer draws — every test that
compares a box to ink is guarding that one invariant.
"""

import numpy as np
import pytest
from svgkit import svg_root, tag

from figlib.gates import _box_at, _label_boxes
from figlib.layout import Transform
from figlib.place import _label_box, place_on_locus
from figlib.render import to_svg
from figlib.scene import Curve, MathLabel, Scene
from figlib.schematic import (Edge, Junction, Node, auto_node, auto_size,
                              label_extent_px, label_overflow, straight)
from figlib.style import DEFAULT_STYLE
from figlib.typeset import apply_register, render_math


def box_scene(*extra) -> Scene:
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    return Scene(items=[Curve(pts, closed=True), *extra], xlim=(0, 10), ylim=(0, 10))


class TestApplyRegister:
    def test_none_is_identity(self):
        assert apply_register(r"x^2", None) == r"x^2"

    def test_single_char_wraps(self):
        assert apply_register("Q", "mono") == r"\mathtt{Q}"
        assert apply_register("Q", "sans") == r"\mathsf{Q}"

    def test_wrapping_is_per_atom(self):
        """`\\mathtt{tok}` is a silent no-op in ziamath — the font command
        reaches only the first atom, so a multi-character run comes back in
        serif italic. Wrapping each character is what actually switches."""
        assert apply_register("tok", "mono") == r"\mathtt{t}\mathtt{o}\mathtt{k}"
        assert apply_register("ab", "sans") == r"\mathsf{a}\mathsf{b}"

    def test_space_becomes_a_math_space(self):
        assert apply_register("a b", "mono") == r"\mathtt{a}\ \mathtt{b}"

    def test_unknown_register_names_the_alternatives(self):
        with pytest.raises(ValueError) as e:
            apply_register("Q", "italic")
        assert "italic" in str(e.value)
        assert "mono" in str(e.value) and "sans" in str(e.value)

    @pytest.mark.parametrize("latex", [r"\alpha", "x^2", "a_i", "{ab}", "50$"])
    def test_latex_machinery_is_rejected(self, latex):
        """A register is per-character, so it has nothing to say about a
        control sequence or a script. Refuse loudly instead of emitting
        `\\mathtt{^}` and letting ziamath produce garbage."""
        with pytest.raises(ValueError) as e:
            apply_register(latex, "mono")
        assert "per-character" in str(e.value).lower()

    def test_plain_punctuation_and_digits_are_allowed(self):
        # They pass through unstyled (STIX has no mono digit reachable via
        # \mathtt), but they render, so they are not an error.
        assert apply_register("gpt-4.o", "mono").count(r"\mathtt{") == 7


class TestZiamathHonorsRegister:
    """The channel is worthless if the engine silently ignores the command.
    `Q` alone is NOT sufficient evidence: single atoms were always honored;
    it was the multi-character run that silently fell back."""

    def test_multichar_mono_differs_from_serif(self):
        serif = render_math("tok", size_pt=11.0)
        mono = render_math(apply_register("tok", "mono"), size_pt=11.0)
        assert mono.width_px > 0.0 and mono.height_px > 0.0
        # mono is materially wider — 3 fixed advances vs. 3 italic ones
        assert mono.width_px > serif.width_px * 1.1

    def test_multichar_sans_changes_the_glyphs(self):
        """Sans and serif happen to have near-identical advances here, so
        width proves nothing — the glyph ids are the evidence."""
        serif = _glyph_ids("tok")
        sans = _glyph_ids(apply_register("tok", "sans"))
        assert len(sans) == len(serif) == 3
        assert sans != serif

    def test_single_char_mono_still_differs(self):
        serif = render_math("Q", size_pt=11.0)
        mono = render_math(apply_register("Q", "mono"), size_pt=11.0)
        assert mono.width_px != pytest.approx(serif.width_px, abs=1e-6)


def _glyph_ids(latex: str, size_pt: float = 11.0) -> list[str]:
    """The glyph symbols ziamath actually reached for — the only proof that
    a font command changed anything when advances happen to match."""
    import re

    import ziamath

    from figlib.typeset import PX_PER_PT
    svg = ziamath.Latex(latex, size=size_pt * PX_PER_PT).svg()
    return re.findall(r'id="(\w+)"', svg)


class TestMathLabelRegister:
    def test_default_is_none(self):
        assert MathLabel("x", (0.0, 0.0)).register is None

    def test_registered_label_still_renders_glyphs(self):
        svg = to_svg(box_scene(MathLabel("Q", (5.0, 5.0), register="mono")),
                     DEFAULT_STYLE, width_px=900)
        root = svg_root(svg)
        assert [e for e in root.iter() if tag(e) == "use"]

    def test_gate_box_tracks_the_register(self):
        """The measured box must change with the register, and must equal the
        typeset extent of the WRAPPED latex — otherwise bboxes drift off ink."""
        plain = box_scene(MathLabel("tok", (5.0, 5.0)))
        mono = box_scene(MathLabel("tok", (5.0, 5.0), register="mono"))
        wp = _box_width(plain)
        wm = _box_width(mono)
        assert wm > wp * 1.1
        expected = render_math(apply_register("tok", "mono"),
                               DEFAULT_STYLE.label_pt(None)).width_px
        assert wm == pytest.approx(expected, abs=1e-6)


def _box_width(scene: Scene) -> float:
    t = Transform(scene, width_px=900)
    boxes = [b for _l, _pt, b in _label_boxes(scene, DEFAULT_STYLE, t)]
    assert len(boxes) == 1
    return boxes[0][2] - boxes[0][0]


class TestSchematicPassThrough:
    def _sole_label(self, obj) -> MathLabel:
        labs = [i for i in obj.items() if isinstance(i, MathLabel)]
        assert len(labs) == 1
        return labs[0]

    def test_node_threads_register(self):
        nd = Node("a", (0.0, 0.0), 2.0, 1.0, label="tok", label_register="mono")
        assert self._sole_label(nd).register == "mono"

    def test_node_default_register_is_none(self):
        nd = Node("a", (0.0, 0.0), 2.0, 1.0, label="x")
        assert self._sole_label(nd).register is None

    def test_edge_threads_register(self):
        e = Edge(straight((0.0, 0.0), (1.0, 0.0)), "map",
                 ((0.0, 0.0), (1.0, 0.0)),
                 label="cat", label_register="sans")
        assert self._sole_label(e).register == "sans"

    def test_junction_threads_register_to_its_glyph(self):
        j = Junction("j", (0.0, 0.0), r"\oplus", label_register="sans")
        glyphs = [i for i in j.items() if isinstance(i, MathLabel)]
        assert len(glyphs) == 1
        assert glyphs[0].register == "sans"

    def test_label_extent_px_honors_register(self):
        plain = label_extent_px("tok", 11.0)
        mono = label_extent_px("tok", 11.0, register="mono")
        assert mono[0] > plain[0] * 1.1
        assert mono[0] == pytest.approx(
            render_math(apply_register("tok", "mono"), 11.0).width_px, abs=1e-6)


class TestEveryMeasurementHelperSeesTheRegister:
    """One test per helper that turns latex into a box. A helper missing
    the register is a silent bbox-vs-ink drift, and this class is the only
    thing standing between a future helper and that bug."""

    def test_gates_box_at(self):
        plain = _box_at("tok", 11.0, 0.0, 0.0, "left", "base")
        mono = _box_at("tok", 11.0, 0.0, 0.0, "left", "base", "mono")
        assert (mono[2] - mono[0]) > (plain[2] - plain[0]) * 1.1

    def test_place_label_box(self):
        plain = _label_box("tok", 11.0, 0.0, 0.0, "left", "base", 0.0)
        mono = _label_box("tok", 11.0, 0.0, 0.0, "left", "base", 0.0, "mono")
        assert (mono[2] - mono[0]) > (plain[2] - plain[0]) * 1.1

    def test_schematic_label_extent_px(self):
        assert label_extent_px("tok", 11.0, "mono")[0] > \
            label_extent_px("tok", 11.0)[0] * 1.1

    def test_schematic_auto_size(self):
        plain = auto_size("tok", scale=50.0, size_pt=11.0)
        mono = auto_size("tok", scale=50.0, size_pt=11.0, register="mono")
        assert mono[0] > plain[0] * 1.05

    def test_place_on_locus_boxes_candidates_with_the_register(self):
        """Two candidates, one hard against the frame: a mono label is wide
        enough to be squeezed there when a serif one is not, so the winner
        changes. If place_on_locus ignored the register it would solve the
        wrong problem silently."""
        scene = box_scene()
        locus = [(1.0, 5.0), (5.0, 5.0)]
        wide = place_on_locus(scene, DEFAULT_STYLE, "tok" * 4, locus,
                              register="mono")
        narrow = place_on_locus(scene, DEFAULT_STYLE, "tok" * 4, locus)
        assert wide.clearance_px != pytest.approx(narrow.clearance_px, abs=1e-6)


class TestAutoNodeSizesForItsRegister:
    """auto_node forwards label_register to the Node; if it did not also
    MEASURE with it, a mono label would be ~20% wider than the box drawn
    for it and label_overflow would flag a node the author never mis-sized."""

    def test_mono_node_is_wider(self):
        plain = auto_node("a", (0.0, 0.0), "tok", scale=50.0)
        mono = auto_node("a", (0.0, 0.0), "tok", scale=50.0,
                         label_register="mono")
        assert mono.width > plain.width * 1.05

    def test_mono_node_label_fits_its_own_box(self):
        nd = auto_node("a", (0.0, 0.0), "token", scale=50.0,
                       label_register="mono")
        assert label_overflow([nd], scale=50.0, size_pt=11.0) == []
