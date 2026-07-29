"""Typographic REGISTER: typeface as a semantic channel.

mono = literal model input / data, sans = human interpretation, default
(None) = mathematics. The channel is only honest if the mechanical gate
measures the SAME typeset string the renderer draws — every test that
compares a box to ink is guarding that one invariant.
"""

import numpy as np
import pytest
from svgkit import svg_root, tag

from figlib.gates import _label_boxes
from figlib.layout import Transform
from figlib.render import to_svg
from figlib.scene import Curve, MathLabel, Scene
from figlib.schematic import Edge, Junction, Node, label_extent_px, straight
from figlib.style import DEFAULT_STYLE
from figlib.typeset import apply_register, render_math


def box_scene(*extra) -> Scene:
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    return Scene(items=[Curve(pts, closed=True), *extra], xlim=(0, 10), ylim=(0, 10))


class TestApplyRegister:
    def test_none_is_identity(self):
        assert apply_register(r"x^2", None) == r"x^2"

    def test_mono_and_sans_wrap(self):
        assert apply_register("Q", "mono") == r"\mathtt{Q}"
        assert apply_register("Q", "sans") == r"\mathsf{Q}"

    def test_unknown_register_names_the_alternatives(self):
        with pytest.raises(ValueError) as e:
            apply_register("Q", "italic")
        assert "italic" in str(e.value)
        assert "mono" in str(e.value) and "sans" in str(e.value)


class TestZiamathHonorsRegister:
    """The channel is worthless if the engine silently ignores the command."""

    def test_mono_metrics_are_nonzero_and_differ_from_serif(self):
        serif = render_math("Q", size_pt=11.0)
        mono = render_math(apply_register("Q", "mono"), size_pt=11.0)
        assert mono.width_px > 0.0 and mono.height_px > 0.0
        assert mono.width_px != pytest.approx(serif.width_px, abs=1e-6)

    def test_sans_differs_from_serif(self):
        serif = render_math("Q", size_pt=11.0)
        sans = render_math(apply_register("Q", "sans"), size_pt=11.0)
        assert sans.width_px > 0.0
        assert sans.width_px != pytest.approx(serif.width_px, abs=1e-6)


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
        plain = box_scene(MathLabel("Q", (5.0, 5.0)))
        mono = box_scene(MathLabel("Q", (5.0, 5.0), register="mono"))
        wp = _box_width(plain)
        wm = _box_width(mono)
        assert wm != pytest.approx(wp, abs=1e-6)
        expected = render_math(apply_register("Q", "mono"),
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
        plain = label_extent_px("Q", 11.0)
        mono = label_extent_px("Q", 11.0, register="mono")
        assert mono[0] != pytest.approx(plain[0], abs=1e-6)
        assert mono[0] == pytest.approx(
            render_math(apply_register("Q", "mono"), 11.0).width_px, abs=1e-6)
