"""The label casing is ONE rect behind the whole label, and only on paper.

Two failures this pins, both visible in a committed baseline before the fix:
a per-glyph casing punched the axis line out in chunks between the glyphs of
`r = -1`, and it painted opaque white blobs onto transparent renders that are
supposed to be ink on alpha.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

import pytest

from figlib.render import to_svg
from figlib.scene import MathLabel, Role, Scene
from figlib.style import Style
from figlib.theme import RISO, opaque_variant
from figlib.typeset import label_box

from svgkit import find_by, svg_root


def _scene(halo: bool) -> Scene:
    s = Scene(xlim=(0, 1), ylim=(0, 1))
    s.add(MathLabel(r"r = -1", anchor=(0.5, 0.5), role=Role.ANNOTATION, halo=halo))
    return s


def _casings(root: ET.Element, ground: str) -> list[ET.Element]:
    """The casing rects — by class, so the page's own ground rect (same
    fill on an opaque theme) is not mistaken for one."""
    return [r for r in find_by(root, "rect", class_="label-casing")
            if (r.get("fill") or "").lower() == ground.lower()]


def _white_stroked(root: ET.Element) -> list[ET.Element]:
    return [el for el in root.iter()
            if (el.get("stroke") or "").lower() == "#ffffff"]


def test_paper_halo_is_one_rect_not_one_per_glyph():
    paper = opaque_variant(RISO)
    root = svg_root(to_svg(_scene(halo=True), style=paper))
    assert len(_casings(root, paper.background)) == 1


def test_paper_halo_draws_no_stroked_glyph_copies():
    """The old casing stroked every glyph; nothing may stroke in the ground
    colour now, or the chunks reappear between letters."""
    paper = opaque_variant(RISO)
    root = svg_root(to_svg(_scene(halo=True), style=paper))
    assert _white_stroked(root) == []


def test_halo_rect_covers_the_label_box():
    paper = opaque_variant(RISO)
    root = svg_root(to_svg(_scene(halo=True), style=paper))
    (rect,) = _casings(root, paper.background)
    rx0, ry0 = float(rect.get("x")), float(rect.get("y"))
    rx1 = rx0 + float(rect.get("width"))
    ry1 = ry0 + float(rect.get("height"))
    # the rect must contain the glyph box it is casing, with room to spare
    lx0, ly0, lx1, ly1 = label_box(r"r = -1", paper.label_pt(None), 0.0, 0.0,
                                   "center", "center")
    w, h = lx1 - lx0, ly1 - ly0
    assert (rx1 - rx0) > w and (ry1 - ry0) > h


def test_transparent_render_has_no_casing():
    """style.transparent means ink on alpha: a casing in the ground colour is
    an opaque blob once the ground is gone."""
    from figlib.theme import RISO_CLEAR
    root = svg_root(to_svg(_scene(halo=True), style=RISO_CLEAR))
    assert RISO_CLEAR.transparent is True
    assert _casings(root, "#ffffff") == []
    assert _white_stroked(root) == []


def test_unhaloed_label_never_gets_a_casing():
    paper = opaque_variant(RISO)
    root = svg_root(to_svg(_scene(halo=False), style=paper))
    assert _casings(root, paper.background) == []
