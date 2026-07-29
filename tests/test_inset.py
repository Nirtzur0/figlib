"""Word-scale inset: affine embedding of one Scene's items into a host.

`embed` is a PRODUCER — it maps a small scene's items into host math
coords and hands back plain items. The semantics under test are the
mapping rules themselves: geometry (coordinates and math-unit lengths)
scales by the linear factor; type and ink (size_pt, offset_px, dash,
width_scale, ...) never scale, by the canvas-px invariant.
"""

import dataclasses

import numpy as np
import pytest

from figlib.inset import embed
from figlib.scene import (AngleMark, Brace, Callout, Curve, FilledCurve,
                          Gradient, MathLabel, Point, RasterField,
                          RightAngleMark, Scene, Vector)
from figlib.style import Role


# --- the four seed behaviors -------------------------------------------------

def test_embed_maps_corners_exactly():
    small = Scene(xlim=(0, 2), ylim=(0, 1))
    small.add(Curve(np.array([[0, 0], [2, 1]]) * 1.0))
    out = embed(small, at=(10.0, 5.0), width=1.0, frame=False)
    c = next(it for it in out if isinstance(it, Curve))
    # dest rect: 1.0 wide, 0.5 tall, centered at (10, 5)
    assert np.allclose(c.pts[0], [9.5, 4.75]) and np.allclose(c.pts[1], [10.5, 5.25])


def test_embed_preserves_type_size_and_scales_geometry():
    small = Scene(xlim=(0, 1), ylim=(0, 1))
    small.add(MathLabel("x", (0.5, 0.5), size_pt=9.0),
              AngleMark((0.5, 0.5), (1, 0), (0, 1), radius=0.2))
    out = embed(small, at=(0.0, 0.0), width=0.5, frame=False)
    lab = next(it for it in out if isinstance(it, MathLabel))
    assert lab.size_pt == 9.0                      # type never scales
    am = next(it for it in out if isinstance(it, AngleMark))
    assert np.isclose(am.radius, 0.1)              # geometry does


def test_embed_requires_lims_and_equal_aspect():
    with pytest.raises(ValueError):
        embed(Scene(), at=(0, 0), width=1.0)
    s = Scene(xlim=(0, 1), ylim=(0, 1), height_px=120.0)
    with pytest.raises(ValueError):
        embed(s, at=(0, 0), width=1.0)


def test_embed_does_not_mutate_source():
    small = Scene(xlim=(0, 1), ylim=(0, 1))
    pts = np.array([[0.0, 0.0], [1.0, 1.0]])
    small.add(Curve(pts))
    embed(small, at=(5.0, 5.0), width=2.0)
    assert np.allclose(small.items[0].pts, pts)


# --- full item coverage ------------------------------------------------------

def _unit_scene():
    return Scene(xlim=(0.0, 1.0), ylim=(0.0, 1.0))


def test_embed_maps_every_item_type():
    small = _unit_scene()
    small.add(
        Curve(np.array([[0.0, 0.0], [1.0, 1.0]])),
        FilledCurve(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
                    holes=(np.array([[0.2, 0.2], [0.4, 0.2], [0.4, 0.4]]),)),
        Vector((0.0, 0.0), (1.0, 0.0)),
        Point((0.5, 0.5)),
        MathLabel("y", (0.25, 0.75)),
        RightAngleMark((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), size=0.1),
        AngleMark((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), radius=0.2),
        Brace((0.0, 0.0), (1.0, 0.0), depth=0.1),
        Callout("c", (0.5, 0.5), (0.9, 0.9)),
        RasterField(np.zeros((2, 2)), extent=(0.0, 1.0, 0.0, 1.0)),
    )
    out = embed(small, at=(4.0, 4.0), width=2.0, frame=False)
    assert len(out) == len(small.items)
    # dest rect: [3, 5] x [3, 5]; linear scale s = 2
    fc = next(it for it in out if isinstance(it, FilledCurve))
    assert np.allclose(fc.pts, [[3, 3], [5, 3], [5, 5]])
    assert np.allclose(fc.holes[0], [[3.4, 3.4], [3.8, 3.4], [3.8, 3.8]])
    v = next(it for it in out if isinstance(it, Vector))
    assert np.allclose(v.tail, (3, 3)) and np.allclose(v.tip, (5, 3))
    p = next(it for it in out if isinstance(it, Point))
    assert np.allclose(p.xy, (4, 4))
    lab = next(it for it in out if isinstance(it, MathLabel))
    assert np.allclose(lab.anchor, (3.5, 4.5))
    ra = next(it for it in out if isinstance(it, RightAngleMark))
    assert np.allclose(ra.corner, (3, 3)) and np.isclose(ra.size, 0.2)
    assert np.allclose(ra.dir1, (1, 0)) and np.allclose(ra.dir2, (0, 1))
    br = next(it for it in out if isinstance(it, Brace))
    assert np.allclose(br.p1, (3, 3)) and np.allclose(br.p2, (5, 3))
    assert np.isclose(br.depth, 0.2)
    co = next(it for it in out if isinstance(it, Callout))
    assert np.allclose(co.anchor, (4, 4)) and np.allclose(co.target, (4.8, 4.8))
    rf = next(it for it in out if isinstance(it, RasterField))
    assert rf.extent == (3.0, 5.0, 3.0, 5.0)


def test_embed_maps_gradient_axis_and_keeps_stops():
    grad = Gradient(stops=((0.0, "#000000"), (1.0, "#ffffff")),
                    p0=(0.0, 0.0), p1=(1.0, 0.0))
    small = _unit_scene()
    small.add(FilledCurve(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
                          gradient=grad))
    out = embed(small, at=(0.0, 0.0), width=1.0, frame=False)
    g = out[0].gradient
    assert np.allclose(g.p0, (-0.5, -0.5)) and np.allclose(g.p1, (0.5, -0.5))
    assert g.stops == grad.stops


def test_embed_leaves_none_brace_depth_alone():
    small = _unit_scene()
    small.add(Brace((0.0, 0.0), (1.0, 0.0)))
    out = embed(small, at=(0.0, 0.0), width=3.0, frame=False)
    assert out[0].depth is None    # auto-depth is span-relative; scales itself


def test_embed_never_touches_ink_quantities():
    small = _unit_scene()
    small.add(Curve(np.array([[0.0, 0.0], [1.0, 1.0]]), width_scale=2.0,
                    dash="dashed", width_profile=(1.0, 3.0), arrow_scale=1.5,
                    arrows=(0.5,)),
              Point((0.5, 0.5), radius_scale=2.0),
              MathLabel("z", (0.5, 0.5), offset_px=(3.0, -4.0)))
    out = embed(small, at=(0.0, 0.0), width=0.25, frame=False)
    c, p, lab = out
    assert (c.width_scale, c.dash, c.width_profile, c.arrow_scale) == \
        (2.0, "dashed", (1.0, 3.0), 1.5)
    assert c.arrows == (0.5,)
    assert p.radius_scale == 2.0
    assert lab.offset_px == (3.0, -4.0)


# --- furniture: frame, leader, key ------------------------------------------

def test_embed_frame_is_a_closed_frame_role_curve_on_the_dest_rect():
    small = _unit_scene()
    small.add(Point((0.5, 0.5)))
    out = embed(small, at=(2.0, 3.0), width=1.0)
    frame = next(it for it in out if isinstance(it, Curve) and it.role is Role.FRAME)
    assert frame.closed
    lo, hi = frame.pts.min(axis=0), frame.pts.max(axis=0)
    assert np.allclose(lo, [1.5, 2.5], atol=1e-9)
    assert np.allclose(hi, [2.5, 3.5], atol=1e-9)


def test_embed_leader_is_an_annotation_vector_from_rect_boundary():
    small = _unit_scene()
    small.add(Point((0.5, 0.5)))
    out = embed(small, at=(0.0, 0.0), width=1.0, frame=False,
                leader_to=(3.0, 0.0))
    leader = next(it for it in out if isinstance(it, Vector))
    assert leader.role is Role.ANNOTATION
    assert np.allclose(leader.tip, (3.0, 0.0))
    assert np.allclose(leader.tail, (0.5, 0.0))   # nearest rect boundary point


def test_embed_forwards_key_onto_every_item():
    small = _unit_scene()
    small.add(Point((0.5, 0.5)), Curve(np.array([[0.0, 0.0], [1.0, 1.0]])))
    out = embed(small, at=(0.0, 0.0), width=1.0, leader_to=(2.0, 2.0),
                key="inset:saddle")
    assert out and all(it.key == "inset:saddle" for it in out)


def test_embed_keeps_item_keys_when_no_key_given():
    small = _unit_scene()
    small.add(Point((0.5, 0.5), key="fixed-point"))
    out = embed(small, at=(0.0, 0.0), width=1.0, frame=False)
    assert out[0].key == "fixed-point"


def test_embed_rejects_unknown_item_types_by_name():
    @dataclasses.dataclass
    class Widget:
        xy: tuple[float, float] = (0.0, 0.0)

    small = _unit_scene()
    small.items.append(Widget())      # bypass the Item union on purpose
    with pytest.raises(TypeError, match="Widget"):
        embed(small, at=(0.0, 0.0), width=1.0)


def test_embed_rejects_degenerate_lims_and_width():
    with pytest.raises(ValueError):
        embed(Scene(xlim=(0.0, 0.0), ylim=(0.0, 1.0)), at=(0, 0), width=1.0)
    with pytest.raises(ValueError):
        embed(_unit_scene(), at=(0, 0), width=0.0)
