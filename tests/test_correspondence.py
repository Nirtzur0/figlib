"""The residual check: what a composite figure holds fixed, and what leaks.

A composite's claim is a predicate over a binding. Declare the binding
(keys on items), declare the one axis of variation (`changes`), and every
other difference across the bound parts is a residual — the mechanical
reason two panels fail to click.
"""

import numpy as np
import pytest

from figlib.correspond import Correspondence, keyed, residual
from figlib.figure import Figure, Panel
from figlib.scene import Curve, MathLabel, Point, Scene
from figlib.style import DEFAULT_STYLE, Role


def _arc(shift: float = 0.0, key: str | None = None, **kw) -> Curve:
    t = np.linspace(0.0, 1.0, 8)
    return Curve(np.column_stack([t + shift, t ** 2]), key=key, **kw)


def _panel(*items, xlim=(-5.0, 5.0), ylim=(-5.0, 5.0)) -> Panel:
    """Explicit lims by default: auto-extents make every panel its own scale,
    which is a frame-drift finding in its own right (tested below) and would
    otherwise contaminate the key tests."""
    return Panel(Scene(items=list(items), xlim=xlim, ylim=ylim))


def _kinds(diags) -> list[str]:
    return [d.kind for d in diags]


def _run(panels, corr, width_px: float = 600.0):
    return residual(Figure(panels=list(panels)), [corr], DEFAULT_STYLE,
                    width_px=width_px)


# --- the binding holds -------------------------------------------------------

def test_identical_keyed_object_across_parts_is_clean():
    fig = [_panel(_arc(key="probe")), _panel(_arc(shift=3.0, key="probe"))]
    assert _run(fig, Correspondence(parts=(0, 1), varies="translation")) == []


def test_position_is_never_part_of_the_fingerprint():
    """The binding says 'same object', not 'same place' — a composite whose
    parts were pixel-identical would have nothing to say."""
    a = _panel(Curve(np.array([[0.0, 0.0], [1.0, 1.0]]), key="ray"))
    b = _panel(Curve(np.array([[0.0, 0.0], [-1.0, 4.0]]), key="ray"))
    assert _run([a, b], Correspondence(parts=(0, 1), varies="the map")) == []


# --- the residual ------------------------------------------------------------

def test_key_missing_from_one_part_is_a_residual():
    fig = [_panel(_arc(key="probe"), _arc(shift=1.0, key="axis")),
           _panel(_arc(key="probe"))]
    diags = _run(fig, Correspondence(parts=(0, 1), varies="the map"))
    assert _kinds(diags) == ["residual"]
    assert "axis" in diags[0].detail
    # the diagnostic contains the fix: which part is short
    assert "[1]" in diags[0].detail


def test_hue_drift_across_the_binding_is_a_residual():
    fig = [_panel(_arc(key="probe", color="#aa3311")),
           _panel(_arc(key="probe", color="#1133aa"))]
    diags = _run(fig, Correspondence(parts=(0, 1), varies="the map"))
    assert _kinds(diags) == ["residual"]
    assert "color" in diags[0].detail
    assert "#aa3311" in diags[0].detail and "#1133aa" in diags[0].detail


def test_role_drift_across_the_binding_is_a_residual():
    fig = [_panel(_arc(key="probe", role=Role.CONTENT)),
           _panel(_arc(key="probe", role=Role.CONSTRUCTION))]
    diags = _run(fig, Correspondence(parts=(0, 1), varies="the map"))
    assert _kinds(diags) == ["residual"]
    assert "role" in diags[0].detail


def test_a_key_covering_several_items_compares_as_a_set():
    """'probe' may be a curve plus its label; the fingerprint is the set of
    facets, so item count alone is not drift."""
    fig = [_panel(_arc(key="probe"), _arc(shift=0.1, key="probe")),
           _panel(_arc(key="probe"))]
    assert _run(fig, Correspondence(parts=(0, 1), varies="the map")) == []


def test_dash_drift_is_a_residual():
    fig = [_panel(_arc(key="branch", dash="dashed")),
           _panel(_arc(key="branch"))]
    diags = _run(fig, Correspondence(parts=(0, 1), varies="r"))
    assert _kinds(diags) == ["residual"]
    assert "dash" in diags[0].detail


# --- the declared variation --------------------------------------------------

def test_declared_change_is_not_a_residual():
    fig = [_panel(_arc(key="roots", color="#aa3311")),
           _panel(_arc(key="roots", color="#1133aa"))]
    corr = Correspondence(parts=(0, 1), varies="r", changes=("roots",))
    assert _run(fig, corr) == []


def test_declared_change_that_never_happens_is_a_defect():
    """The figure says the roots change and then draws them the same. The
    claim is not on the page."""
    fig = [_panel(_arc(key="roots")), _panel(_arc(shift=2.0, key="roots"))]
    corr = Correspondence(parts=(0, 1), varies="r", changes=("roots",))
    diags = _run(fig, corr)
    assert _kinds(diags) == ["stale-change"]
    assert "roots" in diags[0].detail


def test_change_naming_a_key_that_does_not_exist_is_a_defect():
    fig = [_panel(_arc(key="probe")), _panel(_arc(key="probe"))]
    corr = Correspondence(parts=(0, 1), varies="r", changes=("ghost",))
    diags = _run(fig, corr)
    assert _kinds(diags) == ["stale-change"]
    assert "ghost" in diags[0].detail


def test_auto_extents_are_themselves_a_frame_drift():
    """Panels that pad to their own content have no shared scale — the most
    common way a composite silently stops being comparable."""
    a = Panel(Scene(items=[_arc(key="probe")]))
    b = Panel(Scene(items=[Curve(np.array([[0.0, 0.0], [4.0, 4.0]]), key="probe")]))
    diags = _run([a, b], Correspondence(parts=(0, 1), varies="the map"))
    assert _kinds(diags) == ["frame-drift"]


def test_a_declared_correspondence_with_no_keys_is_undeclared():
    fig = [_panel(_arc()), _panel(_arc(shift=1.0))]
    diags = _run(fig, Correspondence(parts=(0, 1), varies="the map"))
    assert _kinds(diags) == ["undeclared-binding"]


# --- the frame ---------------------------------------------------------------

def test_shared_frame_with_unequal_scales_is_a_residual():
    """Same object, same declared frame, different px-per-math-unit: every
    length comparison the reader makes across the panels is a lie."""
    a = _panel(_arc(key="probe"), xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    b = _panel(_arc(key="probe"), xlim=(0.0, 2.0), ylim=(0.0, 2.0))
    diags = _run([a, b], Correspondence(parts=(0, 1), varies="the map"))
    assert _kinds(diags) == ["frame-drift"]
    # the diagnostic carries the number, not just the verdict
    assert "2.0" in diags[0].detail or "0.5" in diags[0].detail


def test_shared_frame_holds_when_extents_match():
    a = _panel(_arc(key="probe"), xlim=(0.0, 2.0), ylim=(0.0, 2.0))
    b = _panel(_arc(key="probe"), xlim=(0.0, 2.0), ylim=(0.0, 2.0))
    assert _run([a, b], Correspondence(parts=(0, 1), varies="the map")) == []


def test_a_declared_rescale_exempts_the_frame_when_nothing_is_bound():
    a = _panel(_arc(key="probe"), xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    b = _panel(_arc(key="probe", color="#123456"), xlim=(0.0, 2.0), ylim=(0.0, 2.0))
    corr = Correspondence(parts=(0, 1), varies="the map", changes=("probe",),
                          frame="the image needs twice the room")
    assert _run([a, b], corr) == []


def test_a_rescale_cannot_exempt_the_fixed_set():
    """`frame=` buys length comparison in general, never the objects the
    figure declared unchanged — those get drawn at two sizes and the
    invariance claim is contradicted by the drawing."""
    a = _panel(_arc(key="anchor"), xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    b = _panel(_arc(key="anchor"), xlim=(0.0, 2.0), ylim=(0.0, 2.0))
    corr = Correspondence(parts=(0, 1), varies="the map",
                          frame="the image needs twice the room")
    diags = _run([a, b], corr)
    assert _kinds(diags) == ["fixed-set-rescaled"]
    assert "anchor" in diags[0].detail
    assert "0.5" in diags[0].detail


# --- shape -------------------------------------------------------------------

def test_parts_must_index_the_figure():
    fig = [_panel(_arc(key="probe"))]
    with pytest.raises(IndexError):
        _run(fig, Correspondence(parts=(0, 4), varies="x"))


def test_a_correspondence_needs_at_least_two_parts():
    with pytest.raises(ValueError):
        Correspondence(parts=(0,), varies="x")


def test_varies_is_required_prose():
    with pytest.raises(ValueError):
        Correspondence(parts=(0, 1), varies="  ")


def test_three_parts_compare_pairwise_against_the_first():
    fig = [_panel(_arc(key="p")), _panel(_arc(key="p")),
           _panel(_arc(key="p", color="#123456"))]
    diags = _run(fig, Correspondence(parts=(0, 1, 2), varies="r"))
    assert _kinds(diags) == ["residual"]
    assert "[2]" in diags[0].detail


def test_filled_vs_hollow_is_identity_not_style():
    """The convention carries the content — a stable fixed point drawn hollow
    in the second panel is a different object, not a restyled one."""
    fig = [_panel(Point((0.0, 0.0), key="fp", filled=True)),
           _panel(Point((1.0, 0.0), key="fp", filled=False))]
    diags = _run(fig, Correspondence(parts=(0, 1), varies="r"))
    assert _kinds(diags) == ["residual"]
    assert "filled" in diags[0].detail


def test_keyed_tags_a_producer_group_in_place():
    items = keyed("branch", _arc(), MathLabel("x", (0.0, 0.0)))
    assert [it.key for it in items] == ["branch", "branch"]


def test_labels_and_points_carry_keys_too():
    fig = [_panel(MathLabel("x", (0.0, 0.0), key="tag"), Point((0.0, 0.0), key="fp")),
           _panel(MathLabel("x", (1.0, 1.0), key="tag"))]
    diags = _run(fig, Correspondence(parts=(0, 1), varies="r"))
    assert _kinds(diags) == ["residual"]
    assert "fp" in diags[0].detail


# --- the semantic half: the only check on `varies` is a reader ---------------

def test_the_readback_prompt_asks_a_composite_what_is_bound():
    from figlib.readback import prompt_for
    fig = Figure(panels=[_panel(_arc()), _panel(_arc())])
    assert "BINDING" in prompt_for("x.png", fig)
    assert "this figure has 2 parts" in prompt_for("x.png", fig)


def test_a_single_scene_figure_gets_the_plain_prompt():
    from figlib.readback import prompt_for
    assert "BINDING" not in prompt_for("x.png", Scene())
    assert "BINDING" not in prompt_for("x.png")
