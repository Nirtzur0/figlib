"""The wash channel and the discrete weight presets.

wash: large-area region fill, opaque, pre-blended toward the paper —
the channel that makes basin/region shading routine instead of a timid
alpha tint that goes muddy on textured paper. Slot-indexed like
categorical(), so a wash can carry correspondence with a stroke hue.

weights: three named levels replacing free width_scale drift; the
actor:background ratio is the hierarchy the grammar asks for.
"""

import pytest

from figlib.color import contrast
from figlib.style import WEIGHT_ACTOR, WEIGHT_BG, WEIGHT_CONTENT
from figlib.theme import CLEAN, RISO, WASH


def test_weight_presets_span_a_real_hierarchy():
    assert WEIGHT_BG < WEIGHT_CONTENT < WEIGHT_ACTOR
    assert WEIGHT_ACTOR / WEIGHT_BG >= 3.0


@pytest.mark.parametrize("theme", [CLEAN, RISO], ids=["clean", "riso"])
def test_wash_is_opaque_tagged_and_perceptible(theme):
    w = theme.wash(0)
    assert w.startswith("#") and len(w) == 7
    assert w.channel == WASH
    # perceptible on the WORST paper stop, at the default strength
    worst = min(contrast(w, p) for p in theme.paper_stops())
    assert worst >= 1.3


def test_wash_slots_are_distinct_and_deterministic():
    a, b = RISO.wash(0), RISO.wash(1)
    assert a != b
    assert a == RISO.wash(0)


def test_wash_neutral_default():
    w = RISO.wash()
    assert w.channel == WASH


def test_wash_slot_out_of_range_raises():
    with pytest.raises(IndexError):
        RISO.wash(99)


def test_wash_strength_monotone():
    """More strength = more ink = more contrast against paper."""
    light = RISO.wash(0, strength=0.15)
    heavy = RISO.wash(0, strength=0.5)
    paper = RISO.paper_stops()[0]
    assert contrast(heavy, paper) > contrast(light, paper)
