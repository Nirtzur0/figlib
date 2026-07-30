"""Brand assets, drawn by the library they brand.

    make brand        # or: uv run python docs/brand/make_brand.py

Three artifacts, all on the theme's stock with its grain, all set in the
same STIX Two Math face the figures use:

    wordmark.{svg,png}   the README header
    mark.{svg,png}       512px square, for an avatar or a favicon
    social.png           1280x640, the card GitHub/Slack/X show for a link

The glyph is three ink circles out of register -- what a two-colour riso
pass actually looks like when the drum registration is off by a millimetre.
It names the house look without illustrating any one subject, and three
filled discs survive being shrunk to a 24px avatar, which no figure does.

Deliberately NOT a figure program: a logo argues nothing, so CLAIM and
EXPOSITION would be ceremony and the gates would be measuring the wrong
thing. It builds a Scene and calls the renderer directly -- same ink, same
grain, same typeface, no pretence that it is part of the corpus.
"""

from pathlib import Path

import numpy as np

from figlib.render import save
from figlib.scene import Curve, FilledCurve, MathLabel, Scene
from figlib.style import Role
from figlib.theme import RISO

HERE = Path(__file__).parent
REPO = HERE.parent.parent

#: (offset, correspondence slot) in DRAW ORDER -- indigo, ochre, then brick
#: last. Order is not cosmetic here: three translucent discs desaturate
#: toward grey wherever all three overlap, and the last pass decides what
#: that centre reads as. Indigo last gives a muddy purple-grey; brick last
#: gives a warm centre and keeps the mark printed rather than smudged.
#: Slots 0 and 1 with the far end of the order: two of the first three
#: (indigo, sky) are both blue, and blue over blue goes muddy instead of
#: printed.
#:
#: Offsets are in units of the circle radius, and small on purpose. At 0.62
#: the three discs read as a Venn diagram, which is a different idea
#: entirely; at 0.30 they read as one disc printed three times with the drum
#: a millimetre out -- which is the thing being named.
LAYERS = [((0.0, 0.30), 0), ((-0.26, -0.15), 1), ((0.26, -0.15), 6)]


def registration_glyph(centre: tuple[float, float], r: float) -> list:
    """Three discs out of register. Multiply-ish overlap comes from opacity,
    not a blend mode: SVG's mix-blend-mode is not safe in every consumer that
    renders a README, and a favicon has to survive being flattened."""
    th = np.linspace(0.0, 2.0 * np.pi, 160)
    ring = np.column_stack([np.cos(th), np.sin(th)])
    cx, cy = centre
    return [FilledCurve(ring * r + (cx + dx * r, cy + dy * r),
                        color=RISO.categorical(slot), opacity=0.72,
                        outline=False, grain=0.35)
            for (dx, dy), slot in LAYERS]


def _lockup(s: Scene, x: float, y: float, scale: float, tagline: str,
            rule: float = 1.95) -> None:
    """Glyph, wordmark, rule, tagline -- one unit, so the three assets cannot
    drift apart. `x, y` is the left edge and the optical centre of the mark."""
    s.add(*registration_glyph((x + 0.26 * scale, y + 0.01 * scale),
                              0.24 * scale))
    tx = x + 0.62 * scale
    s.add(MathLabel(r"\mathrm{figlib}", (tx, y + 0.10 * scale),
                    ha="left", va="baseline", size_pt=44 * scale))
    s.add(Curve(np.array([[tx, y - 0.09 * scale],
                          [tx + rule * scale, y - 0.09 * scale]]),
                role=Role.MUTED, width_scale=0.45))
    s.add(MathLabel(rf"\text{{{tagline}}}", (tx, y - 0.15 * scale),
                    ha="left", va="top", size_pt=13 * scale,
                    role=Role.ANNOTATION))


def wordmark() -> None:
    s = Scene(xlim=(0.0, 2.88), ylim=(-0.36, 0.36), height_px=170)
    _lockup(s, 0.10, 0.03, 1.0, "figures as gated programs")
    save(s, HERE / "wordmark", RISO, width_px=680)


def mark() -> None:
    s = Scene(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
    s.add(*registration_glyph((0.0, 0.0), 0.34))
    save(s, HERE / "mark", RISO, width_px=512)


def social() -> None:
    """1280x640. The card is drawn by figlib; the figure on it is composited
    from its GROUNDLESS render, which is what that variant is for -- the card
    owns the background, so pasting the papered render would seam."""
    from PIL import Image

    s = Scene(xlim=(0.0, 2.0), ylim=(0.0, 1.0), height_px=640)
    _lockup(s, 0.12, 0.66, 0.72, "figures as gated programs", rule=0.80)
    # the pipeline set in the same face as the figures, because it IS the
    # product -- prose here would say less in more room
    s.add(MathLabel(r"\mathrm{compute} \to \mathrm{build} \to "
                    r"\mathrm{gates} \to \mathrm{render}",
                    (0.16, 0.42), ha="left", va="top", size_pt=20))
    s.add(MathLabel(r"\text{github.com/Nirtzur0/figlib}", (0.16, 0.28),
                    ha="left", va="top", size_pt=11, role=Role.ANNOTATION))
    _, png = save(s, HERE / "_social_card", RISO, width_px=1280)

    card = Image.open(png).convert("RGBA")
    fig = Image.open(REPO / "figures" / "out" / "complex"
                     / "vca_fig14_volcanoes_transparent.png").convert("RGBA")
    box_w, box_h = 980, 1080
    k = min(box_w / fig.width, box_h / fig.height)
    fig = fig.resize((round(fig.width * k), round(fig.height * k)),
                     Image.LANCZOS)
    card.alpha_composite(fig, (card.width - fig.width - 60,
                               (card.height - fig.height) // 2))
    card.convert("RGB").save(HERE / "social.png", quality=95)
    for stray in ("_social_card.svg", "_social_card.png"):
        (HERE / stray).unlink()


if __name__ == "__main__":
    wordmark()
    mark()
    social()
    print(f"brand: {HERE}/wordmark.png · mark.png · social.png")
