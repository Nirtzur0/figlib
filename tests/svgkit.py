"""Shared SVG assertion helpers: parse once, query by localname/attrs.

ET.fromstring namespaces every tag ('{http://...svg}path'); each helper
strips that so tests speak plain SVG.
"""

import xml.etree.ElementTree as ET


def svg_root(svg_str: str) -> ET.Element:
    return ET.fromstring(svg_str)


def tag(el: ET.Element) -> str:
    """Namespace-stripped localname."""
    return el.tag.rsplit("}", 1)[-1]


def find_by(root: ET.Element, localname: str | None = None,
            **attrs) -> list[ET.Element]:
    """Every element matching localname (None = any) and all attrs.

    Attr names map Python -> SVG: trailing '_' dropped (class_), then
    '_' -> '-' (fill_opacity -> fill-opacity)."""
    want = {k.rstrip("_").replace("_", "-"): v for k, v in attrs.items()}
    return [el for el in root.iter()
            if (localname is None or tag(el) == localname)
            and all(el.get(k) == v for k, v in want.items())]


def path_cmd_counts(d: str) -> dict[str, int]:
    """Command letter -> count for an SVG path data string."""
    out: dict[str, int] = {}
    for ch in d:
        if ch.isalpha():
            out[ch] = out.get(ch, 0) + 1
    return out


def patterns(root: ET.Element) -> list[ET.Element]:
    """Every <pattern> def in the document."""
    return [el for el in root.iter() if tag(el) == "pattern"]


def has_clip(root: ET.Element) -> bool:
    return any(tag(el) == "clipPath" for el in root.iter())
