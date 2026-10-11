"""Small colour utilities (hex parsing, blending, a perceptual colormap)."""

from __future__ import annotations

import math
from itertools import pairwise


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    if len(c) != 6:
        raise ValueError(f"expected #rrggbb, got {color!r}")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = (max(0, min(255, round(v))) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def blend(a: str, b: str, t: float) -> str:
    """Linear blend from colour ``a`` (t=0) to ``b`` (t=1)."""
    t = max(0.0, min(1.0, t))
    ra, ga, ba = hex_to_rgb(a)
    rb, gb, bb = hex_to_rgb(b)
    return rgb_to_hex((ra + (rb - ra) * t, ga + (gb - ga) * t, ba + (bb - ba) * t))


# Anchor points of matplotlib's "plasma" colormap (perceptually uniform,
# readable on a dark background, colour-blind friendly ordering).
_PLASMA = [
    (13, 8, 135),
    (84, 2, 163),
    (139, 10, 165),
    (185, 50, 137),
    (219, 92, 104),
    (244, 136, 73),
    (254, 188, 43),
    (240, 249, 33),
]


def colormap(value: float) -> str:
    """Map ``value`` in [0, 1] onto the plasma colormap (clamped; NaN -> 0)."""
    if math.isnan(value):
        value = 0.0
    value = max(0.0, min(1.0, value))
    pos = value * (len(_PLASMA) - 1)
    i = min(int(pos), len(_PLASMA) - 2)
    t = pos - i
    a, b = _PLASMA[i], _PLASMA[i + 1]
    return rgb_to_hex((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t))


def star_color(mass_solar: float) -> str:
    """Approximate main-sequence colour for a star of the given mass (solar masses)."""
    # Rough temperature ordering: M (red) -> G (yellow-white) -> B (blue-white).
    stops = [(0.1, "#ff7a4d"), (0.5, "#ffb36b"), (1.0, "#fff1c9"), (2.0, "#e4ecff"), (8.0, "#9fbcff")]
    if mass_solar <= stops[0][0]:
        return stops[0][1]
    for (m0, c0), (m1, c1) in pairwise(stops):
        if mass_solar <= m1:
            t = (math.log(mass_solar) - math.log(m0)) / (math.log(m1) - math.log(m0))
            return blend(c0, c1, t)
    return stops[-1][1]
