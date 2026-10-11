"""Human-readable formatting of SI quantities for display."""

from __future__ import annotations

import math

from .constants import AU, DAY, EARTH_MASS, HOUR, KM, MINUTE, SOLAR_MASS, YEAR


def _num(value: float, digits: int = 3) -> str:
    if value == 0 or not math.isfinite(value):
        return f"{value:g}"
    magnitude = abs(value)
    if 1e-3 <= magnitude < 1e5:
        decimals = max(0, digits - 1 - math.floor(math.log10(magnitude)))
        return f"{value:,.{decimals}f}"
    return f"{value:.{digits - 1}e}"


def format_mass(kg: float, ascii_only: bool = False) -> str:
    sun, earth = ("Msun", "Mearth") if ascii_only else ("M☉", "M⊕")
    if kg >= 0.01 * SOLAR_MASS:
        return f"{_num(kg / SOLAR_MASS)} {sun}"
    if kg >= 1e-6 * EARTH_MASS:
        return f"{_num(kg / EARTH_MASS)} {earth}"
    return f"{_num(kg)} kg"


def format_distance(m: float) -> str:
    if abs(m) >= 0.01 * AU:
        return f"{_num(m / AU)} AU"
    if abs(m) >= KM:
        return f"{_num(m / KM)} km"
    return f"{_num(m)} m"


def format_speed(ms: float) -> str:
    return f"{_num(ms / KM)} km/s" if abs(ms) >= 1.0 else f"{_num(ms)} m/s"


def format_acceleration(ms2: float) -> str:
    return f"{_num(ms2)} m/s²"


def format_density(kg_m3: float) -> str:
    return f"{_num(kg_m3 / 1000.0)} g/cm³"


def format_energy(joules: float) -> str:
    return f"{joules:.4e} J"


def format_duration(seconds: float) -> str:
    """Compact duration for a time step (e.g. ``6 h``, ``0.25 d``, ``2 yr``)."""
    s = abs(seconds)
    if s >= YEAR:
        return f"{_num(seconds / YEAR)} yr"
    if s >= DAY:
        return f"{_num(seconds / DAY)} d"
    if s >= HOUR:
        return f"{_num(seconds / HOUR)} h"
    if s >= MINUTE:
        return f"{_num(seconds / MINUTE)} min"
    return f"{_num(seconds)} s"


def format_sim_time(seconds: float) -> str:
    """Elapsed simulation time as ``Y yr D.D d`` (Julian years)."""
    years = int(seconds // YEAR)
    days = (seconds - years * YEAR) / DAY
    if years:
        return f"{years:,} yr {days:5.1f} d"
    return f"{days:.2f} d"
