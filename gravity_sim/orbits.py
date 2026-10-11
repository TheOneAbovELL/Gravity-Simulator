"""Two-body orbital mechanics helpers (Keplerian elements to state vectors)."""

from __future__ import annotations

import math

import numpy as np


def solve_kepler(mean_anomaly: float, eccentricity: float, tol: float = 1e-14) -> float:
    """Solve Kepler's equation ``E - e sin E = M`` for the eccentric anomaly E (rad)."""
    if not 0 <= eccentricity < 1:
        raise ValueError("only elliptical orbits (0 <= e < 1) are supported")
    m = math.remainder(mean_anomaly, 2 * math.pi)
    e_anom = m if eccentricity < 0.8 else math.pi * math.copysign(1.0, m)
    for _ in range(100):
        f = e_anom - eccentricity * math.sin(e_anom) - m
        step = f / (1 - eccentricity * math.cos(e_anom))
        e_anom -= step
        if abs(step) < tol:
            break
    return e_anom


def true_anomaly_from_mean(mean_anomaly: float, eccentricity: float) -> float:
    e_anom = solve_kepler(mean_anomaly, eccentricity)
    return 2 * math.atan2(
        math.sqrt(1 + eccentricity) * math.sin(e_anom / 2),
        math.sqrt(1 - eccentricity) * math.cos(e_anom / 2),
    )


def elements_to_state(
    a: float,
    e: float,
    inclination: float,
    ascending_node: float,
    arg_periapsis: float,
    true_anomaly: float,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Keplerian elements to position (m) and velocity (m/s).

    Angles are in radians; ``a`` in metres; ``mu = G (M + m)`` in m^3/s^2.
    The reference plane is the x-y plane with x towards the reference
    direction (the ecliptic/equinox frame for solar-system data).
    """
    p = a * (1 - e * e)
    cos_nu, sin_nu = math.cos(true_anomaly), math.sin(true_anomaly)
    r = p / (1 + e * cos_nu)
    r_pf = np.array([r * cos_nu, r * sin_nu, 0.0])
    v_pf = math.sqrt(mu / p) * np.array([-sin_nu, e + cos_nu, 0.0])

    co, so = math.cos(ascending_node), math.sin(ascending_node)
    ci, si = math.cos(inclination), math.sin(inclination)
    cw, sw = math.cos(arg_periapsis), math.sin(arg_periapsis)
    rot = np.array(
        [
            [co * cw - so * sw * ci, -co * sw - so * cw * ci, so * si],
            [so * cw + co * sw * ci, -so * sw + co * cw * ci, -co * si],
            [sw * si, cw * si, ci],
        ]
    )
    return rot @ r_pf, rot @ v_pf


def circular_speed(mu: float, r: float) -> float:
    return math.sqrt(mu / r)


def orbital_period(a: float, mu: float) -> float:
    return 2 * math.pi * math.sqrt(a**3 / mu)
