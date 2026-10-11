"""Built-in scenarios.

Every scenario is generated deterministically from ``(bodies, seed)`` and
comes with a time step, softening and camera framing chosen so that the
leapfrog integrator resolves the fastest relevant orbit with hundreds of
steps.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .bodies import BodySpec
from .colors import star_color
from .constants import AU, DAY, EARTH_MASS, GRAMS_PER_CM3, SOLAR_MASS, G
from .forces import potential_energy
from .orbits import circular_speed, elements_to_state, true_anomaly_from_mean
from .simulation import SimulationConfig


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    description: str
    bodies: tuple[BodySpec, ...]
    config: SimulationConfig
    steps_per_frame: int
    view_radius: float
    """Radius (m) around the focus that the initial camera should frame."""
    focus: str = "com"
    """``com``, ``origin`` or the name of a body."""
    elevation: float = 55.0
    """Initial camera elevation above the x-y plane, degrees."""


@dataclass(frozen=True)
class Preset:
    key: str
    title: str
    description: str
    builder: Callable[[int | None, int], Scenario]
    default_bodies: int | None = None
    """Adjustable body count (None = fixed system)."""
    max_bodies: int | None = None

    def build(self, bodies: int | None = None, seed: int = 1) -> Scenario:
        if bodies is not None:
            if self.default_bodies is None:
                raise ValueError(f"preset {self.key!r} has a fixed number of bodies")
            limit = self.max_bodies or bodies
            if not 2 <= bodies <= limit:
                raise ValueError(f"preset {self.key!r} supports 2..{limit} bodies, got {bodies}")
        return self.builder(bodies if bodies is not None else self.default_bodies, seed)


def to_barycentric(bodies: list[BodySpec]) -> list[BodySpec]:
    """Shift so that the centre of mass is at rest at the origin."""
    m = np.array([b.mass for b in bodies])
    pos = np.array([b.position for b in bodies])
    vel = np.array([b.velocity for b in bodies])
    com = (m[:, None] * pos).sum(0) / m.sum()
    vcom = (m[:, None] * vel).sum(0) / m.sum()
    return [BodySpec(b.name, b.mass, b.position - com, b.velocity - vcom, b.density, b.color, b.kind) for b in bodies]


# --------------------------------------------------------------- solar system
# JPL "Keplerian Elements for Approximate Positions of the Major Planets"
# (Standish), Table 1, valid 1800-2050, at epoch J2000.0, ecliptic frame:
# a [AU], e, I [deg], L (mean longitude) [deg], varpi (longitude of
# perihelion) [deg], Omega (longitude of ascending node) [deg].
# Masses come from the planetary-system GM values of JPL DE440 [km^3/s^2]
# (planet plus its moons, which travel with it). "Earth" is the Earth-Moon
# barycentre, so it carries the Moon's mass too.
_PLANETS = [
    # name,     a,           e,          I,            L,             varpi,         Omega,         GM [km3/s2],  rho [g/cm3], colour
    ("Mercury", 0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593, 22031.868, 5.427, "#b8b2a8"),
    ("Venus", 0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255, 324858.592, 5.243, "#e8cfa0"),
    ("Earth", 1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0, 403503.2356, 5.514, "#5b9bd5"),
    ("Mars", 1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891, 42828.3752, 3.934, "#d9603b"),
    ("Jupiter", 5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909, 126712764.1, 1.326, "#d8b08c"),
    ("Saturn", 9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448, 37940584.8, 0.687, "#e3d39a"),
    ("Uranus", 19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503, 5794556.4, 1.27, "#9fd8e0"),
    ("Neptune", 30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574, 6836527.1, 1.638, "#5b7fe0"),
]

SUN_DENSITY = 1.408 * GRAMS_PER_CM3


def sun() -> BodySpec:
    return BodySpec("Sun", SOLAR_MASS, np.zeros(3), np.zeros(3), SUN_DENSITY, "#ffd34d", "star")


def planet(name: str) -> BodySpec:
    """Heliocentric J2000 state of a major planet (Sun at rest at the origin)."""
    for row in _PLANETS:
        if row[0] == name:
            _, a, e, inc, mean_long, varpi, node, gm_km3, rho, color = row
            break
    else:
        raise KeyError(name)
    rad = math.radians
    mass = gm_km3 * 1e9 / G
    nu = true_anomaly_from_mean(rad(mean_long - varpi), e)
    mu = G * (SOLAR_MASS + mass)
    r, v = elements_to_state(a * AU, e, rad(inc), rad(node), rad(varpi - node), nu, mu)
    return BodySpec(name, mass, r, v, rho * GRAMS_PER_CM3, color, "planet")


def _solar_system(names: list[str]) -> list[BodySpec]:
    return to_barycentric([sun(), *(planet(n) for n in names)])


def _build_inner(_n: int | None, _seed: int) -> Scenario:
    return Scenario(
        key="inner",
        title="Inner solar system",
        description="The Sun with Mercury, Venus, Earth and Mars at their J2000.0 positions.",
        bodies=tuple(_solar_system(["Mercury", "Venus", "Earth", "Mars"])),
        config=SimulationConfig(dt=0.25 * DAY, trail_length=1400, trail_every=2),
        steps_per_frame=4,
        view_radius=1.8 * AU,
        focus="Sun",
        elevation=60.0,
    )


def _build_solar(_n: int | None, _seed: int) -> Scenario:
    return Scenario(
        key="solar",
        title="Solar system",
        description="The Sun and all eight planets at their J2000.0 positions (JPL approximate elements).",
        bodies=tuple(_solar_system([row[0] for row in _PLANETS])),
        config=SimulationConfig(dt=1.0 * DAY, trail_length=3000, trail_every=4),
        steps_per_frame=12,
        view_radius=32 * AU,
        focus="Sun",
        elevation=60.0,
    )


# -------------------------------------------------------------------- binary
def _build_binary(_n: int | None, _seed: int) -> Scenario:
    m1, m2 = 1.0 * SOLAR_MASS, 0.75 * SOLAR_MASS
    sep = 0.4 * AU
    mu = G * (m1 + m2)
    v_rel = circular_speed(mu, sep)
    bodies = [
        BodySpec(
            "Star A",
            m1,
            np.array([-sep * m2 / (m1 + m2), 0, 0]),
            np.array([0, -v_rel * m2 / (m1 + m2), 0]),
            SUN_DENSITY,
            star_color(1.0),
            "star",
        ),
        BodySpec(
            "Star B",
            m2,
            np.array([sep * m1 / (m1 + m2), 0, 0]),
            np.array([0, v_rel * m1 / (m1 + m2), 0]),
            SUN_DENSITY,
            star_color(0.75),
            "star",
        ),
    ]
    for name, r_au, inc_deg, mass, color in [
        ("Tatooine", 1.6, 0.0, 0.8 * EARTH_MASS, "#d9b26f"),
        ("Outer giant", 3.0, 12.0, 300 * EARTH_MASS, "#8fb8de"),
    ]:
        r, v = elements_to_state(r_au * AU, 0.0, math.radians(inc_deg), 0.0, 0.0, math.radians(90 + 40 * r_au), mu)
        bodies.append(BodySpec(name, mass, r, v, 4.0 * GRAMS_PER_CM3, color, "planet"))
    return Scenario(
        key="binary",
        title="Binary star",
        description="Two stars on a tight circular orbit with two circumbinary planets.",
        bodies=tuple(to_barycentric(bodies)),
        config=SimulationConfig(dt=0.1 * DAY, trail_length=1500, trail_every=4),
        steps_per_frame=10,
        view_radius=3.6 * AU,
        focus="com",
        elevation=50.0,
    )


# ---------------------------------------------------------------- figure-8
# Chenciner & Montgomery (2000) periodic three-body choreography, G = m = 1.
_FIG8_X = (-0.97000436, 0.24308753)
_FIG8_V3 = (-0.93240737, -0.86473146)
FIGURE8_PERIOD = 6.32591398  # in N-body time units


def _build_figure8(_n: int | None, _seed: int) -> Scenario:
    m = SOLAR_MASS
    length = AU
    v_unit = math.sqrt(G * m / length)
    x1 = np.array([_FIG8_X[0], _FIG8_X[1], 0.0]) * length
    v3 = np.array([_FIG8_V3[0], _FIG8_V3[1], 0.0]) * v_unit
    colors = ["#ff8a5c", "#7fd1ff", "#ffe27a"]
    states = [(x1, -v3 / 2), (-x1, -v3 / 2), (np.zeros(3), v3)]
    bodies = [BodySpec(f"Star {i + 1}", m, p, v, SUN_DENSITY, colors[i], "star") for i, (p, v) in enumerate(states)]
    return Scenario(
        key="figure8",
        title="Figure-eight three-body orbit",
        description="Three equal suns chasing each other along one figure-eight (Chenciner-Montgomery).",
        bodies=tuple(bodies),
        config=SimulationConfig(dt=0.1 * DAY, trail_length=900, trail_every=4),
        steps_per_frame=10,
        view_radius=1.3 * AU,
        focus="com",
        elevation=90.0,
    )


# ------------------------------------------------------------------- cluster
def _virialize(bodies: list[BodySpec], softening: float, ratio: float = 0.5) -> list[BodySpec]:
    """Scale velocities so that kinetic / |potential| equals ``ratio`` (0.5 = virial equilibrium)."""
    bodies = to_barycentric(bodies)
    m = np.array([b.mass for b in bodies])
    pos = np.array([b.position for b in bodies])
    vel = np.array([b.velocity for b in bodies])
    pot = potential_energy(pos, m, softening)
    kin = 0.5 * float(np.dot(m, (vel * vel).sum(1)))
    scale = math.sqrt(ratio * abs(pot) / kin) if kin > 0 else 0.0
    return [BodySpec(b.name, b.mass, b.position, b.velocity * scale, b.density, b.color, b.kind) for b in bodies]


def _build_cluster(n: int | None, seed: int) -> Scenario:
    n = n or 120
    rng = np.random.default_rng(seed)
    radius = 10 * AU
    softening = 0.2 * AU
    # Uniform density sphere, log-uniform masses in [0.2, 2] solar masses.
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    pos = directions * radius * rng.random(n)[:, None] ** (1 / 3)
    masses = np.exp(rng.uniform(math.log(0.2), math.log(2.0), n))
    vel = rng.normal(size=(n, 3))
    bodies = [
        BodySpec(f"Star {i + 1}", masses[i] * SOLAR_MASS, pos[i], vel[i], SUN_DENSITY, star_color(masses[i]), "star")
        for i in range(n)
    ]
    return Scenario(
        key="cluster",
        title="Star cluster",
        description="A virialised cluster of stars with a range of masses; close encounters eject members.",
        bodies=tuple(_virialize(bodies, softening)),
        config=SimulationConfig(dt=0.5 * DAY, softening=softening, trail_length=120, trail_every=4),
        steps_per_frame=4,
        view_radius=14 * AU,
        focus="com",
        elevation=35.0,
    )


# ----------------------------------------------------------------- galaxies
def _disk(
    prefix: str,
    n: int,
    rng: np.random.Generator,
    bh_mass: float,
    star_mass: float,
    r_min: float,
    r_max: float,
    tilt: float,
    color: str,
) -> list[BodySpec]:
    bodies = [BodySpec(f"{prefix} core", bh_mass, np.zeros(3), np.zeros(3), 1.0, "#ff5fd2", "blackhole")]
    radii = np.sort(rng.uniform(r_min, r_max, n))
    angles = rng.uniform(0, 2 * math.pi, n)
    ct, st = math.cos(tilt), math.sin(tilt)
    rot = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    for k, (r, a) in enumerate(zip(radii, angles, strict=True)):
        enclosed = bh_mass + star_mass * k  # stars inside this radius
        v = circular_speed(G * enclosed, r)
        p = rot @ np.array([r * math.cos(a), r * math.sin(a), 0.0])
        vel = rot @ np.array([-v * math.sin(a), v * math.cos(a), 0.0])
        bodies.append(BodySpec(f"{prefix} star {k + 1}", star_mass, p, vel, SUN_DENSITY, color, "star"))
    return bodies


def _build_galaxies(n: int | None, seed: int) -> Scenario:
    n = n or 200
    rng = np.random.default_rng(seed)
    per = max(1, (n - 2) // 2)
    bh = 1000 * SOLAR_MASS
    star = 1.0 * SOLAR_MASS
    a = _disk("Alpha", per, rng, bh, star, 10 * AU, 60 * AU, 0.0, "#9fc4ff")
    b = _disk("Beta", n - 2 - per, rng, bh, star, 10 * AU, 60 * AU, math.radians(60), "#ffc68a")
    m_a = sum(x.mass for x in a)
    m_b = sum(x.mass for x in b)
    separation = np.array([250 * AU, 60 * AU, 0.0])
    v_escape = math.sqrt(2 * G * (m_a + m_b) / float(np.linalg.norm(separation)))
    v_rel = np.array([-0.55 * v_escape, 0.0, 0.0])
    # Place the pair around their common centre of mass.
    offset_a = -separation * m_b / (m_a + m_b)
    offset_b = separation * m_a / (m_a + m_b)
    vel_a = -v_rel * m_b / (m_a + m_b)
    vel_b = v_rel * m_a / (m_a + m_b)
    bodies = [BodySpec(x.name, x.mass, x.position + offset_a, x.velocity + vel_a, x.density, x.color, x.kind) for x in a]
    bodies += [BodySpec(x.name, x.mass, x.position + offset_b, x.velocity + vel_b, x.density, x.color, x.kind) for x in b]
    return Scenario(
        key="galaxies",
        title="Galaxy collision",
        description="Two star disks orbiting black holes fall into each other and tear apart into tidal tails.",
        bodies=tuple(to_barycentric(bodies)),
        config=SimulationConfig(dt=1.0 * DAY, softening=2 * AU, trail_length=60, trail_every=8),
        steps_per_frame=8,
        view_radius=220 * AU,
        focus="com",
        elevation=40.0,
    )


PRESETS: dict[str, Preset] = {
    p.key: p
    for p in [
        Preset("inner", "Inner solar system", "Sun, Mercury, Venus, Earth and Mars", _build_inner),
        Preset("solar", "Solar system", "Sun and the eight planets", _build_solar),
        Preset("binary", "Binary star", "Circumbinary planets around two stars", _build_binary),
        Preset("figure8", "Figure-eight", "Periodic three-body choreography", _build_figure8),
        Preset("cluster", "Star cluster", "Virialised cluster of stars", _build_cluster, 120, 2000),
        Preset("galaxies", "Galaxy collision", "Two disk galaxies merging", _build_galaxies, 200, 2000),
    ]
}
DEFAULT_PRESET = "inner"


def build_scenario(key: str, bodies: int | None = None, seed: int = 1) -> Scenario:
    try:
        preset = PRESETS[key]
    except KeyError:
        raise ValueError(f"unknown preset {key!r}; choose from {', '.join(PRESETS)}") from None
    return preset.build(bodies, seed)
