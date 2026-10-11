import math

import numpy as np
import pytest

from gravity_sim.constants import AU, DAY, SOLAR_MASS, G
from gravity_sim.forces import potential_energy
from gravity_sim.orbits import elements_to_state, orbital_period, solve_kepler, true_anomaly_from_mean
from gravity_sim.presets import FIGURE8_PERIOD, PRESETS, build_scenario, planet
from gravity_sim.simulation import Simulation

MU_SUN = G * SOLAR_MASS


@pytest.mark.parametrize("e", [0.0, 0.2, 0.7, 0.95])
@pytest.mark.parametrize("m", [-3.0, 0.1, 1.0, 2.5, 7.0])
def test_kepler_solution_satisfies_equation(e: float, m: float) -> None:
    big_e = solve_kepler(m, e)
    assert big_e - e * math.sin(big_e) == pytest.approx(math.remainder(m, 2 * math.pi), abs=1e-12)


def test_elements_to_state_obeys_vis_viva_and_inclination() -> None:
    a, e, inc = 2.0 * AU, 0.3, math.radians(20)
    for nu in np.linspace(0, 2 * math.pi, 7):
        r, v = elements_to_state(a, e, inc, math.radians(40), math.radians(70), float(nu), MU_SUN)
        assert float(v @ v) == pytest.approx(MU_SUN * (2 / np.linalg.norm(r) - 1 / a), rel=1e-12)
        h = np.cross(r, v)
        assert math.acos(h[2] / np.linalg.norm(h)) == pytest.approx(inc, abs=1e-12)


def test_true_anomaly_is_zero_at_periapsis() -> None:
    assert true_anomaly_from_mean(0.0, 0.5) == pytest.approx(0.0)
    assert abs(true_anomaly_from_mean(math.pi, 0.5)) == pytest.approx(math.pi)


@pytest.mark.parametrize(
    ("name", "distance_au", "longitude_deg"),
    [
        ("Earth", 0.9833, 100.4),  # J2000 is two days before perihelion; Earth opposite the Sun (~280 deg)
        ("Jupiter", 4.965, 36.3),
    ],
)
def test_j2000_planet_positions_are_realistic(name: str, distance_au: float, longitude_deg: float) -> None:
    body = planet(name)
    assert np.linalg.norm(body.position) / AU == pytest.approx(distance_au, abs=0.01)
    longitude = math.degrees(math.atan2(body.position[1], body.position[0])) % 360
    assert longitude == pytest.approx(longitude_deg, abs=1.0)


@pytest.mark.parametrize("key", list(PRESETS))
def test_presets_build_in_barycentric_frame_and_run_stably(key: str) -> None:
    scenario = build_scenario(key)
    sim = Simulation(scenario.bodies, scenario.config)
    total_mass = sim.total_mass()
    assert np.linalg.norm(sim.center_of_mass()) < 1e-6 * scenario.view_radius
    assert np.linalg.norm(sim.momentum()) < 1e-9 * total_mass * 1e5
    sim.step(100)
    assert abs(sim.energy().drift) < 1e-4
    focus = scenario.focus
    assert focus in ("com", "origin") or focus in sim.names


def test_generated_presets_are_deterministic_per_seed() -> None:
    a = build_scenario("cluster", 50, seed=7)
    b = build_scenario("cluster", 50, seed=7)
    c = build_scenario("cluster", 50, seed=8)
    np.testing.assert_array_equal(a.bodies[3].position, b.bodies[3].position)
    assert not np.array_equal(a.bodies[3].position, c.bodies[3].position)
    assert len(build_scenario("galaxies", 40).bodies) == 40


def test_cluster_is_virialised() -> None:
    scenario = build_scenario("cluster", 80)
    m = np.array([b.mass for b in scenario.bodies])
    pos = np.array([b.position for b in scenario.bodies])
    vel = np.array([b.velocity for b in scenario.bodies])
    kinetic = 0.5 * float((m * (vel * vel).sum(1)).sum())
    potential = potential_energy(pos, m, scenario.config.softening)
    assert 2 * kinetic / abs(potential) == pytest.approx(1.0, rel=1e-9)


def test_figure_eight_returns_after_one_period() -> None:
    scenario = build_scenario("figure8")
    sim = Simulation(scenario.bodies, scenario.config)
    start = sim.pos.copy()
    period = FIGURE8_PERIOD * math.sqrt(AU**3 / (G * SOLAR_MASS))
    steps = round(period / (0.05 * DAY))
    sim.update_config(dt=period / steps)
    sim.step(steps)
    assert np.abs(sim.pos - start).max() < 0.01 * AU


def test_solar_preset_year_lengths() -> None:
    # Sanity check against Kepler's third law for Earth.
    assert orbital_period(AU, MU_SUN) / DAY == pytest.approx(365.25, rel=1e-3)


@pytest.mark.parametrize(
    ("key", "bodies"),
    [("inner", 10), ("cluster", 1), ("cluster", 100_000), ("nonexistent", None)],
)
def test_invalid_preset_requests(key: str, bodies: int | None) -> None:
    with pytest.raises(ValueError):
        build_scenario(key, bodies)


def test_inner_preset_earth_closes_its_orbit_after_a_sidereal_year() -> None:
    scenario = build_scenario("inner")
    sim = Simulation(scenario.bodies, scenario.config)
    earth, sun = sim.names.index("Earth"), sim.names.index("Sun")
    start = sim.pos[earth] - sim.pos[sun]
    steps = 1461
    sim.update_config(dt=365.256363 * DAY / steps)
    sim.step(steps)
    assert np.linalg.norm(sim.pos[earth] - sim.pos[sun] - start) < 5e-5 * AU
