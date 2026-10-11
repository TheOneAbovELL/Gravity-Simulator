import math

import numpy as np
import pytest

from gravity_sim import simulation as simmod
from gravity_sim.constants import AU, DAY, EARTH_MASS, SOLAR_MASS, G
from gravity_sim.orbits import orbital_period
from gravity_sim.simulation import NumericalInstabilityError, Simulation, SimulationConfig

from .conftest import spec


def sun_earth(dt: float = DAY / 4, **cfg: object) -> Simulation:
    v = math.sqrt(G * (SOLAR_MASS + EARTH_MASS) / AU)
    m_ratio = EARTH_MASS / (SOLAR_MASS + EARTH_MASS)
    bodies = [
        spec("Sun", SOLAR_MASS, [-AU * m_ratio, 0, 0], [0, -v * m_ratio, 0], 1408.0, "star"),
        spec("Earth", EARTH_MASS, [AU * (1 - m_ratio), 0, 0], [0, v * (1 - m_ratio), 0], 5514.0),
    ]
    return Simulation(bodies, SimulationConfig(dt=dt, **cfg))  # type: ignore[arg-type]


def test_circular_orbit_closes_after_one_period() -> None:
    sim = sun_earth()
    period = orbital_period(AU, G * (SOLAR_MASS + EARTH_MASS))
    start = sim.pos[1] - sim.pos[0]
    steps = round(period / sim.config.dt)
    sim.update_config(dt=period / steps)  # land exactly on one period
    sim.step(steps)
    end = sim.pos[1] - sim.pos[0]
    assert np.linalg.norm(end - start) < 1e-4 * AU
    assert sim.time == pytest.approx(period)


def test_energy_error_stays_bounded_over_many_orbits() -> None:
    sim = sun_earth(dt=DAY)
    worst = 0.0
    for _ in range(20):  # ~20 years
        sim.step(365)
        worst = max(worst, abs(sim.energy().drift))
    assert worst < 1e-6  # leapfrog: bounded oscillation, no secular growth


def test_momentum_is_conserved() -> None:
    sim = sun_earth(dt=DAY)
    p0 = sim.momentum()
    sim.step(500)
    scale = SOLAR_MASS * 3e4
    assert np.linalg.norm(sim.momentum() - p0) < 1e-12 * scale


def test_leapfrog_is_time_reversible() -> None:
    sim = sun_earth(dt=DAY)
    start = sim.pos.copy()
    sim.step(200)
    sim.vel *= -1
    sim.step(200)
    np.testing.assert_allclose(sim.pos, start, atol=1e-6 * AU)


def test_first_step_has_no_spurious_kick() -> None:
    # Legacy code applied a full a*dt velocity kick before the first drift.
    sim = Simulation(
        [spec("A", SOLAR_MASS, [0, 0, 0], [0, 0, 0]), spec("B", 1.0, [AU, 0, 0], [0, 0, 0])],
        SimulationConfig(dt=3600.0, collisions="off"),
    )
    a = G * SOLAR_MASS / AU**2
    sim.step(1)
    assert AU - sim.pos[1, 0] == pytest.approx(0.5 * a * 3600.0**2, rel=1e-6)


def test_single_body_simulation_runs() -> None:
    sim = Simulation([spec("Lonely", SOLAR_MASS, [1, 2, 3], [10, 0, 0])])
    sim.step(10)
    assert sim.pos[0] == pytest.approx([1 + 10 * 10 * DAY, 2, 3])


def test_head_on_merge_conserves_mass_and_momentum_and_accounts_energy() -> None:
    bodies = [
        spec("Big", 2 * EARTH_MASS, [-1e7, 0, 0], [3000, 0, 0]),
        spec("Small", EARTH_MASS, [1e7, 0, 0], [-3000, 0, 0]),
        spec("Bystander", EARTH_MASS, [AU, 0, 0], [0, 0, 0]),
    ]
    sim = Simulation(bodies, SimulationConfig(dt=10.0, collisions="merge"))
    m0, p0 = sim.total_mass(), sim.momentum()
    big_id, small_id = sim.ids[0], sim.ids[1]
    sim.step(1200)
    assert sim.n == 2
    assert sim.total_mass() == pytest.approx(m0)
    np.testing.assert_allclose(sim.momentum(), p0, atol=1e-9 * EARTH_MASS * 3000)
    assert sim.merged_into == {small_id: big_id}
    assert sim.resolve_id(small_id) == big_id
    energy = sim.energy()
    # Merging removes the pair's binding energy (-G m1 m2 / r) along with the lost kinetic energy.
    assert abs(energy.collision_energy) > 0.5 * abs(energy.initial)
    # The collision change is accounted for and not reported as numerical error.
    assert abs(energy.drift) < 1e-4


def test_equal_masses_merge() -> None:
    # Legacy code only merged when one body was strictly heavier.
    sim = Simulation(
        [spec("A", EARTH_MASS, [0, 0, 0], [0, 0, 0]), spec("B", EARTH_MASS, [1e6, 0, 0], [0, 0, 0])],
        SimulationConfig(dt=1.0),
    )
    sim.step(1)
    assert sim.n == 1 and sim.mass[0] == pytest.approx(2 * EARTH_MASS)


def test_bounce_mode_keeps_bodies_and_conserves_momentum() -> None:
    bodies = [
        spec("A", EARTH_MASS, [-1e7, 0, 0], [20000, 0, 0]),
        spec("B", EARTH_MASS, [1e7, 0, 0], [-20000, 0, 0]),
    ]  # above escape speed
    sim = Simulation(bodies, SimulationConfig(dt=60.0, collisions="bounce", restitution=1.0))
    p0 = sim.momentum()
    sim.step(200)
    assert sim.n == 2 and sim.collision_events >= 1
    np.testing.assert_allclose(sim.momentum(), p0, atol=1e-9 * EARTH_MASS * 20000)
    assert sim.vel[0, 0] < 0 < sim.vel[1, 0]  # they rebounded


def test_collisions_off_lets_bodies_pass() -> None:
    # Offset by 1 m: unsoftened point masses passing exactly through each other are singular.
    bodies = [spec("A", 1.0, [-10, 0.5, 0], [1, 0, 0]), spec("B", 1.0, [10, -0.5, 0], [-1, 0, 0])]
    sim = Simulation(bodies, SimulationConfig(dt=1.0, collisions="off"))
    sim.step(30)
    assert sim.n == 2 and sim.pos[0, 0] > sim.pos[1, 0]


def test_numerical_instability_rolls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = sun_earth()
    sim.step(3)
    before = (sim.pos.copy(), sim.vel.copy(), sim.time, sim.steps)

    def broken(pos: np.ndarray, mass: np.ndarray, softening: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        return np.full_like(pos, np.nan), np.zeros(len(mass))

    monkeypatch.setattr(simmod, "direct_accelerations", broken)
    with pytest.raises(NumericalInstabilityError):
        sim.step(1)
    np.testing.assert_array_equal(sim.pos, before[0])
    np.testing.assert_array_equal(sim.vel, before[1])
    assert (sim.time, sim.steps) == before[2:]


def test_update_config_validates_and_rebaselines_energy() -> None:
    sim = sun_earth()
    sim.step(50)
    with pytest.raises(ValueError):
        sim.update_config(dt=-1.0)
    with pytest.raises(ValueError):
        sim.update_config(restitution=2.0)
    assert sim.config.dt == DAY / 4  # unchanged after a rejected update
    sim.update_config(softening=0.01 * AU)
    assert sim.energy().drift == pytest.approx(0.0, abs=1e-15)
    assert sim.energy().collision_energy == 0.0


def test_switching_to_barnes_hut_keeps_physics_consistent() -> None:
    sim = sun_earth(dt=DAY)
    sim.update_config(solver="barnes-hut", theta=0.5)
    assert sim.active_solver == "barnes-hut"
    sim.step(365)
    assert abs(sim.energy().drift) < 1e-6  # two bodies: BH is exact
    assert sim.energy().exact


def test_energy_is_marked_approximate_for_large_barnes_hut_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(simmod, "EXACT_ENERGY_MAX_BODIES", 1)
    sim = sun_earth(dt=DAY, solver="barnes-hut")
    assert not sim.energy().exact


def test_energy_cache_refreshes_after_step() -> None:
    sim = sun_earth()
    e1 = sim.energy()
    assert sim.energy() is e1
    sim.step(1)
    assert sim.energy() is not e1


def test_trails_sample_and_resize() -> None:
    sim = sun_earth(trail_length=5, trail_every=2)
    sim.step(20)
    trail = sim.trail(sim.ids[1])
    assert trail.shape == (5, 3)
    np.testing.assert_allclose(trail[-1], sim.pos[1])
    assert len(sim.com_trail()) == 5
    sim.update_config(trail_length=3)
    np.testing.assert_allclose(sim.trail(sim.ids[1]), trail[-3:])
    sim.clear_trails()
    assert len(sim.trail(sim.ids[1])) == 0
    assert sim.trail(999).shape == (0, 3)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(theta=-0.1).validated()
    with pytest.raises(ValueError):
        SimulationConfig(collisions="explode").validated()  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SimulationConfig(trail_every=0).validated()
    with pytest.raises(ValueError):
        Simulation([])


def test_body_view_and_aggregates() -> None:
    sim = sun_earth()
    view = sim.body(1)
    assert view.name == "Earth" and view.speed == pytest.approx(float(np.linalg.norm(sim.vel[1])))
    np.testing.assert_allclose(sim.center_of_mass(), 0.0, atol=1e-3)
    np.testing.assert_allclose(sim.com_velocity(), 0.0, atol=1e-12)
    assert np.linalg.norm(sim.angular_momentum()) > 0
    assert sim.octree().root.mass == pytest.approx(sim.total_mass())
