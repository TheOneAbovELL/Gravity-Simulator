import itertools

import numpy as np
import pytest

from gravity_sim import collisions as col


def brute_force_overlaps(pos: np.ndarray, radii: np.ndarray) -> list[tuple[int, int]]:
    return [
        (i, j) for i, j in itertools.combinations(range(len(pos)), 2) if np.linalg.norm(pos[i] - pos[j]) < radii[i] + radii[j]
    ]


@pytest.mark.parametrize("force_fallback", [False, True])
def test_find_overlaps_matches_brute_force(
    rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch, force_fallback: bool
) -> None:
    if force_fallback:
        monkeypatch.setattr(col, "_SWEEP_MAX_CANDIDATES", 0)
    pos = rng.uniform(0, 100, size=(300, 3))
    radii = rng.uniform(0.5, 4.0, 300)
    assert col.find_overlaps(pos, radii) == brute_force_overlaps(pos, radii)


def test_find_overlaps_trivial_cases() -> None:
    assert col.find_overlaps(np.zeros((1, 3)), np.ones(1)) == []
    far = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    assert col.find_overlaps(far, np.ones(2)) == []


def test_overlap_groups_are_transitive() -> None:
    groups = col.overlap_groups(6, [(0, 1), (1, 2), (4, 5)])
    assert sorted(groups) == [[0, 1, 2], [4, 5]]


def test_merge_conserves_mass_momentum_center_of_mass_and_volume() -> None:
    masses = np.array([3.0, 1.0, 2.0])
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 2, 0]])
    vel = np.array([[1.0, 0, 0], [0.0, -3, 0], [0.0, 0, 5]])
    dens = np.array([1.0, 2.0, 4.0])
    merge = col.plan_merge([0, 1, 2], masses, pos, vel, dens)
    assert merge.survivor == 0 and merge.absorbed == (1, 2)
    assert merge.mass == pytest.approx(6.0)
    np.testing.assert_allclose(merge.mass * merge.velocity, (masses[:, None] * vel).sum(0))
    np.testing.assert_allclose(merge.position, (masses[:, None] * pos).sum(0) / 6.0)
    assert merge.mass / merge.density == pytest.approx((masses / dens).sum())


def test_merge_survivor_is_deterministic_for_equal_masses() -> None:
    m = np.array([1.0, 1.0])
    merge = col.plan_merge([0, 1], m, np.zeros((2, 3)), np.zeros((2, 3)), np.ones(2))
    assert merge.survivor == 0


def _kinetic(m: np.ndarray, v: np.ndarray) -> float:
    return 0.5 * float((m * (v * v).sum(1)).sum())


def test_elastic_bounce_conserves_momentum_and_energy() -> None:
    m = np.array([2.0, 1.0])
    r = np.array([1.0, 1.0])
    pos = np.array([[0.0, 0, 0], [1.5, 0.5, 0]])
    vel = np.array([[1.0, 0.2, 0], [-1.0, 0.0, 0.3]])
    p0, k0 = (m[:, None] * vel).sum(0), _kinetic(m, vel)
    com0 = (m[:, None] * pos).sum(0) / m.sum()
    assert col.resolve_bounces([(0, 1)], m, r, pos, vel, restitution=1.0) == 1
    np.testing.assert_allclose((m[:, None] * vel).sum(0), p0, atol=1e-12)
    assert _kinetic(m, vel) == pytest.approx(k0, rel=1e-12)
    np.testing.assert_allclose((m[:, None] * pos).sum(0) / m.sum(), com0, atol=1e-12)
    assert np.linalg.norm(pos[1] - pos[0]) == pytest.approx(2.0)  # separated to touching


def test_fully_inelastic_bounce_equalises_normal_velocity() -> None:
    m = np.array([1.0, 3.0])
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    vel = np.array([[2.0, 1.0, 0], [-1.0, 0.0, 0]])
    col.resolve_bounces([(0, 1)], m, np.ones(2), pos, vel, restitution=0.0)
    assert vel[0, 0] == pytest.approx(vel[1, 0])
    assert vel[0, 1] == pytest.approx(1.0)  # tangential velocity untouched


def test_separating_bodies_receive_no_impulse() -> None:
    m = np.ones(2)
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    vel = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
    assert col.resolve_bounces([(0, 1)], m, np.ones(2), pos, vel, restitution=1.0) == 0
    np.testing.assert_allclose(vel, [[-1.0, 0, 0], [1.0, 0, 0]])


def test_restitution_out_of_range() -> None:
    with pytest.raises(ValueError):
        col.resolve_bounces([], np.ones(2), np.ones(2), np.zeros((2, 3)), np.zeros((2, 3)), 1.5)
