import numpy as np
import pytest

from gravity_sim import forces
from gravity_sim.constants import AU, SOLAR_MASS, G
from gravity_sim.forces import barnes_hut_accelerations, direct_accelerations, potential_energy, resolve_solver
from gravity_sim.octree import MAX_DEPTH, Octree

from .conftest import random_cluster


def test_two_body_acceleration_matches_newton() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [AU, 0.0, 0.0]])
    mass = np.array([SOLAR_MASS, 1.0e24])
    acc, pot = direct_accelerations(pos, mass)
    assert acc[1] == pytest.approx([-G * SOLAR_MASS / AU**2, 0, 0], rel=1e-12)
    assert acc[0] == pytest.approx([G * 1.0e24 / AU**2, 0, 0], rel=1e-12)
    assert pot[1] == pytest.approx(-G * SOLAR_MASS / AU, rel=1e-12)


def test_direct_sum_obeys_newtons_third_law(rng: np.random.Generator) -> None:
    pos, mass = random_cluster(rng, 200)
    acc, _ = direct_accelerations(pos, mass)
    net = (mass[:, None] * acc).sum(axis=0)
    scale = np.abs(mass[:, None] * acc).sum()
    assert np.linalg.norm(net) < 1e-12 * scale


def test_softened_potential() -> None:
    eps = 0.1 * AU
    pos = np.array([[0.0, 0.0, 0.0], [0.3 * AU, 0.0, 0.0]])
    mass = np.array([SOLAR_MASS, SOLAR_MASS])
    u = potential_energy(pos, mass, eps)
    assert u == pytest.approx(-G * SOLAR_MASS**2 / np.hypot(0.3 * AU, eps), rel=1e-12)


def test_chunked_direct_sum_matches_single_chunk(rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch) -> None:
    pos, mass = random_cluster(rng, 97)
    full_acc, full_pot = direct_accelerations(pos, mass, 0.01 * AU)
    monkeypatch.setattr(forces, "_CHUNK_ELEMENTS", 97 * 7)  # forces 14 chunks
    acc, pot = direct_accelerations(pos, mass, 0.01 * AU)
    np.testing.assert_allclose(acc, full_acc, rtol=1e-12)
    np.testing.assert_allclose(pot, full_pot, rtol=1e-12)


def test_coincident_unsoftened_bodies_produce_finite_forces() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [AU, 0.0, 0.0]])
    mass = np.full(3, SOLAR_MASS)
    acc, pot = direct_accelerations(pos, mass)
    assert np.isfinite(acc).all() and np.isfinite(pot).all()


def test_octree_mass_and_center_of_mass(rng: np.random.Generator) -> None:
    pos, mass = random_cluster(rng, 150)
    tree = Octree(pos, mass)
    assert tree.root.mass == pytest.approx(mass.sum(), rel=1e-12)
    com = (mass[:, None] * pos).sum(0) / mass.sum()
    np.testing.assert_allclose([tree.root.comx, tree.root.comy, tree.root.comz], com, rtol=1e-9, atol=1e-3)
    # Every body is in exactly one leaf.
    leaves = [i for node in tree.iter_nodes() if node.children is None for i in node.bodies]
    assert sorted(leaves) == list(range(150))


def test_barnes_hut_with_theta_zero_is_exact(rng: np.random.Generator) -> None:
    pos, mass = random_cluster(rng, 120)
    exact, exact_pot = direct_accelerations(pos, mass)
    approx, pot, _ = barnes_hut_accelerations(pos, mass, theta=0.0)
    np.testing.assert_allclose(approx, exact, rtol=1e-9)
    np.testing.assert_allclose(pot, exact_pot, rtol=1e-9)


@pytest.mark.parametrize(("theta", "max_rms"), [(0.3, 0.005), (0.5, 0.02), (1.0, 0.08)])
def test_barnes_hut_accuracy_improves_with_smaller_theta(rng: np.random.Generator, theta: float, max_rms: float) -> None:
    pos, mass = random_cluster(rng, 400)
    exact, _ = direct_accelerations(pos, mass)
    approx, _, _ = barnes_hut_accelerations(pos, mass, theta=theta)
    rel = np.linalg.norm(approx - exact, axis=1) / np.linalg.norm(exact, axis=1)
    assert float(np.sqrt((rel**2).mean())) < max_rms


def test_octree_handles_coincident_bodies_without_infinite_recursion() -> None:
    pos = np.zeros((10, 3))
    pos[9] = [AU, AU, AU]
    mass = np.full(10, SOLAR_MASS)
    tree = Octree(pos, mass)
    assert tree.stats()["depth"] <= MAX_DEPTH
    acc, pot = tree.accelerations(0.5)
    assert np.isfinite(acc).all() and np.isfinite(pot).all()
    # The far body feels all nine coincident ones.
    assert np.linalg.norm(acc[9]) == pytest.approx(9 * G * SOLAR_MASS / (3 * AU**2), rel=1e-9)


def test_octree_single_body_has_no_self_force() -> None:
    tree = Octree(np.array([[1.0, 2.0, 3.0]]), np.array([5.0]))
    acc, pot = tree.accelerations(0.5)
    assert not acc.any() and not pot.any()


def test_octree_boxes_filter_and_limit(rng: np.random.Generator) -> None:
    pos, mass = random_cluster(rng, 64)
    tree = Octree(pos, mass)
    stats = tree.stats()
    assert len(tree.boxes("all")) == stats["nodes"]
    assert len(tree.boxes("leaf")) == stats["leaves"]
    assert len(tree.boxes("internal")) == stats["nodes"] - stats["leaves"]
    limited = tree.boxes("all", limit=5)
    assert len(limited) == 5 and limited[0][1] == tree.root.half


def test_octree_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        Octree(np.zeros((0, 3)), np.zeros(0))
    with pytest.raises(ValueError):
        Octree(np.zeros((3, 2)), np.zeros(3))
    with pytest.raises(ValueError):
        Octree(np.zeros((2, 3)), np.zeros(2)).accelerations(-1)


def test_resolve_solver() -> None:
    assert resolve_solver("auto", 10) == "direct"
    assert resolve_solver("auto", forces.AUTO_DIRECT_MAX_BODIES + 1) == "barnes-hut"
    assert resolve_solver("barnes-hut", 10) == "barnes-hut"
    with pytest.raises(ValueError):
        resolve_solver("magic", 10)  # type: ignore[arg-type]
