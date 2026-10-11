from __future__ import annotations

import numpy as np
import pytest

from gravity_sim.bodies import BodySpec
from gravity_sim.constants import AU, SOLAR_MASS


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


def random_cluster(rng: np.random.Generator, n: int, radius: float = 10 * AU) -> tuple[np.ndarray, np.ndarray]:
    pos = rng.normal(size=(n, 3)) * radius
    mass = rng.uniform(0.2, 2.0, n) * SOLAR_MASS
    return pos, mass


def spec(name: str, mass: float, pos: list[float], vel: list[float], density: float = 5500.0, kind: str = "planet") -> BodySpec:
    return BodySpec(name, mass, np.array(pos, dtype=float), np.array(vel, dtype=float), density, "#ffffff", kind)  # type: ignore[arg-type]
