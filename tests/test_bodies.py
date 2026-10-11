import math

import numpy as np
import pytest

from gravity_sim.bodies import BodySpec, radius_from_density, schwarzschild_radius
from gravity_sim.constants import EARTH_MASS, GRAMS_PER_CM3, SOLAR_MASS


def test_earth_radius_from_mass_and_density() -> None:
    r = radius_from_density(EARTH_MASS, 5.514 * GRAMS_PER_CM3)
    assert r == pytest.approx(6.371e6, rel=2e-3)


def test_schwarzschild_radius_of_the_sun_is_about_3_km() -> None:
    assert schwarzschild_radius(SOLAR_MASS) == pytest.approx(2953.3, rel=1e-3)


def test_blackhole_spec_uses_schwarzschild_radius() -> None:
    bh = BodySpec("BH", 10 * SOLAR_MASS, np.zeros(3), np.zeros(3), kind="blackhole")
    assert bh.radius == pytest.approx(schwarzschild_radius(10 * SOLAR_MASS))


def test_integer_vectors_are_converted_to_float() -> None:
    # The legacy code crashed on integer inputs (int64 arrays in-place updated with floats).
    b = BodySpec("A", 1.0, (0, 0, 0), (1, 2, 3))  # type: ignore[arg-type]
    assert b.position.dtype == np.float64 and b.velocity.dtype == np.float64


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mass": 0.0},
        {"mass": -1.0},
        {"mass": math.nan},
        {"density": 0.0},
        {"color": "red"},
        {"kind": "comet"},
        {"position": np.zeros(2)},
        {"velocity": np.array([0.0, math.inf, 0.0])},
        {"name": ""},
    ],
)
def test_invalid_specs_are_rejected(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {"name": "A", "mass": 1.0, "position": np.zeros(3), "velocity": np.zeros(3)}
    base.update(kwargs)
    with pytest.raises(ValueError):
        BodySpec(**base)  # type: ignore[arg-type]
