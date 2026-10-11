"""Body descriptions used to build a simulation.

A :class:`BodySpec` is an immutable, validated description of a body's initial
state. The :class:`~gravity_sim.simulation.Simulation` stores the live state in
contiguous numpy arrays; specs are only the input format.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .constants import C, G

BodyKind = Literal["star", "planet", "blackhole", "particle"]
BODY_KINDS: tuple[BodyKind, ...] = ("star", "planet", "blackhole", "particle")

_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


def radius_from_density(mass: float, density: float) -> float:
    """Radius (m) of a uniform sphere of ``mass`` (kg) and ``density`` (kg/m^3)."""
    return (3.0 * mass / (4.0 * math.pi * density)) ** (1.0 / 3.0)


def schwarzschild_radius(mass: float) -> float:
    """Event-horizon radius (m) of a non-rotating black hole of ``mass`` (kg)."""
    return 2.0 * G * mass / (C * C)


def body_radius(kind: BodyKind, mass: float, density: float) -> float:
    """Physical collision radius for a body of the given kind."""
    if kind == "blackhole":
        return schwarzschild_radius(mass)
    return radius_from_density(mass, density)


def _as_vector(value: object, name: str) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float64)
    if vec.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector, got shape {vec.shape}")
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{name} must be finite, got {vec}")
    return vec


@dataclass(frozen=True)
class BodySpec:
    """Validated initial state of one body (SI units).

    Attributes:
        name: Display name.
        mass: Mass in kg (> 0).
        position: Position in m.
        velocity: Velocity in m/s.
        density: Mean density in kg/m^3 (> 0). Ignored for black holes, whose
            radius is the Schwarzschild radius.
        color: ``#rrggbb`` display color.
        kind: One of ``star``, ``planet``, ``blackhole`` or ``particle``.
    """

    name: str
    mass: float
    position: np.ndarray
    velocity: np.ndarray
    density: float = 5500.0
    color: str = "#ffffff"
    kind: BodyKind = "planet"
    radius: float = field(init=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("body name must not be empty")
        if not (math.isfinite(self.mass) and self.mass > 0):
            raise ValueError(f"{self.name}: mass must be a positive finite number, got {self.mass}")
        if not (math.isfinite(self.density) and self.density > 0):
            raise ValueError(f"{self.name}: density must be a positive finite number, got {self.density}")
        if self.kind not in BODY_KINDS:
            raise ValueError(f"{self.name}: kind must be one of {BODY_KINDS}, got {self.kind!r}")
        if not _HEX_COLOR.match(self.color):
            raise ValueError(f"{self.name}: color must look like '#rrggbb', got {self.color!r}")
        object.__setattr__(self, "position", _as_vector(self.position, f"{self.name}.position"))
        object.__setattr__(self, "velocity", _as_vector(self.velocity, f"{self.name}.velocity"))
        object.__setattr__(self, "radius", body_radius(self.kind, self.mass, self.density))
