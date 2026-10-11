"""The N-body simulation engine.

State is held in contiguous numpy arrays (SI units). Time integration uses the
kick-drift-kick leapfrog (velocity Verlet) scheme, which is second order,
time-reversible and symplectic, so energy errors stay bounded instead of
accumulating for a fixed time step.

Energy bookkeeping distinguishes *numerical* error from deliberate changes
made by collision handling: the mechanical energy removed by collisions is
accumulated in ``collision_energy`` so that :attr:`EnergyReport.drift`
measures only the integration/approximation error. Note that a merge can
*add* mechanical energy (``collision_energy < 0``): the merged pair's mutual
potential energy -G m1 m2 / r disappears, standing in for the heat and
binding energy a point-mass model cannot represent.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from . import collisions as col
from .bodies import BodyKind, BodySpec, body_radius
from .constants import DAY
from .forces import SOLVERS, Solver, barnes_hut_accelerations, direct_accelerations, potential_energy, resolve_solver
from .octree import Octree
from .trails import Trail

EXACT_ENERGY_MAX_BODIES = 5000
"""Above this, energy diagnostics use the solver's (approximate) potentials."""

MAX_BODIES = 20_000


class NumericalInstabilityError(RuntimeError):
    """Raised when a step produces non-finite values. State is rolled back."""


@dataclass(frozen=True)
class SimulationConfig:
    """Tunable simulation parameters.

    Attributes:
        dt: Integration time step in seconds.
        solver: ``auto``, ``direct`` or ``barnes-hut``.
        theta: Barnes-Hut opening angle (0 = exact; 0.3-0.7 typical).
        softening: Plummer softening length in m (0 = pure Newtonian).
        collisions: ``merge``, ``bounce`` or ``off``.
        restitution: Coefficient of restitution for ``bounce`` (0..1).
        trail_length: Number of trail samples kept per body.
        trail_every: Record a trail sample every this many steps.
    """

    dt: float = DAY
    solver: Solver = "auto"
    theta: float = 0.5
    softening: float = 0.0
    collisions: col.CollisionMode = "merge"
    restitution: float = 0.5
    trail_length: int = 600
    trail_every: int = 1

    def validated(self) -> SimulationConfig:
        if not (math.isfinite(self.dt) and self.dt > 0):
            raise ValueError(f"dt must be a positive number of seconds, got {self.dt}")
        if self.solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}, got {self.solver!r}")
        if not (math.isfinite(self.theta) and 0 <= self.theta <= 2):
            raise ValueError(f"theta must be in [0, 2], got {self.theta}")
        if not (math.isfinite(self.softening) and self.softening >= 0):
            raise ValueError(f"softening must be >= 0, got {self.softening}")
        if self.collisions not in col.COLLISION_MODES:
            raise ValueError(f"collisions must be one of {col.COLLISION_MODES}, got {self.collisions!r}")
        if not 0 <= self.restitution <= 1:
            raise ValueError(f"restitution must be in [0, 1], got {self.restitution}")
        if not 1 <= self.trail_length <= 100_000:
            raise ValueError(f"trail_length must be in [1, 100000], got {self.trail_length}")
        if not 1 <= self.trail_every <= 10_000:
            raise ValueError(f"trail_every must be in [1, 10000], got {self.trail_every}")
        return self


@dataclass(frozen=True)
class EnergyReport:
    """Energy diagnostics in joules."""

    kinetic: float
    potential: float
    initial: float
    collision_energy: float
    """Net mechanical energy removed by collision handling (may be negative)."""
    exact: bool

    @property
    def total(self) -> float:
        return self.kinetic + self.potential

    @property
    def drift(self) -> float:
        """Relative numerical energy error ``(E + E_collisions - E0) / |E0|``."""
        if self.initial == 0:
            return math.nan
        return (self.total + self.collision_energy - self.initial) / abs(self.initial)


@dataclass(frozen=True)
class BodyView:
    """Read-only snapshot of one body, for display."""

    id: int
    name: str
    kind: BodyKind
    color: str
    mass: float
    radius: float
    density: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))


class Simulation:
    """Gravitational N-body system integrated with leapfrog."""

    def __init__(self, bodies: Sequence[BodySpec], config: SimulationConfig | None = None) -> None:
        if not bodies:
            raise ValueError("a simulation needs at least one body")
        if len(bodies) > MAX_BODIES:
            raise ValueError(f"too many bodies ({len(bodies)} > {MAX_BODIES})")
        self.config = (config or SimulationConfig()).validated()

        self.pos = np.array([b.position for b in bodies], dtype=np.float64)
        self.vel = np.array([b.velocity for b in bodies], dtype=np.float64)
        self.mass = np.array([b.mass for b in bodies], dtype=np.float64)
        self.density = np.array([b.density for b in bodies], dtype=np.float64)
        self.radius = np.array([b.radius for b in bodies], dtype=np.float64)
        self.names = [b.name for b in bodies]
        self.colors = [b.color for b in bodies]
        self.kinds: list[BodyKind] = [b.kind for b in bodies]
        self.ids = list(range(len(bodies)))
        self._index = {bid: i for i, bid in enumerate(self.ids)}

        self.time = 0.0
        self.steps = 0
        self.collision_energy = 0.0
        self.collision_events = 0
        self.merged_into: dict[int, int] = {}

        self.acc = np.zeros_like(self.pos)
        self._pot = np.zeros(len(bodies))
        self._tree: Octree | None = None
        self._tree_key: tuple[int, int] | None = None
        self.active_solver: Literal["direct", "barnes-hut"] = "direct"
        self._energy_cache: tuple[tuple[object, ...], EnergyReport] | None = None
        self._refresh_forces()

        self._trails = {bid: Trail(self.config.trail_length) for bid in self.ids}
        self._com_trail = Trail(self.config.trail_length)
        self._record_trails()

        kinetic, potential, _ = self._energy_parts()
        self.initial_energy = kinetic + potential

    # ---------------------------------------------------------------- basics
    @property
    def n(self) -> int:
        return len(self.ids)

    def index_of(self, body_id: int) -> int | None:
        return self._index.get(body_id)

    def resolve_id(self, body_id: int) -> int | None:
        """Follow merge history: the id that now carries ``body_id``'s mass."""
        seen = set()
        while body_id not in self._index:
            if body_id in seen or body_id not in self.merged_into:
                return None
            seen.add(body_id)
            body_id = self.merged_into[body_id]
        return body_id

    def body(self, index: int) -> BodyView:
        return BodyView(
            id=self.ids[index],
            name=self.names[index],
            kind=self.kinds[index],
            color=self.colors[index],
            mass=float(self.mass[index]),
            radius=float(self.radius[index]),
            density=float(self.density[index]),
            position=self.pos[index].copy(),
            velocity=self.vel[index].copy(),
            acceleration=self.acc[index].copy(),
        )

    def total_mass(self) -> float:
        return float(self.mass.sum())

    def center_of_mass(self) -> np.ndarray:
        return (self.mass[:, np.newaxis] * self.pos).sum(axis=0) / self.mass.sum()

    def com_velocity(self) -> np.ndarray:
        return (self.mass[:, np.newaxis] * self.vel).sum(axis=0) / self.mass.sum()

    def momentum(self) -> np.ndarray:
        return (self.mass[:, np.newaxis] * self.vel).sum(axis=0)

    def angular_momentum(self) -> np.ndarray:
        return (self.mass[:, np.newaxis] * np.cross(self.pos, self.vel)).sum(axis=0)

    # ---------------------------------------------------------------- config
    def update_config(self, **changes: object) -> None:
        """Change parameters while running (e.g. from the UI)."""
        old = self.config
        new = replace(old, **changes).validated()  # type: ignore[arg-type]
        self.config = new
        if (new.solver, new.theta, new.softening) != (old.solver, old.theta, old.softening):
            self._refresh_forces()
        if new.softening != old.softening:
            # A different softening length is a different energy function: rebaseline.
            kinetic, potential, _ = self._energy_parts()
            self.initial_energy = kinetic + potential
            self.collision_energy = 0.0
        if new.trail_length != old.trail_length:
            self._trails = {k: t.resized(new.trail_length) for k, t in self._trails.items()}
            self._com_trail = self._com_trail.resized(new.trail_length)

    # ---------------------------------------------------------------- forces
    def _refresh_forces(self) -> None:
        cfg = self.config
        self.active_solver = resolve_solver(cfg.solver, self.n)
        if self.active_solver == "direct":
            self.acc, self._pot = direct_accelerations(self.pos, self.mass, cfg.softening)
        else:
            self.acc, self._pot, tree = barnes_hut_accelerations(self.pos, self.mass, cfg.theta, cfg.softening)
            self._tree = tree
            self._tree_key = (self.steps, self.n)

    def octree(self) -> Octree:
        """Octree of the current positions (reused from the solver when available)."""
        key = (self.steps, self.n)
        if self._tree is None or self._tree_key != key:
            self._tree = Octree(self.pos, self.mass)
            self._tree_key = key
        return self._tree

    # ----------------------------------------------------------- integration
    def step(self, count: int = 1) -> None:
        """Advance ``count`` leapfrog steps of ``config.dt`` seconds each.

        Raises:
            NumericalInstabilityError: if a step yields non-finite values. The
                state is restored to the last finite step before raising.
        """
        for _ in range(count):
            dt = self.config.dt
            backup = (self.pos.copy(), self.vel.copy(), self.acc.copy(), self._pot.copy())
            self.vel += 0.5 * dt * self.acc
            self.pos += dt * self.vel
            self._refresh_forces()
            self.vel += 0.5 * dt * self.acc
            if not (np.isfinite(self.pos).all() and np.isfinite(self.vel).all() and np.isfinite(self.acc).all()):
                self.pos, self.vel, self.acc, self._pot = backup
                raise NumericalInstabilityError(
                    "the integration produced non-finite values; reduce the time step or increase softening"
                )
            self.time += dt
            self.steps += 1
            if self.config.collisions != "off" and self.n > 1:
                self._handle_collisions()
            if self.steps % self.config.trail_every == 0:
                self._record_trails()

    def _handle_collisions(self) -> None:
        pairs = col.find_overlaps(self.pos, self.radius)
        if not pairs:
            return
        kinetic, potential, _ = self._energy_parts()
        before = kinetic + potential
        if self.config.collisions == "merge":
            self._apply_merges(col.overlap_groups(self.n, pairs))
        else:
            col.resolve_bounces(pairs, self.mass, self.radius, self.pos, self.vel, self.config.restitution)
        self._refresh_forces()
        kinetic, potential, _ = self._energy_parts()
        self.collision_energy += before - (kinetic + potential)
        self.collision_events += len(pairs)

    def _apply_merges(self, groups: list[list[int]]) -> None:
        keep = np.ones(self.n, dtype=bool)
        for group in groups:
            merge = col.plan_merge(group, self.mass, self.pos, self.vel, self.density)
            s = merge.survivor
            self.mass[s] = merge.mass
            self.pos[s] = merge.position
            self.vel[s] = merge.velocity
            self.density[s] = merge.density
            if any(self.kinds[k] == "blackhole" for k in group):
                self.kinds[s] = "blackhole"
            self.radius[s] = body_radius(self.kinds[s], merge.mass, merge.density)
            for k in merge.absorbed:
                keep[k] = False
                self.merged_into[self.ids[k]] = self.ids[s]
        self._remove(keep)

    def _remove(self, keep: np.ndarray) -> None:
        for i in np.nonzero(~keep)[0]:
            self._trails.pop(self.ids[i], None)
        self.pos = self.pos[keep]
        self.vel = self.vel[keep]
        self.acc = self.acc[keep]
        self._pot = self._pot[keep]
        self.mass = self.mass[keep]
        self.density = self.density[keep]
        self.radius = self.radius[keep]
        idx = np.nonzero(keep)[0].tolist()
        self.names = [self.names[i] for i in idx]
        self.colors = [self.colors[i] for i in idx]
        self.kinds = [self.kinds[i] for i in idx]
        self.ids = [self.ids[i] for i in idx]
        self._index = {bid: i for i, bid in enumerate(self.ids)}
        self._tree = None

    # ---------------------------------------------------------------- energy
    def _energy_parts(self) -> tuple[float, float, bool]:
        kinetic = 0.5 * float(np.dot(self.mass, np.einsum("ij,ij->i", self.vel, self.vel)))
        if self.active_solver == "direct":
            return kinetic, 0.5 * float(np.dot(self.mass, self._pot)), True
        if self.n <= EXACT_ENERGY_MAX_BODIES:
            return kinetic, potential_energy(self.pos, self.mass, self.config.softening), True
        return kinetic, 0.5 * float(np.dot(self.mass, self._pot)), False

    def energy(self) -> EnergyReport:
        """Energy diagnostics for the current state (cached until the state changes)."""
        key = (
            self.steps,
            self.n,
            self.initial_energy,
            self.collision_energy,
            self.config.softening,
            self.config.theta,
            self.active_solver,
        )
        if self._energy_cache is not None and self._energy_cache[0] == key:
            return self._energy_cache[1]
        kinetic, potential, exact = self._energy_parts()
        report = EnergyReport(kinetic, potential, self.initial_energy, self.collision_energy, exact)
        self._energy_cache = (key, report)
        return report

    # ---------------------------------------------------------------- trails
    def _record_trails(self) -> None:
        for i, bid in enumerate(self.ids):
            self._trails[bid].append(self.pos[i])
        self._com_trail.append(self.center_of_mass())

    def trail(self, body_id: int, last: int | None = None) -> np.ndarray:
        trail = self._trails.get(body_id)
        return trail.points(last) if trail is not None else np.empty((0, 3))

    def com_trail(self, last: int | None = None) -> np.ndarray:
        return self._com_trail.points(last)

    def clear_trails(self) -> None:
        for trail in self._trails.values():
            trail.clear()
        self._com_trail.clear()
