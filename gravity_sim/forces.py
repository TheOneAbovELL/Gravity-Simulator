"""Gravitational acceleration solvers.

Two interchangeable solvers return ``(acc, pot)``:

* :func:`direct_accelerations` - exact pairwise sum, vectorised with numpy and
  processed in row chunks so memory stays bounded.
* :func:`barnes_hut_accelerations` - Barnes-Hut octree approximation.

Both use Plummer softening: the pair potential is
``-G m_i m_j / sqrt(r^2 + eps^2)`` and the force is its exact gradient, so
energy remains a conserved quantity of the softened system.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .constants import G
from .octree import Octree

Solver = Literal["auto", "direct", "barnes-hut"]
SOLVERS: tuple[Solver, ...] = ("auto", "direct", "barnes-hut")

AUTO_DIRECT_MAX_BODIES = 5000
"""``auto`` uses the direct solver up to this many bodies.

The direct solver runs in numpy's vectorised C loops while the Barnes-Hut
traversal is pure Python, so here the exact direct sum is faster than
Barnes-Hut at theta=0.5 for every body count up to at least 6000 (measured
with ``scripts/benchmark_solvers.py``; see the README). Barnes-Hut remains
available explicitly, and is what the octree overlay visualises.
"""

_CHUNK_ELEMENTS = 1 << 21  # ~2M pair entries (~50 MB of temporaries) per chunk


def resolve_solver(solver: Solver, n_bodies: int) -> Literal["direct", "barnes-hut"]:
    if solver == "auto":
        return "direct" if n_bodies <= AUTO_DIRECT_MAX_BODIES else "barnes-hut"
    if solver not in ("direct", "barnes-hut"):
        raise ValueError(f"unknown solver {solver!r}; expected one of {SOLVERS}")
    return solver


def direct_accelerations(
    positions: np.ndarray, masses: np.ndarray, softening: float = 0.0, g: float = G
) -> tuple[np.ndarray, np.ndarray]:
    """Exact pairwise accelerations (m/s^2) and specific potentials (J/kg)."""
    n = positions.shape[0]
    acc = np.zeros((n, 3))
    pot = np.zeros(n)
    if n < 2:
        return acc, pot
    eps2 = softening * softening
    # Separate contiguous coordinate arrays are markedly faster than (n, n, 3) strided math.
    x, y, z = (np.ascontiguousarray(positions[:, k]) for k in range(3))
    chunk = max(1, min(n, _CHUNK_ELEMENTS // n))
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        dx = x[np.newaxis, :] - x[start:stop, np.newaxis]  # r_j - r_i
        dy = y[np.newaxis, :] - y[start:stop, np.newaxis]
        dz = z[np.newaxis, :] - z[start:stop, np.newaxis]
        r2 = dx * dx + dy * dy + dz * dz + eps2
        rows = np.arange(stop - start)
        r2[rows, rows + start] = np.inf  # no self-interaction
        r2[r2 == 0.0] = np.inf  # coincident & unsoftened pairs have no defined force
        inv = 1.0 / np.sqrt(r2)
        w = masses[np.newaxis, :] * inv
        pot[start:stop] = -w.sum(axis=1)
        w *= inv * inv
        acc[start:stop, 0] = (w * dx).sum(axis=1)
        acc[start:stop, 1] = (w * dy).sum(axis=1)
        acc[start:stop, 2] = (w * dz).sum(axis=1)
    return g * acc, g * pot


def barnes_hut_accelerations(
    positions: np.ndarray, masses: np.ndarray, theta: float = 0.5, softening: float = 0.0, g: float = G
) -> tuple[np.ndarray, np.ndarray, Octree]:
    """Barnes-Hut accelerations and potentials, plus the tree that produced them."""
    tree = Octree(positions, masses)
    acc, pot = tree.accelerations(theta, softening, g)
    return acc, pot, tree


def potential_energy(positions: np.ndarray, masses: np.ndarray, softening: float = 0.0, g: float = G) -> float:
    """Exact total (softened) gravitational potential energy in J."""
    _, pot = direct_accelerations(positions, masses, softening, g)
    return 0.5 * float(np.dot(masses, pot))
