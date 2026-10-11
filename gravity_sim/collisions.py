"""Collision detection and response.

Bodies collide when their physical spheres overlap. Three behaviours are
supported:

* ``merge`` - perfectly inelastic: every connected group of overlapping
  bodies becomes one body. Mass, linear momentum and centre of mass are
  conserved, as is total volume (so the merged density is the combined
  mass over combined volume).
* ``bounce`` - impulse along the line of centres with a coefficient of
  restitution ``e`` in [0, 1] (1 = elastic, kinetic energy conserved). Bodies
  are also pushed apart, preserving their centre of mass, so they do not
  remain interpenetrating.
* ``off`` - bodies pass through each other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

CollisionMode = Literal["merge", "bounce", "off"]
COLLISION_MODES: tuple[CollisionMode, ...] = ("merge", "bounce", "off")

_CHUNK_ELEMENTS = 1 << 21


_SWEEP_MAX_CANDIDATES = 1 << 20


def find_overlaps(positions: np.ndarray, radii: np.ndarray) -> list[tuple[int, int]]:
    """Return all index pairs ``(i, j)`` with ``i < j`` whose spheres overlap, sorted.

    Uses a sort-and-sweep broad phase along x (O(N log N) when bodies are
    small compared with their separations, the usual case) and falls back to
    a chunked all-pairs test when the sweep is not selective.
    """
    n = positions.shape[0]
    if n < 2:
        return []
    order = np.argsort(positions[:, 0], kind="stable")
    xs = positions[order, 0]
    reach = 2.0 * float(radii.max())
    hi = np.searchsorted(xs, xs + reach, side="right")
    counts = hi - np.arange(n) - 1
    total = int(counts.sum())
    if total == 0:
        return []
    if total > _SWEEP_MAX_CANDIDATES:
        return _all_pairs_overlaps(positions, radii)
    first = np.repeat(np.arange(n), counts)
    # Offsets 1..count for each run, built without a Python loop.
    run_starts = np.repeat(np.cumsum(counts) - counts, counts)
    second = first + (np.arange(total) - run_starts) + 1
    a, b = order[first], order[second]
    d = positions[a] - positions[b]
    reach_ab = radii[a] + radii[b]
    hit = (d * d).sum(axis=1) < reach_ab * reach_ab
    pairs = {(min(i, j), max(i, j)) for i, j in zip(a[hit].tolist(), b[hit].tolist(), strict=True)}
    return sorted(pairs)


def _all_pairs_overlaps(positions: np.ndarray, radii: np.ndarray) -> list[tuple[int, int]]:
    n = positions.shape[0]
    pairs: list[tuple[int, int]] = []
    chunk = max(1, min(n, _CHUNK_ELEMENTS // n))
    cols = np.arange(n)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        d = positions[np.newaxis, :, :] - positions[start:stop, np.newaxis, :]
        dist2 = np.einsum("ijk,ijk->ij", d, d)
        reach = radii[start:stop, np.newaxis] + radii[np.newaxis, :]
        mask = (dist2 < reach * reach) & (cols[np.newaxis, :] > np.arange(start, stop)[:, np.newaxis])
        ii, jj = np.nonzero(mask)
        pairs.extend(zip((ii + start).tolist(), jj.tolist(), strict=True))
    return pairs


def overlap_groups(n: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    """Connected components (size >= 2) of the overlap graph, via union-find."""
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    groups: dict[int, list[int]] = {}
    for k in sorted({k for pair in pairs for k in pair}):
        groups.setdefault(find(k), []).append(k)
    return [g for g in groups.values() if len(g) >= 2]


@dataclass(frozen=True)
class Merge:
    """Result of merging a group: ``survivor`` absorbs ``absorbed``."""

    survivor: int
    absorbed: tuple[int, ...]
    mass: float
    position: np.ndarray
    velocity: np.ndarray
    density: float


def plan_merge(
    group: list[int], masses: np.ndarray, positions: np.ndarray, velocities: np.ndarray, densities: np.ndarray
) -> Merge:
    """Compute the merged state of a group of overlapping bodies.

    The most massive body survives (lowest index on ties, so the result is
    deterministic).
    """
    idx = np.asarray(group)
    m = masses[idx]
    total = float(m.sum())
    survivor = int(idx[np.argmax(m)])
    position = (m[:, np.newaxis] * positions[idx]).sum(axis=0) / total
    velocity = (m[:, np.newaxis] * velocities[idx]).sum(axis=0) / total
    volume = float((m / densities[idx]).sum())
    return Merge(
        survivor=survivor,
        absorbed=tuple(int(k) for k in idx if k != survivor),
        mass=total,
        position=position,
        velocity=velocity,
        density=total / volume,
    )


def resolve_bounces(
    pairs: list[tuple[int, int]],
    masses: np.ndarray,
    radii: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    restitution: float,
) -> int:
    """Apply restitution impulses in place. Returns the number of impulses applied."""
    if not 0.0 <= restitution <= 1.0:
        raise ValueError("restitution must be in [0, 1]")
    applied = 0
    for i, j in pairs:
        delta = positions[j] - positions[i]
        dist = float(np.linalg.norm(delta))
        # Coincident centres have no defined normal: pick a fixed axis deterministically.
        normal = np.array([1.0, 0.0, 0.0]) if dist == 0.0 else delta / dist
        inv_i, inv_j = 1.0 / masses[i], 1.0 / masses[j]
        closing = float(np.dot(velocities[j] - velocities[i], normal))
        if closing < 0.0:  # approaching
            impulse = -(1.0 + restitution) * closing / (inv_i + inv_j)
            velocities[i] -= impulse * inv_i * normal
            velocities[j] += impulse * inv_j * normal
            applied += 1
        # Separate along the normal, weighted by inverse mass (keeps the COM fixed).
        overlap = radii[i] + radii[j] - dist
        if overlap > 0.0 and math.isfinite(overlap):
            share = overlap / (inv_i + inv_j)
            positions[i] -= share * inv_i * normal
            positions[j] += share * inv_j * normal
    return applied
