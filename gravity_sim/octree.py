"""Barnes-Hut octree for O(N log N) gravitational force approximation.

The tree is rebuilt from scratch every step (the standard approach). Each
leaf normally holds a single body; at :data:`MAX_DEPTH` a leaf may hold
several bodies, which guarantees termination for (near-)coincident bodies.

The opening criterion is the classic ``width / distance < theta``. A node that
contains the body being evaluated is always opened, so a body never interacts
with an approximation that includes itself. ``theta = 0`` disables
approximation entirely and yields the exact direct sum.

The implementation deliberately uses plain Python floats: for 3-component
vectors that is several times faster than numpy's per-call overhead.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Literal

import numpy as np

from .constants import G

MAX_DEPTH = 48
"""Maximum subdivision depth; deeper leaves become multi-body buckets."""

BoxKind = Literal["internal", "leaf", "all"]


class OctreeNode:
    """A cubic cell of the octree.

    ``half`` is half the side length. After :meth:`Octree._finalize`,
    ``mass`` is the total mass inside the cell and ``comx/comy/comz`` its
    centre of mass. Leaves have ``children is None`` and list their body
    indices in ``bodies``.
    """

    __slots__ = ("bodies", "children", "comx", "comy", "comz", "cx", "cy", "cz", "depth", "half", "mass")

    def __init__(self, cx: float, cy: float, cz: float, half: float, depth: int) -> None:
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.half = half
        self.depth = depth
        self.mass = 0.0
        self.comx = 0.0
        self.comy = 0.0
        self.comz = 0.0
        self.bodies: list[int] = []
        self.children: list[OctreeNode | None] | None = None

    def contains(self, x: float, y: float, z: float) -> bool:
        h = self.half
        return abs(x - self.cx) <= h and abs(y - self.cy) <= h and abs(z - self.cz) <= h

    def iter_children(self) -> Iterator[OctreeNode]:
        if self.children is not None:
            for child in self.children:
                if child is not None:
                    yield child


class Octree:
    """Barnes-Hut octree built over the given positions (m) and masses (kg)."""

    def __init__(self, positions: np.ndarray, masses: np.ndarray) -> None:
        positions = np.asarray(positions, dtype=np.float64)
        masses = np.asarray(masses, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must have shape (N, 3), got {positions.shape}")
        if masses.shape != (positions.shape[0],):
            raise ValueError("masses must have shape (N,) matching positions")
        if positions.shape[0] == 0:
            raise ValueError("cannot build an octree with no bodies")

        self._pts: list[list[float]] = positions.tolist()
        self._ms: list[float] = masses.tolist()
        self.n = positions.shape[0]

        lo = positions.min(axis=0)
        hi = positions.max(axis=0)
        center = (lo + hi) / 2.0
        half = float((hi - lo).max()) / 2.0
        # Pad slightly so bodies on the boundary are unambiguously inside.
        half = half * (1.0 + 1e-9) + max(half * 1e-12, 1e-6)
        self.root = OctreeNode(float(center[0]), float(center[1]), float(center[2]), half, 0)

        for i in range(self.n):
            self._insert(i)
        self._finalize(self.root)

    # ------------------------------------------------------------------ build
    def _child_for(self, node: OctreeNode, x: float, y: float, z: float) -> OctreeNode:
        assert node.children is not None
        octant = (x >= node.cx) | ((y >= node.cy) << 1) | ((z >= node.cz) << 2)
        child = node.children[octant]
        if child is None:
            q = node.half / 2.0
            child = OctreeNode(
                node.cx + (q if octant & 1 else -q),
                node.cy + (q if octant & 2 else -q),
                node.cz + (q if octant & 4 else -q),
                q,
                node.depth + 1,
            )
            node.children[octant] = child
        return child

    def _insert(self, i: int) -> None:
        x, y, z = self._pts[i]
        node = self.root
        while True:
            if node.children is None:
                if not node.bodies or node.depth >= MAX_DEPTH:
                    node.bodies.append(i)
                    return
                # Occupied leaf: split it and push the existing body down.
                existing = node.bodies
                node.bodies = []
                node.children = [None] * 8
                for j in existing:
                    ex, ey, ez = self._pts[j]
                    self._child_for(node, ex, ey, ez).bodies.append(j)
                continue
            node = self._child_for(node, x, y, z)

    def _finalize(self, node: OctreeNode) -> None:
        m = mx = my = mz = 0.0
        if node.children is None:
            for j in node.bodies:
                mj = self._ms[j]
                px, py, pz = self._pts[j]
                m += mj
                mx += mj * px
                my += mj * py
                mz += mj * pz
        else:
            for child in node.iter_children():
                self._finalize(child)
                m += child.mass
                mx += child.mass * child.comx
                my += child.mass * child.comy
                mz += child.mass * child.comz
        node.mass = m
        if m > 0.0:
            node.comx, node.comy, node.comz = mx / m, my / m, mz / m
        else:
            node.comx, node.comy, node.comz = node.cx, node.cy, node.cz

    # ----------------------------------------------------------------- forces
    def accelerations(self, theta: float, softening: float = 0.0, g: float = G) -> tuple[np.ndarray, np.ndarray]:
        """Approximate gravitational acceleration and potential for every body.

        Args:
            theta: Opening angle. Smaller is more accurate; 0 is exact.
            softening: Plummer softening length (m).
            g: Gravitational constant (override for tests).

        Returns:
            ``(acc, pot)`` where ``acc`` has shape (N, 3) in m/s^2 and ``pot``
            is the specific potential (J/kg) at each body, shape (N,).
        """
        if theta < 0:
            raise ValueError("theta must be >= 0")
        theta2 = theta * theta
        eps2 = softening * softening
        pts = self._pts
        ms = self._ms
        root = self.root
        sqrt = math.sqrt
        acc = np.zeros((self.n, 3))
        pot = np.zeros(self.n)

        for i in range(self.n):
            xi, yi, zi = pts[i]
            ax = ay = az = phi = 0.0
            stack = [root]
            while stack:
                node = stack.pop()
                if node.children is None:
                    for j in node.bodies:
                        if j == i:
                            continue
                        pj = pts[j]
                        dx = pj[0] - xi
                        dy = pj[1] - yi
                        dz = pj[2] - zi
                        r2 = dx * dx + dy * dy + dz * dz + eps2
                        if r2 == 0.0:
                            continue  # exactly coincident and unsoftened: no defined direction
                        inv = 1.0 / sqrt(r2)
                        w = ms[j] * inv
                        phi -= w
                        w *= inv * inv
                        ax += w * dx
                        ay += w * dy
                        az += w * dz
                    continue
                dx = node.comx - xi
                dy = node.comy - yi
                dz = node.comz - zi
                d2 = dx * dx + dy * dy + dz * dz
                width = 2.0 * node.half
                if width * width < theta2 * d2 and not node.contains(xi, yi, zi):
                    inv = 1.0 / sqrt(d2 + eps2)
                    w = node.mass * inv
                    phi -= w
                    w *= inv * inv
                    ax += w * dx
                    ay += w * dy
                    az += w * dz
                else:
                    stack.extend(node.iter_children())
            acc[i, 0] = ax
            acc[i, 1] = ay
            acc[i, 2] = az
            pot[i] = phi
        acc *= g
        pot *= g
        return acc, pot

    # ------------------------------------------------------------- inspection
    def iter_nodes(self) -> Iterator[OctreeNode]:
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.iter_children())

    def boxes(self, kind: BoxKind = "all", limit: int | None = None) -> list[tuple[tuple[float, float, float], float]]:
        """Return ``(center, half_size)`` of nodes for visualisation.

        Args:
            kind: ``internal`` (regional) nodes, ``leaf`` (body) nodes or ``all``.
            limit: Optional cap on the number of boxes returned (shallowest first).
        """
        out: list[tuple[tuple[float, float, float], float]] = []
        # Breadth-first so a limit keeps the most informative, largest boxes.
        frontier = [self.root]
        while frontier and (limit is None or len(out) < limit):
            nxt: list[OctreeNode] = []
            for node in frontier:
                is_leaf = node.children is None
                if kind == "all" or (kind == "leaf") == is_leaf:
                    out.append(((node.cx, node.cy, node.cz), node.half))
                    if limit is not None and len(out) >= limit:
                        break
                nxt.extend(node.iter_children())
            frontier = nxt
        return out

    def stats(self) -> dict[str, int]:
        nodes = leaves = depth = 0
        for node in self.iter_nodes():
            nodes += 1
            depth = max(depth, node.depth)
            if node.children is None:
                leaves += 1
        return {"nodes": nodes, "leaves": leaves, "depth": depth}
