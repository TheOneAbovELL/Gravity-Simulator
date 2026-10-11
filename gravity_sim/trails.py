"""Fixed-capacity position history (ring buffer) used to draw orbit trails."""

from __future__ import annotations

import numpy as np


class Trail:
    """Chronological ring buffer of 3-D positions."""

    __slots__ = ("_buf", "_len", "_start")

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("trail capacity must be >= 1")
        self._buf = np.empty((capacity, 3))
        self._start = 0
        self._len = 0

    @property
    def capacity(self) -> int:
        return self._buf.shape[0]

    def __len__(self) -> int:
        return self._len

    def append(self, point: np.ndarray) -> None:
        cap = self.capacity
        if self._len < cap:
            self._buf[(self._start + self._len) % cap] = point
            self._len += 1
        else:
            self._buf[self._start] = point
            self._start = (self._start + 1) % cap

    def clear(self) -> None:
        self._start = 0
        self._len = 0

    def points(self, last: int | None = None) -> np.ndarray:
        """Return the stored points oldest-first (optionally only the newest ``last``)."""
        n = self._len if last is None else max(0, min(last, self._len))
        if n == 0:
            return np.empty((0, 3))
        begin = (self._start + self._len - n) % self.capacity
        end = begin + n
        if end <= self.capacity:
            return self._buf[begin:end].copy()
        return np.concatenate((self._buf[begin:], self._buf[: end - self.capacity]))

    def resized(self, capacity: int) -> Trail:
        """Return a copy with a different capacity, keeping the newest points."""
        new = Trail(capacity)
        for p in self.points(capacity):
            new.append(p)
        return new
