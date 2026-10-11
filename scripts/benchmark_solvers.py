"""Compare direct-sum and Barnes-Hut force evaluation cost and accuracy.

Usage:
    python scripts/benchmark_solvers.py [N ...]

Prints seconds per force evaluation and the Barnes-Hut RMS relative
acceleration error versus the exact direct sum, for a random cluster.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from functools import partial

import numpy as np

from gravity_sim.constants import AU, SOLAR_MASS
from gravity_sim.forces import barnes_hut_accelerations, direct_accelerations


def _time(fn: Callable[[], object], repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main(sizes: list[int]) -> None:
    rng = np.random.default_rng(0)
    print(f"{'N':>6} {'direct [s]':>11} {'BH t=0.5 [s]':>13} {'BH t=1.0 [s]':>13} {'BH0.5 rms err':>14}")
    for n in sizes:
        pos = rng.normal(size=(n, 3)) * 10 * AU
        mass = rng.uniform(0.2, 2.0, n) * SOLAR_MASS
        exact, _ = direct_accelerations(pos, mass)
        t_direct = _time(partial(direct_accelerations, pos, mass))
        t_bh5 = _time(partial(barnes_hut_accelerations, pos, mass, 0.5), repeat=1)
        t_bh10 = _time(partial(barnes_hut_accelerations, pos, mass, 1.0), repeat=1)
        approx, _, _ = barnes_hut_accelerations(pos, mass, 0.5)
        err = np.linalg.norm(approx - exact, axis=1) / np.linalg.norm(exact, axis=1)
        print(f"{n:>6} {t_direct:>11.4f} {t_bh5:>13.4f} {t_bh10:>13.4f} {float(np.sqrt((err**2).mean())):>14.2e}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [100, 300, 1000, 3000, 6000])
