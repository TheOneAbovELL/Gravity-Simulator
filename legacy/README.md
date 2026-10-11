# Legacy implementation (superseded)

These are the original files of the project, kept unmodified for reference.
They are **not** used by the `gravity_sim` package and are excluded from
linting, type-checking and packaging.

| File | What it was |
|------|-------------|
| `gravitysimulation.py` | Original tkinter + turtle interface (last runnable entry point). |
| `gsim.py` | An in-progress edit of `gravitysimulation.py` that does not compile (the class body lost its `def __init__`). Its intent — a richer class docstring and the project premise ("as accurate as possible, while still being fun and interactive") — is carried into the new package. |
| `Simulation.py` | Original simulation/body classes and velocity-Verlet integrator. |
| `BarnesHut.py` | Original Barnes-Hut octree. |

## Defects found in the legacy code (all fixed in `gravity_sim`)

Verified by running the legacy modules headlessly or by reading the code:

* `Blackhole(...)` raised `TypeError` (`super(Planet, self)` in a non-`Planet` class).
* A simulation with a single body (e.g. after all bodies merge) crashed building the octree.
* Integer-valued positions/velocities crashed the integrator (int64 arrays updated in place with floats).
* The first integration step applied a full `a*dt` velocity kick before drifting.
* Two overlapping bodies of *equal* mass never merged; collisions were only detected
  against single-body octree leaves.
* `restitution_coefficient = 1` produced no bounce at all (velocities were scaled, not reflected).
* Relative ("co-moving") centre-of-mass mode subtracted a velocity from a position (missing `* dt`).
* The displayed "total energy" was the per-body *average*, not the total.
* Colour-mapping mode crashed (tuples rebuilt with missing fields); `max_force` was a tuple.
* `benchmark = True` crashed (printed attributes that were never set).
* Random mode created zero stars (inverted condition).
* Frame capture wrote to a hard-coded `C:/Users/gerri/Desktop/data` path.
* Identical body positions could recurse forever in the octree.
* Turtle-based drawing was very slow, and `show_time = False` let overlay text pile up.

Baseline measured before the rewrite: Sun–Earth over one year at `dt = 1e5 s`
had a relative energy error of 4.0e-4 in the legacy code. The new integrator,
with the inner-solar-system preset, returns Earth to within 2e-5 AU of its
starting point after one sidereal year, with energy drift around 1e-6.

## Running the legacy version

```bash
cd legacy
python gravitysimulation.py
```

It needs `numpy` and `Pillow` and opens a fullscreen window ("leave" button to exit).
