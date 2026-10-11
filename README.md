# Gravity Simulation

An interactive 3-D gravitational N-body simulator: as accurate as practical,
while still being fun to play with. Watch the real solar system evolve from its
J2000.0 configuration, a figure-eight three-body choreography, a star cluster
ejecting members, or two disk galaxies tearing each other into tidal tails, and
see the numerical error of the integration as it happens.

![Inner solar system](docs/inner-solar-system.png)
![Galaxy collision](docs/galaxy-collision.png)

## Features

- **Accurate integration.** Kick-drift-kick leapfrog (velocity Verlet), which is
  symplectic and time-reversible, so energy errors stay bounded instead of
  growing. After one sidereal year (365.256 d), Earth is back within 2×10⁻⁵ AU
  of its start. The residual is real perturbation from the other planets.
- **Two force solvers.** An exact, vectorised direct sum and a **Barnes-Hut
  octree** with an adjustable opening angle θ. You can draw the octree over the scene.
- **Real initial conditions.** The planets start at their J2000.0 positions
  from JPL's approximate Keplerian elements. Other presets are a binary star,
  the figure-eight orbit, a virialised cluster and a galaxy collision.
- **Collisions.** Bodies can merge (mass, momentum, centre of mass and volume
  are conserved), bounce with a coefficient of restitution, or pass through
  each other. Black holes use their Schwarzschild radius.
- **Honest diagnostics.** You see the live relative energy error
  (ΔE/E₀) in the overlay, with energy changed by collisions accounted for
  separately. The inspector also shows momentum, collision counts and octree statistics.
- **Interactive 3-D view.** Orbit, pan and zoom the camera, click a body to
  follow it, and draw trails in the inertial frame or relative to the followed
  body. Bodies can be coloured by speed, acceleration or mass.
- **Headless batch mode.** Run without a window, print diagnostics and write a CSV.

## Quick start

Requires Python 3.10+ with Tk (included in the python.org installers; on
Debian/Ubuntu install `python3-tk`).

```bash
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate`
elsewhere), then:

```bash
pip install -e ".[capture]"
```

```bash
gravity-sim
```

`python -m gravity_sim` works too. The `capture` extra installs Pillow, which is
only needed to save PNG screenshots or record frames.

Useful options:

```bash
gravity-sim --preset solar
```

```bash
gravity-sim --preset galaxies --bodies 400 --steps-per-frame 4
```

```bash
gravity-sim --preset cluster --solver barnes-hut --theta 0.7
```

```bash
gravity-sim --list-presets
```

Run `gravity-sim --help` for the full list (time step, softening, collisions,
fullscreen, frame recording and headless mode).

## Controls

| Input | Action |
|-------|--------|
| Left-drag | Rotate the view |
| Right- or middle-drag | Pan |
| Mouse wheel, `+` / `-` | Zoom |
| Click a body | Select and follow it (click empty space to deselect) |
| `Space` / `S` | Play–pause / single step |
| `←` / `→` | Follow the previous / next body |
| `Home` | Reset the camera |
| `T` `O` `C` `L` `H` | Toggle trails, octree, centre-of-mass marker, labels, overlay |
| `F11` / `Esc` | Toggle / leave fullscreen |
| `F1` or `?` | Show the controls overlay |

The side panel has four tabs:

- **Scenario:** load a preset; the body count and seed apply to generated presets.
- **Physics:** time step, steps per frame, solver, θ, softening, collision mode and restitution.
- **View:** which body to follow, trail options, octree overlay, colouring, field of view and image export.
- **Inspect:** the selected body and system diagnostics.

**Restart** rewinds the scenario but keeps your settings. **Load scenario**
applies the preset's recommended settings.

## Presets

| Key | Scenario | Bodies | Time step |
|-----|----------|--------|-----------|
| `inner` | Sun, Mercury–Mars at J2000.0 | 5 | 6 h |
| `solar` | Sun and all eight planets at J2000.0 | 9 | 1 d |
| `binary` | Two stars (1.0 + 0.75 M☉, 0.4 AU) with two circumbinary planets | 4 | 2.4 h |
| `figure8` | Chenciner–Montgomery figure-eight, three 1 M☉ stars | 3 | 2.4 h |
| `cluster` | Virialised star cluster, masses 0.2–2 M☉, softening 0.2 AU | 120 (2–2000) | 12 h |
| `galaxies` | Two disks of stars around 1000 M☉ black holes, on a collision course | 200 (2–2000) | 1 d |

All presets start in the barycentric frame (no net momentum), and generated
presets are deterministic for a given `--seed`.

## Physics and accuracy

**Units.** Everything is SI internally: G = 6.67430×10⁻¹¹ (CODATA 2018) and
1 AU = 1.495978707×10¹¹ m. Masses of the Sun and planets are defined from
their measured GM values (JPL DE440), which are known about 10⁵ times more
precisely than G itself. That way G·m is exact and orbital periods come out
right. For planets, GM includes their moons.

**Integrator.** Each step is half-kick, drift, force evaluation, half-kick. A
step that produces non-finite values is rolled back and reported. The GUI
pauses with a message, and headless mode exits with code 3. The time step can
be changed while running. Leapfrog stays accurate with a variable step but is
then no longer strictly symplectic.

**Softening.** An optional Plummer softening length ε replaces 1/r with
1/√(r²+ε²) in both the force and the potential, so energy is still conserved
exactly. Changing ε changes the energy function, so the energy baseline resets.

**Solvers.** `auto`, the default, picks the direct sum up to 5000 bodies.
Measured with `scripts/benchmark_solvers.py` on the development machine, in
seconds per force evaluation:

| N | direct (exact) | Barnes-Hut θ=0.5 | Barnes-Hut θ=1.0 | BH θ=0.5 RMS error |
|---|---|---|---|---|
| 100 | 0.0002 | 0.0058 | 0.0034 | 0.67 % |
| 1000 | 0.077 | 0.23 | 0.072 | 0.75 % |
| 3000 | 0.66 | 1.50 | 0.39 | 0.47 % |
| 6000 | 2.09 | 3.49 | 0.90 | 0.44 % |

The direct sum runs inside numpy's compiled loops, while the Barnes-Hut traversal
is pure Python. So in this implementation, Barnes-Hut only beats the direct sum
at loose opening angles or very large N. It is kept because it is the
project's original algorithm, it is genuinely O(N log N), and it is what the
octree overlay visualises. With θ = 0 it reproduces the direct sum exactly.

**Collisions** are detected when the physical spheres overlap, using an
O(N log N) sort-and-sweep pass.

- `merge`: every group of touching bodies becomes one body.
- `bounce`: an impulse along the line of centres, with restitution e ∈ [0, 1].
  Bodies are then separated without moving their centre of mass.
- `off`: bodies pass through each other.

**Energy accounting.** Collisions change the mechanical energy on purpose:
merging removes kinetic energy, and it also removes the pair's mutual
potential energy, so the net change can have either sign. That change is
accumulated as *collision ΔE*. The displayed drift,
`(E + collision ΔE − E₀) / |E₀|`, therefore measures only numerical error.
For more than 5000 bodies under Barnes-Hut, the potential energy comes from
the tree and is marked "approx.".

**Body sizes.** Bodies are drawn exaggerated by default, with screen size
proportional to radius^0.3, because planets would otherwise be sub-pixel.
Enable *True body scale* in the View tab for physical sizes.

### Verification

The test suite (`tests/`) checks physics invariants rather than just running code:

- **Orbits:** a circular orbit closes after one Kepler period. Energy error stays
  below 10⁻⁶ over 20 orbits with no secular growth. Momentum is conserved to
  round-off. Leapfrog is time-reversible, and the first step has no spurious
  kick (a legacy bug).
- **Barnes-Hut:** θ = 0 equals the direct sum. RMS errors stay under 0.5 %,
  2 % and 8 % for θ = 0.3, 0.5 and 1.0, and the octree is correct for
  coincident bodies.
- **Collisions:** merges conserve mass, momentum, centre of mass and volume.
  Elastic bounces conserve kinetic energy. Equal masses merge (a legacy bug).
- **Presets:** J2000 Earth and Jupiter positions agree with the known values,
  Earth closes its orbit after one sidereal year,
  the figure-eight returns to its start after one period, the cluster is
  exactly virialised, and every preset runs stably.
- **Interface:** camera projection, the CLI (including exit codes and the
  CSV), and GUI smoke tests that drive the real Tk window.

## Headless mode

```bash
gravity-sim --headless --preset solar --steps 36525 --report-every 3652 --csv solar.csv
```

This prints a diagnostics table (ASCII only, safe on any console) and optionally
writes a CSV with these columns: `step`, `time_days`, `bodies`, `kinetic_J`,
`potential_J`, `total_J`, `collision_energy_J`, `energy_drift`,
`momentum_kg_m_s`, `angular_momentum`, `collisions` and `solver`.

Exit codes: `0` for success, `2` for invalid arguments, `3` for numerical instability.

## Using the engine as a library

The physics core has no GUI dependency:

```python
from gravity_sim import Simulation, build_scenario

scenario = build_scenario("solar")
sim = Simulation(scenario.bodies, scenario.config)
sim.step(365)  # one year at dt = 1 day
print(sim.energy().drift)  # relative energy error
print(sim.names, sim.pos / 1.496e11)  # positions in AU
```

To build your own system, create `gravity_sim.BodySpec` objects (SI units) and
a `SimulationConfig`.

## Development

```bash
pip install -e ".[dev,capture]"
```

```bash
pytest
```

```bash
ruff check . && ruff format --check . && mypy
```

The GUI tests open a real Tk window and skip automatically when no display
is available. In CI on Linux they run under `xvfb-run`. GitHub Actions
(`.github/workflows/ci.yml`) runs lint and strict mypy, then runs the tests on
Linux, Windows and macOS with Python 3.10 and 3.13, plus a headless smoke run.

Project layout:

```
gravity_sim/
  simulation.py  integrator, collisions, energy bookkeeping, trails
  forces.py      direct-sum and Barnes-Hut solvers
  octree.py      Barnes-Hut octree
  collisions.py  overlap detection, merge and bounce response
  bodies.py      validated body specifications
  presets.py     built-in scenarios (JPL elements, figure-eight, ...)
  orbits.py      Kepler's equation, orbital elements -> state vectors
  camera.py      orbit camera and perspective projection
  app.py         tkinter interface
  cli.py         command line and headless mode
scripts/benchmark_solvers.py   solver speed/accuracy comparison
legacy/                        the original implementation, unmodified
tests/
```

## Limitations

- **Newtonian gravity only.** There is no relativity, tides, radiation or gas.
  Black holes are point masses with a Schwarzschild collision radius.
- **Fixed time step.** There are no adaptive or per-body time steps. Very close
  encounters need softening or a smaller `dt`. If a step goes non-finite, the
  integrator stops and says so.
- **Interactive speed.** Rendering on a Tk canvas stays interactive up to a few
  hundred bodies. The physics alone handles thousands, but the frame rate drops.
- **Approximate ephemeris.** The J2000 planetary elements are JPL's
  approximate set, valid 1800–2050, and "Earth" is the Earth–Moon barycentre.
  The model is accurate to roughly arc-minutes, not an ephemeris replacement.
- **Platform caveats for image capture.** PNG export and `--record` grab the
  screen region of the canvas, so the window must be visible and on top.
  `.eps` export works without Pillow.

## Credits

The original Barnes-Hut gravity simulation is by Gerrit Fritz (PhySimdev) and is
preserved in `legacy/`. Planetary elements come from E. M. Standish, *Keplerian
Elements for Approximate Positions of the Major Planets* (JPL). The figure-eight
solution is from A. Chenciner and R. Montgomery, *Annals of Mathematics* 152 (2000).
