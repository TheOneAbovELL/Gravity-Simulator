"""Command-line entry point: interactive GUI or headless batch runs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TextIO

import numpy as np

from . import __version__
from .collisions import COLLISION_MODES
from .constants import AU, DAY
from .forces import SOLVERS
from .presets import DEFAULT_PRESET, PRESETS, Scenario, build_scenario
from .simulation import NumericalInstabilityError, Simulation, SimulationConfig
from .units import format_sim_time

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNSTABLE = 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gravity-sim",
        description="Interactive 3-D gravitational N-body simulator (leapfrog integrator, direct or Barnes-Hut forces).",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--list-presets", action="store_true", help="list the built-in scenarios and exit")

    s = p.add_argument_group("scenario")
    s.add_argument("--preset", choices=list(PRESETS), default=DEFAULT_PRESET, help="scenario to load (default: %(default)s)")
    s.add_argument("--bodies", type=int, help="number of bodies (cluster and galaxies presets only)")
    s.add_argument("--seed", type=int, default=1, help="random seed for generated presets (default: %(default)s)")

    ph = p.add_argument_group("physics (defaults come from the preset)")
    ph.add_argument("--dt", type=float, metavar="DAYS", help="integration time step in days")
    ph.add_argument("--steps-per-frame", type=int, metavar="N", help="integration steps per rendered frame")
    ph.add_argument("--solver", choices=SOLVERS, help="force solver")
    ph.add_argument("--theta", type=float, help="Barnes-Hut opening angle (0 = exact)")
    ph.add_argument("--softening", type=float, metavar="AU", help="Plummer softening length in AU")
    ph.add_argument("--collisions", choices=COLLISION_MODES, help="collision behaviour")
    ph.add_argument("--restitution", type=float, help="coefficient of restitution for 'bounce' (0..1)")

    g = p.add_argument_group("window")
    g.add_argument("--fullscreen", action="store_true", help="start in fullscreen (toggle with F11)")
    g.add_argument("--paused", action="store_true", help="start paused")
    g.add_argument("--record", type=Path, metavar="DIR", help="save every rendered frame as PNG into DIR (needs Pillow)")
    g.add_argument(
        "--record-frames", type=int, default=3600, metavar="N", help="stop recording after N frames (default: %(default)s)"
    )

    h = p.add_argument_group("headless batch mode")
    h.add_argument("--headless", action="store_true", help="run without a window and print diagnostics")
    h.add_argument("--steps", type=int, default=1000, help="steps to run in headless mode (default: %(default)s)")
    h.add_argument("--report-every", type=int, default=100, metavar="N", help="report interval in steps (default: %(default)s)")
    h.add_argument("--csv", type=Path, metavar="PATH", help="write the diagnostics table to a CSV file")
    return p


def scenario_with_overrides(args: argparse.Namespace) -> tuple[Scenario, int]:
    """Build the requested scenario and apply command-line overrides.

    Raises:
        ValueError: for invalid combinations or values.
    """
    scenario = build_scenario(args.preset, args.bodies, args.seed)
    changes: dict[str, object] = {}
    if args.dt is not None:
        changes["dt"] = args.dt * DAY
    if args.solver is not None:
        changes["solver"] = args.solver
    if args.theta is not None:
        changes["theta"] = args.theta
    if args.softening is not None:
        changes["softening"] = args.softening * AU
    if args.collisions is not None:
        changes["collisions"] = args.collisions
    if args.restitution is not None:
        changes["restitution"] = args.restitution
    config: SimulationConfig = replace(scenario.config, **changes).validated()  # type: ignore[arg-type]
    steps_per_frame = args.steps_per_frame if args.steps_per_frame is not None else scenario.steps_per_frame
    if not 1 <= steps_per_frame <= 1000:
        raise ValueError(f"steps-per-frame must be in [1, 1000], got {steps_per_frame}")
    return replace(scenario, config=config, steps_per_frame=steps_per_frame), steps_per_frame


def _diagnostics_row(sim: Simulation) -> dict[str, float | int | str]:
    energy = sim.energy()
    return {
        "step": sim.steps,
        "time_days": sim.time / DAY,
        "bodies": sim.n,
        "kinetic_J": energy.kinetic,
        "potential_J": energy.potential,
        "total_J": energy.total,
        "collision_energy_J": energy.collision_energy,
        "energy_drift": energy.drift,
        "momentum_kg_m_s": float(np.linalg.norm(sim.momentum())),
        "angular_momentum": float(np.linalg.norm(sim.angular_momentum())),
        "collisions": sim.collision_events,
        "solver": sim.active_solver,
    }


def run_headless(args: argparse.Namespace, scenario: Scenario, out: TextIO) -> int:
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.report_every < 1:
        raise ValueError("--report-every must be >= 1")
    sim = Simulation(scenario.bodies, scenario.config)
    cfg = sim.config
    out.write(
        f"{scenario.title}: {sim.n} bodies, dt = {cfg.dt / DAY:g} d, solver = {sim.active_solver}, "
        f"collisions = {cfg.collisions}, softening = {cfg.softening / AU:g} AU\n"
    )
    out.write(f"{'step':>9} {'time':>16} {'bodies':>6} {'E total [J]':>13} {'dE/E0':>10} {'|p| [kg m/s]':>13}\n")
    rows = [_diagnostics_row(sim)]

    def report(row: dict[str, float | int | str]) -> None:
        out.write(
            f"{row['step']:>9} {format_sim_time(float(row['time_days']) * DAY):>16} {row['bodies']:>6} "
            f"{float(row['total_J']):>13.5e} {float(row['energy_drift']):>10.2e} {float(row['momentum_kg_m_s']):>13.4e}\n"
        )

    report(rows[0])
    status = EXIT_OK
    done = 0
    try:
        while done < args.steps:
            chunk = min(args.report_every, args.steps - done)
            sim.step(chunk)
            done += chunk
            rows.append(_diagnostics_row(sim))
            report(rows[-1])
    except NumericalInstabilityError as exc:
        out.write(f"error at step {sim.steps}: {exc}\n")
        status = EXIT_UNSTABLE

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        out.write(f"wrote {len(rows)} rows to {args.csv}\n")

    final = rows[-1]
    drift = float(final["energy_drift"])
    out.write(
        f"done: {final['step']} steps, {format_sim_time(sim.time)}, {final['bodies']} bodies, "
        f"energy drift {drift:.2e}{'' if math.isfinite(drift) else ' (undefined)'}, "
        f"{len(sim.merged_into)} merges\n"
    )
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_presets:
        for key, preset in PRESETS.items():
            adjustable = (
                f" (--bodies up to {preset.max_bodies}, default {preset.default_bodies})" if preset.default_bodies else ""
            )
            print(f"{key:10s} {preset.title}: {preset.description}{adjustable}")
        return EXIT_OK

    try:
        scenario, _ = scenario_with_overrides(args)
        if args.headless:
            return run_headless(args, scenario, sys.stdout)
    except ValueError as exc:
        parser.print_usage(sys.stderr)
        print(f"gravity-sim: error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        return 130

    if args.record_frames < 1:
        print("gravity-sim: error: --record-frames must be >= 1", file=sys.stderr)
        return EXIT_USAGE

    try:
        from .app import run_app
    except ImportError as exc:  # pragma: no cover - depends on the Python build
        print(f"gravity-sim: error: the GUI needs tkinter ({exc}); use --headless", file=sys.stderr)
        return EXIT_USAGE
    return run_app(
        scenario,
        preset_key=args.preset,
        bodies=args.bodies,
        seed=args.seed,
        fullscreen=args.fullscreen,
        paused=args.paused,
        record_dir=args.record,
        record_frames=args.record_frames,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
