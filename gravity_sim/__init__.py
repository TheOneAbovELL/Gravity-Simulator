"""Interactive 3-D gravitational N-body simulator.

The physics core (:mod:`gravity_sim.simulation`, :mod:`gravity_sim.forces`,
:mod:`gravity_sim.octree`, :mod:`gravity_sim.collisions`) has no GUI
dependency and can be used directly::

    from gravity_sim import Simulation, build_scenario

    scenario = build_scenario("solar")
    sim = Simulation(scenario.bodies, scenario.config)
    sim.step(365)
    print(sim.energy().drift)
"""

from .bodies import BodySpec
from .presets import PRESETS, build_scenario
from .simulation import NumericalInstabilityError, Simulation, SimulationConfig

__version__ = "1.0.0"

__all__ = [
    "PRESETS",
    "BodySpec",
    "NumericalInstabilityError",
    "Simulation",
    "SimulationConfig",
    "__version__",
    "build_scenario",
]
