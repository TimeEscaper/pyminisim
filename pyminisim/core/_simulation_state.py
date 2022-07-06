from dataclasses import dataclass
from typing import List, Dict

import numpy as np

from ._world_state import WorldState
from ._sensor import SensorState


@dataclass
class SimulationState:
    world: WorldState
    sensors: Dict[str, SensorState]
