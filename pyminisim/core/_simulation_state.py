from dataclasses import dataclass
from typing import List, Dict

import numpy as np

from ._world_state import WorldState
from ._sensor import AbstractSensorReading


@dataclass
class SimulationState:
    world: WorldState
    sensors: Dict[str, AbstractSensorReading]
