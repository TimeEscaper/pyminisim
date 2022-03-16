from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ._motion import AbstractRobotMotionModelState
from ._pedestrians_model import AbstractPedestriansModelState


@dataclass
class WorldState:
    robot: Optional[AbstractRobotMotionModelState]
    pedestrians: Optional[AbstractPedestriansModelState]
    robot_to_pedestrians_collisions: Optional[List[int]]
